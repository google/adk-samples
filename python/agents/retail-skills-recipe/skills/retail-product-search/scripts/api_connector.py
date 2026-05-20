#!/usr/bin/env python3
"""Connect to an existing product search API (Path A).

Wraps an external product search API so the agent can call it as a tool.
Handles authentication, pagination, rate limiting, and error retries.

Usage:
    # Test the API connection
    python api_connector.py \
        --project-id my-project \
        --api-url https://service-xyz.a.run.app/v2 \
        --search-endpoint /products/search \
        --query "wireless headphones"

    # Using design-spec.md for defaults
    python api_connector.py \
        --config design-spec.md \
        --query "laptop under 1000"
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

# Add _shared to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
from setup_utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default settings -- override via CLI args or design-spec.md
DEFAULT_AUTH_METHOD = "google_oauth"
DEFAULT_SEARCH_METHOD = "GET"
DEFAULT_SEARCH_ENDPOINT = "/products/search"
DEFAULT_PAGINATION = "page_limit"
DEFAULT_TIMEOUT = 10
DEFAULT_MAX_RETRIES = 1


def get_auth_headers(auth_method: str) -> dict:
    """Get authentication headers based on method.

    Supported: google_oauth, api_key, bearer, iap, none.
    """
    if auth_method == "google_oauth" or auth_method == "Google OAuth (service account)":
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return {"Authorization": f"Bearer {credentials.token}"}

    elif auth_method == "bearer" or auth_method == "Bearer token":
        token = os.environ.get("API_BEARER_TOKEN", "")
        if not token:
            logger.warning("API_BEARER_TOKEN env var not set")
        return {"Authorization": f"Bearer {token}"}

    elif auth_method == "api_key" or auth_method == "API key via API Gateway":
        key = os.environ.get("API_KEY", "")
        if not key:
            logger.warning("API_KEY env var not set")
        return {"x-api-key": key}

    elif auth_method == "iap" or auth_method == "Identity-Aware Proxy (IAP)":
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return {"Authorization": f"Bearer {credentials.token}"}

    return {}


def search_api(
    api_url: str,
    search_endpoint: str,
    query: str,
    http_method: str = "GET",
    auth_method: str = "none",
    query_params: str = "",
    pagination_style: str = "none",
    page: int = 1,
    limit: int = 10,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """Call the external product search API.

    Args:
        api_url: Base URL of the API.
        search_endpoint: Search endpoint path (e.g. /products/search).
        query: Search query string.
        http_method: GET or POST.
        auth_method: Authentication method.
        query_params: Comma-separated parameter names the API accepts.
        pagination_style: page_limit, offset_count, cursor, or none.
        page: Page number (for page_limit pagination).
        limit: Results per page.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on 5xx errors.

    Returns:
        API response as dict with 'products' list and 'total' count.
    """
    url = f"{api_url.rstrip('/')}{search_endpoint}"
    headers = get_auth_headers(auth_method)
    headers["Content-Type"] = "application/json"

    # Build query parameters
    params = {"q": query}

    if pagination_style == "page_limit" or pagination_style == "page + limit":
        params["page"] = page
        params["limit"] = limit
    elif pagination_style == "offset_count" or pagination_style == "offset + count":
        params["offset"] = (page - 1) * limit
        params["count"] = limit

    # Make request with retry
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if http_method.upper() == "POST":
                response = requests.post(
                    url, json=params, headers=headers, timeout=timeout,
                )
            else:
                response = requests.get(
                    url, params=params, headers=headers, timeout=timeout,
                )

            if response.status_code == 200:
                return _parse_response(response.json())

            if response.status_code == 404:
                return {"products": [], "total": 0}

            if response.status_code >= 500 and attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    f"API returned {response.status_code}, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(wait)
                continue

            response.raise_for_status()

        except requests.exceptions.Timeout:
            last_error = f"Request timed out after {timeout}s"
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

    logger.error(f"API call failed after {max_retries + 1} attempts: {last_error}")
    return {"products": [], "total": 0, "error": last_error}


def _parse_response(data: dict | list) -> dict:
    """Normalize API response into a standard format."""
    if isinstance(data, list):
        return {"products": data, "total": len(data)}

    # Handle common response shapes
    products = (
        data.get("products")
        or data.get("results")
        or data.get("items")
        or data.get("data")
        or []
    )
    total = data.get("total") or data.get("total_count") or len(products)

    return {"products": products, "total": total}


def search_all_pages(
    api_url: str,
    search_endpoint: str,
    query: str,
    max_pages: int = 5,
    **kwargs,
) -> list[dict]:
    """Fetch all pages of results from the API."""
    all_products = []

    for page in range(1, max_pages + 1):
        result = search_api(
            api_url=api_url,
            search_endpoint=search_endpoint,
            query=query,
            page=page,
            **kwargs,
        )

        products = result.get("products", [])
        if not products:
            break

        all_products.extend(products)

        total = result.get("total", 0)
        if len(all_products) >= total:
            break

    return all_products


def main():
    parser = argparse.ArgumentParser(
        description="Search products via external API (Path A)"
    )
    parser.add_argument("--config", default="", help="Path to design-spec.md")
    parser.add_argument("--api-url", help="API base URL")
    parser.add_argument("--search-endpoint", default=DEFAULT_SEARCH_ENDPOINT, help="Search endpoint path")
    parser.add_argument("--http-method", default=DEFAULT_SEARCH_METHOD, choices=["GET", "POST"])
    parser.add_argument("--auth-method", default=DEFAULT_AUTH_METHOD, help="Auth: google_oauth, api_key, bearer, iap, none")
    parser.add_argument("--query-params", default="", help="Comma-separated query parameter names")
    parser.add_argument("--pagination", default="none", help="Pagination: page_limit, offset_count, cursor, none")
    parser.add_argument("--query", required=True, help="Search query to test")
    parser.add_argument("--limit", type=int, default=10, help="Results per page")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout (seconds)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries on 5xx")
    parser.add_argument("--all-pages", action="store_true", help="Fetch all pages")

    args = parser.parse_args()

    # Load design-spec.md defaults
    if args.config:
        cfg = load_config(args.config)
        if not args.api_url:
            args.api_url = cfg.get("api_base_url", "")
        if args.search_endpoint == DEFAULT_SEARCH_ENDPOINT and cfg.get("api_search_endpoint"):
            args.search_endpoint = cfg["api_search_endpoint"]
        if args.http_method == DEFAULT_SEARCH_METHOD and cfg.get("api_search_method"):
            args.http_method = cfg["api_search_method"]
        if args.auth_method == DEFAULT_AUTH_METHOD and cfg.get("api_auth_method"):
            args.auth_method = cfg["api_auth_method"]
        if not args.query_params and cfg.get("api_query_params"):
            args.query_params = cfg["api_query_params"]
        if args.pagination == "none" and cfg.get("api_pagination"):
            args.pagination = cfg["api_pagination"]

    if not args.api_url:
        parser.error("--api-url is required (or set api_base_url in design-spec.md)")

    logger.info(f"Searching: {args.api_url}{args.search_endpoint}")
    logger.info(f"Query: {args.query}")
    logger.info(f"Auth: {args.auth_method}")

    if args.all_pages:
        products = search_all_pages(
            api_url=args.api_url,
            search_endpoint=args.search_endpoint,
            query=args.query,
            http_method=args.http_method,
            auth_method=args.auth_method,
            pagination_style=args.pagination,
            limit=args.limit,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        print(json.dumps(products, indent=2))
        logger.info(f"Found {len(products)} products (all pages)")
    else:
        result = search_api(
            api_url=args.api_url,
            search_endpoint=args.search_endpoint,
            query=args.query,
            http_method=args.http_method,
            auth_method=args.auth_method,
            query_params=args.query_params,
            pagination_style=args.pagination,
            limit=args.limit,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        print(json.dumps(result, indent=2))
        logger.info(f"Found {result.get('total', 0)} products")


if __name__ == "__main__":
    main()
