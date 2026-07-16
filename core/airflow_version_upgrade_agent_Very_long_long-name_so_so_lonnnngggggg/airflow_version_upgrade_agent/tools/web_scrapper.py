# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import logging
import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import the Vertex AI Search (Discovery Engine) client
from google.cloud import discoveryengine_v1 as discoveryengine

# Define the path to the .env file
env_file_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_file_path)

# Fetch Data Store configs from env
location = os.getenv("VERTEX_SEARCH_LOCATION", "global")
app_id = os.getenv("VERTEX_AI_SEARCH_APP_ID")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_urls_from_vertex(
    query: str, project_id: str, location: str, app_id: str
) -> list[str]:
    """Get top URLs from Vertex AI Search"""
    try:
        client = discoveryengine.SearchServiceClient()
        serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{app_id}/servingConfigs/default_config"

        # We don't need summaries or snippets as it lacks details
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=3,  # Top 3 pages is usually enough context for Gemini
        )
        response = client.search(request)

        urls = []
        for result in response.results:
            struct_data = result.document.derived_struct_data
            if struct_data and "link" in struct_data:
                urls.append(struct_data["link"])
            elif result.document.uri:
                urls.append(result.document.uri)
        return urls
    except Exception as e:
        logging.error(f"Vertex Search Error: {e}")
        return []


def scrape_urls_content(urls: list[str]) -> str:
    """Scrapes the full text to ensure Gemini sees all code blocks and tables, returning JSON strings."""
    all_content = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for url in urls:
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            if not soup.body:
                logging.warning(
                    f"Could not find <body> tag in {url}. Skipping."
                )
                continue

            for element in soup(
                ["script", "style", "header", "footer", "nav", "aside"]
            ):
                element.decompose()

            content = " ".join(
                soup.body.get_text(separator=" ", strip=True).split()
            )
            safe_json_string = json.dumps({"url": url, "content": content})
            all_content.append(safe_json_string)
        except Exception as e:
            logging.error(f"Scraping failed for {url}: {e}")

    return "\n".join(all_content)


def research_operator_documentation(
    operators: list[str],
    source_version: str,
    target_version: str,
    project_id: str,
) -> list[dict]:
    results = []
    for operator in operators:
        query = f"Airflow {operator} migration parameters code examples {source_version} to {target_version}"
        urls = get_urls_from_vertex(query, project_id, location, app_id)

        if urls:
            full_text = scrape_urls_content(urls)
            results.append(
                {
                    "operator": operator,
                    "urls": urls,
                    "content": full_text,  # Full text passed to Gemini in build_bq_corpus
                }
            )
    return results


# For local testing
# if __name__ == '__main__':
#     # Example usage
#     test_operator = 'airflow.providers.google.cloud.operators.bigquery.BigQueryExecuteQueryOperator'
#     test_source_version = '1.10'
#     test_target_version = '2.10.5'

#     # Ensure these are loading correctly from your .env
#     test_project_id = os.getenv("PROJECT_ID")
#     test_location = os.getenv("VERTEX_SEARCH_LOCATION", "global")
#     test_app_id = os.getenv("VERTEX_AI_SEARCH_APP_ID")

#     if not test_project_id or not test_app_id:
#         print("ERROR: Missing PROJECT_ID or VERTEX_AI_SEARCH_APP_ID in .env file.")
#         exit(1)

#     print(f"Testing Vertex AI Search for operator: {test_operator}...")

#     # Test entire Flow
#     results_test = research_operator_documentation([test_operator], test_source_version, test_target_version, test_project_id)

#     if not results_test:
#         print(f"Could not find any documentation for {test_operator}.")
#     else:
#         print(f"Found {len(results_test[0]['urls'])} URLs.")
#         print("Writing to pages.txt and scraped_content.txt...")

#         with open("pages.txt", 'w', encoding='utf-8') as f:
#             for url in results_test[0]['urls']:
#                 f.write(f"{url}\n")

#         with open("scraped_content.txt", 'w', encoding='utf-8') as f:
#             f.write(results_test[0]['content'])

#         print("Done! Check pages.txt and scraped_content.txt for results.")
