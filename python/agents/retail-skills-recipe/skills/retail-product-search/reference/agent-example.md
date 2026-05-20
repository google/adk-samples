# Example Agent Code

Example `agent.py` and `retrievers.py` implementations for a product search agent.
Adapt these to your specific industry, product fields, and configuration.

## agent.py

```python
"""Product Search Agent.

Tools: retrieve_docs (semantic search), clarify_query (vague queries),
       add_to_cart (optional)
"""

import os

import google
import vertexai
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.retrievers import search_collection

LLM_LOCATION = "global"
LOCATION = os.getenv("GCP_REGION", "us-central1")
LLM = "gemini-2.5-flash"

credentials, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = LLM_LOCATION
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

vertexai.init(project=project_id, location=LOCATION)

vector_search_collection = os.getenv(
    "VECTOR_SEARCH_COLLECTION",
    f"projects/{project_id}/locations/{LOCATION}/collections/YOUR_COLLECTION_NAME",
)


def retrieve_docs(query: str) -> str:
    """Search the product catalog using semantic search.

    Args:
        query: The user's product search query.

    Returns:
        Formatted string containing matching product information.
    """
    try:
        return search_collection(
            query=query,
            collection_path=vector_search_collection,
        )
    except Exception as e:
        return (
            f"Calling retrieval tool with query:\n\n{query}\n\n"
            f"raised the following error:\n\n{type(e)}: {e}"
        )


def clarify_query(original_query: str, clarification_type: str) -> str:
    """Ask a clarifying question when the query is too vague.

    Args:
        original_query: The user's original vague query.
        clarification_type: What to clarify (Price range, Category, Brand).

    Returns:
        A clarifying question to ask the user.
    """
    filter_questions = {
        "Price range": "What price range are you looking for?",
        "Category": "What category are you looking for?",
        "Brand": "What brand are you looking for?",
    }
    question = filter_questions.get(
        clarification_type,
        "Can you tell me more about what you're looking for?",
    )
    return f"I'd like to help you find the right product. {question}"


# Optional: Add to cart via Cloud Run API
def add_to_cart(product_id: str, product_name: str) -> str:
    """Add a product to the user's shopping cart.

    Args:
        product_id: The product ID to add.
        product_name: The product name (for confirmation).

    Returns:
        Confirmation message.
    """
    import requests
    from google.auth.transport.requests import Request

    cart_endpoint = os.getenv("CART_API_ENDPOINT", "")
    if not cart_endpoint:
        return f"Added {product_name} (ID: {product_id}) to your cart."

    auth_req = Request()
    credentials.refresh(auth_req)
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        cart_endpoint, json={"product_id": product_id}, headers=headers, timeout=10,
    )
    response.raise_for_status()
    return f"Added {product_name} to your cart."


tools = [retrieve_docs, clarify_query]
# Uncomment if using cart integration:
# tools.append(add_to_cart)

instruction = """You are a product search assistant.
Your job is to help users find products from the catalog.

## Search behavior
- Use the retrieve_docs tool to search for products matching the user's query.
- When a query is vague (e.g. "I need a gift"), use the clarify_query tool
  to ask about: Price range, Category, Brand (in that priority order).
- Ask at most 2 clarifying questions before searching.

## Results
- Present up to 6 products per response.
- For each product show: name, price, description, category, brand.
- Default sort: Relevance (similarity score).
- Do not show out-of-stock products.

## Filter memory
- Remember filters the user mentioned earlier in the conversation.
- Apply them to subsequent searches unless the user explicitly changes them.

## Constraints
- Only recommend products that exist in the catalog.
- Be accurate about prices and availability.
- Do not make up product details.
"""

root_agent = Agent(
    name="root_agent",
    model=Gemini(model=LLM, retry_options=types.HttpRetryOptions(attempts=3)),
    instruction=instruction,
    tools=tools,
)

app = App(root_agent=root_agent, name="app")
```

## retrievers.py

```python
"""Product search retriever using Vector Search 2.0."""

import os

from google.cloud import vectorsearch_v1beta


def search_collection(
    query: str,
    collection_path: str,
    top_k: int = 10,
) -> str:
    """Search a Vector Search 2.0 collection for products.

    Args:
        query: The product search query text.
        collection_path: Full resource path of the collection.
        top_k: Number of results to return.

    Returns:
        Formatted string with matching product information.
    """
    # For integration tests, return mock data
    if os.getenv("INTEGRATION_TEST") == "TRUE":
        return (
            "## Products found:\n"
            "<Product 0>\n"
            "Name: Sample Product | Price: $29.99 | Category: General\n"
            "</Product 0>"
        )

    client = vectorsearch_v1beta.DataObjectSearchServiceClient()

    request = vectorsearch_v1beta.SearchDataObjectsRequest(
        parent=collection_path,
        semantic_search=vectorsearch_v1beta.SemanticSearch(
            search_text=query,
            search_field="text_embedding",
            task_type="RETRIEVAL_QUERY",
            top_k=top_k,
            output_fields=vectorsearch_v1beta.OutputFields(
                data_fields=[
                    "product_id",
                    "name",
                    "price",
                    "description",
                    "category",
                    "brand",
                    "image_url",
                    "rating",
                    "stock",
                ]
            ),
        ),
    )

    results = client.search_data_objects(request)

    formatted_parts = []
    for i, result in enumerate(results):
        data = result.data_object.data
        name = data.get("name", "Unknown")
        price = data.get("price", "N/A")
        category = data.get("category", "")
        brand = data.get("brand", "")
        description = data.get("description", "")
        rating = data.get("rating", "")
        stock = data.get("stock", "")

        parts = [f"Name: {name}", f"Price: ${price}"]
        if category:
            parts.append(f"Category: {category}")
        if brand:
            parts.append(f"Brand: {brand}")
        if description:
            parts.append(f"Description: {description[:200]}")
        if rating:
            parts.append(f"Rating: {rating}/5")
        if stock is not None and stock != "":
            parts.append(f"In Stock: {'Yes' if int(stock) > 0 else 'No'}")

        formatted_parts.append(
            f"<Product {i}>\n" + " | ".join(parts) + f"\n</Product {i}>"
        )

    if not formatted_parts:
        return "No matching products found."

    return "## Products found:\n" + "\n".join(formatted_parts)
```
