import os

import requests


def web_search_skill(query: str) -> str:
    """
    Live web search using Serper.dev API.
    Ensure SERPER_API_KEY is defined in your environment or .env file.
    """
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        return "⚠️ Error: SERPER_API_KEY not found in environment. Please add it to your .env file."

    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            results = response.json().get("organic", [])
            if not results:
                return f"[Serper] No organic results found for '{query}'."
            summaries = [
                f"- {res.get('title')}: {res.get('snippet')}"
                for res in results[:3]
            ]
            return "### 🔍 Live Google Search Results (Serper):\n" + "\n".join(
                summaries
            )
        return f"[Serper Error] Failed to fetch search results. HTTP Status {response.status_code}."
    except Exception as e:
        return f"[Serper Request Failed] {e}"
