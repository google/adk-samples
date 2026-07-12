import asyncio
import random


async def execute_search(query: str) -> str:
    """Mocks executing a web search query."""
    print(f"RESEARCH: Executing query: '{query}'...")
    await asyncio.sleep(random.uniform(0.5, 1.0))

    result = ""
    if "linkedin" in query.lower():
        result = f"LinkedIn Summary: Professionals are discussing '{query}'."
    if "x" in query.lower() or "twitter" in query.lower():
        result = f"X Summary: The topic '{query}' is currently trending."
    if "reddit" in query.lower():
        result = (
            f"Reddit Summary: A subreddit has a popular thread about '{query}'."
        )
    if "medium" in query.lower():
        result = f"Medium Summary: Several deep-dive articles are available for '{query}'."
    else:
        result = f"General web summary for '{query}': It is a complex topic with varied opinions."

    return result.encode("utf-8", errors="ignore").decode("utf-8")


async def post_to_platform(
    platform: str, content: str, shoutout: bool = False
) -> None:
    """Mock posting to a platform."""
    prefix = "SHOUTOUT" if shoutout else "POST"
    print(
        f"PUBLISH [{prefix}]: Posting to {platform}: '{content[:60].strip()}...'"
    )
    await asyncio.sleep(0.5)
