from collections.abc import AsyncGenerator

from google.adk import Event
from google.adk.agents import InvocationContext
from google.adk.workflow import FunctionNode
from google.genai.types import Content
from ..tools import post_to_platform

# --- Workflow 2 Nodes: Blog and Publish ---

X_MAX_WORDS = 100
LINKEDIN_MAX_WORDS = 300


async def start_blog_node(
    ctx: InvocationContext,
) -> AsyncGenerator[Event | str, None]:
    """Entry node for the blog workflow, takes user thesis from Content object."""
    print("START_WORKFLOW 2  K: Blog post with thesis: ")
    thesis = ctx.state["research_report"]
    # thesis = str(node_input.parts[0].text if node_input.parts else "")
    print(f"START_WORKFLOW 2: Blog post with thesis: '{thesis}'")
    # The thesis is passed as the node_input to the next agent in the chain.
    yield thesis


async def length_router_node(
    node_input: str,
) -> AsyncGenerator[Event | str, None]:
    """Routes the blog post based on its word count."""
    blog_post = node_input
    num_words = len(blog_post.split())
    route = (
        "X"
        if num_words <= X_MAX_WORDS
        else "LINKEDIN"
        if num_words <= LINKEDIN_MAX_WORDS
        else "MEDIUM"
    )

    print(f"ROUTE: Post is {num_words} words. Routing to '{route}'.")
    yield blog_post  # Pass the post content along for the next nodes
    yield Event(route=route)


async def post_node(
    platform: str, node_input: str
) -> AsyncGenerator[Event | str, None]:
    """Posts the main article to a platform and yields a single Event with a list of routes for shoutouts."""
    blog_post = node_input
    await post_to_platform(platform, blog_post)

    routes_for_shoutouts = []
    if platform == "X":
        routes_for_shoutouts.extend(["SHOUTOUT_LINKEDIN", "SHOUTOUT_REDDIT"])
    elif platform == "LINKEDIN":
        routes_for_shoutouts.extend(["SHOUTOUT_X", "SHOUTOUT_REDDIT"])
    elif platform == "MEDIUM":
        routes_for_shoutouts.extend(["SHOUTOUT_X", "SHOUTOUT_LINKEDIN"])

    # Yield a single event with a list of routes
    if routes_for_shoutouts:
        yield Event(route=routes_for_shoutouts)

    # Also yield the blog post content for any potential downstream nodes that don't use routing
    yield blog_post


async def shoutout_node(platform: str, node_input: str) -> None:
    """Posts a shoutout to a given platform."""
    print(f"Shoutout to the following platform {platform} ")
    blog_post_preview = node_input[:40].strip()
    shoutout_msg = f"Check out my new article! '{blog_post_preview}...'"
    await post_to_platform(platform, shoutout_msg, shoutout=True)


# Function node Wrappers
start_blog = FunctionNode(func=start_blog_node, name="Start_Blog_Writing")
route_changer = FunctionNode(func=length_router_node, name="PathFinder")
