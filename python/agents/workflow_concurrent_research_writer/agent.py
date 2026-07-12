# import asyncio
from functools import partial

# from uuid import uuid4
from google.adk.workflow import START

# from google.adk.agents.workflow.join_node import JoinNode
from google.adk.workflow import FunctionNode
from google.adk.workflow import Workflow

# from google.adk.agents.run_config import RunConfig
# from google.adk.sessions.in_memory_session_service import InMemorySessionService
# from google.adk.sessions.session import Session
# from google.adk.agents.invocation_context import InvocationContext
# from google.genai.types import Content, ModelContent, Part
# Local sub-agent and node imports
from .agent_nodes.publishing import generate_blog_post_agent
from .agent_nodes.research import (
    distill_agent,
    parallel_research,
)
from .function_nodes.publishing import (
    post_node,
    route_changer,
    shoutout_node,
    start_blog,
)
from .function_nodes.research import save_node, start_node

# --- 1. Workflow Definitions ---

# Research Workflow: A simple, linear chain. The `research_worker_agent`
# is marked with `parallel_worker=True` so the framework will automatically
# handle fanning out for each query and fanning in the results.
research_workflow = Workflow(
    name="research_workflow",
    edges=[
        (
            START,
            start_node,
            parallel_research,
            distill_agent,
            save_node,
        ),
    ],
)

# Blog Workflow
# Nodes for posting the main article
post_to_x = FunctionNode(func=partial(post_node, "X"), name="Post_to_X")
post_to_linkedin = FunctionNode(
    func=partial(post_node, "LINKEDIN"), name="Post_to_LinkedIn"
)
post_to_medium = FunctionNode(
    func=partial(post_node, "MEDIUM"), name="Post_to_Medium"
)

# Nodes for posting shoutouts
shoutout_to_x = FunctionNode(func=partial(shoutout_node, "X"), name="Shoutout_to_X")
shoutout_to_linkedin = FunctionNode(
    func=partial(shoutout_node, "LINKEDIN"), name="Shoutout_to_LinkedIn"
)
shoutout_to_medium = FunctionNode(
    func=partial(shoutout_node, "MEDIUM"), name="Shoutout_to_Medium"
)
shoutout_to_reddit = FunctionNode(
    func=partial(shoutout_node, "REDDIT"), name="Shoutout_to_Reddit"
)

blog_workflow = Workflow(
    name="blog_workflow",
    edges=[
        # 1. Start, write blog, then route by length
        (START, start_blog, generate_blog_post_agent, route_changer),
        # 2. Post to the primary platform based on the route from route_changer
        (route_changer,
            {
                "X": post_to_x,
                "LINKEDIN": post_to_linkedin,
                "MEDIUM": post_to_medium,
            },
        ),
        # 3. From each primary post, trigger shoutouts based on the new objective rules.
        # If posted to X -> Shoutout to LinkedIn and Reddit
        (
            post_to_x, 
            {
                "SHOUTOUT_LINKEDIN":shoutout_to_linkedin,
                "SHOUTOUT_REDDIT":shoutout_to_reddit,
            },
        ),

        # If posted to LinkedIn -> Shoutout to X and Reddit
        (
            post_to_linkedin,
            {
                "SHOUTOUT_X":shoutout_to_x,
                "SHOUTOUT_REDDIT":shoutout_to_reddit,
            },
        ),
        # If posted to Medium -> Shoutout to X and LinkedIn
        (
            post_to_medium,
            {
                "SHOUTOUT_X":shoutout_to_x,
                "SHOUTOUT_LINKEDIN": shoutout_to_linkedin
            }

        ),        
    ],
)

root_agent = Workflow(
    name="root_agent",
    description="""
        Main workflow contucting the research and pubication phases of blog 
        publication and advertisement.
    """,
    rerun_on_resume=True,
    edges=[("START", research_workflow, blog_workflow)],
)
