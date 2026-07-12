from collections.abc import AsyncGenerator

from google.adk import Event
from google.adk.workflow import FunctionNode
from google.genai.types import Content, ModelContent, Part


async def start_research_node(
    node_input: Content,
) -> AsyncGenerator[Event | list[str], None]:
    """Entry node for the research workflow. Puts the topic in state and yields a list of platforms to research."""
    topic = str(node_input.parts[0].text if node_input.parts else "")
    print(f"START_WORKFLOW 1: Research for topic: '{topic}'")
    yield Event(state={"topic": topic})

    platforms_to_research = ["X", "LinkedIn", "Reddit", "Medium"]
    yield platforms_to_research


async def combine_reports_node(
    node_input: Content,
) -> AsyncGenerator[str, None]:
    """Takes the Content object from parallel agents and joins their text parts into a single string."""
    if node_input.parts is None:
        yield "No reports received from"
    else:
        print(f"DEBUG-ENTIRE node_inut:\n{node_input}")
        report_texts = []
        for part in node_input.parts:
            if part.text:
                report_texts.append(part.text)

        yield "\n\n---\n\n".join(report_texts)


async def save_report_node(
    node_input: str,
) -> AsyncGenerator[Event|Content , None]:
    """Saves the generated report to state and yields it for the user."""
    print(f"STATE_UPDATE: Saving generated report to session state. \n\n {node_input}")
    yield Event(state={"research_report": node_input})
    yield Content(parts=[Part.from_text(text=node_input)])
    print("Finisheed")


# Node Wrappers
start_node = FunctionNode(
   func=start_research_node, name="Start_Research_Node", rerun_on_resume=True
)

combine_reports = FunctionNode(func=combine_reports_node, name="Combine_Reports")

save_node = FunctionNode(
    func=save_report_node, name="Save_Report_Node", rerun_on_resume=False
)
