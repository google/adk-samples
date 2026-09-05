import os

# Enable Vertex AI by default if no API Key is provided
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

from google.adk.agents import Agent, LlmAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.genai import types

from .prompts import (COMPLIANCE_AGENT_INSTRUCTIONS, SUPERVISOR_INSTRUCTIONS,
                      WORKER_AGENT_INSTRUCTIONS)


def query_sds(query: str) -> str:
    """Queries the Safety Data Sheets (SDS) for relevant chemical safety information.

    Args:
        query: The search query or question about chemical safety.

    Returns:
        The retrieved information or answer from the SDS documents.
    """
    query_lower = query.lower()

    # Mocked SDS Knowledge Base
    if "benzene" in query_lower or "carcinogen" in query_lower:
        return (
            "Document: Benzene (C6H6) SDS. CAS Number: 71-43-2. "
            "Section 2 Hazard Identification: Danger. Highly flammable liquid and vapor. Known human carcinogen (May cause cancer). "
            "Section 8 Exposure/PPE: Wear protective gloves, protective clothing, eye protection, and face protection. PEL: 1 ppm. "
            "Section 14 Transport: UN1114, Class 3, Packing Group II. "
            "Section 15 Regulatory: SARA Title III/313 listed. CERCLA RQ: 10 lbs. "
            "Section 13 Disposal: Dispose of contents/container as hazardous chemical waste."
        )
    elif "spill" in query_lower or "acid" in query_lower:
        return (
            "Document: Hydrochloric Acid (HCl) SDS. CAS: 7647-01-0. "
            "Section 2 Hazards: Danger. Corrosive. Causes severe skin burns and eye damage. "
            "Section 6 Accidental Release/Spill: Evacuate personnel. Do not touch spill. "
            "Neutralize spill immediately with sodium bicarbonate or soda ash. Contain and collect spillage with non-combustible absorbent material. "
            "Section 8 PPE: Nitrile Gloves, Safety Goggles (ANSI Z87.1), NIOSH respirator."
        )
        return "Recommended PPE for handling hazardous chemicals: Nitrile gloves, ANSI Z87.1 approved safety goggles, and NIOSH-approved organic vapor respirators when ventilation is insufficient."
    else:
        return "General Safety Note: Always consult the specific SDS for the chemical in question. Ensure adequate ventilation and wear appropriate PPE."


tool_query_sds = FunctionTool(func=query_sds)

# Define sub-agents
worker_agent = LlmAgent(
    name="workplace_safety_agent",
    instruction=WORKER_AGENT_INSTRUCTIONS,
    description="Specialist for on-site worker safety, protective equipment (PPE), first aid, and emergency response.",
    tools=[tool_query_sds],
    model="gemini-3-flash-preview",
)

compliance_agent = LlmAgent(
    name="regulatory_advisor_agent",
    instruction=COMPLIANCE_AGENT_INSTRUCTIONS,
    description="Specialist for SDS regulations, legal compliance, hazard classifications, and formal documentation.",
    tools=[tool_query_sds],
    model="gemini-3-flash-preview",
)

# Define root agent (Supervisor)
root_agent = LlmAgent(
    name="safety_supervisor",
    instruction=SUPERVISOR_INSTRUCTIONS,
    sub_agents=[worker_agent, compliance_agent],
    model="gemini-3-flash-preview",
)

app = App(
    root_agent=root_agent,
    name="hazmat_copilot_app",
)
