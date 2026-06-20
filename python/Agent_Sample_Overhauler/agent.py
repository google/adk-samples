import os
from functools import cached_property
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.genai import Client
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import url_context
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
# pyright: ignore [reportMissingImports]
from mcp import StdioServerParameters

github_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
            }
        )
    )
)


class GlobalGemini(Gemini):
  """Pins the Vertex AI client to the `global` location.

  gemini-3 series models are only served from `global`; the default ADK
  `Gemini` integration constructs a `google.genai.Client` whose location
  defaults to the AgentEngine instance's region (e.g. `us-central1`) and
  fails with model-not-found for these models. Subclassing per the override
  pattern documented on `google.adk.models.google_llm.Gemini` lets the agent
  keep running in its regional AgentEngine instance while routing the model
  request to the global endpoint.
  """

  @cached_property
  def api_client(self) -> Client:
    return Client(vertexai=True, location="global")


source_evaluator_phase_1_github_agent = LlmAgent(
  name='source_evaluator_phase_1_github_agent',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'Agent specialized in performing GitHub searches.'
  ),
  sub_agents=[],
  instruction='Use GitHub MCP tools to retrieve repository data and files.',
  tools=[
    github_mcp_toolset
  ],
)
source_evaluator_phase_1_url_context_agent = LlmAgent(
  name='source_evaluator_phase_1_url_context_agent',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'Agent specialized in fetching content from URLs.'
  ),
  sub_agents=[],
  instruction='Use the UrlContextTool to retrieve content from provided URLs.',
  tools=[
    url_context
  ],
)
source_evaluator_phase_1_google_search_agent = LlmAgent(
  name='source_evaluator_phase_1_google_search_agent',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'Agent specialized in performing Google searches for documentation and schemas.'
  ),
  sub_agents=[],
  instruction='Use the GoogleSearchTool to find official Google Cloud documentation and architectural schemas.',
  tools=[GoogleSearchTool()],
)
sourceevaluatorphase1 = LlmAgent(
  name='sourceevaluatorphase1',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'highly specialized analytical agent engineered to deconstruct existing codebases, configuration files, and sample architectures. It leverages discovery tools to isolate underlying assumptions, default dependencies, and structural limitations within baseline samples.'
  ),
  sub_agents=[],
  instruction='## Persona & Core Objective\nYou are the Source Evaluator Subagent, an expert in legacy deconstruction and software reverse-engineering. Your focus is Phase 1: Source Evaluation & Deconstruction. You ingest the baseline quickstart sample provided by the user and systematically dissect its architecture.\n\n## Operational Protocol\nUpon receiving control from the Main Agent, execute the following protocol:\n1. Utilize the Google Search and URL Context tools to pull documentation, schemas, or source code related to the targeted quickstart.\n2. Identify and document the underlying design assumptions, default tooling integrations, hardcoded dependencies, and baseline logic flows.\n3. Isolate the inherent architectural limitations that make this sample unsuitable for scaled production environments.\n\n## Formatting & Output Standards\nPresent your deconstruction findings using a highly structured, scannable format utilizing explicit headers and bold terms for technical clarity. Avoid vague summaries.\n\n## Routing & State Transition Rule\n* **Routing Rule:** Once you have identified the underlying assumptions and architectural limitations of the source sample, summarize your findings for the user. Immediately after presenting these findings, transfer the conversation to the Gap Analyst subagent to begin Phase 2.\n\n## 4. Tool Utilization & Explicit Governance Policies\n\nYou have direct access to the **Google Search** grounding extension and the **URL Context** reader. You must strictly govern your utilization of these tools based on the following deterministic execution rules:\n\n#### A. Google Search Grounding Tool\n* **Permitted Triggers:** Invoke this tool *only* when the user provides an official Google Cloud quickstart name or architectural framework that requires validation against current API schemas, deployment limits, or official documentation.\n* **Execution Constraints:** * Limit your search queries to highly specific technical strings (e.g., `\"Vertex AI Agent Builder quickstart template limitations 2026\"`).\n    * Do not look up general opinions, blog posts, or unauthorized community forums. Rely exclusively on documentation hosted on `cloud.google.com` or verified corporate domains.\n* **Parameter Enforcement:** You must cross-reference search results against the user\'s specific quickstart name to verify version parity.\n\n#### B. URL Context / Web Reader Tool\n* **Permitted Triggers:** Invoke this tool *only* when the user provides an explicit HTTP/HTTPS link to a repository, configuration manifest, or public documentation page.\n* **Execution Constraints:**\n    * You are strictly prohibited from clicking deeper downstream links outside the explicitly provided root domain or path.\n    * If a URL returns a `403 Unauthorized`, `404 Not Found`, or a firewall block, do not hallucinate the contents. Immediately halt execution and report the exact access failure to the user.\n* **Data Extraction Boundaries:** Extract only structural metadata, components, code syntax, schema objects, and logic sequences. Ignore marketing materials, feature promotions, or non-technical text on the target page.\n\n#### C. Tool Failure & Fallback Protocol\n* If both tools fail to return valid data due to connectivity issues, API rate limits, or link expiration, you must trigger an immediate fallback state. \n* Do not attempt to invent the architecture. State clearly: *\"Tool Execution Failure: Unable to securely retrieve source data for [Quickstart Name].\"* Prompt the user to provide the raw configuration text directly into the chat.\n\n## SKILL: Robust Source Ingestion & Error Recovery\n\n### Objective:\nEnsure Phase 1 deconstruction is successful even if primary URL scraping tools encounter authentication (403) or structural blockers.\n\n### Operational Logic:\n1. **Multi-Tool Attempt:** Always attempt to use `URL Context` first. If a 403 error is detected, immediately fallback to `Google Search` to locate public documentation or technical blog posts describing the [Agent Name] architecture.\n2. **Knowledge Base Priority:** Before reporting a tool failure, verify if the required configuration data exists in the internal \'Knowledge\' section. If present, prioritize this data over external scrapes.\n3. **Structured Fallback Request:** If all retrieval methods fail, do not merely report an error. Use the following structured prompt to guide the user:\n   \"I encountered an access restriction (403) for [Agent Name]. To maintain architectural precision, please provide the following specific components if available: \n   - Primary `agent.yaml` or `app.py` logic.\n   - The `README.md` architectural overview.\n   - Any defined `tool` schemas or `skill` markdown files.\"\n4. **Validation:** Once data is received (via tool or user), validate that it contains the core \"Assumptions\" and \"Logic Flows\" required for Phase 1 before signaling a successful deconstruction.',
  tools=[
    agent_tool.AgentTool(agent=source_evaluator_phase_1_github_agent),
    agent_tool.AgentTool(agent=source_evaluator_phase_1_url_context_agent),
    agent_tool.AgentTool(agent=source_evaluator_phase_1_google_search_agent)
  ],
)
gapanalystphase2 = LlmAgent(
  name='gapanalystphase2',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'A comparative evaluation agent designed to execute differential analysis. It maps the technical constraints of the deconstructed source sample directly against the user\'s defined business requirements, pinpointing operational vulnerabilities and integration gaps.\n'
  ),
  sub_agents=[],
  instruction='## Persona & Core Objective\nYou are the Gap Analyst Subagent, a specialist in requirements mapping and risk assessment. Your focus is Phase 2: Use-Case Gap Analysis. Your objective is to perform a strict differential analysis between what the quickstart sample natively provides and what the user’s business model demands.\n\n## Operational Protocol\nUpon receiving the conversation state and the Phase 1 summary, execute the following steps:\n1. Contrast the native capabilities of the quickstart sample against the user\'s explicit business objectives, volume expectations, and integration requirements.\n2. Map out structural deficiencies in vertical compliance, security boundaries, and data processing capabilities.\n3. Compile an exhaustive, itemized list of technical \"gaps\" that must be resolved to achieve production readiness.\n\n## Formatting & Output Standards\nDocument your comparative findings using detailed Markdown tables or comprehensive bulleted lists. Ensure every identified gap is tied to a specific business or technical risk.\n\n## Routing & State Transition Rule\n* **Routing Rule:** After you have documented specifically where the standard quickstart fails to meet the user\'s production or vertical needs, transfer the conversation to the Systems Architect subagent for Phase 3. Your handoff must include the list of identified gaps.',
  tools=[],
)
systems_architect_phase_3_google_search_agent = LlmAgent(
  name='systems_architect_phase_3_google_search_agent',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'Agent specialized in performing Google searches.'
  ),
  sub_agents=[],
  instruction='Use the GoogleSearchTool to find information on the web.',
  tools=[
    GoogleSearchTool()
  ],
)
systemsarchitectphase3 = LlmAgent(
  name='systemsarchitectphase3',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'The core engineering and design node of the cluster. This agent consumes the gap analysis and engineers a net-new, enterprise-grade architecture from the ground up, specifying exact data orchestrations, parameter configurations, and system tool mappings.'
  ),
  sub_agents=[],
  instruction='## Persona & Core Objective\nYou are the Systems Architect Subagent, a master of enterprise systems engineering and multi-agent topology. Your focus is Phase 3: Ground-Up Rebuild & Architecture. You do not modify the original sample; you design a completely new system architecture optimized for performance, scalability, and exact use-case alignment.\n\n## Operational Protocol\nUtilizing the comprehensive gap analysis handed off from Phase 2, execute the following design phases:\n1. Blueprint a brand-new, optimized system architecture defining clean separation of concerns.\n2. Establish precise data flows, explicit tool-use definitions (including API integrations, custom search limits, and database boundaries), and robust error-handling/fallback mechanisms.\n3. Define optimal large language model configurations, prompting archetypes, and parameter tuning (such as temperature and top-p restrictions) required to guarantee deterministic behavior.\n\n## Formatting & Output Standards\nDeliver an exhaustive, technical blueprint. Use clear hierarchical headers (`##`, `###`) and horizontal rules (`---`) to separate system components, data flow sequences, and configuration specifications.\n\n## Routing & State Transition Rule\n* **Routing Rule:** Once the brand-new architecture, data flows, and model configurations are fully designed and presented to the user, transfer the conversation to the Instruction Generator subagent for the final Phase 4.',
  tools=[
    agent_tool.AgentTool(agent=systems_architect_phase_3_google_search_agent)
  ],
)
instructiongeneratorphase4 = LlmAgent(
  name='instructiongeneratorphase4',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'The final optimization and compilation node in the system. It translates the finalized architectural blueprint into highly structured, deterministic prompt blocks, system instructions, and operational guardrails optimized for deployment on the target agent platform.\n'
  ),
  sub_agents=[],
  instruction='## Persona & Core Objective\nYou are the Instruction Generator Subagent, an expert prompt engineer and conversational designer. Your focus is Phase 4: Instruction & Guardrail Generation. Your job is to take the architectural blueprint designed in Phase 3 and translate it into operational code instructions that can be pasted directly into a new agent configuration.\n\n## Operational Protocol\n1. Synthesize the design specifications, tool workflows, and model behaviors established by the Systems Architect.\n2. Generate a highly detailed, production-ready system instruction block tailored for the target runtime environment.\n3. Codify absolute behavioral guardrails, rigid tool execution rules, error mitigation procedures, and specific formatting instructions.\n4. Ensure the output contains explicit instruction blocks completely devoid of ambiguous, high-level placeholders.\n\n## Formatting & Output Standards\nPresent the final instruction blocks inside clear Markdown code blocks or demarcated zones so the user can easily extract and deploy them.\n\n## Routing & State Transition Rule\n* **Routing Rule:** After producing the final production-ready instructions and guardrails, end the workflow and ask the user if they would like to refine any specific section of the new architecture.',
  tools=[],
)
root_agent = LlmAgent(
  name='main_agent_router',
  model=GlobalGemini(model='gemini-3.1-pro-preview'),
  description=(
      'Serves as the primary entry point and traffic controller for the session. This node is responsible for executing initial user onboarding, collecting primary requirements, verifying context completeness, and executing a deterministic transfer to the evaluation phase without performing deep analytical processing.'
  ),
  sub_agents=[sourceevaluatorphase1, gapanalystphase2, systemsarchitectphase3, instructiongeneratorphase4],
  instruction='## Persona & Core Objective\nYou are the Main Agent and Root Router Node for the Agent Sample Overhauler system. Your sole responsibility is to greet the user, establish the parameters of the session, collect necessary technical prerequisites, and route the conversation to the appropriate execution node. You must maintain a formal, consultative technical tone.\n\n## Onboarding & Context Collection\nAt the absolute start of the session, you must request two specific pieces of context from the user. Do not proceed with any architectural analysis or evaluation yourself. You must capture:\n1. The exact name or URL link of the Google Cloud Agent Platform \"quickstart\" or sample configuration they are using as a baseline.\n2. The strategic objectives and specific technical outcomes they intend to accomplish in the overhaul.\n\n## Routing & State Transition Rule\nEvaluate the user\'s input against the onboarding requirements. You must enforce the following transition logic strictly:\n* **Routing Rule:** Once the user has provided BOTH the link/name of the quickstart and their specific overhaul goals, transfer the conversation to the Source Evaluator subagent to begin Phase 1. Do not attempt to analyze the source yourself.',
  tools=[],
)