import os

# Enable Vertex AI by default if no API Key is provided
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
    # Optionally set your project ID here if ADC doesn't pick it up:
    # os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"


from config import AGENT_MODEL, AGENT_NAME
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.tools import FunctionTool, ToolContext

# 1. CONFIGURATION & SETUP
from prompts import ROOT_AGENT_PROMPT, SALESFORCE_AGENT_PROMPT, SAP_AGENT_PROMPT

# 2. DEFINE MOCK TOOLS FOR SYSTEMS BRIDGE


# --- Salesforce (CRM) Mock Tools ---
def get_salesforce_opportunities() -> list[dict]:
    """Fetch open Salesforce Opportunities ready for quoting."""
    return [
        {
            "Id": "006Ws0000089yYpIAI",
            "Name": "ACME Corp - High Capacity Turbines",
            "StageName": "Proposal/Price Quote",
            "AccountId": "001Ws000053hK4jIAE",
            "Amount": 450000.00,
        }
    ]


def get_opportunity_line_items(opportunity_id: str) -> list[dict]:
    """Fetch line items (products) associated with a Salesforce Opportunity."""
    return [
        {
            "OpportunityId": opportunity_id,
            "ProductCode": "WT-GEN",
            "Quantity": 5,
            "UnitPrice": 90000.00,
        }
    ]


def create_salesforce_quote(opportunity_id: str) -> str:
    """Create a Quote record in Salesforce and generate a PDF."""
    return f"Quote CQ-900823 created successfully for Opportunity {opportunity_id}."


def update_opportunity_status(opportunity_id: str, status: str) -> str:
    """Update the StageName of a Salesforce Opportunity (e.g. to 'Closed Won')."""
    return f"Opportunity {opportunity_id} successfully updated to {status}."


tool_get_sf_opps = FunctionTool(func=get_salesforce_opportunities)
tool_get_sf_lines = FunctionTool(func=get_opportunity_line_items)
tool_create_sf_quote = FunctionTool(
    func=create_salesforce_quote, require_confirmation=True
)
tool_update_sf_opp = FunctionTool(func=update_opportunity_status)


# --- SAP (ERP) Mock Tools ---
def get_sap_credit_status(business_partner_id: str) -> dict:
    """Check the SAP Credit eligibility status and limit for a Business Partner."""
    return {
        "BusinessPartner": business_partner_id,
        "CreditRiskClass": "LOW",
        "CreditLimit": 1000000.00,
        "Status": "Green",
    }


def create_sap_sales_order(business_partner_id: str, items: list[dict]) -> str:
    """Create a Sales Order in SAP S/4HANA."""
    return (
        f"SAP Sales Order SO-2000159 created successfully for BP {business_partner_id}."
    )


tool_get_sap_credit = FunctionTool(func=get_sap_credit_status)
tool_create_sap_order = FunctionTool(
    func=create_sap_sales_order, require_confirmation=True
)

# --- Mapping Tools ---


# mock_sap_credit_status = FunctionTool(func=get_sap_credit_status)
def map_salesforce_account_to_sap_bp(salesforce_account_id: str) -> str:
    """
    Look up and return the SAP Business Partner ID mapped to a Salesforce Account ID.
    Always use this to translate IDs before fetching credit scores in SAP.
    """
    mapping = {
        "001Ws000053hK4jIAE": "1000329",
        "001Ws000053hK4YIAU": "1000330",
        "001Ws000053hK4HIAU": "1000331",
        "001Ws000053hK4OIAU": "1000332",
        "001Ws000053hK4VIAU": "1000333",
        "001Ws000053hK4QIAU": "1000334",
        "001Ws000053hK4dIAE": "1000335",
        "001Ws000053hK4LIAU": "1000336",
        "001Ws000053hK5MIAU": "1000337",
        "001Ws000053hK4mIAE": "1000338",
    }
    bp_id = mapping.get(salesforce_account_id)
    if not bp_id:
        return f"Error: No SAP Business Partner map found for Account ID '{salesforce_account_id}'"
    return bp_id


def map_salesforce_product_to_sap_material(salesforce_product_code: str) -> str:
    """
    Look up and return the SAP Material ID mapped to a Salesforce Product Code.
    Use this to translate product codes during quotation or materials availability check.
    """
    mapping = {
        "B-1000": "TG10",
        "A-100": "TG11",
        "A-1000": "TG12",
        "E-1000": "TG13",
        "A-5000": "TG14",
        "H-36378": "TG15",
        "C-300": "TG16",
        "C-45678": "TG17",
        "M-1000": "TG18",
        "R-100": "TG19",
        "L-1000": "TG20",
        "INST-100": "TG21",
        "L-100": "TG22",
        "S-100": "TG0011",
        "WT-GEN": "TG0012",
        "V-1000": "TG0013",
    }
    material_id = mapping.get(salesforce_product_code)
    if not material_id:
        return f"Error: No SAP Material map found for Product Code '{salesforce_product_code}'"
    return material_id


tool_map_salesforce_product_to_sap_material = FunctionTool(
    func=map_salesforce_product_to_sap_material
)
tool_map_salesforce_account_to_sap_bp = FunctionTool(
    func=map_salesforce_account_to_sap_bp
)

# 4. MULTI-AGENT SUB-SYSTEM DEFINITION

# 3. MULTI-AGENT SUB-SYSTEM DEFINITION

# The Salesforce Agent specializes in CRM operations
salesforce_agent = LlmAgent(
    name="SalesforceAgent",
    model=AGENT_MODEL,
    instruction=SALESFORCE_AGENT_PROMPT,
    tools=[
        tool_get_sf_opps,
        tool_get_sf_lines,
        tool_create_sf_quote,
        tool_update_sf_opp,
        tool_map_salesforce_account_to_sap_bp,
        tool_map_salesforce_product_to_sap_material,
    ],
)

# The SAP Agent specializes in ERP operations
sap_agent = LlmAgent(
    name="SapAgent",
    model=AGENT_MODEL,
    instruction=SAP_AGENT_PROMPT,
    tools=[
        tool_get_sap_credit,
        tool_create_sap_order,
        tool_map_salesforce_account_to_sap_bp,
        tool_map_salesforce_product_to_sap_material,
    ],
)


# --- AgentMail (Manager Approval) Mock Tools ---
def send_email(to: str, subject: str, text: str) -> str:
    """Sends an email to a manager requesting quote approval."""
    print(f"📨 [MOCK EMAIL SENT to {to}]: {subject} - {text}")
    return "Email sent successfully. The manager has responded: APPROVED."


def list_messages() -> str:
    """Check the inbox for manager approval responses."""
    return "Manager Reply: Looks good! APPROVED."


tool_agentmail_send_email = FunctionTool(func=send_email)
tool_agentmail_list_messages = FunctionTool(func=list_messages)


def get_user_identity(tool_context: ToolContext) -> str:
    """Retrieve the identity of the user currently executing the agent."""
    user_id = tool_context.user_id
    # If user_id is an email, strip the domain
    user_name = user_id.split("@")[0] if user_id and "@" in user_id else user_id
    return f"The current executing user is: {user_name}"


def get_manager_email(tool_context: ToolContext) -> str:
    """Dynamically resolve the manager's email address for approvals based on the current user's identity."""
    user_id = tool_context.user_id
    user_name = user_id.split("@")[0] if user_id and "@" in user_id else user_id

    mapping = {
        "example_user1": "example_user1@example.com",
        "example_user2": "example_user2@example.com",
        "example_user3": "example_user3@example.com",
        "user": "example_user4@example.com",
    }

    manager_email = mapping.get(user_name)
    if not manager_email:
        manager_email = user_id

    return manager_email


tool_get_user_identity = FunctionTool(func=get_user_identity)
tool_get_manager_email = FunctionTool(func=get_manager_email)

root_agent = LlmAgent(
    name=AGENT_NAME,
    model=AGENT_MODEL,
    instruction=ROOT_AGENT_PROMPT,
    sub_agents=[salesforce_agent, sap_agent],
    # Coordinator holds approvals, mappings & Mail
    tools=[
        tool_map_salesforce_account_to_sap_bp,
        tool_map_salesforce_product_to_sap_material,
        tool_agentmail_send_email,
        tool_agentmail_list_messages,
        tool_get_user_identity,
        tool_get_manager_email,
    ],
)

# Create the Application
app = App(name="smart_closer_app", root_agent=root_agent)
