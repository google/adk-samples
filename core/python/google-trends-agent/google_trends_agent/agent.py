import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent

from google_trends_agent.prompt import load_agent_instructions
from google_trends_agent.tools import execute_bigquery_sql

# Construct the path to the .env file in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

# Load the environment variables from the .env file.
load_dotenv(dotenv_path)

# Now you can access your environment variables using os.getenv()

MODEL_AGENT = os.getenv("MODEL_AGENT", "gemini-2.5-pro")
MODEL_TOOL = os.getenv("MODEL_TOOL", "gemini-2.5-flash")

# --- Dynamically load agent instructions ---
full_instruction = load_agent_instructions()


# --- 1. Define Sub-Agents for the Pipeline ---

# Google Trends SQL Generator Agent
# Takes the user's question and generates a BigQuery SQL query.
trends_query_generator_agent = LlmAgent(
    name="TrendsQueryGeneratorAgent",
    model=MODEL_AGENT,
    instruction=full_instruction,
    description="Generates a BigQuery SQL query based on the user's question about Google Trends.",
    output_key="generated_sql",  # Stores output in state['generated_sql']
)

# Google Trends SQL Executor Agent
# Takes the generated SQL from the state and executes it using a tool.
trends_query_executor_agent = LlmAgent(
    name="TrendsQueryExecutorAgent",
    model=MODEL_TOOL,
    # This instruction tells the agent how to use the state and the tool.
    instruction="""You are a SQL execution agent.
Your task is to execute the BigQuery SQL query provided in the `{generated_sql}` placeholder.
Use the execute_bigquery_sql tool to run the query.
The query is already written; do not modify it. Simply pass it to the tool.
Read the query results and give insights to the user.
""",
    description="Executes the generated SQL query using the execute_bigquery_sql tool.",
    tools=[execute_bigquery_sql],
)

print(trends_query_executor_agent.instruction)

# --- 2. Create the SequentialAgent ---
# This agent orchestrates the pipeline by running the sub_agents in order.
root_agent = SequentialAgent(
    name="GoogleTrendsAgent",
    sub_agents=[trends_query_generator_agent, trends_query_executor_agent],
    description="""A two-step pipeline that first generates a SQL query for Google Trends and then executes it.
    Format the output as user friendly markdown format. Separete the SQL query and the interpretation of the results with a horizontal line.""",
)
