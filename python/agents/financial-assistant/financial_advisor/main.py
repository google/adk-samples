from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Assuming agent.py is in the same directory and financial_coordinator is the agent instance
from .agent import financial_coordinator

app = FastAPI(
    title="Financial Assistant API",
    description="API to interact with the Financial Assistant agent.",
    version="0.1.0",
)

class InvokeRequest(BaseModel):
    query: str
    # session_id: str | None = None # For future state management

class InvokeResponse(BaseModel):
    response: str | dict # Agent's response can be a string or structured data
    # session_id: str | None = None # For future state management

@app.post("/invoke", response_model=InvokeResponse)
async def invoke_agent(request: InvokeRequest):
    """
    Invoke the Financial Assistant agent with a query.
    """
    # For a single-turn interaction, we might not need complex state management yet.
    # The LlmAgent's invoke method typically handles the interaction.
    # The input format for financial_coordinator.invoke() needs to be checked.
    # It usually takes a dictionary of inputs.
    # Let's assume the agent expects input like: {"user_input": request.query}
    # or based on its prompt/instruction.
    #
    # From the prompt.py (not shown here, but typical for ADK agents),
    # the agent likely processes the 'user_input' or a similar key.
    # For simplicity, let's try passing the query directly or within a common key.

    # This is a guess; ADK agent invocation might require specific input keys
    # based on how the agent and its prompts are structured.
    # If `financial_coordinator.instruction` refers to `user_input`, then this should be `{"user_input": request.query}`
    # If the agent is designed to take direct string, it might just be `request.query`
    # Let's start by trying to pass the query as "user_input" in a dictionary,
    # as this is a common pattern for ADK agents.
    try:
        # ADK LlmAgent.invoke() typically expects a dictionary of inputs
        # The key "user_input" is a common convention for user queries in ADK.
        agent_inputs = {"user_input": request.query}
        agent_response_dict = await financial_coordinator.invoke(agent_inputs)

        # The response from agent.invoke() is a dictionary.
        # We need to extract the actual response text. The key is often 'agent_output' or the agent's name.
        # The financial_coordinator agent has output_key="financial_coordinator_output"
        response_content = agent_response_dict.get("financial_coordinator_output", "Error: Could not retrieve agent output.")
        
        return InvokeResponse(response=response_content)
    except Exception as e:
        # Log the exception details for debugging
        print(f"Error invoking agent: {e}") # Consider proper logging
        return InvokeResponse(response={"error": str(e)})

if __name__ == "__main__":
    # This is for local development/testing of the FastAPI app directly
    # The Docker container will use a different command to run uvicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)
