from google.adk.tools import ToolContext , FunctionTool
from google.adk.agents import Agent
from . import config


def check_condition_and_escalate_tool(tool_context: ToolContext) -> dict:
    """Checks the loop termination condition and escalates if met or max count reached."""
    print("\n--- Checker Tool running ---")

    # Increment loop iteration count using state [3]
    current_loop_count = tool_context.state.get('loop_iteration', 0)
    current_loop_count += 1
    tool_context.state['loop_iteration'] = current_loop_count # Update state [3]

    # Define maximum iterations
    max_iterations = config.MAX_ITERATIONS

    # Get the condition result set by the sequential agent from state [3]
    total_score = tool_context.state.get('total_score', False)

    condition_met= (total_score > config.SCORE_THRESHOLD)

    print(f"  Checking condition for loop iteration: {current_loop_count}/{max_iterations}")
    print(f"  Sequential process condition met: {condition_met}")

    response_message = f"Check iteration {current_loop_count}: Sequential condition met = {condition_met}. "

    # Check if the condition is met OR maximum iterations are reached
    if condition_met:
        print("  Condition met. Setting escalate=True to stop the LoopAgent.")
        tool_context.actions.escalate = True # Signal LoopAgent to terminate [4, 5]
        response_message += "Condition met, stopping loop."
    elif current_loop_count >= max_iterations:
        print(f"  Max iterations ({max_iterations}) reached. Setting escalate=True to stop the LoopAgent.")
        tool_context.actions.escalate = True # Signal LoopAgent to terminate [4, 5]
        response_message += "Max iterations reached, stopping loop."
    else:
        print("  Condition not met and max iterations not reached. Loop will continue.")
        response_message += "Loop continues."

    # Tool functions must return a dictionary [6]
    return {
        "status": "checked",
        "message": response_message
    }

check_tool_instance = FunctionTool(func=check_condition_and_escalate_tool)

# --- 4. Define an Agent that uses the checker tool ---
# This agent will be the second sub_agent in the LoopAgent's list.
checker_agent_instance = Agent(
    name="checker_agent",
    model=config.GENAI_MODEL, 
    instruction="Use the 'check_condition_and_escalate_tool' to evaluate if the loop should continue.",
    tools=[check_tool_instance] 
)

