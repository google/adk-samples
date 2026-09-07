#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""
Economic Research Agent (ERA) - ADK 2.0 Implementation.
Replaces LangChain/LangGraph with native Vertex AI Agent Development Kit.
"""

import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini

# Specialized Skill Imports
from economic_research.tools.bea_skill import fetch_bea_regional_data
from economic_research.tools.bls_skill import (
    labor_force_stats_skill,
    median_hourly_wages_skill,
    state_tax_rate_skill,
    state_union_employment_skill,
)
from economic_research.tools.census_skill import fetch_census_education_stats
from economic_research.tools.eia_skill import fetch_state_electricity_rates
from economic_research.tools.fred_skill import fetch_regional_macro_stats
from economic_research.tools.hud_skill import (
    analyze_housing_affordability,
    fetch_hud_fmr_data,
    fetch_hud_income_limits,
    fetch_hud_usps_crosswalk,
    fetch_hud_chas_data,
)

from economic_research.tools.real_estate_skill import get_real_estate_roi
from economic_research.tools.regulatory_skill import fetch_regulatory_notices
from economic_research.tools.talent_pipeline_skill import (
    get_talent_pipeline_roi,
)
from economic_research.tools.tax_foundation_skill import fetch_state_tax_rates
from economic_research.tools.trade_skill import fetch_regional_trade_data
from economic_research.tools.workforce_exposure_skill import (
    analyze_workforce_exposure,
)
from economic_research.tools.economic_index_skill import (
    fetch_anthropic_economic_index_data,
)
from economic_research.tools.mls_property_analysis_skill import (
    fetch_mls_property_listings,
)
from economic_research.tools.labor_shift_skill import (
    model_labor_shifts,
)
from economic_research.tools.fec_skill import (
    analyze_political_stability,
)
from economic_research.tools.econometrics_skill import (
    run_econometric_regression,
)
from economic_research.tools.underwriting_skill import (
    underwrite_deal_leverage,
)
from economic_research.tools.scorecard_skill import (
    generate_location_scorecard,
)
from economic_research.tools.relocation_skill import (
    estimate_employee_relocation,
)
from economic_research.tools.macro_search_skill import (
    search_macro_series,
)

from .prompt import Prompts






load_dotenv()

prompts = Prompts()
ERA_INSTRUCTIONS = prompts.main_era_instructions()


def set_session_api_key(key_name: str, key_value: str) -> str:
    """
    Sets an API key in the current session's environment variables.
    Use this when the user provides a missing API key in the chat.
    
    Args:
        key_name: The name of the environment variable (e.g., 'FRED_API_KEY').
        key_value: The API key value provided by the user.
        
    Returns:
        A confirmation message.
    """
    allowed_keys = [
        "BEA_API_KEY", "FRED_API_KEY", "CENSUS_API_KEY", "EIA_API_KEY",
        "BLS_API_KEY", "HUD_API_KEY", "FEC_API_KEY", "NEWS_API_KEY",
        "SERPER_API_KEY", "CDC_APP_TOKEN", "OPENFDA_API_KEY"
    ]
    if key_name not in allowed_keys:
        return f"ERROR: Setting {key_name} is not allowed."
        
    os.environ[key_name] = key_value
    return f"Successfully set {key_name} for this session. You can now retry the failed operation."


class ERAAgent:
    agent_framework = "google-adk"

    def __init__(self):
        """Standard container for the Reasoning Engine. State-free to ensure cloud pickling stability."""
        pass

    def get_app(self, model_name: str = "gemini-2.5-flash") -> App:
        """Lazily instantiates the ADK App and Agent only when needed."""
        tools = [
            labor_force_stats_skill,
            median_hourly_wages_skill,
            state_tax_rate_skill,
            state_union_employment_skill,
            fetch_regional_macro_stats,
            fetch_state_electricity_rates,
            get_real_estate_roi,
            get_talent_pipeline_roi,
            fetch_census_education_stats,
            fetch_bea_regional_data,
            fetch_hud_fmr_data,
            fetch_hud_income_limits,
            analyze_housing_affordability,
            fetch_state_tax_rates,
            fetch_regional_trade_data,
            fetch_regulatory_notices,
            set_session_api_key,
            analyze_workforce_exposure,
            fetch_anthropic_economic_index_data,
            fetch_mls_property_listings,
            fetch_hud_usps_crosswalk,
            fetch_hud_chas_data,
            model_labor_shifts,
            analyze_political_stability,
            run_econometric_regression,
            underwrite_deal_leverage,
            generate_location_scorecard,
            estimate_employee_relocation,
            search_macro_series,
        ]





        era_agent = Agent(
            name="economic_research",
            model=Gemini(model_name=model_name),
            instruction=ERA_INSTRUCTIONS,
            tools=tools,
        )
        return App(root_agent=era_agent, name="Economic_Research_Agent")


    def query(self, input: str) -> str:
        """Standard Reasoning Engine entry point."""
        
        # Security Fix: Extract and mask API keys in input to prevent logging
        import re
        allowed_keys = [
            "BEA_API_KEY", "FRED_API_KEY", "CENSUS_API_KEY", "EIA_API_KEY",
            "BLS_API_KEY", "HUD_API_KEY", "FEC_API_KEY", "NEWS_API_KEY",
            "SERPER_API_KEY", "CDC_APP_TOKEN", "OPENFDA_API_KEY"
        ]
        modified_input = input
        for key in allowed_keys:
            pattern = f"{key}=([^\\s]+)"
            match = re.search(pattern, input)
            if match:
                key_value = match.group(1)
                # Set it in environment for the session
                os.environ[key] = key_value
                # Mask it in the input string
                modified_input = re.sub(pattern, f"{key}=**********", modified_input)
                print(f"🔒 [Security] Masked {key} in input and set for session.")

        # Cloud Secrets fallback using Secret Manager
        def get_cloud_secret(key_name):
            val = os.getenv(key_name)
            if val:
                return val
            try:
                from economic_research.shared_libraries.helper import (
                    access_secret_version,
                )

                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                if not project_id:
                    import google.auth

                    try:
                        _, project_id = google.auth.default()
                    except Exception:
                        pass

                if project_id:
                    return access_secret_version(
                        project_id=project_id, secret_id=key_name
                    )
            except Exception:
                return None

        # Provision keys in runtime environment
        env_vars = {
            "BEA_API_KEY": get_cloud_secret("BEA_API_KEY"),
            "FRED_API_KEY": get_cloud_secret("FRED_API_KEY"),
            "CENSUS_API_KEY": get_cloud_secret("CENSUS_API_KEY"),
            "EIA_API_KEY": get_cloud_secret("EIA_API_KEY"),
            "BLS_API_KEY": get_cloud_secret("BLS_API_KEY"),
            "HUD_API_KEY": get_cloud_secret("HUD_API_KEY"),
            "FEC_API_KEY": get_cloud_secret("FEC_API_KEY"),
            "NEWS_API_KEY": get_cloud_secret("NEWS_API_KEY"),
            "SERPER_API_KEY": get_cloud_secret("SERPER_API_KEY"),
            "CDC_APP_TOKEN": get_cloud_secret("CDC_APP_TOKEN"),
            "OPENFDA_API_KEY": get_cloud_secret("OPENFDA_API_KEY"),
        }
        for k, v in env_vars.items():
            if v:
                os.environ[k] = v

        # Classify complexity of input query
        model_name = "gemini-2.5-flash"
        try:
            from google.adk.agents import Agent
            from google.adk.runners import InMemoryRunner
            
            classifier_agent = Agent(
                name="router_supervisor",
                model=Gemini(model_name="gemini-2.5-flash"),
                instruction=prompts.complexity_classifier_instructions()
            )
            classifier_app = App(root_agent=classifier_agent, name="Router_Supervisor")
            classifier_runner = InMemoryRunner(app=classifier_app)
            classifier_runner.auto_create_session = True
            
            classifier_responses = classifier_runner.run(new_message=modified_input)
            classifier_text = ""
            for res in classifier_responses:
                if hasattr(res, "content") and res.content.parts:
                    for part in res.content.parts:
                        if part.text:
                            classifier_text += part.text
            
            import json
            cleaned_text = classifier_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_text)
            complexity = data.get("complexity", "LOW")
            if complexity == "HIGH":
                model_name = "gemini-2.5-pro"
                print("🧠 [Router] Detected high complexity task. Routing to gemini-2.5-pro.")
            else:
                print("⚡ [Router] Detected low complexity task. Routing to gemini-2.5-flash.")
        except Exception as e:
            print(f"⚠️ [Router] Routing failed: {e}. Falling back to gemini-2.5-flash.")

        # Instantiate App & Runner at runtime rather than deploy-time
        app = self.get_app(model_name=model_name)


        from google.adk.runners import InMemoryRunner

        runner = InMemoryRunner(app=app)
        runner.auto_create_session = True

        responses = runner.run(new_message=modified_input)
        full_text = ""
        for res in responses:
            if hasattr(res, "content") and res.content.parts:
                for part in res.content.parts:
                    if part.text:
                        full_text += part.text

        # ⚖️ Active Actor-Critic Loop (Self-Correction)
        try:
            from .sub_agents.agent import JudgeAgent

            judge = JudgeAgent().get_agent()
            judge_app = App(root_agent=judge, name="Judge_Review")
            judge_runner = InMemoryRunner(app=judge_app)
            judge_runner.auto_create_session = True


            # Iteration 1: Judge the initial draft
            judge_prompt = (
                "Please audit this draft report. Use Google Search to verify quantitative claims if needed. "
                "If you find contradictions or hallucinations, start your response with '[REJECT]' and explain exactly what to fix."
                f"\n\nDraft:\n{full_text}"
            )
            judge_responses = judge_runner.run(new_message=judge_prompt)

            judge_text = ""
            for res in judge_responses:
                if hasattr(res, "content") and res.content.parts:
                    for part in res.content.parts:
                        if part.text:
                            judge_text += part.text

            # If rejected, run Researcher again with the correction context!
            if "[REJECT]" in judge_text:
                print(
                    "⚠️ [Actor-Critic] Judge rejected the draft! Self-correcting..."
                )
                correction_prompt = (
                    f"Your previous draft was REJECTED by the Auditor Judge. Please use your tools to FIX the following discrepancies and generate a final report:\n\n"
                    f"### Auditor Feedback:\n{judge_text}\n\n"
                    f"### Previous Draft:\n{full_text}"
                )

                # Reset runner or run again
                retry_responses = runner.run(new_message=correction_prompt)
                corrected_text = ""
                for res in retry_responses:
                    if hasattr(res, "content") and res.content.parts:
                        for part in res.content.parts:
                            if part.text:
                                corrected_text += part.text

                final_report = f"{corrected_text}\n\n---\n### ⚖️ Auditor Judge Verification (Self-Corrected v2)\n{judge_text}"
            else:
                final_report = f"{full_text}\n\n---\n### ⚖️ Auditor Judge Verification (Passed v1)\n{judge_text}"

        except Exception as e:
            final_report = f"{full_text}\n\n---\n⚠️ *Judge verification failed: {e}*"

        # Calculate Economic Primitives of the completed session
        try:
            from google.adk.agents import Agent
            
            evaluator_agent = Agent(
                name="primitives_evaluator",
                model=Gemini(model_name="gemini-2.5-flash"),
                instruction="""
                You are an economic operations analyst. Evaluate the completed interaction between the user and the economic research agent.
                
                Compute the following primitives:
                1. "interaction_type": Classify into: directive, feedback_loop, task_iteration, validation, or learning.
                2. "autonomy_level": Integer from 1 (active collaboration / human-in-the-loop) to 5 (fully autonomous delegation).
                3. "human_only_time_minutes": Estimated time (in minutes) an experienced economic analyst would spend to complete this task manually (e.g., searching FRED/BLS, scraping tax rates, drafting tables, and writing reports).
                4. "human_education_years_required": Estimated years of education/training needed to understand this request (e.g., 12 for high school, 16 for college, 18+ for grad school/PhD).
                5. "task_success": Boolean (true/false) indicating if the agent successfully fulfilled the user request with accurate data.
                
                Output your evaluation as a valid JSON object. Do not include markdown formatting or additional explanation.
                """
            )
            evaluator_app = App(root_agent=evaluator_agent, name="Primitives_Evaluator")
            evaluator_runner = InMemoryRunner(app=evaluator_app)
            evaluator_runner.auto_create_session = True
            
            evaluation_prompt = f"### User Query:\n{modified_input}\n\n### Agent Final Response:\n{final_report}"
            eval_responses = evaluator_runner.run(new_message=evaluation_prompt)
            eval_text = ""
            for res in eval_responses:
                if hasattr(res, "content") and res.content.parts:
                    for part in res.content.parts:
                        if part.text:
                            eval_text += part.text
            
            # Save or log the metrics
            import json
            cleaned_eval = eval_text.replace("```json", "").replace("```", "").strip()
            primitives = json.loads(cleaned_eval)
            
            # Write to a session metadata log file
            log_dir = "/Users/enriq/.gemini/jetski/scratch/observability"
            os.makedirs(log_dir, exist_ok=True)
            import uuid
            session_id = str(uuid.uuid4())
            log_path = os.path.join(log_dir, f"{session_id}.json")
            with open(log_path, "w") as f:
                json.dump({
                    "session_id": session_id,
                    "query": modified_input,
                    "primitives": primitives
                }, f, indent=2)
                
            print(f"📊 [Observability] Logged Economic Primitives to {log_path}: {primitives}")
        except Exception as e:
            print(f"⚠️ [Observability] Failed to evaluate economic primitives: {e}")

        return final_report



export_agent = ERAAgent()

# Also export root_agent for local CLI usage
root_agent = export_agent.get_app().root_agent

# Export the App as 'agent' for run_eval.py
agent = export_agent.get_app()

