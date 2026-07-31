#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""
ERA Evaluation Science (v2.0) - Runner Edition.
"""

import json
import os

import pandas as pd
import vertexai
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types
from vertexai.evaluation import EvalTask, PointwiseMetric

from economic_research.agent import agent

load_dotenv()
vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-east1")

# The judge must also use Vertex AI explicitly
GROUNDING_METRIC = PointwiseMetric(
    metric="era_grounding_fidelity",
    metric_prompt_template="""
    Evaluate the response for institutional-grade economic grounding.
    Score 5/5 if it cites specific numerical tool outputs and correlates regional benchmarks.
    """,
)


def run_benchmarks():
    print("🛰️ ERA EVALUATION SCIENCE: Initiating Benchmarking via Runner...")

    # 1. Initialize Runner (Ensuring Vertex AI mode in the agent)
    # The 'agent' exported from economic_research.agent is already the App.
    runner = InMemoryRunner(app=agent)
    runner.auto_create_session = True

    # 3. Load Golden Set
    golden_path = os.path.join(os.path.dirname(__file__), "golden_set.json")
    with open(golden_path) as f:
        golden_set = json.load(f)

    sim_data = []
    for i, test in enumerate(golden_set):
        q = test["input"]
        print(f"🔬 Simulating: {q[:50]}...")
        try:
            full_response = ""
            msg = types.Content(parts=[types.Part(text=q)])
            # Using blocking run with required identifiers
            responses = runner.run(
                new_message=msg, user_id="eval-user", session_id=f"session-{i}"
            )

            for res in responses:
                if hasattr(res, "content") and res.content.parts:
                    for part in res.content.parts:
                        if part.text:
                            full_response += part.text

            sim_data.append(
                {
                    "prompt": q,
                    "response": full_response,
                    "reference": "Reference benchmark for site-selection grounding.",
                }
            )
            print(f"✅ Question {i + 1} Complete.")
        except Exception as e:
            print(f"❌ Question {i + 1} Failed: {e}")

    if not sim_data:
        return

    print("⚖️ Judging Responses with Gemini 2.5 Flash...")
    eval_df = pd.DataFrame(sim_data)
    eval_task = EvalTask(dataset=eval_df, metrics=[GROUNDING_METRIC])
    # The evaluation judge also needs to point to have access to 2.5 in this project.
    try:
        results = eval_task.evaluate()
        print("\n📈 ERA PERFORMANCE SUMMARY:")
        if hasattr(results, "summary_metrics"):
            print(results.summary_metrics)
        else:
            print(results)
    except Exception as e:
        print(f"❌ Judging Failed: {e}")
        print("\n📉 RAW RESPONSES COLLECTED (First 3):")
        print(eval_df[["prompt", "response"]].head(3))


if __name__ == "__main__":
    run_benchmarks()
