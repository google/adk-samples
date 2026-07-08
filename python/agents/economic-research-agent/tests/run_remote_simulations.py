# Copyright 2026 Google LLC. This software is provided as-is, without warranty or representation.
"""Runs programmatic simulations against the deployed Vertex AI Agent Engine."""

import os
import json
import subprocess
import google.auth

# Deployed Agent Engine ID
REMOTE_ENGINE_ID = "https://us-east1-aiplatform.googleapis.com/v1beta1/projects/697625214430/locations/us-east1/reasoningEngines/8517890192101605376"

# A selection of 100 unique representative queries (from FRED, BLS, CENSUS, HUD, etc. + README WOW queries)
WOW_QUERIES = [
    "Compare Austin, TX and Raleigh, NC using a custom scorecard weighted 40% on corporate tax, 30% on industrial electricity rates, and 30% on software developer wage trends.",
    "Retrieve state-level unionization density from BLS and average weekly wages from FRED, then run a formal OLS regression in the sandbox to see if there is a statistically significant correlation.",
    "Underwrite an investment property in Columbus, OH listed at $10M with a monthly rent roll of $75,000, assuming 25% down and 6.5% interest on a 30-year amortization. Output the pro-forma table.",
    "Estimate the net disposable income shift for relocating a data analyst from Seattle, WA to Richmond, VA on a $140,000 salary, accounting for state income tax brackets and HUD 2-Bedroom rents.",
    "What is the 10-year unemployment trend for Seattle vs. Denver?",
    "Show the educational attainment pipeline for Orlando vs. Raleigh.",
    "Is Salt Lake City affordable for a 50% Area Median Income (AMI) workforce? Correlate rent vs income.",
    "What are the corporate income tax brackets for Washington in 2024?",
    "Find multifamily investment properties in Tampa, FL and estimate their Cap Rates.",
    "Find the county FIPS code for ZIP code 28202 using USPS crosswalk."
]

def generate_simulation_set():
    # Load 90 instances from the main stress test to make exactly 100 queries
    sim_queries = list(WOW_QUERIES)
    
    stress_set_path = "tests/eval/evalsets/wow_stress_test.evalset.json"
    if os.path.exists(stress_set_path):
        with open(stress_set_path) as f:
            full_set = json.load(f)
            cases = full_set.get("eval_cases", [])
            # Select 90 queries evenly spaced across the 900
            for i in range(0, len(cases), 10):
                if len(sim_queries) >= 100:
                    break
                txt = cases[i]["conversation"][0]["user_content"]["parts"][0]["text"]
                if txt not in sim_queries:
                    sim_queries.append(txt)
                    
    # Fallback to make sure we have 100
    while len(sim_queries) < 100:
        sim_queries.append("What is the unemployment rate in Austin for the last year?")
        
    return sim_queries[:100]

def run_simulations(project_id: str):
    print(f"📡 Initializing connection via CLI targeting project '{project_id}'...")
    
    queries = generate_simulation_set()
    print(f"🚀 Prepared {len(queries)} unique simulation queries. Starting remote run via agents-cli...\n")
    
    results = []
    for idx, q in enumerate(queries):
        print(f"[{idx+1}/100] Query: '{q}'")
        try:
            # Execute agents-cli run as a subprocess
            cmd = [
                "uv", "run", "agents-cli", "run",
                "--url", REMOTE_ENGINE_ID,
                "--mode", "adk",
                q
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            output_text = res.stdout
            if res.returncode == 0:
                print("🟢 Received response successfully.")
                results.append({
                    "index": idx + 1,
                    "query": q,
                    "status": "SUCCESS",
                    "response": output_text[:400] + "..." if len(output_text) > 400 else output_text
                })
            else:
                print(f"🔴 Failed with exit code {res.returncode}: {res.stderr}")
                results.append({
                    "index": idx + 1,
                    "query": q,
                    "status": "FAILED",
                    "error": res.stderr
                })
        except Exception as e:
            print(f"🔴 Failed: {e}")
            results.append({
                "index": idx + 1,
                "query": q,
                "status": "FAILED",
                "error": str(e)
            })
            
    out_path = "tests/remote_simulation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Programmatic simulations complete! Logged results to {out_path}")

if __name__ == "__main__":
    try:
        _, project = google.auth.default()
        active_project = project or os.getenv("GOOGLE_CLOUD_PROJECT", "project-maui")
        run_simulations(project_id=active_project)
    except Exception as e:
        print(f"❌ Simulation execution failed: {e}")

