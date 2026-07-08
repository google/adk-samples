# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""Generates a subset of 42 queries from the full 900-query stress test evalset."""

import json

def generate_subset():
    with open("tests/eval/evalsets/wow_stress_test.evalset.json") as f:
        full = json.load(f)
        
    cases = full["eval_cases"]
    
    # We will pick 2 cases from each of the 16 sources, plus 5 cross-source, plus 5 anthropic analyst cases.
    subset_cases = []
    
    sources = [
        "fred", "bea", "census", "hud", "bls", "fec", "usitc", "eia", 
        "register", "tax_f", "workforce", "mls_sourcing", "usps_cross", 
        "chas", "labor_shifts", "anthropic_index"
    ]
    
    for src in sources:
        # filter cases starting with this source prefix
        src_cases = [c for c in cases if c["eval_id"].startswith(src)]
        subset_cases.extend(src_cases[:2])
        
    cross_cases = [c for c in cases if c["eval_id"].startswith("cross_source")]
    subset_cases.extend(cross_cases[:5])
    
    anthropic_cases = [c for c in cases if c["eval_id"].startswith("anthropic_analyst")]
    subset_cases.extend(anthropic_cases[:5])
    
    subset = {
        "eval_set_id": "wow_subset_test",
        "name": "Economic Research Agent 42 Query Verification Subset",
        "description": "Verification subset covering 2 cases per source, 5 cross-source, and 5 Anthropic analyst cases.",
        "eval_cases": subset_cases
    }
    
    with open("tests/eval/evalsets/wow_subset_test.evalset.json", "w") as f:
        json.dump(subset, f, indent=2)
        
    print(f"Created subset of {len(subset_cases)} cases successfully!")

if __name__ == "__main__":
    generate_subset()
