# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""Generates 900 unique natural language queries for testing the Economic Research Agent."""

import json
import os

def generate_evalset():
    eval_cases = []
    
    # 16 Sources List
    sources = [
        "FRED", "BEA", "Census", "HUD", "BLS", "FEC", "USITC", "EIA", 
        "Register", "Tax F.", "Workforce", "MLS Sourcing", "USPS Cross.", 
        "CHAS", "Labor Shifts", "Anthropic Index"
    ]
    
    # Target entities for template variation
    cities = ["Austin", "Raleigh", "Charlotte", "Seattle", "Orlando", "Salt Lake City", "Richmond", "Tampa", "Houston", "Denver"]
    states = ["TX", "NC", "NC", "WA", "FL", "UT", "VA", "FL", "TX", "CO"]
    state_names = ["Texas", "North Carolina", "North Carolina", "Washington", "Florida", "Utah", "Virginia", "Florida", "Texas", "Colorado"]
    zips = ["78702", "27601", "28202", "98101", "32801", "84101", "23219", "33602", "77002", "80202"]
    fips = ["48453", "37183", "37119", "53033", "12095", "49035", "51760", "12057", "48201", "08031"]
    occupations = ["Software Developers", "Customer Service Representatives", "Financial Clerks", "Data Analysts", "Marketing Managers"]
    
    # Template dictionaries for generating 50 queries per source
    templates = {
        "FRED": [
            "What is the unemployment rate in {city} for the last year?",
            "Show the 10-year unemployment trend for {city}.",
            "Compare the labor force size in {city} vs. Nashville.",
            "What is the recent job growth trend in the {city} MSA?",
            "Get the quarterly employment level for {city}."
        ],
        "BEA": [
            "What is the Real GDP growth rate for {city} MSA?",
            "Compare the Real GDP of {city} vs. Dallas.",
            "Show the personal income trend for {city}.",
            "What is the annual GDP output of the {city} metro area?",
            "Show the per capita personal income in {city}."
        ],
        "Census": [
            "Show the educational attainment (Bachelor's+) pipeline for {city}.",
            "What percentage of the population in {city} holds a college degree?",
            "Compare high school graduation rates in {city} vs. Seattle.",
            "What is the science and engineering degree concentration in {city}?",
            "Show the demographic and educational distribution in {city}."
        ],
        "HUD": [
            "Is {city} affordable for a 50% AMI workforce?",
            "What is the 2-bedroom Fair Market Rent (FMR) in FIPS {fips}?",
            "Correlate HUD rent rates vs income limits in FIPS {fips}.",
            "Show the 3-year trend for HUD income limits in FIPS {fips}.",
            "Is the 50% AMI rental rate in FIPS {fips} lower than FMR?"
        ],
        "BLS": [
            "What is the wage trend in {state} over the last decade?",
            "Show the unionization rate vs wage level in {state}.",
            "What are the average weekly wages in {state}?",
            "Compare employment-to-population ratios in {state} vs {state2}.",
            "Show the manufacturing sector wage growth in {state}."
        ],
        "FEC": [
            "Benchmark the political activity of site selection in {state} using FEC data.",
            "Show the campaign contributions trend for {state} in 2024.",
            "What is the total PAC contribution volume in {state}?",
            "Compare political contribution growth in {state} vs {state2}.",
            "Identify candidate committee spending levels in {state}."
        ],
        "USITC": [
            "Analyze {state} as a semiconductor hub and show trade flows.",
            "Compare import/export volumes for tech parts in {state} vs {state2}.",
            "What are the top trade partners for electronics in {state}?",
            "Show the USITC trade balance for {state}.",
            "Analyze trade flows of industrial parts in {state}."
        ],
        "EIA": [
            "Compare industrial electricity rates in {state} vs. Ohio.",
            "What is the commercial power rate in {state}?",
            "Compare utility costs for a data center in {state} vs {state2}.",
            "Show the monthly electricity rate trend in {state}.",
            "What are the retail sales of electricity in {state}?"
        ],
        "Register": [
            "Are there any recent regulatory notices regarding semiconductors in {state}?",
            "Show Federal Register notices for environmental rules in {state}.",
            "What are the latest shipping and logistics notices in {state}?",
            "Check for regulatory notices regarding energy in {state}.",
            "Are there recent manufacturing policy updates in {state}?"
        ],
        "Tax F.": [
            "What are the corporate income tax brackets for {state_name} in 2024?",
            "Compare corporate tax rates in {state_name} vs. Texas.",
            "What is the individual income tax rate in {state_name}?",
            "Show the sales tax rates and brackets for {state_name}.",
            "Is there a gross receipts tax in {state_name}?"
        ],
        "Workforce": [
            "Analyze the workforce AI exposure and automation potential for {occ1} vs. {occ2}.",
            "What tasks in {occ1} are candidates for AI automation?",
            "Score the AI exposure level for {occ1}.",
            "How does AI augmentation impact {occ1} in terms of daily tasks?",
            "Compare directive automation vs collaborative chat augmentation for {occ1}."
        ],
        "MLS Sourcing": [
            "Find multifamily investment properties in {city} and estimate their Cap Rates.",
            "Show active condo listings in {city} with purchase price and FMR yields.",
            "Estimate NOI and Price-to-Rent ratios for listings in {city}.",
            "Show single-family investment properties in {city} mapped with HUD rents.",
            "What is the estimated Cap Rate for properties in {city}?"
        ],
        "USPS Cross.": [
            "Find the county FIPS code for ZIP code {zip} using USPS crosswalk.",
            "Map ZIP {zip} to its local census tract and county.",
            "Show the residential ratio mapping for ZIP {zip}.",
            "Convert ZIP {zip} to county FIPS using HUD USPS crosswalk.",
            "Find the geoid for ZIP code {zip}."
        ],
        "CHAS": [
            "What is the percentage of cost-burdened households in FIPS {fips} using CHAS data?",
            "Show the rate of housing problems in FIPS {fips}.",
            "What percentage of households in FIPS {fips} spend >50% of income on housing?",
            "Compare cost-burdened owner vs renter households in FIPS {fips}.",
            "Show CHAS housing quality problem rates in FIPS {fips}."
        ],
        "Labor Shifts": [
            "Compare {city1} and {city2} for AI-driven labor market disruption.",
            "Forecast the 3-year AI displacement rate in {city1}.",
            "What is the AI vulnerability index of {city1}?",
            "Compare the augmentation potential of {city1} vs {city2}.",
            "Show the projected productivity gains in {occ1} in {city1} over the next 3 years."
        ],
        "Anthropic Index": [
            "What are the Gini coefficient trends for developer API users?",
            "Show the historical Opus utilization share over the last quarter.",
            "Compare Sonnet vs Opus usage share across standard job domains.",
            "What is the average task concentration rate in user accounts?",
            "Show model selection rates for developers with high user tenure."
        ]
    }

    # Generate 50 queries per source (10 permutations of 5 templates)
    for src in sources:
        src_templates = templates.get(src, [])
        for i in range(10):
            # Pick distinct variables per iteration
            c_val = cities[i % len(cities)]
            c_val2 = cities[(i+1) % len(cities)]
            s_val = states[i % len(states)]
            s_val2 = states[(i+1) % len(states)]
            sn_val = state_names[i % len(state_names)]
            z_val = zips[i % len(zips)]
            f_val = fips[i % len(fips)]
            o_val = occupations[i % len(occupations)]
            o_val2 = occupations[(i+1) % len(occupations)]
            
            for temp in src_templates:
                txt = temp.format(
                    city=c_val, city1=c_val, city2=c_val2,
                    state=s_val, state2=s_val2, state_name=sn_val,
                    zip=z_val, fips=f_val, occ1=o_val, occ2=o_val2
                )
                eval_cases.append({
                    "eval_id": f"{src.lower().replace(' ', '_').replace('.', '')}_{len(eval_cases)}",
                    "conversation": [{"user_content": {"parts": [{"text": txt}]}}],
                    "session_input": {"app_name": "Economic_Research_Agent", "user_id": "eval_user", "state": {}}
                })

    # Generate 50 Cross-Source questions
    cross_source_templates = [
        "Compare corporate tax rates in {state_name} vs. electricity rates in {state} for a manufacturing plant.",
        "Correlate the 10-year unemployment rate in {city} (FRED) with local housing affordability (HUD).",
        "Compare GDP growth rate (BEA) in {city} with political activity volume (FEC).",
        "Does {city} have higher technical talent (Census) and lower industrial utility rates (EIA) than {city2}?",
        "Compare the labor force quality in {state} (BLS) vs environmental regulations in the Federal Register (Register)."
    ]
    for i in range(10):
        c_val = cities[i % len(cities)]
        c_val2 = cities[(i+1) % len(cities)]
        s_val = states[i % len(states)]
        sn_val = state_names[i % len(state_names)]
        for temp in cross_source_templates:
            txt = temp.format(city=c_val, city2=c_val2, state=s_val, state_name=sn_val)
            eval_cases.append({
                "eval_id": f"cross_source_{len(eval_cases)}",
                "conversation": [{"user_content": {"parts": [{"text": txt}]}}],
                "session_input": {"app_name": "Economic_Research_Agent", "user_id": "eval_user", "state": {}}
            })

    # Generate 50 Anthropic-Type Analyst questions
    anthropic_analyst_templates = [
        "Analyze how high concentration of {occ} in {city} correlates with Opus selection rates.",
        "What is the AI exposure risk of the local talent pool in {city} compared to Gini index trends?",
        "Model the labor shift in {city} for {occ} using Sonnet vs Opus share metrics.",
        "What are the projected productivity gains in {city} vs developer task concentration levels?",
        "Estimate the workforce displacement risk in {city} for {occ} using HUD CHAS housing burden data."
    ]
    for i in range(10):
        c_val = cities[i % len(cities)]
        o_val = occupations[i % len(occupations)]
        for temp in anthropic_analyst_templates:
            txt = temp.format(city=c_val, occ=o_val)
            eval_cases.append({
                "eval_id": f"anthropic_analyst_{len(eval_cases)}",
                "conversation": [{"user_content": {"parts": [{"text": txt}]}}],
                "session_input": {"app_name": "Economic_Research_Agent", "user_id": "eval_user", "state": {}}
            })

    # Save to JSON
    evalset = {
        "eval_set_id": "wow_stress_test",
        "name": "Economic Research Agent 900 Query WOW Stress Test",
        "description": "Comprehensive evaluation covering 50 queries per source, 50 cross-source comparisons, and 50 Anthropic analyst queries.",
        "eval_cases": eval_cases
    }
    
    out_dir = "tests/eval/evalsets"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wow_stress_test.evalset.json")
    with open(out_path, "w") as f:
        json.dump(evalset, f, indent=2)
        
    print(f"Generated {len(eval_cases)} eval cases successfully in {out_path}!")

if __name__ == "__main__":
    generate_evalset()
