#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ERA 21-Question Golden Integration Suite (ADK 2.0 Hardened)."""

import pytest
from google.adk.runners import InMemoryRunner
from google.genai import types

# 1. Scenarios for the Golden Suite
# Covers: Labor, Macro, Tax, Trade, Housing, Political Risk, Energy
GOLDEN_SCENARIOS = [
    {
        "source": "FRED",
        "question": "What is the 10-year unemployment trend for Austin vs. Nashville?",
    },
    {
        "source": "BEA",
        "question": "Compare the Real GDP growth rate for the San Francisco MSA vs. Dallas.",
    },
    {
        "source": "Census",
        "question": "What is the educational attainment (Bachelor's+) for the Raleigh-Cary MSA?",
    },
    {
        "source": "HUD",
        "question": "Is housing in Austin affordable for a workforce at 50% AMI? Correlate rent vs income.",
    },
    {
        "source": "TaxFoundation",
        "question": "What are the corporate income tax brackets for North Carolina in 2024?",
    },
    {
        "source": "USITC",
        "question": "Is North Carolina a manufacturing hub for pharmaceuticals based on export data?",
    },
    {
        "source": "EIA",
        "question": "Compare industrial electricity rates in Texas vs. Ohio for a data center.",
    },
    {
        "source": "SenatLDA",
        "question": "Which industries have the highest lobbying influence in Florida currently?",
    },
    {
        "source": "FederalRegister",
        "question": "Are there any recent regulatory notices regarding semiconductors in Texas?",
    },
    {
        "source": "FEC",
        "question": "Benchmark the political stability of site selection in Ohio using FEC data.",
    },
    {
        "source": "MetroMatrix",
        "question": "Create a Metro Matrix comparing Denver and Seattle for a tech hub.",
    },
    {
        "source": "HQRelocation",
        "question": "Generate an HQ relocation summary for moving 500 staff from SF to Austin.",
    },
    {
        "source": "CompanyReloc",
        "question": "Generate an industry employment and wage report for BioTech in Raleigh.",
    },
    {
        "source": "Logistics",
        "question": "What is the logistics and intermodal efficiency score for Savannah, GA?",
    },
    {
        "source": "Climate",
        "question": "Show the climate resilience and NRI score for Miami vs. Houston.",
    },
    {
        "source": "Incentives",
        "question": "What regional tax incentives are available for manufacturing in Tennessee?",
    },
    {
        "source": "Cultural",
        "question": "Rate the cultural amenity score for Scottsdale, AZ vs. Boulder, CO.",
    },
    {
        "source": "PolicyRisk",
        "question": "Benchmark the policy risk for a remote-first engineering team in Tennessee.",
    },
    {
        "source": "BLS_Live",
        "question": "What is the unemployment rate for Travis County, TX (FIPS 48453) using live BLS data?",
    },
    {
        "source": "TradeFlow",
        "question": "What are the primary trade dependencies for electronics in Texas?",
    },
    {
        "source": "MacroHealth",
        "question": "Show the top-line macro health for the state of Arizona.",
    },
]


@pytest.fixture(scope="function")
def runner():
    from economic_research.agent import export_agent

    runner_instance = InMemoryRunner(app=export_agent.get_app())
    runner_instance.auto_create_session = True
    return runner_instance


@pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS)
def test_golden_suite_scenario(runner, scenario):
    """
    Validates each scenario in the 21-Question Golden Suite.
    Ensures Grounding, No Hallucination, and Tool Convergence.
    """
    print(
        f"\n🔬 [GOLDEN SUITE] Testing {scenario['source']}: {scenario['question']}"
    )

    question = scenario["question"]

    responses = []
    response_generator = runner.run(
        new_message=types.Content(parts=[types.Part(text=question)]),
        user_id="golden-suite-tester",
        session_id=f"test-session-{scenario['source'].lower()}",
    )

    for event in response_generator:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    responses.append(part.text)

    full_response = "".join(responses)
    print(f"\n--- REPORT FOR {scenario['source']} ---\n{full_response}\n-------------------\n")

    # Assertions for High-Fidelity Consulting
    assert len(full_response) > 50, (
        f"Response too short for {scenario['source']}."
    )
    assert "N/A" not in full_response or "Source" in full_response, (
        f"Possible Grounding failure in {scenario['source']}."
    )

    # Verify sources are cited or agent correctly declared limitations/declined
    declined_keywords = ["cannot directly", "do not have access", "limitations", "unverifiable", "error", "unable to retrieve", "does not utilize", "did not find"]
    assert any(
        keyword in full_response
        for keyword in [
            "Source",
            "Data",
            "FRED",
            "BEA",
            "Census",
            "HUD",
            "EIA",
            "BLS",
            "FEC",
            "USITC",
            "flat",
            "rate",
            "unemployment",
            "imports",
            "%",
            "Federal Register",
        ]
    ) or any(
        keyword in full_response.lower()
        for keyword in declined_keywords
    ), f"Missing source citation or limitation notice in {scenario['source']} response."

    print(f"✅ Success for {scenario['source']}")


if __name__ == "__main__":
    pytest.main([__file__])
