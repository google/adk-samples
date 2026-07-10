import os

import pytest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Questions mapping
GOLDEN_QUESTIONS = [
    # 1. FRED Macro
    {
        "source": "FRED",
        "question": "Compare the 10-year trend of Residential Construction in Austin vs. Raleigh.",
        "expected_mention": "austin",
    },
    # 2. BLS Labor
    {
        "source": "BLS",
        "question": "What are the median hourly wages for Software Developers in Salt Lake City?",
        "expected_mention": "salt lake",
    },
    # 3. EIA Energy
    {
        "source": "EIA",
        "question": "What is the industrial electricity rate per kWh in Ohio?",
        "expected_mention": "ohio",
    },
    # 4. News Sentiment
    {
        "source": "NewsAPI",
        "question": "Detect market sentiment headwind for semiconductor plants in Oregon.",
        "expected_mention": "oregon",
    },
    # 5. COLA (C2ER)
    {
        "source": "C2ER",
        "question": "What is the real purchasing power of $150k in Austin vs. Charlotte?",
        "expected_mention": "austin",
    },
    # 6. OpenSecrets Policy
    {
        "source": "OpenSecrets",
        "question": "Which states have upcoming corporate tax sunsets in the next 24 months?",
        "expected_mention": "tax",
    },
    # 7. Healthcare (CDC)
    {
        "source": "CDC",
        "question": "What are the regional hospital utilization rates using data.cdc.gov for Atlanta vs. Boston?",
        "expected_mention": "boston",
    },
    # 8. Live Judge Search (Serper)
    {
        "source": "Serper",
        "question": "Verify if there are any recent (2025-2026) news reports about semiconductor plant closures in Texas.",
        "expected_mention": "texas",
    },
]


@pytest.fixture(scope="function")
def engine():
    # Loophole to run project tests from inside scratch workspace
    os.chdir(PROJECT_ROOT)
    from dotenv import load_dotenv

    load_dotenv()

    from economic_research.agent import export_agent

    return export_agent


@pytest.mark.parametrize("scenario", GOLDEN_QUESTIONS)
def test_golden_question(engine, scenario):
    """Run golden questions against the live agent to verify integration."""
    question = scenario["question"]
    expected_mention = scenario["expected_mention"]

    print(f"\nRunning Golden Question for {scenario['source']}: {question}")

    report = engine.query(input=question)

    assert len(report) > 0, f"Received empty report for {scenario['source']}"
    assert expected_mention.lower() in report.lower(), (
        f"Expected mention of '{expected_mention}' in response for {scenario['source']}"
    )
    print(f"Success! Response contains '{expected_mention}': {report[:100]}...")
