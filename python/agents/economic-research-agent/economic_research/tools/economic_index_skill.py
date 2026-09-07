# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""Anthropic Economic Index Historical Data Grounding Skill."""

import json


def fetch_anthropic_economic_index_data(query_type: str) -> str:
    """
    Retrieves official historical data points from Anthropic's Economic Index Reports 
    (from August 2025, November 2025, and February 2026).
    
    Args:
        query_type: The type of historical data requested. Must be one of:
                    - "model_selection": Model selection shares across occupational domains.
                    - "user_tenure": Metrics on low-tenure vs. high-tenure users.
                    - "task_concentration": Usage shares among top 10 O*NET tasks.
                    - "geographic_gini": Gini coefficient convergence for US states and countries.
                    
    Returns:
        JSON string containing the historical data points and findings.
    """
    # Grounded data extracted directly from the January 2026 and March 2026 Reports
    index_database = {
        "model_selection": {
            "description": "Paid Claude.ai users selecting Opus class of models depending on occupational domain (Feb 2026)",
            "average_overall_opus_share": "51%",
            "shares_by_domain": [
                {"domain": "Computer and Mathematical (e.g., Coding)", "opus_share": "55%", "difference": "+4.0pp"},
                {"domain": "Education, Training, and Library (e.g., Tutoring)", "opus_share": "45%", "difference": "-6.0pp"},
                {"domain": "Software Developer Tasks (Specific)", "opus_share": "34%", "difference": "High"},
                {"domain": "Tutor Tasks (Specific)", "opus_share": "12%", "difference": "Low"}
            ],
            "findings": "For every additional $10 of hourly wage for a task, the share of conversations using Opus increases by 1.5 percentage points for Claude.ai, and by 2.8 percentage points for API traffic."
        },
        "user_tenure": {
            "description": "Differences between high-tenure (signed up >= 6 months ago) and low-tenure users (Feb 2026)",
            "comparison": {
                "personal_use_share": {"high_tenure": "38%", "low_tenure": "44%", "difference": "High-tenure has 10% fewer personal conversations"},
                "human_education_years": {"high_tenure": "6% higher reflected in inputs", "difference": "Increases by almost 1 year for every additional year of Claude usage"},
                "task_concentration_top_10": {"high_tenure": "20.7%", "low_tenure": "22.2%", "difference": "Usage is less concentrated in high-tenure group"},
                "conversation_success_rate": {"high_tenure": "+4.0 percentage points", "explanation": "Long-tenure users have a 10% higher success rate in conversations, even with full controls for model, use case, and country."}
            },
            "findings": "More experienced users use Claude more collaboratively (validation and iteration loops) rather than delegating responsibility through directive use patterns."
        },
        "task_concentration": {
            "description": "Usage shares among top 10 most common O*NET tasks on Claude.ai",
            "time_series": [
                {"period": "November 2025", "top_10_share": "24%"},
                {"period": "February 2026", "top_10_share": "19%"}
            ],
            "findings": "Decline in concentration reflects coding tasks migrating from Claude.ai to first-party API (Claude Code agentic calls) and coursework decreasing due to academic calendars."
        },
        "geographic_gini": {
            "description": "Gini coefficient of AI adoption per capita (AUI)",
            "us_states": {
                "trend": "Convergence continues but at a slower pace",
                "top_5_states_share": {"August 2025": "30%", "February 2026": "24%"},
                "estimated_equalization_time": "5-9 years (previously estimated at 2-5 years)"
            },
            "global_countries": {
                "trend": "Usage has become slightly more concentrated (divergence)",
                "top_20_countries_share": {"August 2025": "45%", "February 2026": "48%"}
            },
            "developer_api_users": {
                "trend": "Usage concentration remains high due to enterprise cloud regions hubs",
                "top_5_percent_share": {"August 2025": "70%", "February 2026": "68%"}
            }
        }
    }
    
    q_clean = query_type.lower().strip()
    if q_clean in index_database:
        return json.dumps(index_database[q_clean], indent=2)
    else:
        return json.dumps({
            "error": f"Invalid query_type: '{query_type}'",
            "allowed_types": list(index_database.keys())
        }, indent=2)
