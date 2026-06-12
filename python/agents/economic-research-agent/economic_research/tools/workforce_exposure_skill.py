# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""Workforce & AI Task Exposure Analysis Skill."""

import json


def analyze_workforce_exposure(occupations: list[str]) -> str:
    """
    Analyzes AI exposure (automation vs. augmentation) and strategic recommendations 
    for a list of occupational domains or standard job titles.
    
    Args:
        occupations: List of standard occupational categories or job titles 
                    (e.g., ["Software Developers", "Customer Service Representatives", "Financial Analysts", "Retail Sales"]).
                    
    Returns:
        JSON string containing the AI exposure scores, primary impact mode, and strategic action plans.
    """
    # Grounded mapping based on O*NET task classifications and AI labor exposure studies
    exposure_db = {
        "software developers": {
            "soc": "15-1252",
            "exposure_level": "High",
            "impact_mode": "Augmentation (Task Iteration & Validation)",
            "complexity_score": "High (16+ years education required)",
            "key_exposed_tasks": ["Writing/refactoring code", "System design integration", "Unit testing and debugging"],
            "recommendation": "High opportunity for productivity gain. Shift developer hours toward architectural design and system safety."
        },
        "computer and mathematical": {
            "soc": "15-0000",
            "exposure_level": "High",
            "impact_mode": "Augmentation (Task Iteration & Validation)",
            "complexity_score": "High (16+ years education required)",
            "key_exposed_tasks": ["Data analysis", "Statistical modeling", "Algorithmic engineering"],
            "recommendation": "Upskill teams on context caching and collaborative agent programming to accelerate output."
        },
        "customer service representatives": {
            "soc": "43-4051",
            "exposure_level": "High",
            "impact_mode": "Automation (Directive Workflows)",
            "complexity_score": "Medium (12-14 years education required)",
            "key_exposed_tasks": ["Answering billing inquiries", "Resolving standard order complaints", "Ticket routing"],
            "recommendation": "High displacement risk. Automate repetitive tier-1 ticketing via API agents; transition human agents to high-empathy case management."
        },
        "office and administrative support": {
            "soc": "43-0000",
            "exposure_level": "High",
            "impact_mode": "Automation (Directive Workflows)",
            "complexity_score": "Medium (12-14 years education required)",
            "key_exposed_tasks": ["Data entry", "Meeting scheduling", "Document formatting"],
            "recommendation": "Incorporate document-extraction and RAG agents to automate office pipelines."
        },
        "financial analysts": {
            "soc": "13-2051",
            "exposure_level": "Medium-High",
            "impact_mode": "Augmentation (Validation & Learning)",
            "complexity_score": "High (16+ years education required)",
            "key_exposed_tasks": ["Corporate financial modeling", "Market trend analysis", "Investment memo preparation"],
            "recommendation": "Utilize agents for rapid macro-data ingestion (FRED/Census); focus analyst time on risk-assessment and narrative synthesis."
        },
        "management": {
            "soc": "11-0000",
            "exposure_level": "Medium",
            "impact_mode": "Augmentation (Feedback Loops)",
            "complexity_score": "High (16+ years education required)",
            "key_exposed_tasks": ["Strategic decision making", "Team performance reviews", "Inter-department coordination"],
            "recommendation": "Low displacement risk. Deploy conversational dashboards to accelerate executive context-gathering."
        },
        "tutors": {
            "soc": "25-3000",
            "exposure_level": "Medium",
            "impact_mode": "Augmentation (Learning & Feedback)",
            "complexity_score": "Medium-High (14-16 years education required)",
            "key_exposed_tasks": ["Grading assignments", "Curriculum pacing", "Explaining core subjects"],
            "recommendation": "Leverage AI for personalized student pacing and automated grading support; focus human time on mentoring."
        },
        "retail sales": {
            "soc": "41-2031",
            "exposure_level": "Low",
            "impact_mode": "Minimal Impact",
            "complexity_score": "Low (12 years education required)",
            "key_exposed_tasks": ["Processing local payments", "Stocking inventory", "In-person product advice"],
            "recommendation": "Low overall exposure. Focus AI investment on logistics and back-office supply chains rather than consumer interaction."
        }
    }

    results = []
    for occ in occupations:
        occ_lower = occ.lower().strip()
        # Fallback to fuzzy match
        matched_data = None
        for key in exposure_db:
            if key in occ_lower or occ_lower in key:
                matched_data = exposure_db[key]
                matched_data["queried_occupation"] = occ
                break
        
        if matched_data:
            results.append(matched_data)
        else:
            results.append({
                "queried_occupation": occ,
                "soc": "Unknown",
                "exposure_level": "Unknown/Fuzzy Match",
                "impact_mode": "Unknown",
                "complexity_score": "Requires manual review",
                "key_exposed_tasks": ["N/A"],
                "recommendation": f"Data not pre-mapped for '{occ}'. Standard exposure for this role requires custom task-level evaluation."
            })
            
    return json.dumps(results, indent=2)
