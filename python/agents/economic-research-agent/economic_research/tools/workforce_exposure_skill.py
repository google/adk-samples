# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""Workforce & AI Task Exposure Analysis Skill."""

import os
import json
import requests
from google import genai
from google.genai import types

def classify_onet_tasks_with_gemini(title: str, tasks: list[str]) -> dict:
    """Classifies O*NET occupational tasks using Vertex AI / Gemini."""
    try:
        # Load GCP project metadata from environment or fall back to default
        project = os.getenv("GCP_PROJECT", "project-maui")
        location = os.getenv("GCP_LOCATION", "us-central1")
        
        client = genai.Client(vertexai=True, project=project, location=location)
        prompt = f"""
        Analyze the AI exposure and automation potential for the occupation: "{title}".
        Below is the official task list for this role:
        
        {json.dumps(tasks, indent=2)}
        
        Compute the following analysis:
        1. "exposure_level": Rate as High, Medium-High, Medium, Medium-Low, or Low.
        2. "impact_mode": Classify the primary mode, e.g. "Automation (Directive Workflows)", "Augmentation (Task Iteration & Validation)", "Minimal Impact", etc.
        3. "complexity_score": E.g. "High (16+ years education required)", "Medium (12-14 years education required)".
        4. "key_exposed_tasks": Select the top 3 most exposed/impacted tasks from the list above.
        5. "recommendation": Provide a strategic consulting recommendation for organizations employing this role.
        
        Format your response as a valid JSON object with the keys:
        - exposure_level
        - impact_mode
        - complexity_score
        - key_exposed_tasks (list of strings)
        - recommendation (string)
        
        Do not include markdown code block formatting or explanations. Return only the raw JSON.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️ Gemini task analysis failed: {e}")
        return {
            "exposure_level": "High",
            "impact_mode": "Augmentation",
            "complexity_score": "Requires manual review",
            "key_exposed_tasks": tasks[:3] if tasks else ["N/A"],
            "recommendation": f"Default fallback. Error during dynamic classification: {e}"
        }


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
    api_key = os.getenv("ONET_API_KEY", "").strip()

    if api_key:
        headers = {
            "accept": "application/json",
            "X-API-Key": api_key
        }
        
        for occ in occupations:
            occ_clean = occ.strip()
            # Step A: Search for the SOC code
            search_url = "https://api-v2.onetcenter.org/online/search"
            try:
                search_resp = requests.get(search_url, params={"keyword": occ_clean, "limit": 1}, headers=headers, timeout=12)
                if search_resp.status_code == 200:
                    search_data = search_resp.json()
                    occupation_list = search_data.get("occupation", [])
                    if occupation_list:
                        code = occupation_list[0].get("code")
                        official_title = occupation_list[0].get("title")
                        
                        # Step B: Fetch tasks
                        tasks_url = f"https://api-v2.onetcenter.org/online/occupations/{code}/details/tasks"
                        tasks_resp = requests.get(tasks_url, headers=headers, timeout=12)
                        if tasks_resp.status_code == 200:
                            tasks_data = tasks_resp.json()
                            task_items = tasks_data.get("task", [])
                            task_titles = [t.get("title") for t in task_items if t.get("title")][:10]
                            
                            if task_titles:
                                # Step C: Query Gemini to analyze tasks
                                analysis = classify_onet_tasks_with_gemini(official_title, task_titles)
                                results.append({
                                    "soc": code,
                                    "exposure_level": analysis.get("exposure_level", "Medium"),
                                    "impact_mode": analysis.get("impact_mode", "Augmentation"),
                                    "complexity_score": analysis.get("complexity_score", "Requires review"),
                                    "key_exposed_tasks": analysis.get("key_exposed_tasks", task_titles[:3]),
                                    "recommendation": analysis.get("recommendation", "Shift tasks to high-value areas."),
                                    "queried_occupation": occ
                                })
                                continue
            except Exception as e:
                print(f"⚠️ O*NET live fetch/analysis failed for '{occ}': {e}. Falling back to sandbox database.")
                
    # Fallback/Offline logic
    for occ in occupations:
        if any(res.get("queried_occupation") == occ for res in results):
            continue
            
        occ_lower = occ.lower().strip()
        matched_data = None
        for key in exposure_db:
            if key in occ_lower or occ_lower in key:
                matched_data = exposure_db[key].copy()
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

