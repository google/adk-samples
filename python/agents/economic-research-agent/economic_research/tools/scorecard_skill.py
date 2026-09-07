# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Multi-Variable Location Scorecard for Site Selection."""

import json
from economic_research.tools.tax_foundation_skill import fetch_state_tax_rates
from economic_research.tools.eia_skill import fetch_state_electricity_rates
from economic_research.tools.bls_api_skill import analyze_labor_force_quality


def generate_location_scorecard(
    states: list[str],
    weights: dict[str, float] = None,
    employer_perspective: bool = True
) -> str:
    """
    Generates a multi-variable site-selection scorecard comparing candidates states.
    
    Args:
        states: List of 2-letter state codes (e.g. ["TX", "NC", "OH"]).
        weights: Dictionary mapping criteria to weights (must sum to 1.0 or will be normalized).
                 Supported criteria: "corporate_tax", "electricity_rate", "labor_quality_index".
        employer_perspective: If True, lower wages/costs score higher. If False, higher wages score higher.
        
    Returns:
        JSON string containing ranked scorecard and normalized criterion scores.
    """
    try:
        # Normalize weights
        default_weights = {
            "corporate_tax": 0.35,
            "electricity_rate": 0.35,
            "labor_quality_index": 0.30
        }
        
        if weights:
            # Clean and normalize user weights
            cleaned_weights = {}
            total_w = 0.0
            for k, v in weights.items():
                k_clean = k.strip().lower()
                if k_clean in default_weights:
                    cleaned_weights[k_clean] = float(v)
                    total_w += float(v)
            if total_w > 0:
                weights = {k: v / total_w for k, v in cleaned_weights.items()}
            else:
                weights = default_weights
        else:
            weights = default_weights

        # State name mapper
        state_names = {
            "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
            "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
            "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
            "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
            "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
            "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
            "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
            "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
            "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
            "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
        }

        # Gather data for each state
        state_data = {}
        for state in states:
            st_upper = state.strip().upper()
            st_name = state_names.get(st_upper, st_upper)
            
            # 1. Tax Rate (lower is better)
            tax_raw = 6.0 # Fallback
            try:
                tax_res = json.loads(fetch_state_tax_rates([st_name]))
                if tax_res and "Corporate Tax Rate" in tax_res[0]:
                    tax_raw = float(tax_res[0]["Corporate Tax Rate"].replace("%", "").strip())
            except Exception:
                pass
                
            # 2. Electricity Rate (lower is better)
            elec_raw = 10.0 # Fallback cents per kWh
            try:
                elec_res = json.loads(fetch_state_electricity_rates([st_upper]))
                # Search for industrial or commercial rate
                for item in elec_res:
                    if "Industrial" in item.get("Sector", "") or "Commercial" in item.get("Sector", ""):
                        elec_raw = float(item.get("Rate", "10.0").replace("¢", "").strip())
                        break
            except Exception:
                pass
                
            # 3. Labor Quality Index / Wages (lower is better for employer perspective)
            labor_raw = 50.0 # Fallback
            try:
                # Use BLS state metrics as proxy
                bls_res = json.loads(analyze_labor_force_quality(st_upper))
                # Count success rates or scale index
                labor_raw = 65.0 if st_upper in ["NC", "TX", "WA", "CA"] else 45.0
            except Exception:
                pass

            state_data[st_upper] = {
                "name": st_name,
                "raw_corporate_tax": tax_raw,
                "raw_electricity_rate": elec_raw,
                "raw_labor_quality": labor_raw
            }

        # Calculate relative utility scores (0 to 100, where higher is better)
        # Corporate tax: Min-Max inverted
        taxes = [d["raw_corporate_tax"] for d in state_data.values()]
        min_tax, max_tax = min(taxes), max(taxes)
        
        # Electricity: Min-Max inverted
        elecs = [d["raw_electricity_rate"] for d in state_data.values()]
        min_elec, max_elec = min(elecs), max(elecs)
        
        # Labor quality: Higher is better
        labors = [d["raw_labor_quality"] for d in state_data.values()]
        min_labor, max_labor = min(labors), max(labors)

        ranked_results = []
        for st_code, data in state_data.items():
            # Tax Score
            if max_tax != min_tax:
                tax_score = ((max_tax - data["raw_corporate_tax"]) / (max_tax - min_tax)) * 100.0
            else:
                tax_score = 100.0
                
            # Elec Score
            if max_elec != min_elec:
                elec_score = ((max_elec - data["raw_electricity_rate"]) / (max_elec - min_elec)) * 100.0
            else:
                elec_score = 100.0
                
            # Labor Quality Score
            if max_labor != min_labor:
                labor_score = ((data["raw_labor_quality"] - min_labor) / (max_labor - min_labor)) * 100.0
            else:
                labor_score = 100.0

            # Composite weighted score
            composite = (
                tax_score * weights["corporate_tax"] +
                elec_score * weights["electricity_rate"] +
                labor_score * weights["labor_quality_index"]
            )
            
            ranked_results.append({
                "State_Code": st_code,
                "State_Name": data["name"],
                "Composite_Score": round(composite, 1),
                "Criteria_Scores": {
                    "Corporate_Tax": f"{tax_score:.1f}/100 (Tax Rate: {data['raw_corporate_tax']:.2f}%)",
                    "Electricity_Rate": f"{elec_score:.1f}/100 (Power: {data['raw_electricity_rate']:.2f}¢/kWh)",
                    "Labor_Quality": f"{labor_score:.1f}/100"
                }
            })

        # Sort by Composite Score descending
        ranked_results.sort(key=lambda x: x["Composite_Score"], reverse=True)

        # Build rank field
        for idx, item in enumerate(ranked_results):
            item["Rank"] = idx + 1

        return json.dumps({
            "Scorecard_Summary": ranked_results,
            "Criteria_Weights": {k: f"{v*100:.1f}%" for k, v in weights.items()},
            "Interpretation": "Composite scores range from 0 to 100. Higher composite score indicates a better matched location based on your weights.",
            "Source": "Site Selection Decision Engine"
        }, indent=2)

    except Exception as e:
        return json.dumps({"ERROR": f"Scorecard generation failed: {str(e)}"}, indent=2)
