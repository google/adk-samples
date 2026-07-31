#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Bureau of Economic Analysis (BEA). Regional & National GDP/Income."""

import json
import os

import requests

def fetch_bea_regional_data(
    metro_names: list[str], report_type: str = "GDP"
) -> str:
    """
    Fetches regional economic data (GDP or Personal Income) directly from the BEA API.
    Essential for high-fidelity regional economic health assessments.
    """
    h_key = os.getenv("BEA_API_KEY", "").strip()
    bea_key = h_key.replace('"', "").replace("'", "")
    if not bea_key:
        return json.dumps(
            {"ERROR": "BEA_API_KEY not found in environment."}, indent=2
        )

    # MSA FIPS Registry (Hardened for Hub Comparisons)
    msa_fips = {
        "austin": "12420",
        "raleigh": "39580",
        "nashville": "34980",
        "san francisco": "41860",
        "dallas": "19100",
        "seattle": "42660",
        "houston": "26420",
        "denver": "19740",
        "miami": "33100",
    }

    try:
        results = []
        for city in metro_names:
            # Grounding: Clean names for robust mapping (handle 'MSA', 'TX', etc)
            city_clean = city.lower().replace(" msa", "").split(",")[0].strip()
            fips = msa_fips.get(city_clean)

            if fips:
                # Live BEA API Call
                # Dataset: Regional (CAGDP9 = Real GDP by MSA)
                url = (
                    f"https://apps.bea.gov/api/data?UserID={bea_key}"
                    f"&method=GetData&DataSetName=Regional"
                    f"&TableName=CAGDP9"
                    f"&GeoFIPS={fips}"
                    f"&LineCode=1"
                    f"&Year=ALL"  # Get the most recent available year
                    f"&ResultFormat=JSON"
                )

                response = requests.get(url, timeout=12)
                if response.status_code == 200:
                    data = response.json()
                    try:
                        # Parse BEA's nested Results.Data structure
                        val_entries = (
                            data.get("BEAAPI", {})
                            .get("Results", {})
                            .get("Data", [])
                        )
                        if val_entries:
                            # Take the latest year provided
                            latest_entry = val_entries[-1]
                            val = latest_entry["DataValue"]
                            year = latest_entry["TimePeriod"]

                            results.append(
                                {
                                    "City": city,
                                    "Metric": f"Real {report_type} (Millions $)",
                                    "Value": f"${float(val.replace(',', '')):,}",
                                    "Year": year,
                                    "Source": "Bureau of Economic Analysis (BEA) Live API",
                                }
                            )
                        else:
                            # Fallback to FRED for MSA FIPS
                            from economic_research.tools.fred_skill import fetch_regional_macro_stats
                            try:
                                fred_res = fetch_regional_macro_stats([city], series_type="gdp")
                                if "ERROR" not in fred_res and "No FRED data" not in fred_res:
                                    fred_data = json.loads(fred_res)
                                    if fred_data:
                                        item = fred_data[0]
                                        results.append({
                                            "City": city,
                                            "Metric": f"Real {report_type} (Millions $)",
                                            "Value": f"${item['Latest Value']}",
                                            "Year": item['Latest Date'].split("-")[0],
                                            "Source": item['Source'] + " (BEA MSA Fallback)"
                                        })
                                        continue
                            except Exception:
                                pass

                            results.append(
                                {
                                    "City": city,
                                    "Status": "No data items in BEA response.",
                                }
                            )
                    except Exception as e:
                        results.append(
                            {"City": city, "Status": f"Parsing Error: {e!s}"}
                        )
                else:
                    results.append(
                        {
                            "City": city,
                            "Status": f"BEA API Failure ({response.status_code})",
                        }
                    )
            else:
                results.append(
                    {
                        "City": city,
                        "Status": "MSA FIPS not found in Grounded Registry.",
                    }
                )

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"ERROR": str(e)}, indent=2)
