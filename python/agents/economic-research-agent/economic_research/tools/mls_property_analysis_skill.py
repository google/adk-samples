# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""MLS Property Analysis and Real Estate Investment Yield Calculator Skill with RentCast API integration."""

import os
import json
import requests
from economic_research.tools.hud_skill import fetch_hud_fmr_data, fetch_hud_usps_crosswalk

# Grounded city-to-county FIPS mappings for HUD integration (fallback)
CITY_FIPS_MAP = {
    "austin": "48453",      # Travis County, TX
    "raleigh": "37183",     # Wake County, NC
    "dallas": "48113",      # Dallas County, TX
    "columbus": "39049"     # Franklin County, OH
}

PROPERTY_TYPE_MAP = {
    "multifamily": "Multi-Family",
    "multi-family": "Multi-Family",
    "single-family": "Single Family",
    "singlefamily": "Single Family",
    "condo": "Condo",
    "townhouse": "Townhouse",
    "land": "Land",
    "commercial": "Commercial"
}

CITY_STATE_MAP = {
    "austin": "TX",
    "raleigh": "NC",
    "dallas": "TX",
    "columbus": "OH"
}


def fetch_mls_property_listings(
    city_name: str, max_price: float = None, property_type: str = "multifamily"
) -> str:
    """
    Queries MLS listings for a target metropolitan area (either live via RentCast or from 
    the sandbox database fallback) and performs automated investment analysis (Cap Rate, 
    Price-to-Rent Ratio) by correlating listing prices with local HUD Fair Market Rent (FMR) benchmarks.
    
    Args:
        city_name: Name of the target city (e.g., "Austin", "Raleigh", "Columbus", "Dallas").
        max_price: Optional maximum listing price filter in USD.
        property_type: Type of property: "multifamily", "single-family", or "condo".
        
    Returns:
        JSON string containing active listings, estimated local rents, annual expenses, and Cap Rates.
    """
    city_clean = city_name.lower().strip().split(",")[0]
    api_key = os.getenv("RENTCAST_API_KEY", "").strip()
    
    raw_listings = []
    
    # 1. Try fetching from live RentCast API if API key is set
    if api_key:
        url = "https://api.rentcast.io/v1/listings/sale"
        
        parts = city_name.split(",")
        city = parts[0].strip()
        state = parts[1].strip().upper() if len(parts) > 1 else CITY_STATE_MAP.get(city.lower())
        
        mapped_type = PROPERTY_TYPE_MAP.get(property_type.lower().strip(), "Multi-Family")
        
        params = {
            "city": city,
            "propertyType": mapped_type,
            "status": "Active",
            "limit": 3  # Conserve user's 50 requests/month free quota limit
        }
        if state:
            params["state"] = state
            
        try:
            headers = {
                "accept": "application/json",
                "X-Api-Key": api_key
            }
            resp = requests.get(url, params=params, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    raw_listings.append({
                        "address": item.get("formattedAddress"),
                        "price": item.get("price"),
                        "beds": item.get("bedrooms", 2),
                        "baths": item.get("bathrooms", 1.5),
                        "type": property_type.lower(),
                        "zip": item.get("zipCode")
                    })
            else:
                print(f"⚠️ RentCast API returned code {resp.status_code}: {resp.text}. Falling back to sandbox database.")
        except Exception as e:
            print(f"⚠️ RentCast request failed: {e}. Falling back to sandbox database.")
            
    # 2. Fall back to mock active listings if no key was present or no listings were fetched
    if not raw_listings:
        listings_db = {
            "austin": [
                {"address": "1208 Chicon St, Austin, TX 78702", "price": 450000, "beds": 2, "baths": 1.5, "type": "condo"},
                {"address": "7402 Decker Ln, Austin, TX 78724", "price": 380000, "beds": 3, "baths": 2, "type": "single-family"},
                {"address": "1611 E 2nd St, Austin, TX 78702", "price": 650000, "beds": 2, "baths": 2, "type": "multifamily"}
            ],
            "raleigh": [
                {"address": "412 E South St, Raleigh, NC 27601", "price": 310000, "beds": 2, "baths": 1, "type": "condo"},
                {"address": "2910 Avent Ferry Rd, Raleigh, NC 27606", "price": 395000, "beds": 3, "baths": 2.5, "type": "single-family"},
                {"address": "905 S Saunders St, Raleigh, NC 27603", "price": 480000, "beds": 4, "baths": 3, "type": "multifamily"}
            ],
            "columbus": [
                {"address": "84 Indianola Ave, Columbus, OH 43201", "price": 280000, "beds": 2, "baths": 1.5, "type": "condo"},
                {"address": "1042 S High St, Columbus, OH 43206", "price": 340000, "beds": 3, "baths": 2, "type": "single-family"},
                {"address": "512 E Maynard Ave, Columbus, OH 43202", "price": 390000, "beds": 4, "baths": 2, "type": "multifamily"}
            ],
            "dallas": [
                {"address": "2903 Fitzhugh Ave, Dallas, TX 75204", "price": 330000, "beds": 2, "baths": 2, "type": "condo"},
                {"address": "4120 Simpson St, Dallas, TX 75246", "price": 390000, "beds": 3, "baths": 2, "type": "single-family"},
                {"address": "5208 Columbia Ave, Dallas, TX 75214", "price": 550000, "beds": 4, "baths": 3, "type": "multifamily"}
            ]
        }
        
        mock_raw = listings_db.get(city_clean, [])
        for item in mock_raw:
            # Extract ZIP code from end of address string
            try:
                zip_code = item["address"].split(",")[-1].strip().split(" ")[-1]
            except Exception:
                zip_code = None
                
            raw_listings.append({
                "address": item["address"],
                "price": item["price"],
                "beds": item["beds"],
                "baths": item["baths"],
                "type": item["type"],
                "zip": zip_code
            })
            
    if not raw_listings:
        return json.dumps({
            "status": "No listings found",
            "city": city_name,
            "message": f"MLS integration has no active properties for '{city_name}'."
        }, indent=2)
            
    # 3. Filter and analyze listings
    analyzed_listings = []
    for prop in raw_listings:
        # Filter by price
        if max_price and prop["price"] > max_price:
            continue
            
        # Filter by property type
        if property_type and prop["type"].lower() != property_type.lower():
            continue
            
        # Get local HUD FMR data dynamically
        hud_rent_2br = 1500.0  # Default fallback rent
        hud_year = "2025"
        
        # Try dynamic lookup first
        fips = None
        if prop.get("zip"):
            try:
                cross_resp = json.loads(fetch_hud_usps_crosswalk(prop["zip"]))
                if "County_FIPS" in cross_resp:
                    fips = cross_resp["County_FIPS"]
            except Exception:
                pass
                
        # If dynamic FIPS lookup fails, fall back to our city map
        if not fips:
            fips = CITY_FIPS_MAP.get(city_clean)
            
        if fips:
            try:
                hud_resp = json.loads(fetch_hud_fmr_data(fips))
                if "Rent_2BR" in hud_resp:
                    hud_rent_2br = float(hud_resp["Rent_2BR"].replace("$", "").replace(",", ""))
                    hud_year = hud_resp.get("Year", "2025")
            except Exception:
                pass  # Use default fallback rent
                
        # Adjust estimated monthly rent based on bed count (vs 2BR HUD base)
        beds = prop.get("beds") or 2
        bed_multiplier = 1.0
        if beds == 1:
            bed_multiplier = 0.8
        elif beds == 3:
            bed_multiplier = 1.25
        elif beds >= 4:
            bed_multiplier = 1.5
            
        est_monthly_rent = hud_rent_2br * bed_multiplier
        est_annual_rent = est_monthly_rent * 12
        
        # Operational expenses: 35% of gross rent
        est_annual_expenses = est_annual_rent * 0.35
        net_operating_income = est_annual_rent - est_annual_expenses
        
        # Calculate Cap Rate (%)
        cap_rate = (net_operating_income / prop["price"]) * 100
        
        # Price-to-Rent Ratio
        price_to_rent = prop["price"] / est_annual_rent
        
        analyzed_listings.append({
            "Address": prop["address"],
            "Price": f"${prop['price']:,}",
            "Property Type": prop["type"].capitalize(),
            "Beds/Baths": f"{beds}B/{prop['baths']}Ba",
            "HUD FMR (2BR)": f"${hud_rent_2br:,.0f} ({hud_year})",
            "Est. Monthly Rent": f"${est_monthly_rent:,.2f}",
            "Est. Annual Expenses": f"${est_annual_expenses:,.2f}",
            "Net Operating Income": f"${net_operating_income:,.2f}",
            "Price-to-Rent Ratio": f"{price_to_rent:.1f}x",
            "Estimated Cap Rate": f"{cap_rate:.2f}%"
        })
        
    return json.dumps(analyzed_listings, indent=2)
