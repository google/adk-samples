# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evolved Real Estate Portfolio & Investment Advisor Module.
Integrates live RentCast MLS listings, HUD FMR rents, and HUD AMI limits with advanced pro-forma RE investment math (50% Rule, Cash-on-Cash Return).
"""

import json
import logging
import math
import os
import re
import requests
import urllib.request
from typing import Any, Mapping

logger = logging.getLogger(__name__)

from economic_research.tools.dynamic_entity_resolver import resolve_fips
from economic_research.tools.hud_skill import fetch_hud_fmr_data, fetch_hud_income_limits
from economic_research.tools.mls_property_analysis_skill import fetch_mls_property_listings


class RealEstatePortfolioAdvisor:
    def __init__(self, mortgage_rate: float = 0.068, down_payment_pct: float = 0.20):
        self.mortgage_rate = mortgage_rate
        self.down_payment_pct = down_payment_pct

    def calculate_investment_yield(self, prop: dict, hud_rent_2br: float, ami_limit: float = None) -> dict:
        """
        Calculates Pro-Forma ROI, Cap Rate, Gross Rent Multiplier, and Cash-on-Cash Return using the 50% Rule.
        """
        try:
            raw_price = prop.get("Price", "$0").replace("$", "").replace(",", "")
            price = float(raw_price)
        except Exception:
            price = 300000.0

        beds = 2
        try:
            bed_str = prop.get("Beds/Baths", "2B/1.5Ba")
            match = re.search(r'(\d+)B', bed_str)
            if match:
                beds = int(match.group(1))
        except Exception:
            pass

        # Rent scaling based on bedroom count
        bed_multiplier = 1.0
        if beds == 1:
            bed_multiplier = 0.8
        elif beds == 3:
            bed_multiplier = 1.25
        elif beds >= 4:
            bed_multiplier = 1.5

        est_monthly_rent = hud_rent_2br * bed_multiplier
        est_annual_rent = est_monthly_rent * 12

        # 50% Rule for Operating Expenses (Maintenance, taxes, insurance, management, vacancy)
        est_annual_expenses = est_annual_rent * 0.50
        noi = est_annual_rent - est_annual_expenses

        # Cap Rate
        cap_rate = (noi / price) * 100 if price > 0 else 0.0

        # Gross Rent Multiplier (GRM)
        grm = price / est_annual_rent if est_annual_rent > 0 else 0.0

        # Cash-on-Cash Return (CoC)
        down_payment = price * self.down_payment_pct
        closing_costs = price * 0.03 # 3% closing costs
        total_cash_invested = down_payment + closing_costs

        loan_amount = price - down_payment
        # Monthly mortgage payment (P&I)
        r = self.mortgage_rate / 12
        n = 30 * 12
        if r > 0:
            monthly_pi = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            monthly_pi = 0.0

        annual_debt_service = monthly_pi * 12
        cash_flow = noi - annual_debt_service
        coc_return = (cash_flow / total_cash_invested) * 100 if total_cash_invested > 0 else 0.0

        # Affordability Index (Rent vs 50% AMI level)
        affordability_status = "Market"
        if ami_limit and ami_limit > 0:
            monthly_ami_50 = (ami_limit / 12)
            if est_monthly_rent <= (monthly_ami_50 * 0.30):
                affordability_status = "Affordable (Section 8 Eligible)"

        return {
            "Address": prop.get("Address", "Unknown"),
            "Price": f"${price:,.0f}",
            "Beds/Baths": prop.get("Beds/Baths", "2B/1.5Ba"),
            "Est. Monthly Rent": f"${est_monthly_rent:,.2f}",
            "Net Operating Income (50% Rule)": f"${noi:,.2f}",
            "Annual Cash Flow (After Debt)": f"${cash_flow:,.2f}",
            "Gross Rent Multiplier": f"{grm:.1f}x",
            "Cap Rate": f"{cap_rate:.2f}%",
            "Cash-on-Cash Return": f"{coc_return:.2f}%",
            "Affordability Status": affordability_status
        }

    def evaluate_city(self, city_name: str, property_type: str = "single-family") -> list[dict]:
        """
        Scans MLS, correlates HUD data dynamically, and calculates advanced investment metrics.
        """
        raw_listings_json = fetch_mls_property_listings(city_name=city_name, property_type=property_type)
        try:
            raw_listings = json.loads(raw_listings_json)
        except Exception:
            return []

        if isinstance(raw_listings, dict) and "status" in raw_listings:
            return []

        clean_city = city_name.split(",")[0].strip()
        fips = resolve_fips(clean_city)

        hud_rent_2br = 1500.0
        ami_limit = 0.0
        
        if fips:
            try:
                hud_data = json.loads(fetch_hud_fmr_data(fips))
                if "Rent_2BR" in hud_data:
                    hud_rent_2br = float(hud_data["Rent_2BR"].replace("$", "").replace(",", ""))
            except Exception:
                pass

            try:
                income_data = json.loads(fetch_hud_income_limits(fips))
                if "AMI_50_Level" in income_data:
                    ami_limit = float(income_data["AMI_50_Level"].replace("$", "").replace(",", ""))
            except Exception:
                pass

        results = []
        for prop in raw_listings:
            investment_data = self.calculate_investment_yield(prop, hud_rent_2br, ami_limit)
            results.append(investment_data)

        try:
            results.sort(key=lambda x: float(x["Cash-on-Cash Return"].replace("%", "")), reverse=True)
        except Exception:
            pass

        return results

    def generate_investment_brief(self, city_names: list[str], property_type: str = "single-family") -> str:
        """
        Generates a high-fidelity Markdown and HTML real estate investment brief.
        """
        brief_sections = []
        brief_sections.append(f"# 🏢 Corporate Real Estate & Yield Investment Brief")
        brief_sections.append(f"**Target Property Type**: {property_type.capitalize()} listings compared across target MSAs.")
        brief_sections.append(f"**Mortgage Assumptions**: 30-Year Fixed at {self.mortgage_rate*100:.1f}%, {self.down_payment_pct*100:.0f}% Down Payment (+3% Closing Costs).")
        brief_sections.append(f"**Operational Accounting**: Assumes the **50% Rule** for Operating Expenses (Maintenance, taxes, insurance, vacancy, and management).")
        brief_sections.append("\n---\n")

        all_results = {}
        for city in city_names:
            yields = self.evaluate_city(city, property_type)
            all_results[city] = yields

            brief_sections.append(f"## 📍 Market Profile: {city}")
            if not yields:
                brief_sections.append(f"_No active listings or payload retrieved for {city}._")
                continue

            brief_sections.append(f"| Address | Price | Beds/Baths | Est. Rent | Annual Cash Flow | Cap Rate | Cash-on-Cash | Affordability |")
            brief_sections.append(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for y in yields:
                brief_sections.append(f"| {y['Address']} | {y['Price']} | {y['Beds/Baths']} | {y['Est. Monthly Rent']} | {y['Annual Cash Flow (After Debt)']} | **{y['Cap Rate']}** | **{y['Cash-on-Cash Return']}** | {y['Affordability Status']} |")
            brief_sections.append("\n")

        try:
            from google import genai
            from google.genai import types
            client = genai.Client()
            
            comparison_prompt = f"""
            You are a Managing Director of a Real Estate Private Equity firm. 
            Review the following collected yield data for: {', '.join(city_names)}.
            
            Data Collected:
            {json.dumps(all_results, indent=2)}
            
            Synthesize this data into a professional Markdown investment brief for our acquisition committee.
            Structure it with:
            # Executive Summary (Highlight the MSA with the highest Cash-on-Cash return)
            # Market Yield Comparison (Compare the Cap Rates and Gross Rent Multipliers)
            # Risk & Affordability Analysis (Highlight any listings that are Section 8 / Affordable Housing eligible based on their AMI status)
            # Acquisition Recommendations (Top 2 specific property addresses to target for purchase)
            
            Provide a hyperlinked Sources & Citations header at the bottom citing HUD User API and RentCast.
            """
            
            response = client.models.generate_content(
                model=os.getenv("MODEL_NAME_GENERATED_1"),
                contents=comparison_prompt
            )
            report_text = response.text
            brief_sections.append(report_text)
            
        except Exception as e:
            brief_sections.append(f"## ⚖️ Portfolio Synthesis\nError executing PE firm synthesis: {e}")

        return "\n".join(brief_sections)
