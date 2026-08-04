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
"""Universal Whitepaper Generation Orchestrator.
Features an Adaptive LLM Router that classifies the user's research topic into one of 4 Strategic Pillars and dispatches tailored, high-fidelity data harvesting and synthesis prompts to generate premium corporate whitepapers for ANY 'Wow Factor' query in the README.
"""

import json
import logging
import math
import os
import re
import sys
from typing import Any, Mapping

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types


def classify_topic(topic: str) -> dict:
    """
    Classifies the research topic into one of four strategic pillars via a quick, deterministic LLM turn.
    """
    try:
        client = genai.Client()
        router_prompt = f"""
        Analyze the following economic research topic or query:
        "{topic}"
        
        Classify it into EXACTLY ONE of the following four Strategic Pillars. 
        Return your answer as a JSON object containing "pillar" (A, B, C, or D) and "rationale".
        
        Pillars:
        * A: "SITE_SELECTION" - Comparing cities for corporate relocation, facility site selection, utility infrastructure, and metro matrices.
        * B: "REAL_ESTATE" - Multifamily/residential investments, Cap Rates, HUD FMR rents, AMI affordability limits, and deal underwriting.
        * C: "WORKFORCE_AI" - AI automation risk, task exposure, labor market disruption, and 3-year workforce outlooks.
        * D: "FISCAL_TRADE_POLICY" - Corporate tax climates, international trade flows (USITC), semiconductor supply chains, and regulatory federal register notices.
        
        Do not include markdown tags.
        """
        response = client.models.generate_content(
            model=os.getenv("MODEL_NAME"),
            contents=router_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text.strip())
        return data
    except Exception as e:
        logger.warning(f"Topic classification failed: {e}. Falling back to SITE_SELECTION.")
        return {"pillar": "A", "rationale": "Fallback"}


def get_adaptive_prompts(pillar: str, topic: str) -> tuple[str, str]:
    """
    Returns the tailored Phase 1 (Harvesting) and Phase 2 (Synthesis) prompts based on the Strategic Pillar.
    """
    clean_pillar = str(pillar).strip().upper()
    
    if clean_pillar == "B" or "REAL_ESTATE" in clean_pillar:
        harvest_prompt = f"""
        Gather raw data and listings for the real estate investment query: "{topic}".
        Specifically, find and output:
        1. Active property listings (prices, beds/baths, types) for the target MSA(s) using available MLS tools.
        2. Local HUD Fair Market Rents (FMR) for 2BR/3BR and HUD Area Median Income (AMI) limits.
        3. USPS county FIPS crosswalks and CHAS housing burden metrics if relevant.
        """
        synth_prompt = f"""
        You are a Global Managing Director and Senior Partner at a Tier-1 Strategy Consulting Firm (McKinsey/BCG/Bain). 
        Synthesize the collected real estate and HUD data into a Multi-Million Dollar Corporate Real Estate Investment Brief, suitable for direct publication on a premium corporate blog or PE prospectus.
        RESEARCH TOPIC: {topic}
        
        ### 🏛️ Formatting & Persona Constraints:
        - **Premium Executive Tone**: Use the MECE framework (Mutually Exclusive, Collectively Exhaustive). Frame every data point with high-level corporate strategy.
        - **Rich Styling & Data Density**: Utilize rich markdown, bold strategic highlights, and extensive side-by-side Markdown Tables to present your analysis.
        - **Zero Hallucination Grounding**: Cite exact sources and endpoint URLs at the bottom of the brief.

        Your output MUST be a formal Markdown publication structured with:
        # Executive Summary (Highlight the highest Cash-on-Cash Return opportunity using a gorgeous summary table)
        # Market Yield Deep Dive (Display Cap Rates, GRMs, and NOI using the 50% Rule in a dense Markdown table)
        # Affordability & Workforce Housing Analysis (Correlate live HUD FMR rents against the 50% AMI limit for Section 8 underwriting)
        # Strategic SWOT & Acquisition Recommendations
        # Sources & Citations
        """
        
    elif clean_pillar == "C" or "WORKFORCE_AI" in clean_pillar:
        harvest_prompt = f"""
        Gather raw data and task analysis for the workforce AI exposure query: "{topic}".
        Specifically, find and output:
        1. O*NET task listings and AI exposure/automation potential vectors for the target occupations.
        2. BLS employment figures, median hourly wages, and unionization rates for those sectors.
        3. Live web searches for recent AI disruption studies and corporate adoption announcements.
        """
        synth_prompt = f"""
        You are a Chief Labor Economist and Senior Partner at a Tier-1 Strategy Consulting Firm. 
        Synthesize the collected O*NET and BLS data into a Multi-Million Dollar Workforce Adaptation & AI Disruption Whitepaper, suitable for direct publication on a premium HBR or corporate blog.
        RESEARCH TOPIC: {topic}
        
        ### 🏛️ Formatting & Persona Constraints:
        - **Premium Executive Tone**: Use the MECE framework. Frame every data point with high-level corporate reskilling and automation strategy.
        - **Rich Styling & Data Density**: Utilize rich markdown, bold strategic highlights, and extensive Markdown Tables to present your O*NET and BLS metrics.
        - **Zero Hallucination Grounding**: Cite exact sources and endpoint URLs at the bottom of the brief.

        Your output MUST be a formal Markdown publication structured with:
        # Executive Summary (Highlight the occupations with the highest automation risk vs augmentation potential)
        # O*NET Task Deep Dive & Augmentation Metrics (Display a dense Markdown table of tasks, wage impact, and exposure scores)
        # 3-Year Displacement & Reskilling Outlook
        # Strategic HR & Operational SWOT Recommendations
        # Sources & Citations
        """
        
    elif clean_pillar == "D" or "FISCAL_TRADE_POLICY" in clean_pillar:
        harvest_prompt = f"""
        Gather raw data for the policy, fiscal, and supply chain query: "{topic}".
        Specifically, find and output:
        1. State and local corporate income tax brackets, phases, and credits (e.g. Tax Foundation).
        2. USITC international trade flows, state export/import values, and semiconductor/commodity HS codes.
        3. Federal Register regulatory notices and FEC campaign contribution benchmarks for the target region.
        """
        synth_prompt = f"""
        You are a Global Managing Director of Supply Chain & Regulatory Policy at a Tier-1 Strategy Consulting Firm. 
        Synthesize the collected fiscal, USITC, and Federal Register data into a Multi-Million Dollar Corporate Supply Chain & Fiscal Policy Whitepaper, suitable for direct publication on a premium corporate blog.
        RESEARCH TOPIC: {topic}
        
        ### 🏛️ Formatting & Persona Constraints:
        - **Premium Executive Tone**: Use the PESTLE framework. Frame every data point with high-level corporate risk, trade, and tax strategy.
        - **Rich Styling & Data Density**: Utilize rich markdown, bold strategic highlights, and extensive Markdown Tables to compare state tax regimes and trade corridors.
        - **Zero Hallucination Grounding**: Cite exact sources and endpoint URLs at the bottom of the brief.

        Your output MUST be a formal Markdown publication structured with:
        # Executive Summary (Highlight supply chain dependencies and fiscal runway)
        # Trade Flow & Supply Chain Analysis (Display USITC export/import metrics and HS Code analysis in a table)
        # Fiscal & Regulatory Climate Deep Dive (Tax phases, abatements, and recent Federal Register policy shifts)
        # Strategic Supply Chain, Tax Mitigation & SWOT Recommendations
        # Sources & Citations
        """
        
    else: # Pillar A: SITE_SELECTION
        harvest_prompt = f"""
        Gather raw data for the corporate relocation and site-selection comparison: "{topic}".
        Specifically, find and output:
        1. Labor force metrics (BLS employment, wages) and macro indicators (FRED real GDP, unemployment trends).
        2. EIA industrial electricity rates and CoStar commercial office/industrial lease rates and vacancies.
        3. Corporate and state tax climates from the Tax Foundation.
        """
        synth_prompt = f"""
        You are a Senior Partner and Chief Economist at a Tier-1 Strategy Consulting Firm (McKinsey/BCG/Bain). 
        Synthesize the collected data into a Multi-Million Dollar Corporate Relocation & Site Selection Whitepaper, suitable for direct publication on a premium corporate blog.
        RESEARCH TOPIC: {topic}
        
        ### 🏛️ Formatting & Persona Constraints:
        - **Premium Executive Tone**: Use the MECE and SWOT frameworks. Frame every data point with high-level site selection, operational efficiency, and ROI strategy.
        - **Rich Styling & Data Density**: Utilize rich markdown, bold strategic highlights, and extensive Markdown Tables for side-by-side metro comparisons.
        - **Derived Scorecards**: Blend metrics into a 0-100 Weighted Site Suitability Index table.
        - **Zero Hallucination Grounding**: Cite exact sources and endpoint URLs at the bottom of the brief.

        Your output MUST be a formal Markdown publication structured with:
        # Executive Summary
        # Methodology
        # Data Analysis & Deep Dive (Display labor, tax, and utility data in Markdown tables)
        # Cross-Source Correlations & Derived Scorecard (Blend metrics into a 0-100 Site Suitability Index)
        # Strategic SWOT Recommendations
        # Sources & Citations
        """
        
    return harvest_prompt, synth_prompt


def solve(eval_inputs: Mapping[str, Any]) -> str:
    """
    Universally orchestrates Deep Research Whitepaper generation for ANY Wow Factor query.
    """
    topic = eval_inputs.get("research_topic", "")
    if not topic:
        return "ERROR: No research topic provided."

    try:
        from economic_research.agent import export_agent
        
        # Step 1: Adaptive Pillar Routing
        routing_info = classify_topic(topic)
        pillar = routing_info.get("pillar", "A")
        logger.info(f"🧬 Routed Topic '{topic}' to Pillar: {pillar} ({routing_info.get('rationale')})")
        
        harvest_prompt, synth_prompt = get_adaptive_prompts(pillar, topic)
        
        # Step 2: Adaptive Data Harvesting
        print(f"🚀 [Phase 1] Harvesting Data for Pillar {pillar}...")
        raw_research_data = export_agent.query(harvest_prompt)
        
        # Step 3: Adaptive McKinsey/PE Synthesis
        print(f"🚀 [Phase 2] Synthesizing Whitepaper for Pillar {pillar}...")
        synthesis_input = f"""
        {synth_prompt}
        
        RAW RESEARCH DATA GATHERED:
        {raw_research_data}
        """
        
        final_whitepaper = export_agent.query(synthesis_input)
        return final_whitepaper
        
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"Failed Universal Whitepaper Orchestration: {tb_str}")
        return f"Error executing universal whitepaper pipeline: {e}"
