#!/usr/bin/env python3
"""
Populate the application_quotes and issued_policies BigQuery tables with
~50 realistic sample records each for Western & Southern (Fabric App).

The data is designed so that the stalled_apps_view produces all health
tiers (Issued, Recent, Stalled, Abandoned) and the "Millennial Medical
History Drop-off" demo scenario works out-of-the-box.

Prerequisites:
  - pip install google-cloud-bigquery python-dotenv
  - GCP credentials configured (gcloud auth application-default login)
  - Dataset + tables already created via setup_bigquery.sh
"""

import os
import random
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Load environment variables securely from parent directory
load_dotenv(dotenv_path="../.env")
PROJECT_ID = os.getenv("PROJECT_ID")

if not PROJECT_ID:
    raise SystemExit(
        "ERROR: No GCP project configured.\n"
        "  Ensure PROJECT_ID is set in your ../.env file."
    )

DATASET = "fabric_dtc_data"
TABLE_QUOTES = f"{PROJECT_ID}.{DATASET}.application_quotes"
TABLE_POLICIES = f"{PROJECT_ID}.{DATASET}.issued_policies"

TODAY = date.today()
NOW = datetime.now(timezone.utc)

random.seed(42)  # reproducible data

# ---------------------------------------------------------------------------
# Reference data for FS Corp / Fabric App
# ---------------------------------------------------------------------------
STATES =["OH", "TX", "CA", "NY", "IL", "FL", "PA"]
DEMOGRAPHICS =["Millennial_Parent", "Gen_Z_Parent", "Gen_X_Parent", "Empty_Nester"]
PRODUCT_TYPES =["Term_Life_10Yr", "Term_Life_20Yr", "Term_Life_30Yr", "Simplified_Issue_Term"]
DROP_OFF_STAGES =["Basic_Info", "Beneficiary_Naming", "Medical_History", "Payment_Gateway", "Completed"]

def _quote_id(n: int) -> str:
    return f"QTE-{n:04d}"

def _random_date_between(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 0)))

# ---------------------------------------------------------------------------
# 1. Build application_quotes rows (The "Inventory")
# ---------------------------------------------------------------------------
quote_rows: list[dict] =[]
quote_counter = 1

# ── Core demo rows: Millennial Parents stalled at Medical History ──────────
# These will be forced into the "Stalled App" tier (10-25 days ago, NO POLICY)
demo_cohort =[]
for _ in range(8):
    quote_date = TODAY - timedelta(days=random.randint(12, 28))
    demo_cohort.append({
        "quote_id": _quote_id(quote_counter),
        "demographic": "Millennial_Parent",
        "product_type": "Term_Life_20Yr",
        "drop_off_stage": "Medical_History",
        "face_amount": random.choice([500000, 750000, 1000000]),
        "premium_quoted": round(random.uniform(35.0, 85.0), 2),
        "applicant_info": {
            "age": random.randint(28, 41),
            "state": random.choice(STATES),
            "smoker_status": "Non-Tobacco"
        },
        "quote_date": str(quote_date),
        "last_updated": NOW.isoformat(),
    })
    quote_counter += 1

quote_rows.extend(demo_cohort)
# Track these IDs so we explicitly DO NOT issue policies for them
stalled_demo_ids = {item["quote_id"] for item in demo_cohort}

# ── Background quote rows ─────────────────────────────────────────────
# Generate background noise across all demographics and drop-off stages
for _ in range(42):
    demo = random.choice(DEMOGRAPHICS)
    prod = random.choice(PRODUCT_TYPES)
    stage = random.choice(DROP_OFF_STAGES)
    quote_date = _random_date_between(TODAY - timedelta(days=90), TODAY)
    
    quote_rows.append({
        "quote_id": _quote_id(quote_counter),
        "demographic": demo,
        "product_type": prod,
        "drop_off_stage": stage,
        "face_amount": random.choice([100000, 250000, 500000, 1000000]),
        "premium_quoted": round(random.uniform(20.0, 150.0), 2),
        "applicant_info": {
            "age": random.randint(22, 65),
            "state": random.choice(STATES),
            "smoker_status": random.choice(["Non-Tobacco", "Tobacco"])
        },
        "quote_date": str(quote_date),
        "last_updated": NOW.isoformat(),
    })
    quote_counter += 1

print(f"  application_quotes: {len(quote_rows)} rows prepared")

# ---------------------------------------------------------------------------
# 2. Build issued_policies rows (The "Sales")
# ---------------------------------------------------------------------------
policy_rows: list[dict] =[]

# Separate background quotes (excluding our core Millennial Stalled cohort)
background_quotes = [q for q in quote_rows if q["quote_id"] not in stalled_demo_ids]
random.shuffle(background_quotes)

# Assign conversion tiers to background quotes
n = len(background_quotes)
issued_quotes = background_quotes[: int(n * 0.40)]       # ~16 items convert to policies
stalled_quotes = background_quotes[int(n * 0.40): int(n * 0.70)] # ~12 items (stalled)
abandoned_quotes = background_quotes[int(n * 0.70):]     # ~14 items (abandoned)

policy_counter = 1

def _issue_policy(quote: dict, days_delay: int) -> dict:
    global policy_counter
    # Policy is issued a few days after the quote date
    issue_date = date.fromisoformat(quote["quote_date"]) + timedelta(days=days_delay)
    if issue_date > TODAY:
        issue_date = TODAY # Cap it at today
        
    issue_ts = datetime.combine(
        issue_date,
        datetime.min.time().replace(hour=random.randint(9, 17), minute=random.randint(0, 59)),
        tzinfo=timezone.utc,
    )
    policy = {
        "policy_id": f"POL-{policy_counter:04d}",
        "quote_id": quote["quote_id"],
        "issue_timestamp": issue_ts.isoformat(),
        "premium_collected": quote["premium_quoted"],
    }
    policy_counter += 1
    return policy

# Issue policies for the "Issued" cohort
for quote in issued_quotes:
    # Most policies take 1-5 days to issue
    policy_rows.append(_issue_policy(quote, random.randint(1, 5)))

# Explicitly NO policies issued for stalled_quotes, abandoned_quotes, and demo_cohort

print(f"  issued_policies: {len(policy_rows)} rows prepared")

# ---------------------------------------------------------------------------
# 3. Insert into BigQuery
# ---------------------------------------------------------------------------
client = bigquery.Client(project=PROJECT_ID)

print(f"\n>>> Inserting into {TABLE_QUOTES} ...")
errors = client.insert_rows_json(TABLE_QUOTES, quote_rows)
if errors:
    print(f"  ERROR inserting quote rows: {errors}")
else:
    print(f"  ✓ {len(quote_rows)} rows inserted into application_quotes")

print(f"\n>>> Inserting into {TABLE_POLICIES} ...")
errors = client.insert_rows_json(TABLE_POLICIES, policy_rows)
if errors:
    print(f"  ERROR inserting policy rows: {errors}")
else:
    print(f"  ✓ {len(policy_rows)} rows inserted into issued_policies")

print("\nDone! Data successfully staged for Agent Analysis.")
