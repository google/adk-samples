#!/bin/bash
# =============================================================================
# BigQuery Setup Script — FS Corp Fabric App Quotes & Policies Data Architecture
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Load Environment Variables from the parent directory .env file
# ---------------------------------------------------------------------------
if [ -f "../.env" ]; then
    source ../.env
else
    echo "ERROR: .env file not found in the parent directory."
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET="fabric_dtc_data"
LOCATION="us-central1"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: PROJECT_ID is empty. Check your .env file."
  exit 1
fi

echo "============================================="
echo " BigQuery Setup for FS Corp Fabric Demo"
echo " Project : ${PROJECT_ID}"
echo " Dataset : ${DATASET}"
echo " Location: ${LOCATION}"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. Create the dataset
# ---------------------------------------------------------------------------
echo ""
echo ">>> Checking / creating dataset '${DATASET}' ..."

if bq --project_id="${PROJECT_ID}" show "${DATASET}" > /dev/null 2>&1; then
  echo "    Dataset '${DATASET}' already exists — skipping."
else
  bq --project_id="${PROJECT_ID}" mk \
    --dataset \
    --location="${LOCATION}" \
    --description="Fabric App DTC pipeline and stalled application data" \
    "${DATASET}"
  echo "    Dataset '${DATASET}' created."
fi

# ---------------------------------------------------------------------------
# 2. Create the application_quotes table
# ---------------------------------------------------------------------------
TABLE_QUOTES="${DATASET}.application_quotes"

echo ""
echo ">>> Creating table '${TABLE_QUOTES}' ..."

SCHEMA_FILE=$(mktemp /tmp/fabric_quotes_schema.XXXXXX.json)
cat > "${SCHEMA_FILE}" <<'SCHEMA'
[
  {"name": "quote_id",          "type": "STRING",    "mode": "REQUIRED"},
  {"name": "demographic",       "type": "STRING"},
  {"name": "product_type",      "type": "STRING"},
  {"name": "drop_off_stage",    "type": "STRING"},
  {"name": "face_amount",       "type": "NUMERIC"},
  {"name": "premium_quoted",    "type": "NUMERIC"},
  {"name": "applicant_info",    "type": "RECORD", "fields":[
    {"name": "age", "type": "INT64"},
    {"name": "state",  "type": "STRING"},
    {"name": "smoker_status", "type": "STRING"}
  ]},
  {"name": "quote_date",        "type": "DATE"},
  {"name": "last_updated",      "type": "TIMESTAMP"}
]
SCHEMA

bq --project_id="${PROJECT_ID}" mk \
  --table \
  --description="Tracks started term-life quotes and user demographics" \
  "${TABLE_QUOTES}" \
  "${SCHEMA_FILE}" \
  2>/dev/null \
  && echo "    Table '${TABLE_QUOTES}' created." \
  || echo "    Table '${TABLE_QUOTES}' already exists — skipping."

rm -f "${SCHEMA_FILE}"

# ---------------------------------------------------------------------------
# 3. Create the issued_policies table
# ---------------------------------------------------------------------------
TABLE_POLICIES="${DATASET}.issued_policies"

echo ""
echo ">>> Creating table '${TABLE_POLICIES}' ..."

bq --project_id="${PROJECT_ID}" mk \
  --table \
  --description="Records successfully issued policies (converted quotes)" \
  "${TABLE_POLICIES}" \
  'policy_id:STRING,
   quote_id:STRING,
   issue_timestamp:TIMESTAMP,
   premium_collected:NUMERIC' \
  2>/dev/null \
  && echo "    Table '${TABLE_POLICIES}' created." \
  || echo "    Table '${TABLE_POLICIES}' already exists — skipping."

# ---------------------------------------------------------------------------
# 4. Create the stalled_apps_view
# ---------------------------------------------------------------------------
VIEW_STALLED_APPS="${DATASET}.stalled_apps_view"

echo ""
echo ">>> Creating view '${VIEW_STALLED_APPS}' ..."

bq --project_id="${PROJECT_ID}" mk \
  --use_legacy_sql=false \
  --view "
SELECT
  q.quote_id,
  q.demographic,
  q.product_type,
  q.drop_off_stage,
  q.face_amount,
  q.premium_quoted,
  q.applicant_info.age,
  q.applicant_info.state,
  q.applicant_info.smoker_status,
  q.quote_date,
  p.issue_timestamp,
  COALESCE(
    DATE_DIFF(CURRENT_DATE(), DATE(p.issue_timestamp), DAY),
    DATE_DIFF(CURRENT_DATE(), q.quote_date, DAY)
  ) AS days_since_quote,
  CASE
    WHEN p.issue_timestamp IS NOT NULL THEN 'Issued'
    WHEN DATE_DIFF(CURRENT_DATE(), q.quote_date, DAY) <= 7 THEN 'Recent Quote'
    WHEN DATE_DIFF(CURRENT_DATE(), q.quote_date, DAY) <= 30 THEN 'Stalled App'
    ELSE 'Abandoned'
  END AS conversion_status,
  CASE
    WHEN p.issue_timestamp IS NOT NULL THEN 'Policy Active - No Action'
    WHEN DATE_DIFF(CURRENT_DATE(), q.quote_date, DAY) <= 7 THEN 'Standard Email Nurture'
    WHEN DATE_DIFF(CURRENT_DATE(), q.quote_date, DAY) <= 30 THEN 'Trigger Accelerated Frictionless Campaign'
    ELSE 'Archive / Long-term Retargeting'
  END AS recommendation
FROM
  \`${PROJECT_ID}.${DATASET}.application_quotes\` AS q
LEFT JOIN (
  SELECT
    quote_id,
    MIN(issue_timestamp) AS issue_timestamp
  FROM
    \`${PROJECT_ID}.${DATASET}.issued_policies\`
  GROUP BY
    quote_id
) AS p
  ON q.quote_id = p.quote_id
ORDER BY
  days_since_quote DESC
" \
  --description="Automatically flags unissued quotes by conversion status (Issued / Recent / Stalled)" \
  "${VIEW_STALLED_APPS}" \
  2>/dev/null \
  && echo "    View '${VIEW_STALLED_APPS}' created." \
  || echo "    View '${VIEW_STALLED_APPS}' already exists — skipping."

echo ""
echo "============================================="
echo " Setup complete!"
echo "============================================="