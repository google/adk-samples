#!/bin/bash
# Deploy to Google Cloud Run
set -euo pipefail

cd "$(dirname "$0")"

echo "Deploying PDF Conversion Endpoint to Google Cloud Run..."

ENV_FILE="../../.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found." >&2
    exit 1
fi

read_env() {
    local key="$1"
    grep -E "^${key}=" "$ENV_FILE" | tail -n1 | cut -d= -f2- | tr -d '[:space:]'
}

require_env() {
    local key="$1"
    local val
    val="$(read_env "$key")"
    if [[ -z "$val" ]]; then
        echo "Error: ${key} is not set in $ENV_FILE." >&2
        exit 1
    fi
    echo "$val"
}

REGION="$(require_env GOOGLE_CLOUD_LOCATION)"
PROJECT_ID="$(require_env GOOGLE_CLOUD_PROJECT)"
OUTPUT_BUCKET="$(require_env ADK_OUTPUT_BUCKET)"

SERVICE_NAME="md-to-pdf-converter"

gcloud run deploy $SERVICE_NAME \
    --source . \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --allow-unauthenticated \
    --memory 1Gi

# Grant the Cloud Run runtime SA (default compute SA) read/write on the output
# bucket so the converter can fetch the source markdown and write the PDF back.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Granting roles/storage.objectAdmin on gs://${OUTPUT_BUCKET} to ${RUNTIME_SA}..."
gcloud storage buckets add-iam-policy-binding "gs://${OUTPUT_BUCKET}" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/storage.objectAdmin" \
    --project "$PROJECT_ID" >/dev/null

# Required for the converter to mint V4 signed URLs (blob.generate_signed_url)
# without a downloaded key — calls IAM signBlob on itself.
echo "Granting roles/iam.serviceAccountTokenCreator on ${RUNTIME_SA} to itself..."
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/iam.serviceAccountTokenCreator" \
    --project "$PROJECT_ID" >/dev/null

echo "Deployment complete!"
