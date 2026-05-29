#!/bin/bash

# Smoke-test the markdown-to-PDF Cloud Run conversion endpoint.
#
# Usage:
#   BASE_URL=https://<your-cloud-run-service>.a.run.app \
#   GCS_URI=gs://<your-output-bucket>/<path>/<file>.md \
#       ./test_endpoint.sh
#
# Note: Ensure you are authenticated with gcloud (gcloud auth login) before running.

BASE_URL="${BASE_URL:-https://your-conversion-service-url.a.run.app}"
GCS_URI="${GCS_URI:-gs://your-output-bucket/path/to/file.md}"
TOKEN=$(gcloud auth print-identity-token 2>/dev/null)

echo "========================================================="
echo "Testing Conversion Endpoint"
echo "Base URL: $BASE_URL"
echo "GCS URI:  $GCS_URI"
echo "========================================================="

# Prepare curl command as an array to easily inject the auth header if present
CURL_CMD=(curl -X POST -G "${BASE_URL}/" --data-urlencode "gcs_uri=${GCS_URI}" -H "accept: application/json")

if [ -n "$TOKEN" ]; then
    echo "Using Bearer authentication token."
    CURL_CMD+=(-H "Authorization: Bearer ${TOKEN}")
fi

echo -e "\nExecuting command:"
echo "${CURL_CMD[@]}"
echo -e "\nResponse:"

"${CURL_CMD[@]}" -w "\n\nHTTP Status: %{http_code}\n"
