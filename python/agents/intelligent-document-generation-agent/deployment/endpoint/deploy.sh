#!/bin/bash
# Deploy to Google Cloud Run

echo "Deploying PDF Conversion Endpoint to Google Cloud Run..."

SERVICE_NAME="md-to-pdf-converter"
REGION="us-central1"

gcloud run deploy $SERVICE_NAME \
    --source . \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi

echo "Deployment complete!"
