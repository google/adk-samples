@echo off
REM Deploy to Google Cloud Run

echo Deploying PDF Conversion Endpoint to Google Cloud Run...

set SERVICE_NAME=md-to-pdf-converter
set REGION=us-central1

gcloud run deploy %SERVICE_NAME% ^
    --source . ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory 1Gi

echo Deployment complete!
pause
