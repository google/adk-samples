#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# Deploy Script for Ad Campaign Agent to Vertex AI Agent Engine
# =============================================================================
# This script deploys the ad campaign agent to Google Cloud's Vertex AI
# Agent Engine using the ADK CLI.
#
# Agent Engine provides:
#   - Managed container lifecycle (no Dockerfile needed)
#   - Managed sessions via VertexAiSessionService
#   - Built-in memory bank for persistent memory
#   - Cloud Trace integration for observability
#   - Query API for programmatic access
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - adk CLI installed (pip install google-adk)
#   - GCS bucket created for staging and assets
#   - google-cloud-aiplatform[adk,agent_engines] installed
#
# Usage:
#   ./scripts/deploy_ae.sh                    # Deploy with defaults
#   ./scripts/deploy_ae.sh --dry-run          # Show commands without executing
#   ./scripts/deploy_ae.sh --update           # Update existing deployment
#   ./scripts/deploy_ae.sh --trace            # Enable Cloud Trace
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-kaggle-on-gcp}"
REGION="${REGION:-us-central1}"
GCS_BUCKET="${GCS_BUCKET:-kaggle-on-gcp-ad-campaign-assets}"
STAGING_BUCKET="gs://${GCS_BUCKET}"
DISPLAY_NAME="Ad Campaign Agent"
DESCRIPTION="Fashion retail ad campaign management with Veo 3.1 video generation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory (for relative paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AGENT_DIR="$PROJECT_DIR/app"

# =============================================================================
# Helper Functions
# =============================================================================
print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}  $1${NC}"
}

# =============================================================================
# Parse Arguments
# =============================================================================
DRY_RUN=false
ENABLE_TRACE=false
UPDATE_MODE=false
AGENT_ENGINE_ID=""

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --trace)
            ENABLE_TRACE=true
            shift
            ;;
        --update)
            UPDATE_MODE=true
            shift
            ;;
        --agent-engine-id=*)
            AGENT_ENGINE_ID="${arg#*=}"
            UPDATE_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run              Show deployment command without executing"
            echo "  --trace                Enable Cloud Trace for observability"
            echo "  --update               Update existing Agent Engine instance"
            echo "  --agent-engine-id=ID   Specify Agent Engine ID to update"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Environment Variables (optional overrides):"
            echo "  GOOGLE_CLOUD_PROJECT   GCP project ID (default: kaggle-on-gcp)"
            echo "  REGION                 Deployment region (default: us-central1)"
            echo "  GCS_BUCKET             GCS bucket name (default: kaggle-on-gcp-ad-campaign-assets)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Fresh deployment"
            echo "  $0 --trace                   # Deploy with Cloud Trace enabled"
            echo "  $0 --dry-run                 # Preview the deployment command"
            echo "  $0 --agent-engine-id=12345   # Update existing deployment"
            exit 0
            ;;
    esac
done

# Allow environment variable overrides
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-kaggle-on-gcp}"
REGION="${REGION:-us-central1}"
GCS_BUCKET="${GCS_BUCKET:-kaggle-on-gcp-ad-campaign-assets}"
STAGING_BUCKET="gs://${GCS_BUCKET}"

# =============================================================================
# Pre-flight Checks
# =============================================================================
print_header "Pre-flight Checks"

# Check gcloud
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi
print_success "gcloud CLI found"

# Check adk - prefer .venv-deploy version, then system
if [ -f "$PROJECT_DIR/.venv-deploy/bin/adk" ]; then
    ADK_CMD="$PROJECT_DIR/.venv-deploy/bin/adk"
    print_success "adk CLI found: .venv-deploy/bin/adk"
elif [ -f "$PROJECT_DIR/.venv/bin/adk" ]; then
    ADK_CMD="$PROJECT_DIR/.venv/bin/adk"
    print_success "adk CLI found: .venv/bin/adk"
elif command -v adk &> /dev/null; then
    ADK_CMD="adk"
    print_success "adk CLI found (system)"
else
    print_error "adk CLI not found. Please install: pip install google-adk"
    exit 1
fi

# Check agent directory
if [ ! -d "$AGENT_DIR" ]; then
    print_error "Agent directory not found: $AGENT_DIR"
    exit 1
fi
print_success "Agent directory found: $AGENT_DIR"

# Check agent_engine_app.py exists
if [ ! -f "$AGENT_DIR/agent_engine_app.py" ]; then
    print_warning "agent_engine_app.py not found. Using agent.py with root_agent."
fi

# Check if authenticated
if ! gcloud auth print-identity-token &> /dev/null; then
    print_warning "Not authenticated with gcloud. Running: gcloud auth login"
    gcloud auth login
fi
print_success "gcloud authenticated"

# Set project
gcloud config set project "$GOOGLE_CLOUD_PROJECT" 2>/dev/null
print_success "Project set to: $GOOGLE_CLOUD_PROJECT"

# =============================================================================
# Configuration Summary
# =============================================================================
print_header "Agent Engine Deployment Configuration"

echo ""
echo "Configuration:"
print_info "Project:         $GOOGLE_CLOUD_PROJECT"
print_info "Region:          $REGION"
print_info "Display Name:    $DISPLAY_NAME"
print_info "Staging Bucket:  $STAGING_BUCKET"
print_info "Agent Path:      $AGENT_DIR"
print_info "Cloud Trace:     $ENABLE_TRACE"
print_info "Env File:        app/.env (telemetry, model location)"
if [ "$UPDATE_MODE" = true ]; then
    print_info "Mode:            UPDATE (Agent Engine ID: $AGENT_ENGINE_ID)"
else
    print_info "Mode:            NEW DEPLOYMENT"
fi
echo ""

# =============================================================================
# Build Deploy Command
# =============================================================================

# Base command
DEPLOY_CMD="$ADK_CMD deploy agent_engine"
DEPLOY_CMD="$DEPLOY_CMD --project=$GOOGLE_CLOUD_PROJECT"
DEPLOY_CMD="$DEPLOY_CMD --region=$REGION"
DEPLOY_CMD="$DEPLOY_CMD --staging_bucket=$STAGING_BUCKET"
DEPLOY_CMD="$DEPLOY_CMD --display_name=\"$DISPLAY_NAME\""
DEPLOY_CMD="$DEPLOY_CMD --description=\"$DESCRIPTION\""

# Optional: Cloud Trace
if [ "$ENABLE_TRACE" = true ]; then
    DEPLOY_CMD="$DEPLOY_CMD --trace_to_cloud"
fi

# Optional: Update existing deployment
if [ "$UPDATE_MODE" = true ] && [ -n "$AGENT_ENGINE_ID" ]; then
    DEPLOY_CMD="$DEPLOY_CMD --agent_engine_id=$AGENT_ENGINE_ID"
fi

# Environment variables are read from app/.env file automatically
# The .env file contains:
# - GOOGLE_CLOUD_LOCATION=global (for Gemini 3.x models)
# - GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true (observability dashboard)
# - OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true (prompt/response logging)
# - GCS_BUCKET (asset storage)

# ADK CLI will auto-generate agent_engine_app.py and import root_agent from agent.py
# No need to specify --adk_app or --adk_app_object - using defaults is correct

# Agent path (must be last)
DEPLOY_CMD="$DEPLOY_CMD $AGENT_DIR"

# =============================================================================
# Deploy
# =============================================================================
print_header "Deployment Command"

echo ""
echo "Command to execute:"
echo -e "${YELLOW}$DEPLOY_CMD${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "Dry run mode - not executing"
    echo ""
    echo "To deploy, run without --dry-run:"
    echo "  ./scripts/deploy_ae.sh"
    exit 0
fi

# Confirm before deploying
read -p "Proceed with Agent Engine deployment? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

print_header "Deploying to Agent Engine"

# Change to project directory and deploy
cd "$PROJECT_DIR"
eval $DEPLOY_CMD

# =============================================================================
# Post-Deployment
# =============================================================================
print_header "Deployment Complete!"

echo ""
echo -e "${GREEN}Agent Engine Deployment Successful!${NC}"
echo ""
echo "Your agent is now deployed to Vertex AI Agent Engine."
echo ""
echo -e "${GREEN}Query API Endpoint:${NC}"
echo -e "  ${CYAN}https://${REGION}-aiplatform.googleapis.com/v1/projects/${GOOGLE_CLOUD_PROJECT}/locations/${REGION}/reasoningEngines/{RESOURCE_ID}:query${NC}"
echo ""
echo -e "${GREEN}Stream Query API Endpoint:${NC}"
echo -e "  ${CYAN}https://${REGION}-aiplatform.googleapis.com/v1/projects/${GOOGLE_CLOUD_PROJECT}/locations/${REGION}/reasoningEngines/{RESOURCE_ID}:streamQuery${NC}"
echo ""
echo "Replace {RESOURCE_ID} with the ID shown in the deployment output above."
echo ""
echo "Useful commands:"
echo "  # List deployed agents"
echo "  gcloud ai reasoning-engines list --project=$GOOGLE_CLOUD_PROJECT --region=$REGION"
echo ""
echo "  # Query your agent (Python)"
echo "  from vertexai import agent_engines"
echo "  agent = agent_engines.get('{RESOURCE_ID}')"
echo "  response = agent.query(input='Hello, help me with campaigns')"
echo ""
echo "  # Delete agent"
echo "  gcloud ai reasoning-engines delete {RESOURCE_ID} --project=$GOOGLE_CLOUD_PROJECT --region=$REGION"
echo ""
echo "  # Update deployment"
echo "  ./scripts/deploy_ae.sh --agent-engine-id={RESOURCE_ID}"
echo ""

# =============================================================================
# Comparison with Cloud Run
# =============================================================================
print_header "Deployment Options"

echo ""
echo "You now have two deployment options:"
echo ""
echo "  1. ${GREEN}Agent Engine${NC} (this script):"
echo "     ./scripts/deploy_ae.sh"
echo "     - Managed sessions and memory"
echo "     - Query API (no web UI)"
echo "     - Automatic scaling"
echo ""
echo "  2. ${GREEN}Cloud Run${NC} (original script):"
echo "     ./scripts/deploy.sh"
echo "     - ADK Web UI at /dev-ui"
echo "     - Full container control"
echo "     - Custom memory/CPU settings"
echo ""
