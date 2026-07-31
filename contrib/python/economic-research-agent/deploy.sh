#!/usr/bin/env bash
# Copyright 2026 Google LLC
# Production Deploy and Registration Script for Evolved Economic Research Agent (ERA)
set -e

# Load local .env if it exists
if [ -f ".env" ]; then
    echo "🔑 Loading local .env parameters..."
    export $(grep -v '^#' .env | xargs)
fi

# Usage/Help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --gemini-enterprise-app-id ID    Full Gemini Enterprise app resource name"
    echo "  --region REGION                  Target GCP Region (default: read from agents-cli-manifest.yaml)"
    echo "  --no-register                    Deploy to Agent Runtime only (skip Gemini Enterprise registration)"
    echo "  --help                           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh --gemini-enterprise-app-id projects/123/locations/us/collections/default_collection/engines/my-engine"
    echo "  ./deploy.sh --no-register"
    exit 0
}

# Parse Arguments
APP_ID="${GEMINI_ENTERPRISE_APP_ID:-}"
NO_REGISTER=false
DRY_RUN=false
REGION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gemini-enterprise-app-id)
            APP_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-register)
            NO_REGISTER=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "⚠️ Unknown parameter: $1"
            show_help
            ;;
    esac
done


# Step 1: Verify agents-cli installation
if ! command -v agents-cli &> /dev/null; then
    echo "❌ Error: agents-cli tool not found!"
    echo "Please install it using: uv tool install google-agents-cli"
    exit 1
fi

# Step 2: Extract all API keys from .env to compile the update-env-vars string
echo "🛰️ Compiling live API grounding keys from .env..."
UPDATE_VARS=""
API_KEYS=(
    "BEA_API_KEY" "FRED_API_KEY" "CENSUS_API_KEY" "BLS_API_KEY" 
    "FEC_API_KEY" "HUD_API_KEY" "EIA_API_KEY" "NEWS_API_KEY" 
    "SERPER_API_KEY" "RENTCAST_API_KEY" "ONET_API_KEY"
)

for key in "${API_KEYS[@]}"; do
    eval value=\$$key
    if [ ! -z "$value" ]; then
        if [ ! -z "$UPDATE_VARS" ]; then
            UPDATE_VARS="${UPDATE_VARS},"
        fi
        UPDATE_VARS="${UPDATE_VARS}${key}=${value}"
    fi
done

# Step 3: Execute deployment to Vertex AI Agent Runtime
DEPLOY_CMD="agents-cli deploy --no-confirm-project"
if [ "$DRY_RUN" = true ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --dry-run"
fi
if [ ! -z "$REGION" ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --region ${REGION}"
fi
if [ ! -z "$UPDATE_VARS" ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --update-env-vars=\"${UPDATE_VARS}\""
fi


echo "🚀 Executing Agent Runtime deployment..."
echo "Command: $DEPLOY_CMD"
eval "$DEPLOY_CMD"

# Step 4: Execute registration to Gemini Enterprise (Unless skipped or dry-run)
if [ "$NO_REGISTER" = false ] && [ "$DRY_RUN" = false ]; then
    echo ""
    echo "🤖 Registering Agent to Gemini Enterprise..."
    
    REG_CMD="agents-cli publish gemini-enterprise"
    if [ ! -z "$APP_ID" ]; then
        REG_CMD="${REG_CMD} --gemini-enterprise-app-id=\"${APP_ID}\""
    fi
    
    echo "Command: $REG_CMD"
    eval "$REG_CMD"
else
    echo ""
    if [ "$DRY_RUN" = true ]; then
        echo "ℹ️ Skipping Gemini Enterprise registration (Dry-run mode enabled)."
    else
        echo "ℹ️ Skipping Gemini Enterprise registration (--no-register flag provided)."
    fi
fi


echo ""
echo "🥂 Deployed & Registered Successfully!"
