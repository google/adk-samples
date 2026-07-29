#!/bin/bash
# deploy-to-cloud-run.sh
# Deploys the YouTube Analyst agent to Google Cloud Run.
set -e

# Self-detect the repository root
# The script is located at: python/agents/youtube-analyst/deployment/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

echo "🚀 Deploying to Cloud Run from: ${REPO_ROOT}"
cd "${REPO_ROOT}"

export AGENT_NAME=youbuddy-cr-${RANDOM}
export UV_NO_CONFIG=1
export UV_INDEX_URL="https://pypi.org/simple"

# Create the project using Agent Starter Pack
uvx agent-starter-pack==0.15.4 create ${AGENT_NAME} \
    -d cloud_run \
    -ag \
    -a local@python/agents/youtube-analyst

cd "${AGENT_NAME}"

# Build and deploy the backend
make install
make backend
