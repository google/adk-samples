#!/bin/bash
# deploy-to-agent-engine.sh
# Deploys the YouTube Analyst agent to Vertex AI Agent Engine.
set -e

# Self-detect the repository root
# The script is located at: python/agents/youtube-analyst/deployment/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

echo "🚀 Deploying to Agent Engine from: ${REPO_ROOT}"
cd "${REPO_ROOT}"

export AGENT_NAME=youbuddy-${RANDOM}
export UV_NO_CONFIG=1
export UV_INDEX_URL="https://pypi.org/simple"
uvx agent-starter-pack==0.15.4 create ${AGENT_NAME} -d agent_engine -ag -a local@python/agents/youtube-analyst
cd "${AGENT_NAME}"
make install
make backend
