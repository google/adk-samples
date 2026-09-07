
# ==============================================================================
# Installation & Setup
# ==============================================================================

# Install dependencies using uv package manager
install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh; source $HOME/.local/bin/env; }
	uv sync

# ==============================================================================
# Playground Targets
# ==============================================================================

# Launch local dev playground
playground:
	@echo "==============================================================================="
	@echo "| Starting your agent playground...                                            |"
	@echo "|                                                                             |"
	@echo "| Try asking: What's the weather in San Francisco?                             |"
	@echo "|                                                                             |"
	@echo "| IMPORTANT: Select the 'app' folder to interact with your agent.              |"
	@echo "==============================================================================="
	uv run adk web . --port 8501 --reload_agents

# ==============================================================================
# Local Development Commands
# ==============================================================================

# Launch local development server with hot-reload (Cloud Run entry point)
# Usage: make local-server [PORT=8000]
local-server:
	uv run uvicorn app.fast_api_app:app --host localhost --port $(or $(PORT),8000) --reload

# ==============================================================================
# Deployment Targets
# ==============================================================================

# Deploy to Vertex AI Agent Engine
# Usage: make deploy-agent-engine [AGENT_IDENTITY=true] [SECRETS="KEY=SECRET_ID,..."]
deploy-agent-engine:
	# Export dependencies to requirements file using uv export.
	(uv export --no-hashes --no-header --no-dev --no-emit-project --no-annotate > app/app_utils/.requirements.txt 2>/dev/null || \
	uv export --no-hashes --no-header --no-dev --no-emit-project > app/app_utils/.requirements.txt) && \
	uv run -m app.app_utils.deploy \
		--source-packages=./app \
		--entrypoint-module=app.agent_engine_app \
		--entrypoint-object=agent_engine \
		--requirements-file=app/app_utils/.requirements.txt \
		$(if $(AGENT_IDENTITY),--agent-identity) \
		$(if $(filter command line,$(origin SECRETS)),--set-secrets="$(SECRETS)")

# Deploy to Cloud Run
# Usage: make deploy-cloud-run [IAP=true] [PORT=8080]
deploy-cloud-run:
	PROJECT_ID=$$(gcloud config get-value project) && \
	gcloud beta run deploy memory-bank-sample \
		--source . \
		--memory "4Gi" \
		--project $$PROJECT_ID \
		--region "us-west1" \
		--no-allow-unauthenticated \
		--no-cpu-throttling \
		--labels "created-by=adk" \
		--update-build-env-vars "AGENT_VERSION=$(shell awk -F'"' '/^version = / {print $$2}' pyproject.toml || echo '0.0.0')" \
		$(if $(IAP),--iap) \
		$(if $(PORT),--port=$(PORT))

# ==============================================================================
# Testing & Code Quality
# ==============================================================================

# Run unit and integration tests
test:
	uv sync --dev
	uv run pytest tests/unit && uv run pytest tests/integration

# Run code quality checks (codespell, ruff, ty)
lint:
	uv sync --dev --extra lint
	uv run codespell
	uv run ruff check . --diff
	uv run ruff format . --check --diff
	uv run ty check .
