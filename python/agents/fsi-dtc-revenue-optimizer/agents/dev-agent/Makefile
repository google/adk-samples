
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
	@echo "| 🚀 Starting your agent playground...                                        |"
	@echo "|                                                                             |"
	@echo "| 💡 Try asking: Help me debug this issue                                     |"
	@echo "|                                                                             |"
	@echo "| 🔍 IMPORTANT: Select the 'app' folder to interact with your agent.          |"
	@echo "==============================================================================="
	uv run adk web . --port 8501 --reload_agents

# ==============================================================================
# Local Development Commands
# ==============================================================================

# Launch local development server with hot-reload
# Usage: make local-backend [PORT=8000] - Specify PORT for parallel scenario testing
local-backend:
	uv run uvicorn app.fast_api_app:app --host localhost --port $(or $(PORT),8000) --reload

# ==============================================================================
# A2A Protocol Inspector
# ==============================================================================

# Launch A2A Protocol Inspector to test your agent implementation
inspector: setup-inspector-if-needed build-inspector-if-needed
	@echo "==============================================================================="
	@echo "| 🔍 A2A Protocol Inspector                                                  |"
	@echo "==============================================================================="
	@echo "| 🌐 Inspector UI: http://localhost:5001                                     |"
	@echo "|                                                                             |"
	@echo "| 💡 Testing Locally:                                                         |"
	@echo "|    Paste this URL into the inspector:                                      |"
	@echo "|    http://localhost:8000/.well-known/agent-card.json              |"
	@echo "|                                                                             |"
	@echo "| 💡 Testing Remote Deployment:                                               |"
	@echo "|    <SERVICE_URL>/.well-known/agent-card.json"
	@echo "|    (Get SERVICE_URL from 'make deploy' output or Cloud Console)            |"
	@echo "|                                                                             |"
	@echo "|    🔐 Auth: Expand 'Authentication & Headers', select 'Bearer Token',       |"
	@echo "|       and paste output of: gcloud auth print-identity-token                |"
	@echo "==============================================================================="
	@echo ""
	cd tools/a2a-inspector/backend && uv run app.py

# Internal: Setup inspector if not already present (runs once)
# TODO: Update to --branch v1.0.0 when a2a-inspector publishes releases
setup-inspector-if-needed:
	@if [ ! -d "tools/a2a-inspector" ]; then \
		echo "" && \
		echo "📦 First-time setup: Installing A2A Inspector..." && \
		echo "" && \
		mkdir -p tools && \
		git clone --quiet https://github.com/a2aproject/a2a-inspector.git tools/a2a-inspector && \
		(cd tools/a2a-inspector && git -c advice.detachedHead=false checkout --quiet 893e4062f6fbd85a8369228ce862ebbf4a025694) && \
		echo "📥 Installing Python dependencies..." && \
		(cd tools/a2a-inspector && uv sync --quiet) && \
		echo "📥 Installing Node.js dependencies..." && \
		(cd tools/a2a-inspector/frontend && npm install --silent) && \
		echo "🔨 Building frontend..." && \
		(cd tools/a2a-inspector/frontend && npm run build --silent) && \
		echo "" && \
		echo "✅ A2A Inspector setup complete!" && \
		echo ""; \
	fi

# Internal: Build inspector frontend if needed
build-inspector-if-needed:
	@if [ -d "tools/a2a-inspector" ] && [ ! -f "tools/a2a-inspector/frontend/public/script.js" ]; then \
		echo "🔨 Building inspector frontend..."; \
		cd tools/a2a-inspector/frontend && npm run build; \
	fi

# ==============================================================================
# Secret Manager
# ==============================================================================

# Secrets that should be stored in Secret Manager (not as plain-text env vars)
SECRET_KEYS := GOOGLE_CHAT_WEBHOOK_URL JIRA_API_TOKEN

# Create or update secrets in Secret Manager from local .env values (one-time setup)
create-secrets:
	@echo "==============================================================================="
	@echo "| 🔐 Creating/updating secrets in Secret Manager                              |"
	@echo "==============================================================================="
	@PROJECT_ID=$$(gcloud config get-value project 2>/dev/null) && \
	for SECRET_NAME in $(SECRET_KEYS); do \
		VALUE=$$(grep "^$$SECRET_NAME=" .env | head -1 | sed 's/^[^=]*=//' | sed 's/^["'\'']*//;s/["'\'']*$$//') ; \
		if [ -z "$$VALUE" ]; then \
			echo "⚠️  Skipping $$SECRET_NAME (not found in .env)"; \
			continue; \
		fi; \
		if gcloud secrets describe $$SECRET_NAME --project=$$PROJECT_ID >/dev/null 2>&1; then \
			printf '%s' "$$VALUE" | gcloud secrets versions add $$SECRET_NAME --data-file=- --project=$$PROJECT_ID >/dev/null; \
			echo "✅ Updated secret: $$SECRET_NAME"; \
		else \
			printf '%s' "$$VALUE" | gcloud secrets create $$SECRET_NAME --data-file=- --replication-policy=automatic --project=$$PROJECT_ID >/dev/null; \
			echo "✅ Created secret: $$SECRET_NAME"; \
		fi; \
	done
	@echo ""
	@echo "Done. Run 'make grant-secret-access' to grant Cloud Run access."

# Grant the Cloud Run service account access to read the secrets
grant-secret-access:
	@echo "==============================================================================="
	@echo "| 🔑 Granting Cloud Run secret access                                        |"
	@echo "==============================================================================="
	@PROJECT_ID=$$(gcloud config get-value project 2>/dev/null) && \
	PROJECT_NUMBER=$$(gcloud projects describe $$PROJECT_ID --format="value(projectNumber)" 2>/dev/null) && \
	SA="$$PROJECT_NUMBER-compute@developer.gserviceaccount.com" && \
	echo "Service account: $$SA" && \
	for SECRET_NAME in $(SECRET_KEYS); do \
		if gcloud secrets add-iam-policy-binding $$SECRET_NAME \
			--member="serviceAccount:$$SA" \
			--role="roles/secretmanager.secretAccessor" \
			--project=$$PROJECT_ID --quiet > /dev/null 2>&1; then \
			echo "✅ Granted access to $$SECRET_NAME"; \
		else \
			echo "⚠️  Could not grant access to $$SECRET_NAME (may already be set, or insufficient permissions)"; \
		fi; \
	done

# ==============================================================================
# Backend Deployment Targets
# ==============================================================================

# Deploy the agent remotely
# Usage: make deploy [IAP=true] [PORT=8080] - Set IAP=true to enable Identity-Aware Proxy, PORT to specify container port
deploy:
	PROJECT_ID=$$(gcloud config get-value project) && \
	PROJECT_NUMBER=$$(gcloud projects describe $$PROJECT_ID --format="value(projectNumber)") && \
	ENV_VARS=$$(if [ -f .env ]; then awk -F= '!/^#/ && NF && $$1!="GOOGLE_CHAT_WEBHOOK_URL" && $$1!="JIRA_API_TOKEN" { key=$$1; val=$$2; for(i=3; i<=NF; i++) val=val"="$$i; gsub(/^"|"$$/, "", val); gsub(/^'\''|'\''$$/, "", val); printf "~%s=%s", key, val; }' .env; fi) && \
	SECRETS="" && \
	if [ -f .env ] && ! grep -q "^SKIP_JIRA=true" .env; then \
		SECRETS="JIRA_API_TOKEN=JIRA_API_TOKEN:latest"; \
	fi && \
	if [ -f .env ] && ! grep -q "^SKIP_CHAT=true" .env; then \
		if [ -n "$$SECRETS" ]; then SECRETS="GOOGLE_CHAT_WEBHOOK_URL=GOOGLE_CHAT_WEBHOOK_URL:latest,$$SECRETS"; else SECRETS="GOOGLE_CHAT_WEBHOOK_URL=GOOGLE_CHAT_WEBHOOK_URL:latest"; fi; \
	fi && \
	SECRETS_FLAG="" && \
	if [ -n "$$SECRETS" ]; then \
		SECRETS_FLAG="--set-secrets=$$SECRETS"; \
	else \
		SECRETS_FLAG="--clear-secrets"; \
	fi && \
	gcloud beta run deploy dev-agent \
		--source . \
		--memory 8Gi \
		--cpu 2 \
		--project $$PROJECT_ID \
		--region us-central1 \
		--no-allow-unauthenticated \
		--no-cpu-throttling \
		--max-instances=1 \
		--labels created-by=adk \
		--update-env-vars "^~^APP_URL=https://dev-agent-$$PROJECT_NUMBER.us-central1.run.app~AGENT_VERSION=$(shell awk -F'\"' '/^version = / {print $$2}' pyproject.toml || echo '1.0.0')$${ENV_VARS}" \
		$$SECRETS_FLAG \
		$(if $(IAP),--iap) \
		$(if $(PORT),--port=$(PORT))

# Deploy the agent to Vertex AI Agent Engine
# Usage: make deploy-agent-engine [AGENT_ENGINE_ID=<id>] - Specify AGENT_ENGINE_ID to update an existing instance
deploy-agent-engine:
	PROJECT_ID=$$(gcloud config get-value project) && \
	uv run python deploy.py \
		--project=$$PROJECT_ID \
		--region="us-central1" \
		--display_name="dev-agent" \
		$(if $(AGENT_ENGINE_ID),--agent_engine_id=$(AGENT_ENGINE_ID))

# Alias for 'make deploy' for backward compatibility
backend: deploy

# ==============================================================================
# Testing & Code Quality
# ==============================================================================

# Run unit and integration tests
test:
	uv sync --dev
	uv run pytest tests/unit && uv run pytest tests/integration

# ==============================================================================
# Agent Evaluation
# ==============================================================================

# Run agent evaluation using ADK eval
# Usage: make eval [EVALSET=tests/eval/evalsets/basic.evalset.json] [EVAL_CONFIG=tests/eval/eval_config.json]
eval:
	@echo "==============================================================================="
	@echo "| Running Agent Evaluation                                                    |"
	@echo "==============================================================================="
	uv sync --dev --extra eval
	uv run adk eval ./app $${EVALSET:-tests/eval/evalsets/basic.evalset.json} \
		$(if $(EVAL_CONFIG),--config_file_path=$(EVAL_CONFIG),$(if $(wildcard tests/eval/eval_config.json),--config_file_path=tests/eval/eval_config.json,))

# Run evaluation with all evalsets
eval-all:
	@echo "==============================================================================="
	@echo "| Running All Evalsets                                                        |"
	@echo "==============================================================================="
	@for evalset in tests/eval/evalsets/*.evalset.json; do \
		echo ""; \
		echo "▶ Running: $$evalset"; \
		$(MAKE) eval EVALSET=$$evalset || exit 1; \
	done
	@echo ""
	@echo "✅ All evalsets completed"

# Run code quality checks (codespell, ruff, ty)
lint:
	uv sync --dev --extra lint
	uv run codespell
	uv run ruff check . --diff
	uv run ruff format . --check --diff
	uv run ty check .

# ==============================================================================
# Gemini Enterprise Integration
# ==============================================================================

# Register the deployed agent to Gemini Enterprise
# Usage: make register-gemini-enterprise (interactive - will prompt for required details)
# For non-interactive use, set env vars: ID or GEMINI_ENTERPRISE_APP_ID (full GE resource name)
# Optional env vars: GEMINI_DISPLAY_NAME, GEMINI_DESCRIPTION, AGENT_CARD_URL
register-gemini-enterprise:
	@PROJECT_ID=$$(gcloud config get-value project 2>/dev/null) && \
	PROJECT_NUMBER=$$(gcloud projects describe $$PROJECT_ID --format="value(projectNumber)" 2>/dev/null) && \
	uvx agent-starter-pack@0.38.0 register-gemini-enterprise \
		--agent-card-url="https://dev-agent-$$PROJECT_NUMBER.us-central1.run.app/.well-known/agent-card.json" \
		--deployment-target="cloud_run" \
		--project-number="$$PROJECT_NUMBER"
