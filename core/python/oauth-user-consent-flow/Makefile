
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
	@echo "| 💡 Try asking: What's the weather in San Francisco?                         |"
	@echo "|                                                                             |"
	@echo "| 🔍 IMPORTANT: Select the 'app' folder to interact with your agent.          |"
	@echo "==============================================================================="
	uv run adk web . --port 8501 --reload_agents

# ==============================================================================
# Backend Deployment Targets
# ==============================================================================

# Deploy the agent remotely
# Usage: make deploy [AGENT_IDENTITY=true] [SECRETS="KEY=SECRET_ID,..."] - Set AGENT_IDENTITY=true to enable per-agent IAM identity (Preview)
deploy:
	# Export dependencies to requirements file using uv export.
	(uv export --no-hashes --no-header --no-dev --no-emit-project --no-annotate > adk_ae_oauth/app_utils/.requirements.txt 2>/dev/null || \
	uv export --no-hashes --no-header --no-dev --no-emit-project > adk_ae_oauth/app_utils/.requirements.txt) && \
	uv run -m adk_ae_oauth.app_utils.deploy \
		--source-packages=./adk_ae_oauth \
		--entrypoint-module=adk_ae_oauth.agent_engine_app \
		--entrypoint-object=agent_engine \
		--requirements-file=adk_ae_oauth/app_utils/.requirements.txt \
		$(if $(AGENT_IDENTITY),--agent-identity) \
		$(if $(filter command line,$(origin SECRETS)),--set-secrets="$(SECRETS)")

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
	uv run adk eval ./adk_ae_oauth $${EVALSET:-tests/eval/evalsets/basic.evalset.json} \
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

# Register OAuth authorization resource for Gemini Enterprise
# Usage: make register-oauth
register-oauth:
	uv run python tools/register_oauth.py

# Register the deployed agent to Gemini Enterprise
# Usage:
#   make register-gemini-enterprise                          (interactive)
#   make register-gemini-enterprise \                        (non-interactive)
#     AUTH_ID_RESOURCE="projects/<PROJECT_NUMBER>/locations/<LOCATION>/authorizations/google-drive-auth" \
#     GE_APP_ID="projects/<PROJECT_NUMBER>/locations/<LOCATION>/collections/default_collection/engines/<ENGINE_ID>" \
#     DISPLAY_NAME="Drive Reader Agent" \
#     DESCRIPTION="Reads files from Google Drive" \
#     TOOL_DESCRIPTION="Read content of a Google Drive file on behalf of the user"
register-gemini-enterprise:
	@uvx agent-starter-pack@0.39.4 register-gemini-enterprise \
		$(if $(AUTH_ID_RESOURCE),--authorization-id="$(AUTH_ID_RESOURCE)") \
		$(if $(GE_APP_ID),--gemini-enterprise-app-id="$(GE_APP_ID)") \
		$(if $(DISPLAY_NAME),--display-name="$(DISPLAY_NAME)") \
		$(if $(DESCRIPTION),--description="$(DESCRIPTION)") \
		$(if $(TOOL_DESCRIPTION),--tool-description="$(TOOL_DESCRIPTION)") \
		$(if $(YES),-y)

# Unregister (delete) an agent from Gemini Enterprise so you can re-register
# Usage: make unregister-gemini-enterprise AGENT_NAME="projects/.../agents/<agent_id>"
unregister-gemini-enterprise:
	@if [ -z "$(AGENT_NAME)" ]; then \
		echo "❌ AGENT_NAME is required."; \
		echo "   Usage: make unregister-gemini-enterprise AGENT_NAME=\"projects/.../agents/<id>\""; \
		echo "   You can find the agent name from the register output or the console."; \
		exit 1; \
	fi
	@echo "Deleting agent registration: $(AGENT_NAME)"
	@curl -s -X DELETE \
		-H "Authorization: Bearer $$(gcloud auth print-access-token)" \
		-H "X-Goog-User-Project: $$(gcloud config get-value project 2>/dev/null)" \
		"https://discoveryengine.googleapis.com/v1alpha/$(AGENT_NAME)" | python3 -m json.tool
	@echo "✅ Agent unregistered. You can now re-register with 'make register-gemini-enterprise'."