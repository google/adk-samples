#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
# --- Economic Research Agent (ERA) Makefile ---

PROJECT_ID ?= $(shell gcloud config get-value project)
LOCATION ?= us-east1
VENV_PATH = .venv
MODERN_REPO = economic_research

.PHONY: venv install test deploy clean help apply-fixes

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create a virtual environment using uv
	@echo "🏗️ Creating virtual environment..."
	uv venv $(VENV_PATH)

install: venv ## Install dependencies using uv
	@echo "📦 Installing ERA dependencies..."
	uv sync

test: ## Run unit tests with coverage
	@echo "🧪 Running ERA test suite..."
	uv run pytest --cov=$(MODERN_REPO) tests/unit/

run: ## Run the agent locally in interactive mode
	@echo "🧠 Starting Interactive ERA Consultant Session..."
	uv run python3 -m economic_research.agent

mcp: ## Run the agent as an MCP Server
	@echo "🛰️ Starting ERA MCP Server..."
	uv run python3 mcp_server.py

streamlit: ## Launch the Economic Research Agent Dashboard (Streamlit)
	@echo "🖥️ Launching Economic Research Agent Dashboard..."
	uv run streamlit run streamlit_app.py

# Note: Serve target removed as server.py was deprecated in 2.0 structure.

test-integration: ## Run integration tests (Requires API keys)
	@echo "🛰️ Running ERA integration tests..."
	uv run pytest tests/integration/

deploy: ## Deploy the agent to Vertex AI Reasoning Engine (Direct Vertex SDK)
	@echo "🚀 Deploying ERA to Vertex (Direct SDK)..."
	PYTHONPATH=. uv run python3 economic_research/deployment/deploy.py




register-gemini-enterprise: ## Register the agent with Gemini Enterprise (Reasoning Engine Spec)
	@echo "🛰️ Registering ERA with Gemini Enterprise..."
	@uvx agent-starter-pack register-gemini-enterprise
lint: ## Run ruff check without fixing
	@echo "🔍 Running ruff checks..."
	uvx ruff check .

apply-fixes: ## Run ruff and auto-fix linting issues
	@echo "🩹 Applying auto-fixes and formatting..."
	uv run ruff check --fix .
	uv run ruff format .

clean: ## Clean up build artifacts and cache
	rm -rf $(VENV_PATH) .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
