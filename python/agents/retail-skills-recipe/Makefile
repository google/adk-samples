
# ==============================================================================
# Installation & Setup
# ==============================================================================

install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh; source $$HOME/.local/bin/env; }
	uv sync

# ==============================================================================
# Playground Targets
# ==============================================================================

playground:
	@echo "==============================================================================="
	@echo "| Starting retail skills agent playground...                                   |"
	@echo "|                                                                             |"
	@echo "| Try asking: I want to build a product search agent for my e-commerce site   |"
	@echo "|                                                                             |"
	@echo "| Select the 'retail-product-search/app' folder to interact with the agent.   |"
	@echo "==============================================================================="
	uv run adk web skills/retail-product-search --port 8501 --reload_agents

# ==============================================================================
# Eval Targets
# ==============================================================================

eval:
	@if [ -z "$(SKILL)" ]; then echo "Usage: make eval SKILL=retail-product-search PROJECT_ID=your-project"; exit 1; fi
	@echo "Evals not included in this build. See the main repo for eval tooling."

# ==============================================================================
# Testing & Code Quality
# ==============================================================================

test:
	uv sync --dev
	uv run pytest skills/retail-product-search/tests/unit
	uv run pytest skills/retail-virtual-tryon/tests/unit
	uv run pytest skills/retail-product-recommendation/tests/unit
	uv run pytest skills/retail-content-generation/tests/unit

test-search:
	uv run pytest skills/retail-product-search/tests/unit -v

test-tryon:
	uv run pytest skills/retail-virtual-tryon/tests/unit -v

lint:
	uv sync --dev
	uv run ruff check . --diff
	uv run ruff format . --check --diff

# ==============================================================================
# Deployment
# ==============================================================================

deploy-cloud-run:
	PROJECT_ID=$$(gcloud config get-value project) && \
	gcloud beta run deploy retail-skills-recipe \
		--source . \
		--memory "4Gi" \
		--project $$PROJECT_ID \
		--region "us-central1" \
		--no-allow-unauthenticated \
		--no-cpu-throttling \
		--labels "created-by=adk"
