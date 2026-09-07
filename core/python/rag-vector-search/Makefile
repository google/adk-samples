install:
	uv sync --dev --extra eval

playground:
	uv run adk web . --port 8501 --reload_agents

# Provision the Vector Search 2.0 Collection (+ pipeline bucket) via terraform.
setup-infra:
	cd infra/terraform && terraform init && terraform apply -var-file=vars/env.tfvars

# Run the data-ingestion pipeline locally (embeds + ingests into the VS 2.0
# collection) using KFP SubprocessRunner. Reads project/region/collection-id
# from .env.
data-ingestion:
	set -a && . ./.env && set +a && \
	cd data_ingestion && uv run python data_ingestion_pipeline/submit_pipeline.py --local \
		--project "$$PROJECT_ID" --region "$$REGION" --collection-id "$$VECTOR_SEARCH_COLLECTION_ID"

test:
	uv sync --dev
	INTEGRATION_TEST=TRUE uv run pytest tests/integration

lint:
	uv sync --extra lint
	uv run ruff check .
	uv run ruff format . --check
