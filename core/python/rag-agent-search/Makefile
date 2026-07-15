install:
	uv sync --dev --extra eval

playground:
	uv run adk web . --port 8501 --reload_agents

# Provision the Agent Platform Search datastore (GCS connector + search engine).
setup-infra:
	cd infra/terraform && terraform init && terraform apply -var-file=vars/env.tfvars

# Upload the bundled sample documents to the docs bucket.
upload-sample-data:
	gcloud storage cp sample_data/* "gs://$$(cd infra/terraform && terraform output -raw docs_bucket_name)/"

# Trigger an immediate connector sync (otherwise it waits for the daily
# refresh). Reads project/region from .env.
ingest:
	set -a && . ./.env && set +a && \
	uv run infra/terraform/scripts/start_connector_run.py \
		--project "$$GOOGLE_CLOUD_PROJECT" --region "$$DATA_STORE_REGION" \
		--collection-id rag-agent-search-collection --wait

test:
	uv sync --dev
	INTEGRATION_TEST=TRUE uv run pytest tests/integration

lint:
	uv sync --extra lint
	uv run ruff check .
	uv run ruff format . --check
