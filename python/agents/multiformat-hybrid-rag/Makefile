
# ==============================================================================
# 1. Installation & Setup
# ==============================================================================

install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh; . $HOME/.local/bin/env; }
	uv sync

# ==============================================================================
# 2. Infrastructure (Terraform) — all variables sourced from config.env
# ==============================================================================

define TF_VARS
. ./config.env && \
export TF_VAR_project_id=$${PROJECT_ID} && \
export TF_VAR_project_name=$${PROJECT_NAME:-multiformat-hybrid-rag} && \
export TF_VAR_region=$${DEFAULT_REGION:-us-central1} && \
export TF_VAR_gcs_bucket=$${GCS_BUCKET} && \
export TF_VAR_gcs_prefix=$${GCS_PREFIX:-documents/} && \
export TF_VAR_bq_dataset=$${BQ_DATASET:-rag_pipeline} && \
export TF_VAR_bq_object_table=$${BQ_OBJECT_TABLE:-gcs_objects} && \
export TF_VAR_bq_preprocessed_table=$${BQ_PREPROCESSED_TABLE:-preprocessed} && \
export TF_VAR_bq_chunks_table=$${BQ_CHUNKS_TABLE:-chunks} && \
export TF_VAR_bq_gcs_connection_id=$${BQ_GCS_CONNECTION_ID:-rag-gcs-connection} && \
export TF_VAR_vs_collection_id=$${VS_COLLECTION_ID:-multiformat-hybrid-rag-collection} && \
export TF_VAR_vs_documents_collection_id=$${VS_DOCUMENTS_COLLECTION_ID:-multiformat-hybrid-rag-documents} && \
export TF_VAR_markdown_converter_gemini_model=$${MARKDOWN_CONVERTER_GEMINI_MODEL:-gemini-2.5-flash} && \
export TF_VAR_relevance_gemini_model=$${RELEVANCE_GEMINI_MODEL:-gemini-2.5-flash-lite} && \
export TF_VAR_contextual_chunking_gemini_model=$${CONTEXTUAL_CHUNKING_GEMINI_MODEL:-gemini-2.5-flash-lite}
endef

setup-infra:
	@if [ ! -f ./config.env ]; then echo "Error: config.env not found"; exit 1; fi
	@. ./config.env && \
		echo "Setting project..." && \
		gcloud config set project $$PROJECT_ID --quiet && \
		gcloud auth application-default set-quota-project $$PROJECT_ID && \
		echo "Enabling bootstrap APIs..." && \
		gcloud services enable cloudresourcemanager.googleapis.com serviceusage.googleapis.com \
			--project=$$PROJECT_ID --quiet
	@echo "Provisioning infrastructure (APIs, BQ, VS2, Cloud Run, IAM)..."
	@$(TF_VARS) && cd infra/terraform/dev && terraform init && terraform apply -auto-approve
	@echo "Building all images in parallel..."
	@set -a && . ./config.env && set +a && { \
	gcloud builds submit --config=data_ingestion_pipeline/preprocess_service/cloudbuild.yaml \
		--substitutions=_IMAGE="$$PREPROCESS_SERVICE_IMAGE" --project="$$PROJECT_ID" . & \
	gcloud builds submit --config=data_ingestion_pipeline/chunk_index_service/cloudbuild.yaml \
		--substitutions=_IMAGE="$$CHUNK_INDEX_SERVICE_IMAGE" --project="$$PROJECT_ID" . & \
	gcloud builds submit --config=data_ingestion_pipeline/cloudbuild.yaml \
		--substitutions=_IMAGE="$$PIPELINE_IMAGE" --project="$$PROJECT_ID" . & \
	wait && echo "All images built."; }
	@echo "Deploying Cloud Run services..."
	@set -a && . ./config.env && set +a && { \
	gcloud run deploy $${PROJECT_NAME:-multiformat-hybrid-rag}-preprocess \
		--image="$$PREPROCESS_SERVICE_IMAGE" \
		--region="$$DEFAULT_REGION" \
		--project="$$PROJECT_ID" \
		--quiet & \
	gcloud run deploy $${PROJECT_NAME:-multiformat-hybrid-rag}-chunk-index \
		--image="$$CHUNK_INDEX_SERVICE_IMAGE" \
		--region="$$DEFAULT_REGION" \
		--project="$$PROJECT_ID" \
		--quiet & \
	wait && echo "All services deployed."; }

tf-plan:
	@if [ ! -f ./config.env ]; then echo "Error: config.env not found"; exit 1; fi
	@$(TF_VARS) && cd infra/terraform/dev && terraform plan

tf-destroy:
	@if [ ! -f ./config.env ]; then echo "Error: config.env not found"; exit 1; fi
	@$(TF_VARS) && cd infra/terraform/dev && terraform destroy

# ==============================================================================
# 3. RAG Data Ingestion Pipeline
# ==============================================================================

data-ingestion-remote:
	@if [ ! -f ./config.env ]; then echo "Error: config.env not found"; exit 1; fi
	@. ./config.env && \
	PYTHONPATH=.:data_ingestion_pipeline PIPELINE_IMAGE=$$PIPELINE_IMAGE \
	uv run python data_ingestion_pipeline/data_ingestion_pipeline/submit_pipeline.py \
		--service-account=$$SERVICE_ACCOUNT \
		--pipeline-root=$$PIPELINE_ROOT \
		--disable-caching \
		--cron-schedule="$${PIPELINE_CRON:-0 2 * * *}"


# ==============================================================================
# 4. Playground
# ==============================================================================

playground:
	@echo "==============================================================================="
	@echo "| 🚀 Starting your agent playground...                                        |"
	@echo "|                                                                             |"
	@echo "| 💡 Try asking: How to save a pandas dataframe to CSV?                       |"
	@echo "|                                                                             |"
	@echo "| 🔍 IMPORTANT: Select the 'app' folder to interact with your agent.          |"
	@echo "==============================================================================="
	uv run adk web . --port 8501 --reload_agents

# ==============================================================================
# 5. Deployment
# ==============================================================================

deploy:
	@if [ ! -f ./config.env ]; then echo "Error: config.env not found"; exit 1; fi
	@. ./config.env && \
	gcloud beta run deploy $${PROJECT_NAME:-multiformat-hybrid-rag} \
		--source . \
		--memory "4Gi" \
		--cpu "4" \
		--project $$PROJECT_ID \
		--region $$DEFAULT_REGION \
		--service-account=$${PROJECT_NAME:-multiformat-hybrid-rag}-app@$$PROJECT_ID.iam.gserviceaccount.com \
		--no-allow-unauthenticated \
		--no-cpu-throttling \
		--labels "created-by=adk" \
		--update-build-env-vars "AGENT_VERSION=$(shell awk -F'"' '/^version = / {print $$2}' pyproject.toml || echo '0.0.0')" \
		$(if $(IAP),--iap)

# ==============================================================================
# 6. Local Development
# ==============================================================================

backend: local-backend

local-backend:
	uv run uvicorn app.fast_api_app:app --host localhost --port $(or $(PORT),8000) --reload

mcp-server:
	uv run python -m app.mcp_server --transport sse --port $(or $(MCP_PORT),8081)

# ==============================================================================
# 7. Testing & Code Quality
# ==============================================================================

test:
	uv sync --dev
	uv run pytest tests/unit && uv run pytest tests/integration

lint:
	uv sync --dev --extra lint
	uv run codespell
	uv run ruff check . --diff
	uv run ruff format . --check --diff
	uv run ty check .
