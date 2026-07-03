PROJECT_ID ?= $(shell gcloud config get-value project 2>/dev/null)
REGION ?= us-east1
FRONTEND_SERVICE ?= expense-approval-ui
NOTIFICATION_EMAIL ?= your-email@example.com
REGISTRY = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/expense-agent
FRONTEND_IMAGE = $(REGISTRY)/frontend

# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/0.6.12/install.sh | sh; source $$HOME/.local/bin/env; }
	uv sync

install-frontend:
	cd frontend && \
		command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/0.6.12/install.sh | sh; source $$HOME/.local/bin/env; } && \
		uv sync

dev:
	uv run python expense_agent/fast_api_app.py

dev-frontend:
	cd frontend && BACKEND_URL=http://localhost:8080 uv run python main.py

playground:
	uv run adk web --port 8501

test:
	uv run python -m pytest tests/ -xvs

lint:
	uv run codespell . && \
		uv run ruff check . --fix && \
		uv run ruff format . && \
		uv run mypy .

# ---------------------------------------------------------------------------
# Cloud deployment
#
# Two-step process:
#   1. Deploy the agent to Agent Runtime via agents-cli.
#   2. Deploy the frontend + Pub/Sub + IAM + monitoring via Terraform.
#
# Usage:
#   make deploy NOTIFICATION_EMAIL=finance@example.com
#
# agents-cli writes the Agent Runtime resource name to deployment_metadata.json
# after a successful deploy; Terraform reads it from there.
# ---------------------------------------------------------------------------

deploy:
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "Error: PROJECT_ID is not set."; \
		echo "Run: gcloud config set project <your-project-id>"; \
		exit 1; \
	fi
	@if [ "$(NOTIFICATION_EMAIL)" = "your-email@example.com" ]; then \
		echo "Warning: Using default NOTIFICATION_EMAIL. Set your own:"; \
		echo "  make deploy NOTIFICATION_EMAIL=you@example.com"; \
		echo ""; \
	fi
	@command -v agents-cli >/dev/null 2>&1 || { echo "Error: agents-cli is not installed. See https://goo.gle/agents-cli"; exit 1; }
	@echo "==> [1/3] Deploying agent to Agent Runtime..."
	agents-cli deploy \
		--project $(PROJECT_ID) \
		--region $(REGION) \
		--no-confirm-project
	@echo ""
	@echo "==> [2/3] Building frontend image..."
	gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com \
		--project=$(PROJECT_ID) --quiet
	-gcloud artifacts repositories create expense-agent \
		--repository-format=docker \
		--location=$(REGION) \
		--project=$(PROJECT_ID) 2>/dev/null
	gcloud builds submit frontend/ \
		--tag $(FRONTEND_IMAGE) \
		--project=$(PROJECT_ID) --quiet
	@echo ""
	@echo "==> [3/3] Deploying frontend + Pub/Sub infrastructure..."
	$(eval AGENT_RUNTIME_RESOURCE := $(shell python3 -c "import json; d=json.load(open('deployment_metadata.json')); print(d['remote_agent_runtime_id'])"))
	@echo "Agent Runtime resource: $(AGENT_RUNTIME_RESOURCE)"
	cd terraform && terraform init -input=false && terraform apply -auto-approve \
		-var=project_id=$(PROJECT_ID) \
		-var=region=$(REGION) \
		-var=frontend_service_name=$(FRONTEND_SERVICE) \
		-var=frontend_image=$(FRONTEND_IMAGE) \
		-var=agent_runtime_resource_name="$(AGENT_RUNTIME_RESOURCE)" \
		-var=notification_email=$(NOTIFICATION_EMAIL)
	@echo ""
	@echo "==> Deployment complete!"
	@echo ""
	@cd terraform && \
		echo "  Agent Runtime resource: $(AGENT_RUNTIME_RESOURCE)" && \
		echo "  Trigger endpoint:  $$(terraform output -raw trigger_endpoint)" && \
		echo "  Frontend (approval UI): $$(terraform output -raw frontend_url)/approval" && \
		echo "  Topic:     expense-reports" && \
		echo "  Alerts:    Expenses >= \$$100 -> $(NOTIFICATION_EMAIL)"

remote-test:
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "Error: PROJECT_ID is not set."; \
		echo "Run: gcloud config set project <your-project-id>"; \
		exit 1; \
	fi
	@echo "Publishing test expense (>= \$$100, requires review)..."
	gcloud pubsub topics publish expense-reports \
		--project=$(PROJECT_ID) \
		--message='{"amount":250.00,"submitter":"alice@company.com","category":"travel","description":"Flight to NYC for client meeting","date":"2026-04-10"}' \
		--attribute=source=make-test
	@echo ""
	@echo "Check Cloud Logging for results:"
	@echo "  gcloud logging read 'resource.type=\"aiplatform.googleapis.com/ReasoningEngine\" jsonPayload.alert_type=\"expense_review\"' --project=$(PROJECT_ID) --limit=5"

clean:
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "Error: PROJECT_ID is not set."; \
		echo "Run: gcloud config set project <your-project-id>"; \
		exit 1; \
	fi
	@echo "==> Tearing down supporting infrastructure (Pub/Sub, IAM, monitoring, frontend)..."
	$(eval AGENT_RUNTIME_RESOURCE := $(shell python3 -c "import json; d=json.load(open('deployment_metadata.json')); print(d['remote_agent_runtime_id'])" 2>/dev/null || echo "placeholder"))
	cd terraform && terraform destroy -auto-approve \
		-var=project_id=$(PROJECT_ID) \
		-var=region=$(REGION) \
		-var=frontend_service_name=$(FRONTEND_SERVICE) \
		-var=frontend_image=$(FRONTEND_IMAGE) \
		-var=agent_runtime_resource_name="$(AGENT_RUNTIME_RESOURCE)" \
		-var=notification_email=$(NOTIFICATION_EMAIL) \
	|| (echo "" && \
		echo "Retrying in 60s (GCP needs time to propagate alert policy deletion)..." && \
		sleep 60 && \
		cd terraform && terraform destroy -auto-approve \
			-var=project_id=$(PROJECT_ID) \
			-var=region=$(REGION) \
			-var=frontend_service_name=$(FRONTEND_SERVICE) \
			-var=frontend_image=$(FRONTEND_IMAGE) \
			-var=agent_runtime_resource_name="$(AGENT_RUNTIME_RESOURCE)" \
			-var=notification_email=$(NOTIFICATION_EMAIL))
	@echo ""
	@echo "==> Deleting Agent Runtime engine..."
	@RESOURCE_NAME=$$(python3 -c "import json; d=json.load(open('deployment_metadata.json')); print(d['remote_agent_runtime_id'])") && \
		gcloud ai reasoning-engines delete $$RESOURCE_NAME --region=$(REGION) --quiet 2>/dev/null || \
		echo "Agent Runtime engine not found or already deleted."
	@echo "Cleanup complete."
