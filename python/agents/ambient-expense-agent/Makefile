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
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; source $$HOME/.local/bin/env; }
	uv sync

install-frontend:
	cd frontend && uv sync

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
# Three-step process (all handled by `make deploy`):
#   1. Terraform creates all infrastructure including Agent Runtime (with a
#      placeholder source) and wires Pub/Sub → Agent Runtime passthrough.
#   2. agents-cli deploy uploads the real source code to Agent Runtime.
#   3. gcloud builds submit builds and pushes the frontend container image.
#
# Usage:
#   make deploy NOTIFICATION_EMAIL=finance@example.com
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
	@echo "==> [1/3] Enabling required APIs and setting up Artifact Registry..."
	gcloud services enable \
		aiplatform.googleapis.com \
		artifactregistry.googleapis.com \
		cloudbuild.googleapis.com \
		--project=$(PROJECT_ID) --quiet
	-gcloud artifacts repositories create expense-agent \
		--repository-format=docker \
		--location=$(REGION) \
		--project=$(PROJECT_ID) 2>/dev/null
	@echo ""
	@echo "==> [2/3] Deploying infrastructure via Terraform..."
	@echo "   (Creates Agent Runtime skeleton, Pub/Sub, IAM, monitoring, approval UI)"
	cd terraform && terraform init -input=false && terraform apply -auto-approve \
		-var=project_id=$(PROJECT_ID) \
		-var=region=$(REGION) \
		-var=frontend_service_name=$(FRONTEND_SERVICE) \
		-var=frontend_image=$(FRONTEND_IMAGE) \
		-var=notification_email=$(NOTIFICATION_EMAIL)
	@echo ""
	@echo "==> [3/3] Deploying agent source to Agent Runtime..."
	@echo "   (agents-cli deploy finds the Agent Runtime by display name and updates the source)"
	agents-cli deploy \
		--project $(PROJECT_ID) \
		--region $(REGION) \
		--no-confirm-project
	@echo ""
	@echo "==> Building and pushing frontend image..."
	gcloud builds submit frontend/ \
		--tag $(FRONTEND_IMAGE) \
		--project=$(PROJECT_ID) --quiet
	@echo ""
	@echo "==> Updating frontend with built image..."
	cd terraform && terraform apply -auto-approve \
		-var=project_id=$(PROJECT_ID) \
		-var=region=$(REGION) \
		-var=frontend_service_name=$(FRONTEND_SERVICE) \
		-var=frontend_image=$(FRONTEND_IMAGE) \
		-var=notification_email=$(NOTIFICATION_EMAIL)
	@echo ""
	@echo "==> Deployment complete!"
	@cd terraform && \
		echo "  Trigger endpoint: $$(terraform output -raw trigger_endpoint)" && \
		echo "  Approval UI:      $$(terraform output -raw frontend_url)/approval" && \
		echo "  Topic:            expense-reports" && \
		echo "  Alerts:           Expenses >= \$$100 -> $(NOTIFICATION_EMAIL)"

remote-test:
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "Error: PROJECT_ID is not set."; \
		exit 1; \
	fi
	@echo "Publishing test expense (>= \$$100, requires review)..."
	gcloud pubsub topics publish expense-reports \
		--project=$(PROJECT_ID) \
		--message='{"amount":250.00,"submitter":"alice@company.com","category":"travel","description":"Flight to NYC for client meeting","date":"2026-04-10"}' \
		--attribute=source=make-test
	@echo ""
	@echo "Check Cloud Logging:"
	@echo "  gcloud logging read 'resource.type=\"aiplatform.googleapis.com/ReasoningEngine\"' --project=$(PROJECT_ID) --limit=10"

clean:
	@if [ -z "$(PROJECT_ID)" ]; then echo "Error: PROJECT_ID is not set."; exit 1; fi
	@echo "==> Tearing down all infrastructure..."
	cd terraform && terraform destroy -auto-approve \
		-var=project_id=$(PROJECT_ID) \
		-var=region=$(REGION) \
		-var=frontend_service_name=$(FRONTEND_SERVICE) \
		-var=frontend_image=$(FRONTEND_IMAGE) \
		-var=notification_email=$(NOTIFICATION_EMAIL) \
	|| (echo "Retrying in 60s..." && sleep 60 && \
		cd terraform && terraform destroy -auto-approve \
			-var=project_id=$(PROJECT_ID) \
			-var=region=$(REGION) \
			-var=frontend_service_name=$(FRONTEND_SERVICE) \
			-var=frontend_image=$(FRONTEND_IMAGE) \
			-var=notification_email=$(NOTIFICATION_EMAIL))
	@echo "Cleanup complete."
