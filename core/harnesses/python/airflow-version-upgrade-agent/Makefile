# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


.PHONY: deploy

deploy:
	@echo "Starting automated deployment..."
	@bash -c ' \
		if [ ! -f .env ]; then \
			echo "Error: .env file not found in project root."; \
			exit 1; \
		fi; \
		echo "Loading environment variables from .env..."; \
		source .env; \
		echo "Checking if state bucket $$TF_STATE_BUCKET_NAME exists..."; \
		if ! gcloud storage buckets describe gs://$$TF_STATE_BUCKET_NAME >/dev/null 2>&1; then \
			echo "Creating state bucket $$TF_STATE_BUCKET_NAME..."; \
			gcloud storage buckets create gs://$$TF_STATE_BUCKET_NAME --location=US; \
			if [ $$? -ne 0 ]; then echo "Failed to create bucket"; exit 1; fi; \
		else \
			echo "State bucket already exists."; \
		fi; \
		echo "Navigating to deployment/terraform/..."; \
		cd deployment/terraform; \
		echo "Initializing Terraform with backend bucket: $$TF_STATE_BUCKET_NAME..."; \
		terraform init -backend-config="bucket=$$TF_STATE_BUCKET_NAME"; \
		if [ $$? -ne 0 ]; then echo "Terraform init failed"; exit 1; fi; \
		echo "Terraform will now create core infrastructure:"; \
		echo " GCS Buckets for source and destination DAGs"; \
		echo " BigQuery Dataset and Table for migration corpus"; \
		echo " IAM Role Bindings for Compute SA and User"; \
		echo "Applying Terraform configuration..."; \
		terraform apply; \
		if [ $$? -ne 0 ]; then echo "Terraform apply failed"; exit 1; fi; \
		echo "Navigating back to deployment directory..."; \
		cd ..; \
		echo "Python script will now setup Vertex AI Search:"; \
		echo " Create Website Data Store ($$VERTEX_AI_WEB_DATASTORE_ID)"; \
		echo " Create BigQuery Data Store ($$VERTEX_AI_BQ_DATASTORE_ID)"; \
		echo " Create Search App ($$VERTEX_AI_SEARCH_APP_ID) and link Web Data Store"; \
		echo "Running setup_vertex_ai_search.py..."; \
		python setup_vertex_ai_search.py; \
		if [ $$? -ne 0 ]; then echo "Vertex AI Search setup failed"; exit 1; fi; \
		echo "Deployment completed successfully!"; \
		echo "Summary of changes:"; \
		echo " BigQuery dataset and table created: $$BIGQUERY_DATASET.$$BIGQUERY_TABLE"; \
		echo " Created source bucket: $$PROJECT_ID-dags-source-bucket"; \
		echo " Created destination bucket: $$PROJECT_ID-dags-destination-bucket"; \
		echo " Put 2 deprecated 1.10 version DAGs in source bucket dags/ folder"; \
		echo " Created Vertex AI Data Store for BigQuery: $$VERTEX_AI_BQ_DATASTORE_ID"; \
		echo " Created Vertex AI Search App: $$VERTEX_AI_SEARCH_APP_ID (linked to Web Data Store)"; \
		echo " Added permissions to Compute Engine default service account"; \
		echo " Added permissions to user for necessary access"; \
	'