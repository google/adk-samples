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

# --- BQ Observability Agent Hooks ---
# These targets inject interactive dataset setup into the Agent Starter Pack lifecycle

.PHONY: setup_dataset grant_permissions


# We declare `grant_permissions` to entirely override or pre-empt the standard deployment 
# so we can securely inject the env var.
# Overrides default ASP backend target to orchestrate setup and deployment sequentially.
backend: setup_dataset grant_permissions

setup_dataset:
	@echo "==============================================================================="
	@echo "| BigQuery Dataset Setup                                                      |"
	@echo "==============================================================================="
	@touch .env
	@if ! grep -q "^GOOGLE_CLOUD_PROJECT=" .env; then \
		PROJECT_ID=$$(gcloud config get-value project); \
		echo "GOOGLE_CLOUD_PROJECT=$$PROJECT_ID" >> .env; \
	fi
	@if ! grep -q "^GOOGLE_CLOUD_LOCATION=" .env; then \
		echo "GOOGLE_CLOUD_LOCATION=us-east1" >> .env; \
	fi
	@if grep -q "^BQ_ANALYTICS_DATASET_ID=" .env; then \
		if [ -f "agent_observability_bq/shared_libraries/prepare_dataset.py" ]; then \
			DATASET_VAL=$$(grep "^BQ_ANALYTICS_DATASET_ID=" .env | cut -d '=' -f2- | tr -d "'\""); \
			BQ_ANALYTICS_DATASET_ID=$$DATASET_VAL uv run python agent_observability_bq/shared_libraries/prepare_dataset.py; \
			RET=$$?; \
			if [ $$RET -eq 2 ]; then \
				read -p "Enter a different BigQuery Dataset ID: " dataset_id; \
				if [ -n "$$dataset_id" ]; then \
					if [ "$$(uname)" = "Darwin" ]; then \
						sed -i '' "s/^BQ_ANALYTICS_DATASET_ID=.*/BQ_ANALYTICS_DATASET_ID=$$dataset_id/" .env; \
					else \
						sed -i "s/^BQ_ANALYTICS_DATASET_ID=.*/BQ_ANALYTICS_DATASET_ID=$$dataset_id/" .env; \
					fi; \
					$(MAKE) setup_dataset; \
				else \
					echo "No dataset ID provided. Exiting."; \
					exit 1; \
				fi \
			elif [ $$RET -ne 0 ]; then \
				exit $$RET; \
			fi \
		fi \
	else \
		read -p "Do you want to prepare the BigQuery dataset for this agent? (y/n) " prepare_dataset; \
		if [ "$$prepare_dataset" = "y" ] || [ "$$prepare_dataset" = "Y" ]; then \
			read -p "Enter BigQuery Dataset ID [default: adk_agent_analytics]: " dataset_id; \
			dataset_id=$${dataset_id:-adk_agent_analytics}; \
			if grep -q "^BQ_ANALYTICS_DATASET_ID=" .env; then \
				if [ "$$(uname)" = "Darwin" ]; then \
					sed -i '' "s/^BQ_ANALYTICS_DATASET_ID=.*/BQ_ANALYTICS_DATASET_ID=$$dataset_id/" .env; \
				else \
					sed -i "s/^BQ_ANALYTICS_DATASET_ID=.*/BQ_ANALYTICS_DATASET_ID=$$dataset_id/" .env; \
				fi \
			else \
				[ -n "$(tail -c 1 .env)" ] && echo "" >> .env; \
				echo "BQ_ANALYTICS_DATASET_ID=$$dataset_id" >> .env; \
			fi; \
			if [ -f "agent_observability_bq/shared_libraries/prepare_dataset.py" ]; then \
				BQ_ANALYTICS_DATASET_ID=$$dataset_id uv run python agent_observability_bq/shared_libraries/prepare_dataset.py; \
			fi \
		fi \
	fi

grant_permissions:
	@echo "==============================================================================="
	@echo "| Deploying Agent Engine with BQ Observability & Granting Permissions         |"
	@echo "==============================================================================="
	@if grep -q "^BQ_ANALYTICS_DATASET_ID=" .env; then \
		DATASET_VAL=$$(grep "^BQ_ANALYTICS_DATASET_ID=" .env | cut -d '=' -f2- | tr -d "'\""); \
	else \
		DATASET_VAL="adk_agent_analytics"; \
	fi; \
	if grep -q "^GOOGLE_CLOUD_PROJECT=" .env; then \
		PROJECT_VAL=$$(grep "^GOOGLE_CLOUD_PROJECT=" .env | cut -d '=' -f2- | tr -d "'\""); \
	else \
		PROJECT_VAL=$$(gcloud config get-value project); \
	fi; \
	if grep -q "^GOOGLE_CLOUD_LOCATION=" .env; then \
		LOCATION_VAL=$$(grep "^GOOGLE_CLOUD_LOCATION=" .env | cut -d '=' -f2- | tr -d "'\""); \
	else \
		LOCATION_VAL="us-east1"; \
	fi; \
	echo "Injecting BQ_ANALYTICS_DATASET_ID=$$DATASET_VAL and GOOGLE_CLOUD_LOCATION=$$LOCATION_VAL via --set-env-vars..."; \
	(uv export --no-hashes --no-header --no-dev --no-emit-project --no-annotate > agent_observability_bq/app_utils/.requirements.txt 2>/dev/null || \
	uv export --no-hashes --no-header --no-dev --no-emit-project > agent_observability_bq/app_utils/.requirements.txt) && \
	uv run -m agent_observability_bq.app_utils.deploy \
		--source-packages=./agent_observability_bq \
		--entrypoint-module=agent_observability_bq.agent_engine_app \
		--entrypoint-object=agent_engine \
		--requirements-file=agent_observability_bq/app_utils/.requirements.txt \
		--set-env-vars="BQ_ANALYTICS_DATASET_ID=$$DATASET_VAL,GOOGLE_CLOUD_LOCATION=$$LOCATION_VAL" \
		$(if $(AGENT_IDENTITY),--agent-identity) \
		$(if $(filter command line,$(origin SECRETS)),--set-secrets="$(SECRETS)"); \
	if [ -f "agent_observability_bq/shared_libraries/grant_permissions.sh" ]; then \
		bash agent_observability_bq/shared_libraries/grant_permissions.sh; \
	else \
		echo "Warning: agent_observability_bq/shared_libraries/grant_permissions.sh not found."; \
	fi
