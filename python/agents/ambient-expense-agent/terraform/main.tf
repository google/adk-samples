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

# ---------------------------------------------------------------------------
# Terraform configuration for the Ambient Expense Agent infrastructure.
#
# This provisions the supporting GCP infrastructure for the ambient expense
# agent deployed on Agent Runtime: the frontend approval UI (Cloud Run),
# Pub/Sub triggers, Cloud Monitoring alerts, and IAM bindings.
#
# The agent backend is deployed separately via:
#   agents-cli deploy --project <PROJECT_ID> --region <REGION>
#
# After deploying, pass the printed Agent Runtime resource name to
# terraform apply with -var=agent_runtime_resource_name=<resource_name>.
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region

  default_labels = {
    goog-terraform-provisioned = "true"
    app                        = "ambient-expense-agent"
  }
}

locals {
  # Enable required GCP APIs.
  required_apis = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com",
    "iap.googleapis.com",
    "monitoring.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.required_apis)

  project = var.project_id
  service = each.value

  disable_on_destroy = false
}
