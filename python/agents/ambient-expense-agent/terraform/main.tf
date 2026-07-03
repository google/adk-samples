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
# Ambient Expense Agent — Terraform infrastructure.
#
# Provisions all GCP resources for the ambient expense agent:
#
#   1. Agent Runtime (google_vertex_ai_reasoning_engine) — created with a
#      placeholder source; `agents-cli deploy` uploads the real code after.
#   2. Pub/Sub topic + authenticated push subscription → Agent Runtime API.
#   3. IAM — service accounts for the agent, Pub/Sub invoker, and frontend.
#   4. Cloud Run — the approval UI (frontend).
#   5. Cloud Monitoring — log-based metric + alert for expense reviews.
#
# Deploy flow (see Makefile):
#   make deploy NOTIFICATION_EMAIL=finance@example.com
#   1. terraform apply  (creates Agent Runtime skeleton + all supporting infra)
#   2. agents-cli deploy (uploads real source code to Agent Runtime)
#   3. gcloud builds submit frontend/ (builds + pushes frontend image)
# ---------------------------------------------------------------------------

locals {
  required_apis = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com",
    "iap.googleapis.com",
    "iam.googleapis.com",
    "monitoring.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "logging.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.required_apis)

  project = var.project_id
  service = each.value

  disable_on_destroy = false
}

resource "google_project_service_identity" "vertex_sa" {
  provider = google-beta
  project  = var.project_id
  service  = "aiplatform.googleapis.com"

  depends_on = [google_project_service.apis]
}

data "google_project" "project" {
  project_id = var.project_id
}
