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
# IAM: service accounts with least-privilege permissions.
# ---------------------------------------------------------------------------

# --- Agent Runtime service account ---

resource "google_service_account" "app_sa" {
  account_id   = "${var.project_name}-app"
  display_name = "Ambient Expense Agent - Agent Runtime SA"
  project      = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "app_sa_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent",
    "roles/serviceusage.serviceUsageConsumer",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}

# Grant the Vertex AI service identity the same roles.
resource "google_project_iam_member" "vertex_sa_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/logging.logWriter",
  ])

  project = var.project_id
  role    = each.value
  member  = google_project_service_identity.vertex_sa.member
}

# --- Pub/Sub → Agent Runtime ---

# Service account Pub/Sub uses to authenticate push deliveries to the
# Agent Runtime API passthrough endpoint.
resource "google_service_account" "pubsub_invoker" {
  account_id   = "expense-agent-invoker"
  display_name = "Ambient Expense Agent - Pub/Sub → Agent Runtime Invoker"
  project      = var.project_id

  depends_on = [google_project_service.apis]
}

# Grant the invoker permission to call the Agent Runtime API.
# The passthrough endpoint requires aiplatform.user.
resource "google_project_iam_member" "pubsub_invoker_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.pubsub_invoker.email}"
}

# Allow the GCP-managed Pub/Sub service agent to create OIDC tokens for
# authenticated push delivery to the Agent Runtime API.
resource "google_service_account_iam_member" "pubsub_token_creator" {
  service_account_id = google_service_account.pubsub_invoker.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# --- Frontend → Agent Runtime ---

resource "google_service_account" "frontend_invoker" {
  account_id   = "approval-ui-invoker"
  display_name = "Approval UI - Agent Runtime Invoker"
  project      = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "frontend_invoker_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.frontend_invoker.email}"
}

# --- IAP for the approval UI ---

resource "google_iap_web_cloud_run_service_iam_member" "approval_access" {
  project                = var.project_id
  location               = var.region
  cloud_run_service_name = google_cloud_run_v2_service.frontend.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = "user:${var.notification_email}"

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "iap_invoker" {
  name     = google_cloud_run_v2_service.frontend.name
  location = var.region
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-iap.iam.gserviceaccount.com"
}
