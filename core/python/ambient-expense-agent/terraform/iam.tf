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
# IAM: Dedicated service accounts with least-privilege permissions.
# ---------------------------------------------------------------------------

# --- Backend → Vertex AI ---

# The backend Cloud Run service uses the default compute SA.
# Grant it Vertex AI User so it can call Gemini.
resource "google_project_iam_member" "backend_vertex_ai" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# --- Pub/Sub → Backend ---

# Service account for Pub/Sub push to invoke the backend Cloud Run service.
resource "google_service_account" "pubsub_invoker" {
  account_id   = "expense-agent-invoker"
  display_name = "Ambient Expense Agent - Pub/Sub Invoker"
  project      = var.project_id
}

# Grant the invoker permission to call the backend Cloud Run service.
resource "google_cloud_run_v2_service_iam_member" "pubsub_invoker" {
  name     = google_cloud_run_v2_service.backend.name
  location = var.region
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_invoker.email}"
}

# Allow the GCP-managed Pub/Sub service agent to create OIDC tokens
# for authenticated push delivery.
data "google_project" "project" {
  project_id = var.project_id
}

resource "google_service_account_iam_member" "pubsub_token_creator" {
  service_account_id = google_service_account.pubsub_invoker.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# --- Frontend → Backend ---

# Service account for the approval UI to invoke the backend.
resource "google_service_account" "frontend_invoker" {
  account_id   = "approval-ui-invoker"
  display_name = "Approval UI - Backend Invoker"
  project      = var.project_id
}

# Grant the frontend permission to call the backend Cloud Run service.
resource "google_cloud_run_v2_service_iam_member" "frontend_invoker" {
  name     = google_cloud_run_v2_service.backend.name
  location = var.region
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.frontend_invoker.email}"
}

# --- IAP access ---

# Grant the notification email access to the frontend via IAP.
# This ensures the manager who receives expense alerts can open the
# approval UI without additional IAM setup.
# Note: Cloud Run's native IAP requires a service-level binding, not
# the project-level google_iap_web_iam_member.
resource "google_iap_web_cloud_run_service_iam_member" "approval_access" {
  project                = var.project_id
  location               = var.region
  cloud_run_service_name = google_cloud_run_v2_service.frontend.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = "user:${var.notification_email}"

  depends_on = [google_project_service.apis]
}

# The IAP service agent needs run.invoker on the frontend so it can
# proxy authenticated requests through to the Cloud Run service.
resource "google_cloud_run_v2_service_iam_member" "iap_invoker" {
  name     = google_cloud_run_v2_service.frontend.name
  location = var.region
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-iap.iam.gserviceaccount.com"
}
