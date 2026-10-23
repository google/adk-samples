resource "google_service_account" "cloud_run" {
  account_id   = "${var.service_name}-run"
  display_name = "Cloud Run runtime SA for ${var.service_name}"
  project      = var.project_id
}

resource "google_service_account" "scheduler" {
  account_id   = "${var.service_name}-scheduler"
  display_name = "Cloud Scheduler invoker for ${var.service_name}"
  project      = var.project_id
}

locals {
  cloud_run_roles = [
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/aiplatform.user",
  ]
}

resource "google_project_iam_member" "cloud_run" {
  for_each = toset(local.cloud_run_roles)

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Cloud Run SA doubles as the sandbox caller (LHA_SANDBOX_CALLER_SA in
# cloud_run.tf). mint_sandbox_token calls generate_access_token on this
# SA, which requires the caller to hold serviceAccountTokenCreator on the
# target — here, itself.
resource "google_service_account_iam_member" "cloud_run_token_creator_self" {
  service_account_id = google_service_account.cloud_run.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_cloud_run_v2_service_iam_member" "scheduler_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_service.app.location
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

# Web frontend proxies all browser requests to the backend with a minted
# ID token; only this SA may invoke the backend.
resource "google_cloud_run_v2_service_iam_member" "web_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_service.app.location
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.cloud_run_web.email}"
}
