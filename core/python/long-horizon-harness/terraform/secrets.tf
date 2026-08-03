# Per-user secrets are created at runtime by the app (one Secret Manager secret
# per user). The Cloud Run SA needs to create those secrets and add/access
# versions. No predefined role grants only secrets.create, so use a custom role.
# (Secret Manager API is already enabled in main.tf.) Version pruning is not yet
# implemented, so enable/disable/destroy are intentionally omitted — add them
# alongside the code that uses them.

resource "google_project_iam_custom_role" "lha_user_secrets" {
  role_id = "lhaUserSecrets"
  title   = "LHA User Secrets Manager"
  project = var.project_id
  permissions = [
    "secretmanager.secrets.create",
    "secretmanager.secrets.get",
    "secretmanager.versions.add",
    "secretmanager.versions.access",
  ]
}

resource "google_project_iam_member" "cloud_run_user_secrets" {
  project = var.project_id
  role    = google_project_iam_custom_role.lha_user_secrets.id
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}
