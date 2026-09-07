# GCS bucket backing the ADK artifact service (LOGS_BUCKET_NAME). Durable
# artifact storage is best practice: with the in-memory service, artifacts are
# lost on restart/teardown and can be unavailable when the A2A FilePart is
# emitted (the interceptor reloads them at emit time). GCS keeps them loadable.
resource "google_storage_bucket" "artifacts" {
  # GCS bucket names are a single global namespace, so scope with the
  # (globally unique) project id or a second deployer / taken name 409s.
  name          = coalesce(var.artifacts_bucket_name, "${var.project_id}-lha-artifacts-${var.env}")
  project       = var.project_id
  location      = var.location
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  # `make destroy` should tear down the bucket even after artifacts are written.
  force_destroy = true

  labels = {
    app = "lha"
    env = var.env
  }
}

# Runtime SA reads/writes artifact objects.
resource "google_storage_bucket_iam_member" "artifacts_object_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}
