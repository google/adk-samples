output "artifacts_bucket_name" {
  value       = google_storage_bucket.artifacts.name
  description = "GCS bucket backing the ADK artifact service."
}

output "cloud_run_url" {
  value       = google_cloud_run_v2_service.app.uri
  description = "Public URL of the Cloud Run service."
}

output "cloud_sql_connection_name" {
  value       = google_sql_database_instance.db.connection_name
  description = "Cloud SQL connection name (project:region:instance) — paste into LHA_REMINDER_DB_URL host param for local proxy use."
}

output "scheduler_audience" {
  value       = local.scheduler_audience
  description = "Audience string used in OIDC tokens (matches LHA_SCHEDULER_AUDIENCE in Cloud Run)."
}
