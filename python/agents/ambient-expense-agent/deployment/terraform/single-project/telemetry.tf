# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# BigQuery dataset for telemetry external tables
resource "google_bigquery_dataset" "telemetry_dataset" {
  project       = var.project_id
  dataset_id    = replace("${var.project_name}_telemetry", "-", "_")
  friendly_name = "${var.project_name} Telemetry"
  location      = var.region
  description   = "Dataset for GenAI telemetry data stored in GCS"
  depends_on    = [google_project_service.services]
}

# BigQuery connection for accessing GCS telemetry data
resource "google_bigquery_connection" "genai_telemetry_connection" {
  project       = var.project_id
  location      = var.region
  connection_id = "${var.project_name}-genai-telemetry"
  friendly_name = "${var.project_name} GenAI Telemetry Connection"

  cloud_resource {}

  depends_on = [google_project_service.services]
}

# Wait for the BigQuery connection service account to propagate in IAM
resource "time_sleep" "wait_for_bq_connection_sa" {
  create_duration = "10s"

  depends_on = [google_bigquery_connection.genai_telemetry_connection]
}

# Grant the BigQuery connection service account access to read from the logs bucket
resource "google_storage_bucket_iam_member" "telemetry_connection_access" {
  bucket = google_storage_bucket.logs_data_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_bigquery_connection.genai_telemetry_connection.cloud_resource[0].service_account_id}"

  depends_on = [time_sleep.wait_for_bq_connection_sa]
}

# ====================================================================
# Log Sinks — route GenAI and feedback logs directly to BigQuery
# ====================================================================

# Log sink to route GenAI telemetry logs directly to BigQuery
resource "google_logging_project_sink" "genai_logs_to_bq" {
  name        = "${var.project_name}-genai-logs"
  project     = var.project_id
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.telemetry_dataset.dataset_id}"
  filter      = "log_name=\"projects/${var.project_id}/logs/gen_ai.client.inference.operation.details\" AND (labels.\"gen_ai.input.messages_ref\" =~ \".*${var.project_name}.*\" OR labels.\"gen_ai.output.messages_ref\" =~ \".*${var.project_name}.*\")"

  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }

  depends_on = [google_bigquery_dataset.telemetry_dataset]
}

# Log sink for user feedback logs — routes to the same BigQuery dataset
resource "google_logging_project_sink" "feedback_logs_to_bq" {
  name        = "${var.project_name}-feedback"
  project     = var.project_id
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.telemetry_dataset.dataset_id}"
  filter      = var.feedback_logs_filter

  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }

  depends_on = [google_bigquery_dataset.telemetry_dataset]
}

# Grant log sink service accounts write access to the BigQuery dataset
resource "google_bigquery_dataset_iam_member" "genai_logs_bq_writer" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.telemetry_dataset.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.genai_logs_to_bq.writer_identity
}

resource "google_bigquery_dataset_iam_member" "feedback_logs_bq_writer" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.telemetry_dataset.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.feedback_logs_to_bq.writer_identity
}

# ====================================================================
# Completions External Table (GCS-based)
# ====================================================================

# External table for completions data (messages/parts) stored in GCS
resource "google_bigquery_table" "completions_external_table" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.telemetry_dataset.dataset_id
  table_id            = "completions"
  deletion_protection = false

  external_data_configuration {
    autodetect            = false
    source_format         = "NEWLINE_DELIMITED_JSON"
    source_uris           = ["gs://${google_storage_bucket.logs_data_bucket.name}/completions/*"]
    connection_id         = google_bigquery_connection.genai_telemetry_connection.name
    ignore_unknown_values = true
    max_bad_records       = 1000
  }

  # Schema matching the ADK completions format
  schema = jsonencode([
    {
      name = "parts"
      type = "RECORD"
      mode = "REPEATED"
      fields = [
        { name = "type", type = "STRING", mode = "NULLABLE" },
        { name = "content", type = "STRING", mode = "NULLABLE" },
        { name = "mime_type", type = "STRING", mode = "NULLABLE" },
        { name = "uri", type = "STRING", mode = "NULLABLE" },
        { name = "data", type = "BYTES", mode = "NULLABLE" },
        { name = "id", type = "STRING", mode = "NULLABLE" },
        { name = "name", type = "STRING", mode = "NULLABLE" },
        { name = "arguments", type = "JSON", mode = "NULLABLE" },
        { name = "response", type = "JSON", mode = "NULLABLE" }
      ]
    },
    { name = "role", type = "STRING", mode = "NULLABLE" },
    { name = "index", type = "INTEGER", mode = "NULLABLE" }
  ])

  depends_on = [
    google_storage_bucket.logs_data_bucket,
    google_bigquery_connection.genai_telemetry_connection,
    google_storage_bucket_iam_member.telemetry_connection_access
  ]
}

# ====================================================================
# GenAI Log Export Table (pre-created for the completions view)
# ====================================================================

# Pre-create the log export table so the completions_view can be created
# immediately on first deploy. Cloud Logging appends to this table via the
# sink and adds new columns as needed (BQ schema evolution).
# Labels are flattened: dots in label keys become underscores (e.g.
# gen_ai.conversation.id → labels.gen_ai_conversation_id).

resource "google_bigquery_table" "genai_logs_table" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.telemetry_dataset.dataset_id
  table_id            = "gen_ai_client_inference_operation_details"
  deletion_protection = false
  description         = "GenAI inference logs exported directly from Cloud Logging"

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  # Cloud Logging BQ export schema (shared between cicd and single-project).
  # Top-level fields are camelCase (Cloud Logging's LogEntry protobuf schema).
  # Labels are snake_case (OTel attribute keys with dots flattened to underscores).
  # All fields NULLABLE to match Cloud Logging's default export behavior and
  # avoid sink write failures for optional fields (e.g. trace, spanId, labels).
  schema = file("${path.module}/../shared/genai_logs_schema.json")

  depends_on = [google_bigquery_dataset.telemetry_dataset]
}

# ====================================================================
# Completions View (Joins BQ log export with GCS Data)
# ====================================================================

# View that joins BigQuery log export data with GCS-stored completions data
resource "google_bigquery_table" "completions_view" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.telemetry_dataset.dataset_id
  table_id            = "completions_view"
  description         = "View of GenAI completion logs joined with the GCS prompt/response external table"
  deletion_protection = false

  view {
    query = templatefile("${path.module}/../shared/completions.sql", {
      project_id                 = var.project_id
      dataset_id                 = google_bigquery_dataset.telemetry_dataset.dataset_id
      completions_external_table = google_bigquery_table.completions_external_table.table_id
    })
    use_legacy_sql = false
  }

  depends_on = [
    google_bigquery_table.completions_external_table,
    google_bigquery_table.genai_logs_table,
    google_logging_project_sink.genai_logs_to_bq
  ]
}
