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

resource "google_bigquery_dataset" "migration_kb" {
  project    = local.project_id
  dataset_id = var.bigquery_dataset_id
  location   = "US" # Match this to your Vertex AI Search Data Store location if regional
  description = "Dataset for Airflow migration knowledge base."

  depends_on = [
    google_project_service.apis["bigquery.googleapis.com"],
  ]
}

resource "google_bigquery_table" "migration_corpus" {
  project    = local.project_id
  dataset_id = google_bigquery_dataset.migration_kb.dataset_id
  table_id   = var.bigquery_table_id
  description = "A knowledge base for migrating Apache Airflow operators between versions, populated by an AI agent."

  schema = jsonencode([
    {
      name        = "operator_path"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "The full import path of the source operator, e.g., 'airflow.contrib.operators.bigquery_operator.BigQueryOperator'"
    },
    {
      name        = "source_version"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "The source Airflow version for the migration, e.g., '1.10.15'"
    },
    {
      name        = "target_version"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "The target Airflow version for the migration, e.g., '2.8.0'"
    },
    {
      name        = "is_deprecated"
      type        = "BOOLEAN"
      description = "True if the operator is deprecated or has been moved"
    },
    {
      name        = "new_operator_path"
      type        = "STRING"
      description = "The new import path if the operator was moved or replaced"
    },
    {
      name        = "parameter_changes"
      type        = "JSON"
      description = "A JSON object detailing changes to operator parameters (renamed, removed, new defaults)"
    },
    {
      name        = "summary"
      type        = "STRING"
      description = "A brief summary of the required changes for the migration"
    },
    {
      name        = "code_example_before"
      type        = "STRING"
      description = "A snippet of Python code showing usage in the source version"
    },
    {
      name        = "code_example_after"
      type        = "STRING"
      description = "A snippet of Python code showing the updated usage in the target version"
    },
    {
      name        = "source_urls"
      type        = "STRING"
      mode        = "REPEATED"
      description = "A list of source URLs used to generate this knowledge entry"
    },
    {
      name        = "created_at"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "The timestamp when this knowledge entry was created or last updated"
    }
  ])

  labels = {
    component = "airflow-migration-agent"
  }

  clustering = ["operator_path", "source_version", "target_version"]

  depends_on = [
    google_bigquery_dataset.migration_kb,
  ]
}
