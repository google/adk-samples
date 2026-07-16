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

output "gcs_source_bucket_name_output" {
  description = "Name of the GCS bucket for source DAGs."
  value       = local.source_bucket_name
}

output "gcs_destination_bucket_name_output" {
  description = "Name of the GCS bucket for destination DAGs."
  value       = local.destination_bucket_name
}

output "bigquery_table_id_output" {
  description = "Fully qualified BigQuery table ID for the knowledge base."
  value       = "${google_bigquery_table.migration_corpus.project}.${google_bigquery_table.migration_corpus.dataset_id}.${google_bigquery_table.migration_corpus.table_id}"
}

# vertex_ai_search_data_store_id already outputted in vertex_ai_search.tf
# cloud_run_service_url already outputted in cloud_run.tf
