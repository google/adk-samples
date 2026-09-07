
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

output "data_store_id" {
  description = "Data store ID"
  value       = data.external.data_store_id.result.data_store_id
}

output "data_store_collection" {
  description = "Collection that contains the data store (set as DATA_STORE_COLLECTION in .env)"
  value       = data.external.data_store_id.result.collection_id
}

output "data_store_path" {
  description = "Full data store resource path used by the agent"
  value       = "projects/${var.project_id}/locations/${var.data_store_region}/collections/${data.external.data_store_id.result.collection_id}/dataStores/${data.external.data_store_id.result.data_store_id}"
}

output "search_engine_id" {
  description = "Search engine ID"
  value       = google_discovery_engine_search_engine.search_engine.engine_id
}

output "docs_bucket_name" {
  description = "Document bucket name"
  value       = google_storage_bucket.docs_bucket.name
}

