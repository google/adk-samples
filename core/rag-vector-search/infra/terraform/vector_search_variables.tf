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


variable "pipelines_roles" {
  description = "List of roles to assign to the Vertex AI runner service account"
  type        = list(string)
  default = [
    "roles/storage.admin",
    "roles/run.invoker",
    "roles/aiplatform.user",
    "roles/logging.logWriter",
    "roles/artifactregistry.writer",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/bigquery.readSessionUser",
    "roles/bigquery.connectionAdmin",
    "roles/vectorsearch.dataObjectWriter"
  ]
}

variable "vector_search_location" {
  type        = string
  description = "The location for the Vector Search 2.0 Collection."
  default     = "us-central1"
}

variable "vector_search_collection_id" {
  type        = string
  description = "The ID for the Vector Search 2.0 Collection."
  default     = "rag-vector-search-collection"
}

