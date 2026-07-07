
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

variable "data_store_region" {
  type        = string
  description = "Region for the Agent Platform Search data store."
  default     = "global"
}

variable "data_connector_refresh_interval" {
  type        = string
  description = "Refresh interval for the GCS Data Connector periodic sync."
  default     = "86400s"
}

variable "data_connector_data_schema" {
  type        = string
  description = "Data schema for the GCS Data Connector. Use 'content' for unstructured files (PDF, HTML, TXT), 'document' for NDJSON/JSONL, 'csv' for CSV, or 'custom' for custom JSON."
  default     = "content"
}

