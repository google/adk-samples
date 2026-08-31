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

variable "project_id" {
  type        = string
  description = "The GCP project ID where the datastore is provisioned."
}

variable "project_name" {
  type        = string
  description = "Base name used for datastore resource naming."
  default     = "rag-agent-search"
}

variable "region" {
  type        = string
  description = "The GCP region for resources."
  default     = "us-central1"
}
