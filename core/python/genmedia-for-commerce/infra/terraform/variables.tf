# Copyright 2025 Google LLC
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
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run and Cloud Build"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "genmedia-for-commerce"
}

variable "memory" {
  description = "Memory allocation for Cloud Run"
  type        = string
  default     = "32Gi"
}

variable "cpu" {
  description = "CPU allocation for Cloud Run"
  type        = string
  default     = "8"
}

variable "timeout" {
  description = "Request timeout in seconds"
  type        = number
  default     = 3600
}

variable "concurrency" {
  description = "Max concurrent requests per instance"
  type        = number
  default     = 7
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 2
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

