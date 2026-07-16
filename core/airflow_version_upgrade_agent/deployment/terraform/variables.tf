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

variable "project_id" {
  description = "The GCP project ID."
  type        = string
  default     = ""
}

variable "region" {
  description = "The GCP region for deploying resources (e.g. global)."
  type        = string
  default     = "global" # Set a default, but allow override
}

variable "gcs_source_bucket_name" {
  description = "Name for the GCS bucket where source Airflow DAGs are located."
  type        = string
  default     = ""
}

variable "gcs_destination_bucket_name" {
  description = "Name for the GCS bucket where converted Airflow DAGs will be stored."
  type        = string
  default     = ""
}

variable "bigquery_dataset_id" {
  description = "The ID for the BigQuery dataset to store migration knowledge."
  type        = string
  default     = "airflow_migration_kb"
}

variable "bigquery_table_id" {
  description = "The ID for the BigQuery table to store migration knowledge."
  type        = string
  default     = "migration_corpus"
}

variable "vertex_ai_data_store_display_name" {
  description = "The display name for the Vertex AI Search data store."
  type        = string
  default     = "Airflow Migration Knowledge Base"
}

variable "vertex_ai_data_store_location" {
  description = "The location of the Vertex AI Search data store (must match the BigQuery dataset location for BigQuery sources)."
  type        = string
  default     = "global" # Vertex AI Search data stores can be global or regional
}


variable "gcs_source_dags_folder" {
  description = "The folder path within the source GCS bucket where DAGs are located (e.g., 'dags/')."
  type        = string
  default     = "dags/"
}

variable "gcs_destination_dags_folder" {
  description = "The folder path within the destination GCS bucket where converted DAGs will be stored (e.g., 'dags/')."
  type        = string
  default     = "dags/"
}

variable "create_source_bucket" {
  description = "Whether to create the GCS bucket for source Airflow DAGs. Set to false if it already exists."
  type        = bool
  default     = true
}

variable "create_destination_bucket" {
  description = "Whether to create the GCS bucket for destination Airflow DAGs. Set to false if it already exists or is the same as the source bucket."
  type        = bool
  default     = true
}

variable "upload_sample_dags" {
  description = "Whether to upload sample DAGs from the local 'input_dags' directory into the source bucket."
  type        = bool
  default     = true
}

variable "user_email" {
  description = "The email address of the user to grant impersonation permissions."
  type        = string
  default     = ""
}
