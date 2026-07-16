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

resource "google_storage_bucket" "source_dags_bucket" {
  count                       = var.create_source_bucket ? 1 : 0
  project                     = local.project_id
  name                        = local.source_bucket_name
  location                    = "US" # Or a specific region/multi-region
  uniform_bucket_level_access = true
  force_destroy               = false # Set to true for easier deletion during dev/test, but be cautious!

  # Ensures API services are enabled before bucket creation
  depends_on = [
    google_project_service.apis["storage.googleapis.com"],
  ]
}

resource "google_storage_bucket_object" "source_dags" {
  for_each = var.upload_sample_dags ? fileset("${path.module}/input_dags", "*") : toset([])
  name     = "${var.gcs_source_dags_folder}${each.value}"
  bucket   = local.source_bucket_name
  source   = "${path.module}/input_dags/${each.value}"
  depends_on = [google_storage_bucket.source_dags_bucket]
}


resource "google_storage_bucket" "destination_dags_bucket" {
  count                       = var.create_destination_bucket ? 1 : 0
  project                     = local.project_id
  name                        = local.destination_bucket_name
  location                    = "US" # Or a specific region/multi-region
  uniform_bucket_level_access = true
  force_destroy               = false # Set to true for easier deletion during dev/test, but be cautious!

  # Ensures API services are enabled before bucket creation
  depends_on = [
    google_project_service.apis["storage.googleapis.com"],
  ]
}

resource "google_storage_bucket_object" "destination_dags_folder" {
  name    = var.gcs_destination_dags_folder
  content = " " # Space is sufficient to create an empty folder prefix in GCS
  bucket  = local.destination_bucket_name
  depends_on = [google_storage_bucket.destination_dags_bucket]
}


