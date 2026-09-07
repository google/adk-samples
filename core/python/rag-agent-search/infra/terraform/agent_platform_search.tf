
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

# GCS bucket for documents to be ingested
resource "google_storage_bucket" "docs_bucket" {
  name                        = "${var.project_id}-${var.project_name}-docs"
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = true

  depends_on = [resource.google_project_service.services]
}

# Set up GCS Data Connector
resource "null_resource" "data_connector" {
  triggers = {
    project_id      = var.project_id
    location        = var.data_store_region
    collection_id   = "${var.project_name}-collection"
    scripts_dir     = "${path.module}/scripts"
  }

  provisioner "local-exec" {
    command = "uv run ${path.module}/scripts/setup_data_connector.py ${var.project_id} ${var.data_store_region} ${var.project_name}-collection ${var.project_name} gs://${google_storage_bucket.docs_bucket.name} --refresh-interval ${var.data_connector_refresh_interval} --data-schema ${var.data_connector_data_schema}"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "uv run ${self.triggers.scripts_dir}/delete_data_connector.py ${self.triggers.project_id} ${self.triggers.location} ${self.triggers.collection_id}"
  }

  depends_on = [google_storage_bucket.docs_bucket]
}

# Retrieve the auto-created data store ID
data "external" "data_store_id" {
  program = ["uv", "run", "${path.module}/scripts/get_data_store_id.py"]

  query = {
    project_id    = var.project_id
    location      = var.data_store_region
    collection_id = "${var.project_name}-collection"
  }

  depends_on = [null_resource.data_connector]
}

# Search engine app over the connector's auto-created data store. The data
# store's actual collection is resolved by get_data_store_id.py (a connector
# places its data store in its own collection, not default_collection).
resource "google_discovery_engine_search_engine" "search_engine" {
  project        = var.project_id
  engine_id      = "${var.project_name}-search"
  collection_id  = data.external.data_store_id.result.collection_id
  location       = var.data_store_region
  display_name   = "Search Engine App"
  data_store_ids = [data.external.data_store_id.result.data_store_id]
  search_engine_config {
    search_tier = "SEARCH_TIER_ENTERPRISE"
  }
  provider   = google.billing_override
  depends_on = [null_resource.data_connector]
}

