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

resource "google_storage_bucket" "data_ingestion_PIPELINE_GCS_ROOT" {
  name                        = "${var.project_id}-${var.project_name}-rag"
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = true

  depends_on = [resource.google_project_service.services]
}

# Set up Vector Search 2.0 Collection
resource "null_resource" "vector_search_collection" {
  triggers = {
    project_id    = var.project_id
    location      = var.vector_search_location
    collection_id = var.vector_search_collection_id
    scripts_dir   = "${path.module}/scripts"
  }

  provisioner "local-exec" {
    command = "uv run ${path.module}/scripts/setup_vector_search_collection.py ${var.project_id} ${var.vector_search_location} ${var.vector_search_collection_id}"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "uv run ${self.triggers.scripts_dir}/delete_vector_search_collection.py ${self.triggers.project_id} ${self.triggers.location} ${self.triggers.collection_id}"
  }

  depends_on = [resource.google_project_service.services]
}

