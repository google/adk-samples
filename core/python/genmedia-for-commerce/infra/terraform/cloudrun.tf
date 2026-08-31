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

# Cloud Run service
resource "google_cloud_run_v2_service" "main" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      # Initial placeholder image - will be replaced by Cloud Build
      image = "gcr.io/cloudrun/placeholder"

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
        cpu_idle = false  # no-cpu-throttling equivalent
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "LOCATION"
        value = var.region
      }
    }

    service_account                  = google_service_account.cloudrun.email
    timeout                          = "${var.timeout}s"
    max_instance_request_concurrency = var.concurrency
  }

  depends_on = [
    google_project_service.apis,
    google_service_account.cloudrun,
  ]

  lifecycle {
    ignore_changes = [
      template,  # Entire template managed by Cloud Build deploys
      client,
      client_version,
      invoker_iam_disabled,
    ]
  }
}
