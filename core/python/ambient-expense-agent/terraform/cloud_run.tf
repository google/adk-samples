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

# ---------------------------------------------------------------------------
# Cloud Run services — managed by Terraform.
#
# The Makefile builds container images via Cloud Build, then Terraform
# creates both services with the correct env vars, service accounts,
# and wires BACKEND_URL automatically.
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "backend" {
  name                = var.backend_service_name
  location            = var.region
  project             = var.project_id
  deletion_protection = false

  template {
    scaling {
      min_instance_count = 1
    }

    containers {
      image = var.backend_image

      resources {
        cpu_idle = false
      }

}
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service" "frontend" {
  name                = var.frontend_service_name
  location            = var.region
  project             = var.project_id
  deletion_protection = false
  iap_enabled         = true

  template {
    service_account = google_service_account.frontend_invoker.email

    scaling {
      min_instance_count = 1
    }

    containers {
      image = var.frontend_image

      resources {
        cpu_idle = false
      }

      env {
        name  = "BACKEND_URL"
        value = google_cloud_run_v2_service.backend.uri
      }
      env {
        name  = "USE_SERVICE_AUTH"
        value = "true"
      }
      env {
        name  = "PUBSUB_SUBSCRIPTION"
        value = google_pubsub_subscription.expense_push.name
      }
      env {
        name  = "APP_NAME"
        value = var.agent_name
      }
    }
  }

  depends_on = [google_project_service.apis]
}
