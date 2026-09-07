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
# Agent Runtime — the ambient expense agent backend.
#
# Terraform creates the reasoning engine with a placeholder source archive.
# The real source code is deployed (and updated) by:
#   agents-cli deploy --project PROJECT_ID --region REGION
#
# After Terraform creates this resource, the Pub/Sub subscription (pubsub.tf)
# uses the resource name to build the push endpoint automatically.
# ---------------------------------------------------------------------------

locals {
  dummy_source_b64 = trimspace(file("${path.module}/shared/dummy_source.b64"))
}

resource "google_vertex_ai_reasoning_engine" "app" {
  display_name = var.project_name
  description  = "Ambient expense agent — processes expense reports via Pub/Sub triggers"
  region       = var.region
  project      = var.project_id

  spec {
    agent_framework = "google-adk"
    service_account = google_service_account.app_sa.email

    deployment_spec {
      min_instances         = 1
      max_instances         = 10
      container_concurrency = 8

      resource_limits = {
        cpu    = "1"
        memory = "4Gi"
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = "global"
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "True"
      }

      env {
        name  = "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"
        value = "true"
      }
    }

    source_code_spec {
      inline_source {
        source_archive = local.dummy_source_b64
      }
      image_spec {}
    }
  }

  # Terraform creates this resource with a placeholder source build.
  # `agents-cli deploy` overwrites source_code_spec + deployment_spec with
  # the real code. Ignore changes so Terraform never reverts the deployed agent.
  lifecycle {
    ignore_changes = [
      spec[0].container_spec,
      spec[0].source_code_spec,
      spec[0].deployment_spec,
    ]
  }

  depends_on = [
    google_project_service.apis,
    google_service_account.app_sa,
  ]
}
