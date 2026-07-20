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
# Pub/Sub topic + authenticated push subscription → /trigger/pubsub
#
# Messages published to the "expense-reports" topic are automatically
# pushed to the agent's trigger endpoint on Cloud Run.
# ---------------------------------------------------------------------------

resource "google_pubsub_topic" "expense_reports" {
  name    = "expense-reports"
  project = var.project_id

  depends_on = [google_project_service.apis]
}

# Dead-letter topic for messages that fail after max delivery attempts.
resource "google_pubsub_topic" "dead_letter" {
  name    = "expense-reports-dead-letter"
  project = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_subscription" "expense_push" {
  name    = "expense-reports-push"
  project = var.project_id
  topic   = google_pubsub_topic.expense_reports.id

  push_config {
    push_endpoint = "${google_cloud_run_v2_service.backend.uri}/apps/${var.agent_name}/trigger/pubsub"

    oidc_token {
      service_account_email = google_service_account.pubsub_invoker.email
      audience              = google_cloud_run_v2_service.backend.uri
    }
  }

  # 10-minute ack deadline (maximum for push subscriptions).
  ack_deadline_seconds = 600

  # Retry with exponential backoff on failed deliveries.
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  # Route failed messages to the dead-letter topic after 5 attempts.
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }

  expiration_policy {
    ttl = ""
  }
}
