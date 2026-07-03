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
# Pub/Sub topic + authenticated push subscription → Agent Runtime passthrough
#
# Messages published to the "expense-reports" topic are automatically
# pushed to the agent's trigger endpoint via the Agent Runtime API.
#
# Cloud Scheduler publishes to the topic on a cron schedule (optional).
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

# Allow the Pub/Sub dead-letter publisher SA to publish to the dead-letter topic.
resource "google_pubsub_topic_iam_member" "dead_letter_publisher" {
  topic   = google_pubsub_topic.dead_letter.name
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# Allow the Pub/Sub SA to ack messages on the main subscription (for dead-letter).
resource "google_pubsub_subscription_iam_member" "dead_letter_subscriber" {
  subscription = google_pubsub_subscription.expense_push.name
  project      = var.project_id
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_subscription" "expense_push" {
  name    = "expense-reports-push"
  project = var.project_id
  topic   = google_pubsub_topic.expense_reports.id

  push_config {
    # Agent Runtime API passthrough: routes to /api/apps/{agent}/trigger/pubsub
    # inside the container. Requires aiplatform.user on the invoker SA.
    push_endpoint = "https://${var.region}-aiplatform.googleapis.com/reasoningEngines/v1/${var.agent_runtime_resource_name}/api/apps/${var.agent_name}/trigger/pubsub"

    oidc_token {
      service_account_email = google_service_account.pubsub_invoker.email
      # Audience must be the Vertex AI API base URL for the region.
      audience = "https://${var.region}-aiplatform.googleapis.com/"
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

  depends_on = [
    google_pubsub_topic_iam_member.dead_letter_publisher,
  ]
}
