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

output "backend_url" {
  description = "URL of the backend Cloud Run service (ADK agent)."
  value       = google_cloud_run_v2_service.backend.uri
}

output "frontend_url" {
  description = "URL of the frontend Cloud Run service (approval UI)."
  value       = google_cloud_run_v2_service.frontend.uri
}

output "pubsub_topic" {
  description = "Pub/Sub topic for publishing expense reports."
  value       = google_pubsub_topic.expense_reports.id
}

output "dead_letter_topic" {
  description = "Dead-letter topic for failed expense processing."
  value       = google_pubsub_topic.dead_letter.id
}

output "trigger_endpoint" {
  description = "Full trigger endpoint URL for Pub/Sub."
  value       = "${google_cloud_run_v2_service.backend.uri}/apps/${var.agent_name}/trigger/pubsub"
}

output "alert_policy" {
  description = "Cloud Monitoring alert policy for expense reviews."
  value       = google_monitoring_alert_policy.expense_reviews.display_name
}
