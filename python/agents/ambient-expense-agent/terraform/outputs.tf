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

output "agent_runtime_resource_name" {
  description = "Agent Runtime resource name (used by agents-cli deploy to update source)."
  value       = google_vertex_ai_reasoning_engine.app.name
}

output "agent_runtime_api_base" {
  description = "Agent Runtime API base URL (Agent Runtime passthrough). Append /apps/{agent}/... for agent endpoints."
  value       = local.agent_runtime_api_base
}

output "trigger_endpoint" {
  description = "Pub/Sub push endpoint — the Agent Runtime trigger URL."
  value       = "${local.agent_runtime_api_base}/apps/${var.agent_name}/trigger/pubsub"
}

output "frontend_url" {
  description = "URL of the approval UI (Cloud Run)."
  value       = google_cloud_run_v2_service.frontend.uri
}

output "pubsub_topic" {
  description = "Pub/Sub topic for publishing expense reports."
  value       = google_pubsub_topic.expense_reports.id
}

output "alert_policy" {
  description = "Cloud Monitoring alert policy for expense reviews."
  value       = google_monitoring_alert_policy.expense_reviews.display_name
}
