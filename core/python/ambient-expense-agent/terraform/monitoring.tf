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
# Cloud Monitoring: log-based metric + alert policy + notification channel
#
# When the agent flags an expense >= $100 for review it emits a structured
# JSON log. Cloud Logging ingests it, a log-based metric counts it, and
# an alert policy sends an email notification.
# ---------------------------------------------------------------------------

resource "google_logging_metric" "expense_reviews" {
  name    = "expense-review-alerts"
  project = var.project_id

  description = "Counts expense review alerts from the expense agent."

  filter = <<-EOT
    resource.type="cloud_run_revision"
    resource.labels.service_name="${var.backend_service_name}"
    jsonPayload.alert_type="expense_review"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
  }
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "Expense Agent - Review Alerts"
  project      = var.project_id
  type         = "email"

  labels = {
    email_address = var.notification_email
  }

  depends_on = [google_project_service.apis]
}

resource "google_monitoring_alert_policy" "expense_reviews" {
  display_name = "Expense Agent - High-Value Expense Review"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Expense review count > 0"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.expense_reviews.name}\" AND resource.type=\"cloud_run_revision\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_COUNT"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.id
  ]

  documentation {
    content   = <<-EOT
## Expense Review Required

One or more expenses of **$$100 or more** have been flagged for review by the ambient expense agent.

### What to do

1. **[Open the Approval UI](${google_cloud_run_v2_service.frontend.uri}/approval)** to review pending expenses
2. Check the amount, submitter, category, and the LLM's risk assessment
3. Click **Approve** or **Reject** — the agent will log your decision and resume the workflow

EOT
    mime_type = "text/markdown"
  }

  depends_on = [google_project_service.apis]
}
