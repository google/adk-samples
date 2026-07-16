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

locals {
  compute_sa = "${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# Grant roles to the default Compute Engine service account
resource "google_project_iam_member" "compute_sa_roles" {
  for_each = toset([
    "roles/storage.admin",
    "roles/bigquery.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/aiplatform.user",
    "roles/discoveryengine.editor",
    "roles/logging.logWriter",
  ])
  project = local.project_id
  role    = each.key
  member  = "serviceAccount:${local.compute_sa}"
}

# Allow user to impersonate the default Compute Engine service account
resource "google_service_account_iam_member" "user_impersonation" {
  for_each = toset([
    "roles/iam.serviceAccountTokenCreator",
    "roles/iam.serviceAccountUser",
  ])
  service_account_id = "projects/${local.project_id}/serviceAccounts/${local.compute_sa}"
  role               = each.key
  member             = "user:${local.user_email}"
}

# Grant Service Usage Consumer to the user to allow using quota project
resource "google_project_iam_member" "user_service_usage" {
  project = local.project_id
  role    = "roles/serviceusage.serviceUsageConsumer"
  member  = "user:${local.user_email}"
}
