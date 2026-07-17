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

data "external" "env_vars" {
  program = ["python3", "-c", "import os, json; print(json.dumps({'project': os.environ.get('PROJECT_ID', ''), 'account': os.environ.get('USER_EMAIL', '')}))"]
}

locals {
  fetched_project = data.external.env_vars.result["project"]
  fetched_account = data.external.env_vars.result["account"]

  project_id = var.project_id != "" ? var.project_id : local.fetched_project
  user_email = var.user_email != "" ? var.user_email : local.fetched_account

  source_bucket_name      = var.gcs_source_bucket_name != "" ? var.gcs_source_bucket_name : "${local.project_id}-dags-source-bucket"
  destination_bucket_name = var.gcs_destination_bucket_name != "" ? var.gcs_destination_bucket_name : "${local.project_id}-dags-destination-bucket"
}
