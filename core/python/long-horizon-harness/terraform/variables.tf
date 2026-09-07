variable "project_id" {
  type        = string
  description = "GCP project that owns the bucket. Required — supply via -var or TF_VAR_project_id."
}

variable "env" {
  type        = string
  description = "Environment suffix appended to the bucket name (dev/staging/prod)."
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.env)
    error_message = "env must be one of: dev, staging, prod."
  }
}

variable "location" {
  type        = string
  description = "Bucket location. US multi-region by default for cross-region read latency parity with Vertex 'global'."
  default     = "US"
}

variable "artifacts_bucket_name" {
  type        = string
  default     = null
  description = "Override the artifact GCS bucket name. Defaults to <project_id>-lha-artifacts-<env> (bucket names are a single global namespace). Set to an existing bucket name to reuse it instead of creating a new one."
}

variable "skills_bucket_name" {
  type        = string
  default     = null
  description = "Override the skills GCS bucket name. Defaults to <project_id>-lha-skills-<env>. Set to an existing bucket name to reuse it."
}

variable "region" {
  type        = string
  description = "Region for Cloud Run, Cloud SQL, and Cloud Scheduler resources."
  default     = "us-central1"
}

variable "image_uri" {
  type        = string
  description = "Container image for the Cloud Run service (e.g. us-central1-docker.pkg.dev/PROJECT/lha/lha:TAG)."
}

variable "lha_web_image" {
  type        = string
  description = "Container image for the IAP-fronted web frontend Cloud Run service (e.g. us-central1-docker.pkg.dev/PROJECT/lha/web:TAG). Built from web/server/Dockerfile."
}

variable "iap_users" {
  type        = list(string)
  description = "Principals granted roles/iap.httpsResourceAccessor on lha-web. Each entry must be a fully-qualified IAM member string (e.g. \"user:alice@example.com\", \"group:team@example.com\"). Empty by default — grant access explicitly via -var or TF_VAR_iap_users."
  default     = []
}

variable "lha_runtime_image" {
  type        = string
  description = "Sandbox shim container image used by Vertex Agent Engine sandboxes (e.g. REGION-docker.pkg.dev/PROJECT/lha-sandbox/runtime:TAG). Required — supply via -var or TF_VAR_lha_runtime_image; bump the tag to roll out a new runtime version."
}

variable "agent_engine_display_name" {
  type        = string
  description = "Shared Vertex Agent Engine display name. Hosts Memory Bank (cross-session memory) and per-user sandboxes. Discovered or created on every `terraform apply` via the external data source in agent_engine.tf — terraform calls scripts/provision_agent_engine.py which is idempotent."
  default     = "lha-agent-engine"
}

variable "agent_engine_location" {
  type        = string
  description = "Region for the shared Agent Engine. Memory Bank is regional; `global` is rejected. us-central1 is the default."
  default     = "us-central1"
}

variable "service_name" {
  type        = string
  description = "Cloud Run service name."
  default     = "lha"
}

variable "agent_version" {
  type        = string
  description = "Value of AGENT_VERSION env var. Stamped onto the A2A agent card. Keep in sync with pyproject.toml version."
  default     = "0.1.0"
}

variable "db_instance_name" {
  type        = string
  description = "Cloud SQL instance name."
  default     = "lha-db"
}

variable "db_tier" {
  type        = string
  description = "Cloud SQL machine tier. db-custom-1-3840 (1 vCPU, 3.75 GB) is the prod minimum: it has the RAM to back db_max_connections (~5-10 MB/conn) across all Cloud Run instances. db-f1-micro (shared core, ~600 MB) is dev-only — its ~25-connection ceiling exhausts under multi-instance load. HA (REGIONAL) also requires a dedicated tier."
  default     = "db-custom-1-3840"
}

variable "db_max_connections" {
  type        = number
  description = "Postgres max_connections. Must satisfy (per-instance pool cap) x (Cloud Run max-instances) + ops reserve <= this, AND fit the tier's RAM (~5-10 MB/conn). 300 covers the app's theoretical cap (30 x 8 = 240) plus ~60 for superuser/migrations/psql/proxy on db-custom-1-3840 (3.75 GB); raise the tier before raising this much further."
  default     = 300
}

variable "db_availability_type" {
  type        = string
  description = "ZONAL (single node, cheapest) for dev; REGIONAL (hot standby + automatic failover, ~2x compute cost) for prod. REGIONAL fails over during maintenance so an outage becomes a seconds-long blip, but requires a dedicated db_tier."
  default     = "ZONAL"

  validation {
    condition     = contains(["ZONAL", "REGIONAL"], var.db_availability_type)
    error_message = "db_availability_type must be ZONAL or REGIONAL."
  }
}

variable "db_edition" {
  type        = string
  description = "Cloud SQL edition. null uses the provider default (ENTERPRISE). ENTERPRISE_PLUS adds near-zero-downtime planned maintenance but requires perf-optimized machine tiers (pricier) — don't pair it with shared-core db_tier."
  default     = null

  validation {
    # Ternary, not ||: TF 1.5.x eagerly evaluates contains() on the null default
    # and errors; the ternary only evaluates the branch it takes.
    condition     = var.db_edition == null ? true : contains(["ENTERPRISE", "ENTERPRISE_PLUS"], var.db_edition)
    error_message = "db_edition must be null, ENTERPRISE, or ENTERPRISE_PLUS."
  }
}

variable "db_maintenance_window" {
  type = object({
    day          = number # 1 = Monday … 7 = Sunday
    hour         = number # 0–23, in the instance's local time
    update_track = string # "stable" (later track) or "canary" (earlier)
  })
  description = "Optional Cloud SQL maintenance window. null lets Google choose the time; set it (e.g. Sunday 04:00 on the stable track) to confine maintenance to off-hours."
  default     = null

  validation {
    # Ternary, not ||: avoids eagerly dereferencing .update_track on the null default.
    condition     = var.db_maintenance_window == null ? true : contains(["stable", "canary"], var.db_maintenance_window.update_track)
    error_message = "db_maintenance_window.update_track must be stable or canary."
  }
}

variable "dream_review_user_ids" {
  type        = list(string)
  description = "User IDs the daily dream-review scheduler targets. Empty (the default) means 'every user with a session in the lookback window' — the endpoint auto-discovers them from the session store."
  default     = []
}

variable "snapshot_user_ids" {
  type        = list(string)
  description = "User IDs the daily sandbox-snapshot scheduler targets. Empty (the default) means 'every active user' (auto-discovered server-side). The /scheduler/snapshot endpoint no-ops unless LHA_SNAPSHOT_ENABLED is set on the backend — Phase C is gated on the snapshot probe."
  default     = []
}

variable "agent_app_name" {
  type        = string
  description = "Advisory app_name in the dream-review request body. The endpoint uses the runner's actual app name (where sessions are keyed); this is kept for observability."
  default     = "lha"
}

variable "cloud_run_deletion_protection" {
  type        = bool
  description = "Cloud Run delete guard. True for prod; set false to allow replacement during smoke tests."
  default     = true
}

variable "db_deletion_protection" {
  type        = bool
  description = "Cloud SQL delete guard. True for prod; set false only when deliberately tearing down."
  default     = true
}

variable "noncurrent_version_retention_days" {
  type        = number
  description = "Days to keep noncurrent object versions before deletion."
  default     = 30
}

variable "gcp_oauth_client_id" {
  type        = string
  description = "Connect Google OAuth: Web client ID. Empty (default) disables the feature."
  default     = ""
}

variable "gcp_oauth_redirect_uri" {
  type        = string
  description = "Connect Google OAuth: redirect URI, e.g. https://<web-host>/lha/gcp/callback. Empty disables the feature."
  default     = ""
}

variable "gcp_oauth_client_secret" {
  type        = string
  description = "Connect Google OAuth: Web client secret. Empty disables the feature. Supply via TF_VAR_gcp_oauth_client_secret or a tfvars kept out of VCS."
  default     = ""
  sensitive   = true
}
