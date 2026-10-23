resource "google_storage_bucket" "skills" {
  # Global namespace — scope with the project id (see artifact_bucket.tf).
  name          = coalesce(var.skills_bucket_name, "${var.project_id}-lha-skills-${var.env}")
  project       = var.project_id
  location      = var.location
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  # `make destroy` should tear down the bucket even after objects are written.
  force_destroy = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      days_since_noncurrent_time = var.noncurrent_version_retention_days
      with_state                 = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    app = "lha"
    env = var.env
  }
}
