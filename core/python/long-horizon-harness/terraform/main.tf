terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source = "hashicorp/google"
      # 7.21+: adds `iap_enabled` on google_cloud_run_v2_service for direct IAP.
      version = ">= 7.21, < 8.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_services = [
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "cloudscheduler.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "iap.googleapis.com",
    "aiplatform.googleapis.com",       # Vertex: Gemini + Memory Bank + sandbox
    "artifactregistry.googleapis.com", # agents-cli / Cloud Build image push
    "cloudbuild.googleapis.com",       # agents-cli deploy + build-web
  ]
}

resource "google_project_service" "enabled" {
  for_each = toset(local.required_services)

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}
