# lha-web: tiny Express service that fronts the backend.
# - Serves the Vite static build (web/dist) baked into the image.
# - Validates IAP JWTs on incoming browser requests.
# - Reverse-proxies API paths to the backend with an ID token + the
#   verified IAP JWT forwarded as X-LHA-IAP-Assertion (backend re-verifies it
#   in LHA_AUTH_MODE=iap).

resource "google_service_account" "cloud_run_web" {
  account_id   = "${var.service_name}-web-run"
  display_name = "Cloud Run runtime SA for ${var.service_name}-web"
  project      = var.project_id
}

resource "google_project_iam_member" "cloud_run_web" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.cloud_run_web.email}"
}

resource "google_cloud_run_v2_service" "lha_web" {
  name     = "${var.service_name}-web"
  location = var.region
  project  = var.project_id

  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.cloud_run_deletion_protection
  iap_enabled         = true

  template {
    service_account = google_service_account.cloud_run_web.email

    session_affinity = true

    scaling {
      # Keep one instance warm so first-paint after idle doesn't pay a cold
      # start — matters for IAP-gated browser sessions where the cold start
      # blocks every initial / and /lha/* request.
      min_instance_count = 1
      max_instance_count = 5
    }

    containers {
      image = var.lha_web_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "512Mi"
        }
        cpu_idle = true
        # Faster Node/Express cold start for the IAP-gated first paint.
        startup_cpu_boost = true
      }

      env {
        name  = "LHA_BACKEND_URL"
        value = google_cloud_run_v2_service.app.uri
      }

      env {
        name  = "LHA_IAP_AUDIENCE"
        value = local.lha_web_iap_audience
      }
    }
  }

  depends_on = [
    google_project_service.enabled,
    google_project_iam_member.cloud_run_web,
  ]

  lifecycle {
    ignore_changes = [
      client,
      client_version,
      template[0].containers[0].image,
    ]
  }
}

locals {
  lha_web_iap_audience = "/projects/${data.google_project.current.number}/locations/${var.region}/services/${var.service_name}-web"
}

# IAP fronts the web service: the IAP service agent itself invokes Cloud Run
# on behalf of authenticated browser users, so it needs run.invoker.
resource "google_cloud_run_v2_service_iam_member" "lha_web_iap_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_service.lha_web.location
  name     = google_cloud_run_v2_service.lha_web.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-iap.iam.gserviceaccount.com"

  depends_on = [google_project_service.enabled]
}

# Grant human / group access through IAP. Populate via the iap_users var.
resource "google_iap_web_cloud_run_service_iam_member" "lha_web_users" {
  for_each = toset(var.iap_users)

  project                = var.project_id
  location               = var.region
  cloud_run_service_name = google_cloud_run_v2_service.lha_web.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.key

  depends_on = [google_project_service.enabled]
}

output "lha_web_url" {
  value       = google_cloud_run_v2_service.lha_web.uri
  description = "Public URL of the IAP-fronted web frontend."
}
