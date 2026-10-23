resource "google_sql_database_instance" "db" {
  name             = var.db_instance_name
  database_version = "POSTGRES_15"
  region           = var.region
  project          = var.project_id

  deletion_protection = var.db_deletion_protection

  settings {
    tier              = var.db_tier
    edition           = var.db_edition
    availability_type = var.db_availability_type
    disk_size         = 10
    disk_autoresize   = true

    # Set max_connections explicitly instead of inheriting the tiny tier default
    # (~25 on shared-core). Sized with db_tier so the app's bounded pools across
    # all Cloud Run instances fit under it. Changing this restarts the instance.
    database_flags {
      name  = "max_connections"
      value = tostring(var.db_max_connections)
    }

    # Off-hours maintenance + later (stable) update track for prod; unset in dev
    # so Google schedules it whenever.
    dynamic "maintenance_window" {
      for_each = var.db_maintenance_window == null ? [] : [var.db_maintenance_window]
      content {
        day          = maintenance_window.value.day
        hour         = maintenance_window.value.hour
        update_track = maintenance_window.value.update_track
      }
    }

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
      start_time                     = "03:00"
    }

    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }
  }

  depends_on = [google_project_service.enabled]
}

resource "google_sql_database" "lha" {
  name     = "lha"
  instance = google_sql_database_instance.db.name
  project  = var.project_id
}

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "google_sql_user" "lha" {
  name     = "lha"
  instance = google_sql_database_instance.db.name
  password = random_password.db.result
  project  = var.project_id

  # Postgres refuses to DROP a role that owns objects, and the app's session
  # tables are owned by this one — a plain destroy would fail once the app has
  # served traffic. Dropping the instance removes the user with it.
  deletion_policy = "ABANDON"
}

resource "google_secret_manager_secret" "db_password" {
  secret_id = "${var.service_name}-db-password"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db.result
}

resource "google_secret_manager_secret_iam_member" "cloud_run_db_password" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_secret_manager_secret" "db_url" {
  secret_id = "${var.service_name}-db-url"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_version" "db_url" {
  secret      = google_secret_manager_secret.db_url.id
  secret_data = "postgres://${google_sql_user.lha.name}:${random_password.db.result}@/${google_sql_database.lha.name}?host=/cloudsql/${google_sql_database_instance.db.connection_name}"
}

resource "google_secret_manager_secret_iam_member" "cloud_run_db_url" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.db_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

# ADK 2.x DatabaseSessionService always uses `create_async_engine` — same
# requirement as DatabaseTaskStore, so this URL takes the asyncpg driver too.
resource "google_secret_manager_secret" "session_db_url" {
  secret_id = "${var.service_name}-session-db-url"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_version" "session_db_url" {
  secret      = google_secret_manager_secret.session_db_url.id
  secret_data = "postgresql+asyncpg://${google_sql_user.lha.name}:${random_password.db.result}@/${google_sql_database.lha.name}?host=/cloudsql/${google_sql_database_instance.db.connection_name}"
}

resource "google_secret_manager_secret_iam_member" "cloud_run_session_db_url" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.session_db_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

# DatabaseTaskStore wraps `create_async_engine` — needs the asyncpg
# SQLAlchemy driver scheme.
resource "google_secret_manager_secret" "task_db_url" {
  secret_id = "${var.service_name}-task-db-url"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_version" "task_db_url" {
  secret      = google_secret_manager_secret.task_db_url.id
  secret_data = "postgresql+asyncpg://${google_sql_user.lha.name}:${random_password.db.result}@/${google_sql_database.lha.name}?host=/cloudsql/${google_sql_database_instance.db.connection_name}"
}

resource "google_secret_manager_secret_iam_member" "cloud_run_task_db_url" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.task_db_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}
