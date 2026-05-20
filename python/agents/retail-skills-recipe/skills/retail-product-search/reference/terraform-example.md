# Example Terraform Configuration

Example `products.tf` for provisioning product search infrastructure.
Adapt the schema and resource names to your project.

## products.tf

```hcl
# Product search infrastructure

# BigQuery dataset and products table
resource "google_bigquery_dataset" "products" {
  dataset_id = "retail_skill_products"
  project    = var.project_id
  location   = var.region

  labels = {
    environment = "dev"
  }
}

resource "google_bigquery_table" "products" {
  dataset_id = google_bigquery_dataset.products.dataset_id
  table_id   = "products"
  project    = var.project_id

  # Adjust schema to match your product fields level (Basic/Standard/Extended/Full)
  schema = jsonencode([
    {"name": "product_id",  "type": "STRING",  "mode": "REQUIRED"},
    {"name": "name",        "type": "STRING",  "mode": "REQUIRED"},
    {"name": "price",       "type": "FLOAT64", "mode": "REQUIRED"},
    {"name": "description", "type": "STRING",  "mode": "REQUIRED"},
    {"name": "category",    "type": "STRING",  "mode": "NULLABLE"},
    {"name": "brand",       "type": "STRING",  "mode": "NULLABLE"},
    {"name": "image_url",   "type": "STRING",  "mode": "NULLABLE"},
    {"name": "rating",      "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "stock",       "type": "INT64",   "mode": "NULLABLE"}
  ])
}

# Pub/Sub for real-time product sync (optional -- only if catalog changes in real-time)
resource "google_pubsub_topic" "product_changes" {
  name    = "retail-skill-product-changes"
  project = var.project_id
}

resource "google_pubsub_subscription" "product_changes_sub" {
  name    = "retail-skill-product-changes-sync"
  topic   = google_pubsub_topic.product_changes.name
  project = var.project_id

  ack_deadline_seconds = 60

  push_config {
    push_endpoint = google_cloudfunctions2_function.pubsub_sync.uri
  }
}

resource "google_cloudfunctions2_function" "pubsub_sync" {
  name     = "retail-skill-pubsub-sync"
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python311"
    entry_point = "handle_product_event"

    source {
      storage_source {
        bucket = google_storage_bucket.logs.name
        object = "functions/pubsub_sync.zip"
      }
    }
  }

  service_config {
    max_instance_count = 10
    available_memory   = "256M"
    timeout_seconds    = 120

    environment_variables = {
      GOOGLE_CLOUD_PROJECT     = var.project_id
      GOOGLE_CLOUD_LOCATION    = var.region
      VECTOR_SEARCH_COLLECTION = "projects/${var.project_id}/locations/${var.region}/collections/${var.project_name}-collection"
    }

    service_account_email = google_service_account.app_sa.email
  }
}

# Agent traces table (evaluation and observability)
resource "google_bigquery_table" "agent_traces" {
  dataset_id = google_bigquery_dataset.products.dataset_id
  table_id   = "agent_traces"
  project    = var.project_id

  schema = jsonencode([
    {"name": "session_id",        "type": "STRING",  "mode": "REQUIRED"},
    {"name": "turn_count",        "type": "INT64",   "mode": "NULLABLE"},
    {"name": "duration_seconds",  "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "final_filters",     "type": "STRING",  "mode": "NULLABLE"},
    {"name": "conversation",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "timestamp",         "type": "FLOAT64", "mode": "NULLABLE"}
  ])
}
```
