# Product Search - Data Ingestion Scripts Reference

Reference documentation for the scripts in `scripts/`. These are standalone
Python scripts that handle product data ingestion and vector search indexing.

All scripts support `--config design-spec.md` to load defaults from your
project configuration.

---

## Script 1: ingest_bigquery.py

**Purpose**: Load product catalog from CSV/JSON into BigQuery
**Location**: `scripts/ingest_bigquery.py`

### Usage

```bash
# From GCS
python scripts/ingest_bigquery.py \
    --project-id my-project \
    --gcs-bucket my-project-products \
    --gcs-path products.csv

# From local file
python scripts/ingest_bigquery.py \
    --project-id my-project \
    --local-file data/products.json \
    --format json

# Using design-spec.md for defaults
python scripts/ingest_bigquery.py \
    --config design-spec.md \
    --local-file data/products.csv
```

### Schema

The script uses a fixed schema matching the Extended product field level:

| Field | Type | Mode | Notes |
|-------|------|------|-------|
| product_id | STRING | REQUIRED | Unique identifier |
| name | STRING | REQUIRED | Product name |
| price | FLOAT64 | REQUIRED | Price in configured currency |
| description | STRING | REQUIRED | Product description |
| category | STRING | NULLABLE | Product category |
| brand | STRING | NULLABLE | Brand name |
| image_url | STRING | NULLABLE | Product image URL |
| rating | FLOAT64 | NULLABLE | Rating (0-5) |
| stock | INT64 | NULLABLE | Stock quantity |

To customize the schema (e.g. for Basic or Full field levels), edit
`REQUIRED_FIELDS`, `OPTIONAL_FIELDS`, and `SCHEMA` at the top of the script.

### Validation

The script validates each row before loading:
- Required fields must be present and non-empty
- `price` must be numeric
- `stock` must be an integer
- Invalid rows are skipped with warnings (not fatal)

---

## Script 2: ingest_vertex_search.py

**Purpose**: Create a Vector Search 2.0 Collection and ingest products
**Location**: `scripts/ingest_vertex_search.py`

Uses Vector Search 2.0 Collections with auto-embeddings. No manual
embedding generation or GCS bucket needed -- the Collection's configured
embedding model generates embeddings automatically when data objects
are inserted.

### Usage

```bash
# Basic usage
python scripts/ingest_vertex_search.py \
    --project-id my-project \
    --collection-id retail-skill-products-collection

# Using design-spec.md for defaults
python scripts/ingest_vertex_search.py --config design-spec.md
```

### Pipeline

1. Fetch products from BigQuery (`retail_skill_products.products`)
2. Create Vector Search 2.0 Collection if it doesn't exist (with auto-embedding config)
3. Batch-insert products as data objects (250 per batch)
4. VS 2.0 auto-generates embeddings from the configured text template

### Configuration

| Flag | Default | Config key |
|------|---------|------------|
| `--collection-id` | retail-skill-products-collection | `collection_id` |
| `--embedding-model` | gemini-embedding-001 | `embedding_model` |
| `--embedding-fields` | name,description,category,brand | `embedding_fields` |
| `--location` | us-central1 | `gcp_region` |

### Output

After ingestion, set the `VECTOR_SEARCH_COLLECTION` environment variable
in your agent to the collection path printed by the script:

```bash
export VECTOR_SEARCH_COLLECTION="projects/PROJECT_ID/locations/us-central1/collections/retail-skill-products-collection"
```

---

## Script 3: ingest_gcs.py

**Purpose**: Upload product images to GCS and create image manifest
**Location**: `scripts/ingest_gcs.py`

Only needed for multimodal search (text + images).

### Usage

```bash
# From local directory
python scripts/ingest_gcs.py \
    --project-id my-project \
    --bucket-name my-project-images \
    --source-type local \
    --source-path ./product_images/

# From URL file
python scripts/ingest_gcs.py \
    --project-id my-project \
    --bucket-name my-project-images \
    --source-type urls \
    --url-file image_urls.txt
```

### Output

Creates `image_manifest.json` mapping product IDs to GCS URIs:

```json
[
  {
    "product_id": "prod-001",
    "image_uris": ["gs://bucket/products/prod-001.jpg"],
    "primary_image": "gs://bucket/products/prod-001.jpg",
    "image_count": 1
  }
]
```

This manifest is used when configuring multimodal search.

---

## Script 4: validate_schema.py

**Purpose**: Validate product data files before ingestion
**Location**: `scripts/validate_schema.py`

### Usage

```bash
# Validate CSV at Extended level (matches sample-products.csv)
python scripts/validate_schema.py \
    --file data/products.csv \
    --fields-level Extended

# Validate JSON at Standard level
python scripts/validate_schema.py \
    --file data/products.json \
    --format json \
    --fields-level Standard
```

### Field Levels

| Level | Required | Optional |
|-------|----------|----------|
| Basic | product_id, name, price, description | (none) |
| Standard | Basic | category, brand, image_url |
| Extended | Basic | Standard + rating, stock, manufacturer |
| Full | Basic | Extended + variants, tags, specifications, reviews |

---

## Script 5: pubsub_sync.py

**Purpose**: Cloud Function for real-time vector index sync
**Location**: `scripts/pubsub_sync.py`

Only needed when `catalog_change_frequency` is "Real-time (Cloud Pub/Sub trigger)".

### Deploy

```bash
gcloud functions deploy pubsub-sync \
    --gen2 \
    --runtime python311 \
    --trigger-topic retail-skill-product-changes \
    --entry-point handle_product_event \
    --region us-central1
```

### Event Format

```json
{"action": "created", "product_id": "prod-001"}
{"action": "updated", "product_id": "prod-001"}
{"action": "deleted", "product_id": "prod-001"}
```

---

## Ingestion Order

Run scripts in this order:

1. `validate_schema.py` -- validate your data file
2. `ingest_bigquery.py` -- load to BigQuery
3. `ingest_gcs.py` -- upload images (multimodal only)
4. `ingest_vertex_search.py` -- create collection + ingest products

**Do NOT use `agents-cli data-ingestion`** -- it runs a generic document
pipeline that is incompatible with product catalogs.

---

## Customizing the Schema

To change the product schema:

1. Edit `REQUIRED_FIELDS`, `OPTIONAL_FIELDS`, and `SCHEMA` in `ingest_bigquery.py`
2. Edit `DEFAULT_EMBEDDING_FIELDS` and `PRODUCT_DATA_FIELDS` in `ingest_vertex_search.py`
3. Update `validate_schema.py` field level definitions if needed
4. Update `design-spec.md` with the new field level and embedding fields
