# Container image for KFP pipeline components.
#
# This image is used as the base_image for all three KFP components
# (preprocess, chunk_and_index, cleanup). When Vertex AI Pipelines
# executes a step, it pulls this image and runs the serialized
# component function inside it.
#
# Key design decisions:
#   - python:3.11-slim (not 3.12) to match the project's uv.lock
#   - uv pip install --system (not uv sync) because KFP's executor
#     runs on the system Python (/usr/local/bin/python3), NOT inside
#     a .venv. Using uv sync would install to .venv and KFP would
#     get ModuleNotFoundError at runtime.
#   - PYTHONPATH=/app so that `from pipeline.xxx import yyy` resolves
#     to /app/pipeline/xxx.py inside the container.
#
# Build from project root (context must be repo root for COPY paths):
#   docker build -f data_ingestion/Dockerfile -t data-pipeline:latest .
#
# Or via Cloud Build (no local Docker needed):
#   gcloud builds submit --config=data_ingestion/cloudbuild.yaml \
#     --substitutions=_IMAGE="$PIPELINE_IMAGE" --project="$PROJECT_ID" .
#
# Push to Artifact Registry:
#   docker tag data-pipeline:latest \
#     ${REGION}-docker.pkg.dev/${PROJECT_ID}/${PROJECT_NAME}-pipeline/data-pipeline:latest
#   docker push \
#     ${REGION}-docker.pkg.dev/${PROJECT_ID}/${PROJECT_NAME}-pipeline/data-pipeline:latest

FROM python:3.11-slim

# Install uv for fast dependency resolution (pinned version for reproducibility)
RUN pip install --no-cache-dir uv==0.8.13

WORKDIR /app

# Install all project dependencies into the system Python.
# We copy pyproject.toml + app/ first (app/ is needed because the project
# is defined as a package in pyproject.toml and must be installable).
# uv pip install --system targets /usr/local/lib/python3.11/ directly,
# making packages available to KFP's executor without activating a venv.
COPY pyproject.toml README.md uv.lock* ./
COPY app/ app/
RUN uv pip install --system --no-cache .

# Copy the shared src package — this is where all business logic lives.
# KFP components import from src.document_preprocessing, src.chunking,
# src.removal.propagate_gcs_deletions, and src.utils at runtime inside the container.
COPY src/ src/

# config.env is needed at runtime for any code that reads configuration
# via python-dotenv (e.g. model names, embedding dimensions).
COPY config.env .

# Copy the KFP pipeline package (DAG definition + component wrappers).
# Note: this is only needed if running submit_pipeline.py from inside
# the container. For normal Vertex AI execution, KFP serializes the
# component function body and sends it to the container directly.
COPY data_ingestion_pipeline/data_ingestion_pipeline/ data_ingestion_pipeline/

# Set PYTHONPATH so that `from src.xxx` and `from data_ingestion_pipeline.xxx`
# resolve correctly regardless of the working directory.
ENV PYTHONPATH=/app
