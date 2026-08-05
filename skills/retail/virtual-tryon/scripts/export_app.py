#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production exporter utility.

Copies backend scripts, UI static files, and writes container configurations
to target export folders for production Cloud Run deployments.
"""

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

try:
    from _setup_utils import load_config as _shared_load_config
except ImportError:
    _shared_load_config = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from design-spec.md YAML frontmatter.

    Only closes the frontmatter on a line that is exactly `---` (with optional
    trailing whitespace). Naive `str.split("---", ...)` mis-splits on
    section-header comments like `# --- Required ---` inside the frontmatter.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found at: {config_path}")
        return {}

    if _shared_load_config is not None:
        return _shared_load_config(config_path) or {}

    try:
        content = path.read_text()
        if content.startswith("---"):
            parts = re.split(r"^---\s*$", content, flags=re.MULTILINE)
            if len(parts) >= 3:
                return yaml.safe_load(parts[1]) or {}
        return yaml.safe_load(content) or {}
    except Exception as e:
        logger.error(f"Failed to parse config: {e}")
        return {}


def export_app(config_path: str, skill_dir: str):
    """Package VTO sandbox into a standalone Cloud Run deployment folder."""
    config = load_config(config_path)
    if not config:
        logger.error("Empty or invalid design specification configuration.")
        return False

    project_id = config.get("gcp_project_id")
    if not project_id:
        logger.error("gcp_project_id is required in design-spec.md for export.")
        return False

    region = config.get("gcp_region", "us-west1")
    catalog_path = config.get("tryon_catalog_path", "catalog_images")
    output_bucket = (
        config.get("tryon_output_bucket") or f"{project_id}-tryon-output"
    )
    upload_bucket = (
        config.get("tryon_upload_bucket") or f"{project_id}-tryon-uploads"
    )
    model_name = config.get("tryon_model", "flash")

    # Destination directory path
    dest_val = config.get("export_directory") or "./vto-retail-app"
    dest_dir = Path(dest_val).resolve()
    logger.info(f"Exporting production-ready application to: {dest_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    ui_dest_dir = dest_dir / "ui"
    ui_dest_dir.mkdir(exist_ok=True)

    src_path = Path(skill_dir)

    # 1. Export UI assets
    ui_src_dir = src_path / "assets" / "ui"
    if ui_src_dir.exists():
        for f in ["index.html", "styles.css", "app.js"]:
            src_file = ui_src_dir / f
            if src_file.exists():
                shutil.copy2(src_file, ui_dest_dir / f)
                logger.info(f"Copied static asset: {f} -> {ui_dest_dir}")
    else:
        logger.error(f"UI source assets folder not found at: {ui_src_dir}")
        return False

    # 2. Export and Refactor Python source files
    scripts_src_dir = src_path / "scripts"
    py_files = [
        "server.py",
        "tryon_processor.py",
        "tryon_agent.py",
        "scan_catalog.py",
        "setup_tryon.py",
    ]

    for f in py_files:
        src_file = scripts_src_dir / f
        if not src_file.exists():
            logger.warning(f"Python script not found: {src_file}")
            continue

        with open(src_file) as rfile:
            code = rfile.read()

        # Refactor imports from "scripts.X" to "X" for standalone root directory execution
        code = re.sub(
            r"from\s+scripts\.(?P<mod>\w+)\s+import",
            r"from \g<mod> import",
            code,
        )
        code = re.sub(
            r"import\s+scripts\.(?P<mod>\w+)", r"import \g<mod>", code
        )

        # In server.py, adjust the static asset mounts path
        if f == "server.py":
            # Change UI_DIR resolution to point directly to "./ui" in root
            code = code.replace(
                'UI_DIR = Path(__file__).resolve().parent.parent / "assets" / "ui"',
                'UI_DIR = Path(__file__).resolve().parent / "ui"',
            )

        dest_file = dest_dir / f
        with open(dest_file, "w") as wfile:
            wfile.write(code)
        # Preserve executability
        shutil.copymode(src_file, dest_file)
        logger.info(f"Exported python module: {f} -> {dest_dir}")

    # 3. Create requirements.txt
    reqs = """fastapi>=0.100.0
uvicorn>=0.20.0
python-multipart>=0.0.6
google-genai>=1.0.0
google-cloud-storage>=2.0.0
pillow>=9.0.0
pyyaml>=6.0
httpx>=0.23.0
"""
    with open(dest_dir / "requirements.txt", "w") as f:
        f.write(reqs)
    logger.info("Generated: requirements.txt")

    # 4. Create Dockerfile
    dockerfile = f"""# Stage 1: Build virtual environment
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime container
FROM python:3.11-slim AS runner
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables from project configuration
ENV GOOGLE_CLOUD_PROJECT="{project_id}"
ENV GCP_REGION="{region}"
ENV TRYON_CATALOG_PATH="{catalog_path}"
ENV TRYON_OUTPUT_BUCKET="{output_bucket}"
ENV TRYON_UPLOAD_BUCKET="{upload_bucket}"
ENV GEMINI_IMAGE_MODEL="{model_name}"
ENV PORT=8080

COPY . .
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
"""
    with open(dest_dir / "Dockerfile", "w") as f:
        f.write(dockerfile)
    logger.info("Generated: Dockerfile")

    # 5. Create .dockerignore
    dockerignore = """__pycache__/
.pytest_cache/
.catalog_cache/
catalog_images/
tmp/
.git/
.gitignore
deploy_cloudrun.sh
"""
    with open(dest_dir / ".dockerignore", "w") as f:
        f.write(dockerignore)
    logger.info("Generated: .dockerignore")

    # 6. Create cloudbuild.yaml
    cloudbuild = f"""steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/vto-retail-app:$COMMIT_SHA', '.']
  
  # Push the image to Artifact Registry / GCR
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/vto-retail-app:$COMMIT_SHA']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'vto-retail-app'
      - '--image'
      - 'gcr.io/$PROJECT_ID/vto-retail-app:$COMMIT_SHA'
      - '--region'
      - '{region}'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
images:
  - 'gcr.io/$PROJECT_ID/vto-retail-app:$COMMIT_SHA'
"""
    with open(dest_dir / "cloudbuild.yaml", "w") as f:
        f.write(cloudbuild)
    logger.info("Generated: cloudbuild.yaml")

    # 7. Create deploy_cloudrun.sh script
    deploy_script = f"""#!/bin/bash
# Standalone Cloud Run deployment script fallback

set -e

PROJECT_ID="{project_id}"
REGION="{region}"
SERVICE_NAME="vto-retail-app"

echo "=========================================================="
echo "DEPLOYING VIRTUAL TRY-ON APP TO GOOGLE CLOUD RUN"
echo "=========================================================="
echo "GCP Project:   $PROJECT_ID"
echo "Region:        $REGION"
echo "Service Name:  $SERVICE_NAME"
echo "=========================================================="

# 1. Enable APIs
echo "Enabling Cloud Run and Cloud Build APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com --project="$PROJECT_ID" --quiet

# 2. Deploy directly from source (uses local Dockerfile)
echo "Deploying source container via Cloud Build and Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \\
  --source . \\
  --region "$REGION" \\
  --project "$PROJECT_ID" \\
  --allow-unauthenticated \\
  --quiet

echo "=========================================================="
echo "DEPLOYMENT COMPLETED SUCCESSFULLY"
echo "=========================================================="
"""

    deploy_file = dest_dir / "deploy_cloudrun.sh"
    with open(deploy_file, "w") as f:
        f.write(deploy_script)
    # Make script executable
    os.chmod(deploy_file, 0o755)
    logger.info("Generated: deploy_cloudrun.sh")

    logger.info("EXPORT COMPLETED SUCCESSFULLY. Standalone source generated.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export virtual try-on app codebase for production"
    )
    parser.add_argument(
        "--config", required=True, help="Path to design-spec.md config"
    )
    parser.add_argument(
        "--skill-dir",
        required=True,
        help="Path to this skill installation directory",
    )
    args = parser.parse_args()

    success = export_app(args.config, args.skill_dir)
    if not success:
        sys.exit(1)
