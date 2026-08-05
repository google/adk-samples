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

"""FastAPI uvicorn launcher.

Runs the VTO catalog sandbox uvicorn server on localhost. Loads design-spec.md
into the environment before starting so server.py reads the user's project /
buckets instead of falling back to ADC defaults or failing.
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Add parent of scripts directory to path to enable package resolution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_config(config_path: str) -> dict:
    """Parse YAML frontmatter from design-spec.md. Returns {} on any failure."""
    try:
        import yaml

        content = Path(config_path).read_text()
        parts = re.split(r"^---\s*$", content, flags=re.MULTILINE)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}
    except Exception:
        pass
    return {}


def apply_config_to_env(config_path: str) -> None:
    """Export design-spec values into os.environ. Uses setdefault so caller env wins."""
    cfg = _load_config(config_path)
    if not cfg:
        return
    project_id = cfg.get("gcp_project_id")
    if project_id:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    if cfg.get("gcp_region"):
        os.environ.setdefault("GCP_REGION", cfg["gcp_region"])
    output_bucket = cfg.get("tryon_output_bucket") or (
        f"{project_id}-tryon-output" if project_id else None
    )
    if output_bucket:
        os.environ.setdefault("TRYON_OUTPUT_BUCKET", output_bucket)
    upload_bucket = cfg.get("tryon_upload_bucket") or (
        f"{project_id}-tryon-uploads" if project_id else None
    )
    if upload_bucket:
        os.environ.setdefault("TRYON_UPLOAD_BUCKET", upload_bucket)
    catalog_path = cfg.get("tryon_catalog_path")
    if catalog_path:
        if catalog_path.lower() == "demo":
            catalog_path = "catalog_images"
        os.environ.setdefault("TRYON_CATALOG_PATH", catalog_path)
    if cfg.get("tryon_model"):
        os.environ.setdefault("GEMINI_IMAGE_MODEL", cfg["tryon_model"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./design-spec.md")
    args = parser.parse_args()
    apply_config_to_env(args.config)

    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    print("\n============================================================")
    print("Starting Virtual Try-On Sandbox Dashboard")
    print(f"Access the web interface at:  http://localhost:{port}")
    print("============================================================\n")

    # Run the FastAPI server
    uvicorn.run("scripts.server:app", host="0.0.0.0", port=port, reload=True)
