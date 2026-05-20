"""Shared utilities for skill setup scripts.

Provides load_config() for reading YAML frontmatter from design-spec.md
and run_step() for executing pipeline commands.
"""

import logging
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load design-spec.md YAML frontmatter."""
    if not Path(config_path).exists():
        return {}
    text = Path(config_path).read_text()
    if text.startswith("---"):
        lines = text.split("\n")
        yaml_lines = []
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                    continue
                else:
                    break
            if in_frontmatter:
                yaml_lines.append(line)
        yaml_text = "\n".join(yaml_lines)
        return yaml.safe_load(yaml_text) or {}
    return yaml.safe_load(text) or {}


def run_step(description: str, cmd: list, dry_run: bool = False) -> bool:
    """Run a pipeline step. Returns True on success."""
    logger.info(f"\n  {description}")
    if dry_run:
        logger.info(f"    [dry-run] {' '.join(cmd)}")
        return True
    logger.info(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"    FAILED (exit code {result.returncode})")
        return False
    return True
