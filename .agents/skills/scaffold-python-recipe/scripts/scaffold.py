#!/usr/bin/env python3
# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0

import argparse
import os
import shutil

# Junk that can accumulate in the templates directory locally (e.g. from
# running tooling there) and must never be copied into a scaffolded recipe.
COPY_IGNORE_PATTERNS = (
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "*.pyc",
    ".DS_Store",
)


def is_safe_recipe_name(name: str) -> bool:
    """Return True if ``name`` is a single, safe path component.

    Rejects empty names, ``.``/``..``, absolute paths, and any name containing
    a path separator, so scaffolding can never escape the output directory.
    """
    if not name or name in (".", ".."):
        return False
    if os.path.isabs(name):
        return False
    return "/" not in name and "\\" not in name


def replace_in_file(filepath: str, replacements: dict[str, str]) -> None:
    """Reads a file, replaces placeholder tokens, and writes it back."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def scaffold(
    name: str,
    output_dir: str,
) -> bool:
    # Reject unsafe names so scaffolding can never escape output_dir.
    if not is_safe_recipe_name(name):
        print(
            f"Error: Invalid recipe name '{name}'. The name must be a single "
            "directory component (no '/', '\\', '..', or absolute path)."
        )
        return False

    # Setup paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    skill_dir = os.path.dirname(script_dir)
    templates_dir = os.path.join(skill_dir, "resources", "templates")

    if not os.path.isdir(templates_dir):
        print(f"Error: Templates directory not found: {templates_dir}")
        return False

    # Target directory under the workspace
    target_dir = os.path.abspath(os.path.join(output_dir, name))

    if os.path.exists(target_dir):
        print(f"Error: Target directory {target_dir} already exists.")
        return False

    # Copy templates, skipping local caches / junk artifacts.
    shutil.copytree(
        templates_dir,
        target_dir,
        ignore=shutil.ignore_patterns(*COPY_IGNORE_PATTERNS),
    )
    print(f"Copied templates to {target_dir}")

    # Define placeholder replacements
    replacements = {
        "<RECIPE_NAME>": name,
        "<OUTPUT_DIRECTORY>": output_dir.rstrip("/"),
    }

    # Walk through the target directory and apply replacements in all files
    for root, _, files in os.walk(target_dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                replace_in_file(filepath, replacements)
            except Exception as e:
                print(f"Warning: Could not process placeholders in {file}: {e}")

    print(f"Successfully scaffolded recipe '{name}' at {target_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaffold a new Python ADK sample."
    )
    parser.add_argument("--name", required=True, help="Name of the recipe")
    parser.add_argument(
        "--output-dir",
        default="contrib",
        help="Output directory inside repository",
    )

    args = parser.parse_args()
    scaffold(
        name=args.name,
        output_dir=args.output_dir,
    )
