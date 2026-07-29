"""
Load a value from .github/policy.yml by dotted key path.

Usage:
    load_policy.py <dotted.key.path>

Examples:
    load_policy.py recipe_size_limits.core.default.max_files
    # -> 500

    load_policy.py excluded_paths.python.dirs
    # -> __pycache__
    #    .venv
    #    ...

Output:
    Scalars are printed as-is on a single line.
    Lists are printed one item per line so callers can `mapfile -t` them
    into bash arrays.

Exit codes:
    0  success
    1  key path not found, or resolves to a dict (which cannot be printed)
    2  usage error, or PyYAML not installed

Runtime dependency:
    PyYAML. In CI, invoke via `uv run --with pyyaml python3 ...` so uv
    fetches PyYAML on demand (cached after first call).
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "error: PyYAML not installed. Run via `uv run --with pyyaml python3 ...`.",
        file=sys.stderr,
    )
    sys.exit(2)

POLICY_PATH = Path(__file__).resolve().parents[1] / "policy.yml"


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "usage: load_policy.py <dotted.key.path>",
            file=sys.stderr,
        )
        sys.exit(2)

    key_path_str = sys.argv[1]

    try:
        with open(POLICY_PATH, "rb") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"error: {POLICY_PATH} not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"error: {POLICY_PATH} is not valid YAML: {exc}", file=sys.stderr)
        sys.exit(1)

    value = data
    walked: list[str] = []
    for key in key_path_str.split("."):
        walked.append(key)
        if not isinstance(value, dict) or key not in value:
            print(
                f"error: key path '{key_path_str}' not found in {POLICY_PATH.name} "
                f"(failed at '{'.'.join(walked)}')",
                file=sys.stderr,
            )
            sys.exit(1)
        value = value[key]

    if isinstance(value, dict):
        print(
            f"error: key path '{key_path_str}' resolves to a dict, not a scalar or list",
            file=sys.stderr,
        )
        sys.exit(1)

    if isinstance(value, list):
        for item in value:
            print(item)
    else:
        print(value)


if __name__ == "__main__":
    main()
