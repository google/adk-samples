"""
Validates a recipe's pyproject.toml against the repo's [project] metadata rules.

Rules enforced (see .github/workflows/validate-python-recipe.yml):

  - B2-name: [project].name must exist and equal the recipe folder basename.
  - B1:      [project].requires-python must declare a lower bound >= 3.11
             (per AGENTS.md: "Minimum python version: 3.11").
  - B2-desc: If [project].description is set, it must equal manifest.description
             from the same recipe's manifest.yaml (after .strip(), exact match).
             The field is optional; if absent, this check is skipped.

Note: rule A1 (forbid [tool.ruff*] blocks in recipe pyproject.toml) is enforced
by a grep in the workflow itself, not here.

Usage: python check_recipe_pyproject.py <recipe-dir>

Output format (one record per line, for the shell caller to parse):
  PASS::<path>::<message>
  FAIL::<path>::<message>

Exits 0 always. The workflow decides pass/fail from the emitted records so that
a missing pyproject.toml (which is caught by a separate required-files check)
does not cause a redundant error here.
"""

import sys
from pathlib import Path

import tomllib
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

MIN_PYTHON = (3, 11)
MIN_PYTHON_STR = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
# Any Python release that predates MIN_PYTHON. If the recipe's specifier
# permits any of these, its lower bound is too low.
BELOW_MIN = [
    Version(f"{MIN_PYTHON[0]}.{minor}")
    for minor in range(MIN_PYTHON[1])  # 3.0, 3.1, ..., 3.(min-1)
] + [Version(f"{MIN_PYTHON[0] - 1}.99")]  # e.g. 2.99, catches Python 2.x


def emit(kind: str, file: Path, msg: str) -> None:
    """Print one PASS/FAIL record. Newlines are collapsed so the shell can split
    stdout line-by-line without worrying about multi-line messages."""
    msg = msg.replace("\n", " ")
    print(f"{kind}::{file}::{msg}")


def check_name(project: dict, pyproject_path: Path, folder: str) -> None:
    """B2-name: [project].name must equal the recipe folder basename."""
    name = project.get("name")
    if not name:
        emit(
            "FAIL",
            pyproject_path,
            f"[project].name is missing; it must equal the recipe folder "
            f"name '{folder}'.",
        )
    elif name != folder:
        emit(
            "FAIL",
            pyproject_path,
            f"[project].name = '{name}' does not match the recipe folder "
            f"name '{folder}'.",
        )
    else:
        emit(
            "PASS",
            pyproject_path,
            f"[project].name matches folder name: '{name}'.",
        )


def check_requires_python(project: dict, pyproject_path: Path) -> None:
    """B1: [project].requires-python must not permit Python < MIN_PYTHON.

    Interpretation A: the repo standard is a FLOOR. A recipe that requires
    Python 3.12+ (e.g. `>=3.12`) is fine — the recipe author has legitimately
    chosen a stricter minimum. A recipe that PERMITS versions below 3.11
    (e.g. `>=3.10`, `~=3.10`, `!=3.11`, `<=3.12`, unpinned) is a violation.

    Uses packaging.specifiers.SpecifierSet (the PEP 440 reference
    implementation) so every legal operator (>=, >, ~=, ==, !=, <, <=, and
    combinations) is handled correctly.
    """
    requires_python = project.get("requires-python")
    if not requires_python:
        emit(
            "FAIL",
            pyproject_path,
            f"[project].requires-python is missing; it must declare a "
            f"lower bound of >= {MIN_PYTHON_STR} (per AGENTS.md).",
        )
        return

    try:
        spec = SpecifierSet(requires_python)
    except InvalidSpecifier as e:
        emit(
            "FAIL",
            pyproject_path,
            f"[project].requires-python = '{requires_python}' is not a valid "
            f"PEP 440 version specifier ({e}).",
        )
        return

    # If any pre-MIN_PYTHON version satisfies the spec, the lower bound is
    # too loose (e.g. '>=3.10', '~=3.10', '!=3.11', '<=3.12', unpinned).
    permits_older = [v for v in BELOW_MIN if v in spec]
    if permits_older:
        emit(
            "FAIL",
            pyproject_path,
            f"[project].requires-python = '{requires_python}' permits Python "
            f"versions below {MIN_PYTHON_STR} (e.g. {permits_older[0]}); "
            f"lower bound must be >= {MIN_PYTHON_STR} (per AGENTS.md).",
        )
        return

    emit(
        "PASS",
        pyproject_path,
        f"[project].requires-python lower bound is >= {MIN_PYTHON_STR} "
        f"('{requires_python}').",
    )


def check_description(
    project: dict, pyproject_path: Path, manifest_path: Path
) -> None:
    """B2-desc: if [project].description is set, must equal manifest.description.

    The field is optional; skipped entirely when absent.
    """
    description = project.get("description")
    if description is None:
        return

    if not manifest_path.is_file():
        emit(
            "FAIL",
            pyproject_path,
            "[project].description is set but manifest.yaml is missing; "
            "cannot verify consistency.",
        )
        return

    # Imported lazily so a recipe with no [project].description does not
    # require pyyaml at all.
    try:
        import yaml
    except ImportError:
        emit(
            "FAIL",
            manifest_path,
            "pyyaml is required to verify [project].description against "
            "manifest.description but is not installed. Ensure the workflow "
            "invokes this script with pyyaml available (e.g. "
            "`uv run --with pyyaml`).",
        )
        return

    try:
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        emit("FAIL", manifest_path, f"Failed to parse manifest.yaml: {e}")
        return

    py_desc = description.strip()
    mf_desc = (manifest.get("description") or "").strip()
    if py_desc != mf_desc:
        emit(
            "FAIL",
            pyproject_path,
            f"[project].description does not match manifest.description. "
            f"pyproject: {py_desc!r} | manifest: {mf_desc!r}. Update "
            f"whichever is out of date so both match (or drop "
            f"[project].description from pyproject.toml, since it is "
            f"optional).",
        )
    else:
        emit(
            "PASS",
            pyproject_path,
            "[project].description matches manifest.description.",
        )


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <recipe-dir>", file=sys.stderr)
        return 2

    recipe_dir = Path(sys.argv[1])
    pyproject_path = recipe_dir / "pyproject.toml"
    manifest_path = recipe_dir / "manifest.yaml"

    if not pyproject_path.is_file():
        # A separate required-files check in the workflow reports this; stay
        # silent here so we don't double up on the same failure.
        return 0

    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        emit("FAIL", pyproject_path, f"Failed to parse pyproject.toml: {e}")
        return 0

    project = pyproject.get("project") or {}
    check_name(project, pyproject_path, recipe_dir.name)
    check_requires_python(project, pyproject_path)
    check_description(project, pyproject_path, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
