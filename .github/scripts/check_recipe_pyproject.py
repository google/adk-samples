"""
Validates a recipe's pyproject.toml against the repo's metadata rules.

Rules enforced (see .github/workflows/python-validate-recipe.yml):

  - project-name-matches-folder: [project].name must equal the recipe
    folder basename.
  - python-version-floor: [project].requires-python must not permit any
    Python version below 3.11 (per AGENTS.md).
  - description-matches-manifest: if [project].description is set, it must
    equal manifest.description from the recipe's manifest.yaml (after
    .strip(), exact match). Optional; skipped when absent.
  - default-pypi-index: [[tool.uv.index]] must have an entry with
    default=true whose url is public PyPI (https://pypi.org/simple[/]).
    Required so `uv sync` works on Google corp workstations without
    Airlock auth (see the block comment in the root pyproject.toml for
    the full rationale).

Note: no-local-ruff-config (forbid [tool.ruff*] blocks in recipe
pyproject.toml) is enforced by a grep in the workflow itself, not here.

MAINTENANCE NOTE — keep in sync with the align skill. These four rules are
also implemented (as auto-fixes) by
.agents/skills/align-recipe-pyproject/scripts/align_pyproject.py. This script
only READS/validates (stdlib tomllib + pyyaml); that one REWRITES
(comment-preserving tomlkit + ruamel.yaml). They are intentionally separate
but MUST stay semantically in sync — if you change a rule's meaning here,
mirror it there (and vice versa).

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
# Representative Python versions strictly below MIN_PYTHON, used to probe
# whether a requires-python specifier admits anything under the floor. For
# each minor series below MIN_PYTHON we include BOTH the .0 release and a
# very-high micro (`.9999`): the .0 catches plain floors like `>=3.10`, while
# the high micro catches micro-version floors like `>=3.10.5` / `~=3.10.2`
# whose lower bound sits above X.Y.0 — a single `.0` probe would sail past
# them and wrongly report OK. The trailing 2.99 catches Python 2.x.
# Known residual: an exact micro pin (e.g. `==3.10.5`) is not detected, since
# no finite probe set can hit an arbitrary pinned micro; such pins are
# effectively nonexistent in real requires-python declarations.
# NOTE: mirrored in .agents/skills/align-recipe-pyproject/scripts/
# align_pyproject.py — keep in sync.
_PROBE_MICRO = 9999  # synthetic "very high" micro (see note above)
_BELOW_MIN_MINORS = range(MIN_PYTHON[1])  # 3.0, 3.1, ..., 3.(min-1)
BELOW_MIN = (
    [Version(f"{MIN_PYTHON[0]}.{m}") for m in _BELOW_MIN_MINORS]
    + [
        Version(f"{MIN_PYTHON[0]}.{m}.{_PROBE_MICRO}")
        for m in _BELOW_MIN_MINORS
    ]
    + [Version(f"{MIN_PYTHON[0] - 1}.99")]
)

# Acceptable URLs for the required default `[[tool.uv.index]]` entry.
# Both trailing-slash and non-trailing-slash forms are legal.
PYPI_URLS = frozenset(
    {
        "https://pypi.org/simple",
        "https://pypi.org/simple/",
    }
)


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
        # Only surface a real witness version, never a synthetic `.9999`
        # probe (which is the only match for a micro floor like `>=3.10.5`).
        real = [v for v in permits_older if v.micro != _PROBE_MICRO]
        example = f" (e.g. {real[0]})" if real else ""
        emit(
            "FAIL",
            pyproject_path,
            f"[project].requires-python = '{requires_python}' permits Python "
            f"versions below {MIN_PYTHON_STR}{example}; lower bound must be "
            f">= {MIN_PYTHON_STR} (per AGENTS.md).",
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


def check_default_pypi_index(pyproject: dict, pyproject_path: Path) -> None:
    """default-pypi-index: [[tool.uv.index]] with default=true → public PyPI.

    Required so `uv sync` works on Google corp workstations without the
    developer having to authenticate to the corp Airlock proxy that the
    system-wide /etc/uv/uv.toml would otherwise redirect to. Per uv's
    config-file merge rules, project-level indexes are concatenated ahead
    of system-level ones — declaring public PyPI here as default puts it
    first in the merged index list.

    Fail modes:
      - No [[tool.uv.index]] at all: FAIL.
      - Entries exist but none has default=true: FAIL.
      - A default entry exists but its url is not public PyPI: FAIL (the
        recipe author may have a legitimate reason but must justify it;
        default here is strictness so CI protects the repo standard).
    """
    indexes = pyproject.get("tool", {}).get("uv", {}).get("index") or []
    if not indexes:
        emit(
            "FAIL",
            pyproject_path,
            "Missing required `[[tool.uv.index]]` block. Every recipe "
            "must declare public PyPI as its default index so `uv sync` "
            "works without corp Airlock auth. Add:\n"
            "  [[tool.uv.index]]\n"
            '  url = "https://pypi.org/simple/"\n'
            "  default = true",
        )
        return

    # `[tool.uv.index]` (single brackets) parses as a dict; `[[tool.uv.index]]`
    # (double brackets, array-of-tables) parses as a list of dicts. Only the
    # latter is what uv accepts. Guard against the single-bracket mistake so
    # we emit a clean FAIL instead of crashing with AttributeError when we
    # iterate the dict and try `.get()` on a key-string below.
    if not isinstance(indexes, list):
        emit(
            "FAIL",
            pyproject_path,
            "`[tool.uv.index]` is declared as a single table but uv "
            "requires an array of tables. Use double brackets:\n"
            "  [[tool.uv.index]]   ← not [tool.uv.index]\n"
            '  url = "https://pypi.org/simple/"\n'
            "  default = true",
        )
        return

    for entry in indexes:
        if entry.get("default") is True:
            url = entry.get("url")
            if url and str(url).lower() in PYPI_URLS:
                emit(
                    "PASS",
                    pyproject_path,
                    f"[[tool.uv.index]] with default=true points at "
                    f"public PyPI ('{url}').",
                )
                return
            emit(
                "FAIL",
                pyproject_path,
                f"[[tool.uv.index]] has default=true but url='{url}' is "
                f"not public PyPI. Change it to "
                f"'https://pypi.org/simple/' (both trailing-slash and "
                f"non-trailing-slash forms are accepted).",
            )
            return

    emit(
        "FAIL",
        pyproject_path,
        "One or more `[[tool.uv.index]]` entries exist but none is marked "
        "`default = true`. Mark the public-PyPI entry as default, or add "
        "one:\n"
        "  [[tool.uv.index]]\n"
        '  url = "https://pypi.org/simple/"\n'
        "  default = true",
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
    check_default_pypi_index(pyproject, pyproject_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
