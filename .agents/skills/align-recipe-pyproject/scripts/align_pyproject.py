"""
Aligns a Python recipe's pyproject.toml with the repo's standards.

Enforces the same pyproject.toml rules that
.github/workflows/python-validate-recipe.yml checks in CI, plus one critical
additional check.

MAINTENANCE NOTE — keep in sync with the CI validator. Four of these rules
(project-name-matches-folder, python-version-floor, description-matches-
manifest, default-pypi-index) are also implemented independently by
.github/scripts/check_recipe_pyproject.py. That script only READS/validates
(stdlib tomllib + pyyaml, so CI needs no extra deps); this one REWRITES
(comment-preserving tomlkit + ruamel.yaml). They are intentionally separate
but MUST stay semantically in sync — if you change a rule's meaning here,
mirror it there (and vice versa).

  - no-local-ruff-config
        Recipe pyproject.toml must not declare any [tool.ruff*] table. Ruff
        configuration is centralized in the root pyproject.toml.
        Auto-fix: remove the tables.
        (Standalone ruff.toml / .ruff.toml files are also forbidden but are
        outside this skill's scope — see the workflow's Check 7.)
  - python-version-floor
        [project].requires-python must not permit any Python version below
        3.11 (per AGENTS.md "Minimum python version: 3.11"). Recipes that
        require Python 3.12+ are the author's choice and are left alone.
        Auto-fix: raise the lower bound to >=3.11 while preserving every
        upper bound, exclusion, compatible-release ceiling, and pin (only the
        pure lower-bound operators >= and > are dropped). If the rewrite
        would produce a self-contradictory result — because the recipe's own
        ceiling/pin/exclusion excludes 3.11 (e.g. `>=3.10,!=3.11` or
        `==3.10.*`) — refuse to apply and return NEEDS_INPUT.
  - project-name-matches-folder
        [project].name must equal the recipe folder basename.
        Auto-fix: set it.
  - description-matches-manifest
        If [project].description is set, it must equal manifest.description.
        Auto-fix requires --description-source={pyproject,manifest,delete}
        to resolve; refuses to touch description without an explicit choice.
  - build-system-present (report-only)
        [build-system] must have both `requires` and `build-backend`.
        Without it, `uv build` and `pip install .` fail. Not auto-fixed
        because the canonical backend is an editorial choice (hatchling vs
        uv_build).
  - default-pypi-index
        [[tool.uv.index]] must have an entry with default=true pointing at
        public PyPI (https://pypi.org/simple[/]). Required so `uv sync` works
        on Google corp workstations without corp Airlock auth.
        Auto-fix: when no default index is declared, promote an existing
        public-PyPI entry to default=true, or append one if none exists.
        Report-only when a default entry exists but points elsewhere (private
        mirror, TestPyPI) — the divergence may be intentional.

Usage:

    python align_pyproject.py --recipe-dir <RECIPE_DIR> --dry-run
    python align_pyproject.py --recipe-dir <RECIPE_DIR>
    python align_pyproject.py --recipe-dir <RECIPE_DIR> \\
        --description-source={pyproject,manifest,delete}

Output:

    Prints a JSON report to stdout. The caller (a coding agent) is expected
    to render it into markdown for the user. Exit code:

      0  dry-run: always. apply: nothing left unfixed that could have been.
      1  apply mode found at least one issue it could not resolve (typically
         a description-matches-manifest mismatch without
         --description-source, or a bug).
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomlkit
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import Version
from ruamel.yaml import YAML

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
# NOTE: mirrored in .github/scripts/check_recipe_pyproject.py — keep in sync.
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

# Status values for a Check entry.
OK = "ok"  # nothing to do
WOULD_FIX = "would_fix"  # dry-run: a fix is available and would be applied
FIXED = "fixed"  # apply mode: the fix was applied
NEEDS_INPUT = "needs_input"  # dry-run or apply: cannot proceed without human
REPORT_ONLY = "report_only"  # dry-run or apply: informational, no auto-fix
ERROR = "error"  # unexpected failure


@dataclass
class Check:
    """One entry in the report — one rule's outcome for this recipe."""

    id: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    recipe_dir: str
    mode: str  # "dry-run" or "apply"
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # A manifest.yaml write staged by the description-matches-manifest fix
    # (--description-source=pyproject). Held here rather than written inside
    # the check so run() can write pyproject.toml and manifest.yaml together
    # at the end — otherwise a later pyproject write failure would leave
    # manifest.yaml already modified on disk (an inconsistent state the old
    # error message wrongly reported as "nothing changed"). Not serialised.
    pending_manifest: tuple[Path, Any, YAML] | None = None

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def stage_manifest_write(self, path: Path, data: Any, yaml: YAML) -> None:
        self.pending_manifest = (path, data, yaml)

    def to_json(self) -> str:
        return json.dumps(
            {
                "recipe_dir": self.recipe_dir,
                "mode": self.mode,
                "checks": [asdict(c) for c in self.checks],
                "notes": self.notes,
            },
            indent=2,
        )


# ---------- no-local-ruff-config: no [tool.ruff*] tables -------------------


def check_no_local_ruff_config(
    pyproject_path: Path, doc: tomlkit.TOMLDocument, apply: bool
) -> Check:
    """Remove any [tool.ruff*] table from pyproject.toml.

    Assumes doc["tool"] (if present) is a table — run() validates that the
    top-level [project]/[tool]/[build-system] keys are tables inline, before
    dispatching any check.
    """
    tool = doc.get("tool")
    if tool is None or "ruff" not in tool:
        return Check(
            "no-local-ruff-config",
            OK,
            "No [tool.ruff*] block in pyproject.toml.",
        )

    subtables = _enumerate_ruff_subtables(tool["ruff"])
    listing = ", ".join(subtables) if subtables else "[tool.ruff]"
    if apply:
        del tool["ruff"]
        # If [tool] is now empty, remove it too so the file stays tidy.
        if not tool:
            del doc["tool"]
        return Check(
            "no-local-ruff-config",
            FIXED,
            f"Removed [tool.ruff*] tables from pyproject.toml ({listing}).",
            {"removed_tables": subtables},
        )
    return Check(
        "no-local-ruff-config",
        WOULD_FIX,
        f"Would remove [tool.ruff*] tables from pyproject.toml ({listing}).",
        {"removed_tables": subtables},
    )


def _enumerate_ruff_subtables(
    ruff_table: Any, prefix: str = "tool.ruff"
) -> list[str]:
    """Return every table path under [tool.ruff] as dotted strings."""
    names = [f"[{prefix}]"]
    for key, val in ruff_table.items():
        if hasattr(val, "items"):  # nested table
            names.extend(_enumerate_ruff_subtables(val, f"{prefix}.{key}"))
    return names


# ---------- python-version-floor: requires-python floor must be >= 3.11 ----


def _add_missing_python_floor(
    doc: tomlkit.TOMLDocument, project: Any, apply: bool
) -> Check:
    """Handle the case where [project].requires-python is absent."""
    target = f">={MIN_PYTHON_STR}"
    verb = "Added" if apply else "Would add"
    if apply:
        if project is None:
            doc["project"] = tomlkit.table()
            project = doc["project"]
        project["requires-python"] = target
    return Check(
        "python-version-floor",
        FIXED if apply else WOULD_FIX,
        f"{verb} [project].requires-python = '{target}'.",
        {"from": None, "to": target},
    )


def _validate_and_apply_python_floor_rewrite(
    project: Any,
    current: str,
    spec: SpecifierSet,
    permits_older: list[Version],
    apply: bool,
) -> Check:
    """Rewrite requires-python, validating the result is not degenerate."""
    target = _rewrite_requires_python(spec)

    # If the mechanical result no longer admits MIN_PYTHON (e.g.
    # `>=3.10,!=3.11` -> `>=3.11,!=3.11`), the recipe author's exclusion
    # collides with our required floor. Refuse to apply rather than emit a
    # self-contradictory specifier.
    try:
        new_spec = SpecifierSet(target)
    except InvalidSpecifier as e:
        return Check(
            "python-version-floor",
            ERROR,
            f"Mechanical rewrite of '{current}' produced an invalid "
            f"specifier '{target}' ({e}); fix by hand.",
            {"current": current, "attempted_rewrite": target},
        )
    if Version(MIN_PYTHON_STR) not in new_spec:
        return Check(
            "python-version-floor",
            NEEDS_INPUT,
            f"[project].requires-python = '{current}' permits Python "
            f"versions below {MIN_PYTHON_STR}, but a mechanical rewrite "
            f"would produce '{target}' which excludes {MIN_PYTHON_STR} "
            f"itself (the recipe's own upper bound, pin, or exclusion "
            f"contradicts the required floor). Fix by hand.",
            {"current": current, "attempted_rewrite": target},
        )

    # Prefer a real witness version for the message; fall back to a generic
    # phrase rather than surfacing a synthetic `.9999` probe (which would only
    # appear for a micro-version floor like `>=3.10.5`).
    real = [v for v in permits_older if v.micro != _PROBE_MICRO]
    reason = (
        f"permits Python {real[0]}"
        if real
        else f"permits Python below {MIN_PYTHON_STR}"
    )
    verb = "Rewrote" if apply else "Would rewrite"
    if apply:
        project["requires-python"] = target
    return Check(
        "python-version-floor",
        FIXED if apply else WOULD_FIX,
        f"{verb} [project].requires-python: '{current}' -> '{target}' "
        f"({reason}).",
        {"from": current, "to": target, "reason": reason},
    )


def check_python_version_floor(
    pyproject_path: Path, doc: tomlkit.TOMLDocument, apply: bool
) -> Check:
    """Ensure [project].requires-python does not permit versions < MIN_PYTHON.

    Interpretation A (per AGENTS.md discussion): the repo standard is a FLOOR.
    A recipe that requires Python 3.12+ is fine and is left untouched.
    A recipe that permits Python < 3.11 is rewritten so its lower bound
    becomes >=3.11 (upper bounds and exclusions preserved). If the rewrite
    would produce a self-contradictory result (e.g. `>=3.10,!=3.11`), the
    script refuses to apply and returns NEEDS_INPUT.
    """
    project = doc.get("project")
    current = None if project is None else project.get("requires-python")

    # Treat an empty / whitespace-only string the same as a missing field:
    # both mean "no floor declared". Routing it here avoids the generic
    # rewrite path emitting a misleading "permits Python 3.0" reason for a
    # value that never mentioned 3.0.
    if current is None or (isinstance(current, str) and not current.strip()):
        return _add_missing_python_floor(doc, project, apply)

    try:
        spec = SpecifierSet(current)
    except InvalidSpecifier as e:
        return Check(
            "python-version-floor",
            ERROR,
            f"[project].requires-python = '{current}' is not a valid PEP 440 "
            f"specifier ({e}); fix by hand.",
            {"current": current},
        )

    permits_older = [v for v in BELOW_MIN if v in spec]

    # Interpretation A: only rewrite when the spec permits versions below
    # MIN_PYTHON. Higher lower bounds (e.g. `>=3.12`) are the recipe author's
    # deliberate choice and are left alone.
    if not permits_older:
        return Check(
            "python-version-floor",
            OK,
            f"[project].requires-python lower bound is >= {MIN_PYTHON_STR} "
            f"('{current}').",
        )

    return _validate_and_apply_python_floor_rewrite(
        project, current, spec, permits_older, apply
    )


def _rewrite_requires_python(spec: SpecifierSet) -> str:
    """Return a new requires-python string with the lower bound at MIN_PYTHON.

    Strategy: drop only the *pure* lower-bound operators (`>=`, `>`) and
    prepend `>=MIN_PYTHON`. Every other operator is kept verbatim so its
    constraint survives the rewrite:

      - Pure upper bounds / exclusions (`<`, `<=`, `!=`) are preserved as-is.
      - Compound and pin operators (`~=`, `==`, `===`) are *also kept*,
        because each encodes an upper bound or a pin — not just a floor.
        Dropping them (as an earlier version did) silently discarded the
        ceiling: `~=3.10` (== `>=3.10,<4`) became an unbounded `>=3.11`, and
        `==3.10.*` became `>=3.11`. Keeping them lets the caller's
        contradiction guard (_validate_and_apply_python_floor_rewrite) act on
        the real result — a compatible-release ceiling that still admits
        MIN_PYTHON is preserved (`~=3.10` -> `>=3.11,~=3.10` == `>=3.11,<4`),
        while a pin that excludes MIN_PYTHON (`==3.10.*` -> `>=3.11,==3.10.*`)
        is correctly rejected as NEEDS_INPUT instead of silently broadened.
    """
    pure_lower_bound_ops = {">=", ">"}
    kept = [
        str(s)
        for s in spec
        if Specifier(str(s)).operator not in pure_lower_bound_ops
    ]
    return ",".join([f">={MIN_PYTHON_STR}", *kept])


# ---------- project-name-matches-folder ------------------------------------


def check_project_name_matches_folder(
    recipe_dir: Path,
    pyproject_path: Path,
    doc: tomlkit.TOMLDocument,
    apply: bool,
) -> Check:
    folder = recipe_dir.name
    project = doc.get("project")
    current = None if project is None else project.get("name")

    if current == folder:
        return Check(
            "project-name-matches-folder",
            OK,
            f"[project].name matches folder name: '{folder}'.",
        )

    if apply:
        if project is None:
            doc["project"] = tomlkit.table()
            project = doc["project"]
        project["name"] = folder
        if current is None:
            msg = f"Added [project].name = '{folder}'."
        else:
            msg = f"Rewrote [project].name: '{current}' -> '{folder}'."
        return Check(
            "project-name-matches-folder",
            FIXED,
            msg,
            {"from": current, "to": folder},
        )

    if current is None:
        msg = f"Would add [project].name = '{folder}' (recipe folder basename)."
    else:
        msg = (
            f"Would rewrite [project].name: '{current}' -> '{folder}' "
            f"(recipe folder basename)."
        )
    return Check(
        "project-name-matches-folder",
        WOULD_FIX,
        msg,
        {"from": current, "to": folder},
    )


# ---------- description-matches-manifest -----------------------------------


def _load_manifest(manifest_path: Path) -> tuple[Any, YAML] | Check:
    """Load manifest.yaml with ruamel.yaml (comment-preserving).

    Returns (data, yaml_instance) on success, or an ERROR Check on failure.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    try:
        with open(manifest_path) as f:
            return (yaml.load(f) or {}, yaml)
    except Exception as e:
        return Check(
            "description-matches-manifest",
            ERROR,
            f"Failed to parse manifest.yaml: {e}",
        )


def _apply_desc_source_pyproject(
    report: Report,
    manifest_path: Path,
    manifest: Any,
    yaml: YAML,
    py_desc: str,
    details: dict,
    apply: bool,
) -> Check:
    """Overwrite manifest.description with the pyproject value.

    In apply mode the manifest is mutated in memory and its write is *staged*
    on the report — run() flushes it only after pyproject.toml is safely on
    disk, so the two files are updated together rather than manifest-first.
    """
    verb = "Overwrote" if apply else "Would overwrite"
    if apply:
        manifest["description"] = py_desc
        report.stage_manifest_write(manifest_path, manifest, yaml)
    return Check(
        "description-matches-manifest",
        FIXED if apply else WOULD_FIX,
        f"{verb} manifest.description with pyproject value: {py_desc!r}.",
        details,
    )


def _apply_desc_source_manifest(
    project: Any, mf_desc: str, details: dict, apply: bool
) -> Check:
    """Overwrite [project].description with the manifest value."""
    verb = "Overwrote" if apply else "Would overwrite"
    if apply:
        project["description"] = mf_desc
    return Check(
        "description-matches-manifest",
        FIXED if apply else WOULD_FIX,
        f"{verb} [project].description with manifest value: {mf_desc!r}.",
        details,
    )


def _apply_desc_source_delete(
    project: Any, details: dict, apply: bool
) -> Check:
    """Delete [project].description; manifest becomes single source of truth."""
    verb = "Deleted" if apply else "Would delete"
    tail = (
        "manifest.description remains the single source of truth."
        if apply
        else "manifest.description would remain the single source of truth."
    )
    if apply:
        del project["description"]
    return Check(
        "description-matches-manifest",
        FIXED if apply else WOULD_FIX,
        f"{verb} [project].description ({tail}).",
        details,
    )


def _load_manifest_description(
    manifest_path: Path, py_desc_raw: str
) -> tuple[Any, Any, YAML] | Check:
    """Load manifest.yaml and return (mf_desc_raw, manifest, yaml).

    Returns an ERROR Check instead if the manifest is missing, unparseable,
    or its `description` is a non-string value. Extracted from
    check_description_matches_manifest to keep that function's return count
    within the repo's ruff limit.
    """
    if not manifest_path.is_file():
        return Check(
            "description-matches-manifest",
            ERROR,
            "[project].description is set but manifest.yaml is missing; "
            "cannot verify consistency.",
            {"pyproject_description": py_desc_raw},
        )

    loaded = _load_manifest(manifest_path)
    if isinstance(loaded, Check):
        loaded.details = {"pyproject_description": py_desc_raw}
        return loaded
    manifest, yaml = loaded

    mf_desc_raw = manifest.get("description")
    if mf_desc_raw is not None and not isinstance(mf_desc_raw, str):
        return Check(
            "description-matches-manifest",
            ERROR,
            f"manifest.description must be a string but is a "
            f"{type(mf_desc_raw).__name__}; fix manifest.yaml by hand.",
            {"manifest_description": str(mf_desc_raw)},
        )
    return (mf_desc_raw, manifest, yaml)


def check_description_matches_manifest(
    recipe_dir: Path,
    pyproject_path: Path,
    doc: tomlkit.TOMLDocument,
    description_source: str | None,
    apply: bool,
    report: Report,
) -> Check:
    project = doc.get("project")
    py_desc_raw = None if project is None else project.get("description")
    manifest_path = recipe_dir / "manifest.yaml"

    if py_desc_raw is None:
        return Check(
            "description-matches-manifest",
            OK,
            "[project].description is not set — check skipped "
            "(field is optional).",
        )

    # tomlkit's String is a str subclass, so a well-formed description passes
    # this. A description written as a non-string (array, integer, inline
    # table — all syntactically valid TOML) would otherwise crash on .strip()
    # below; catch it here with an actionable message instead.
    if not isinstance(py_desc_raw, str):
        return Check(
            "description-matches-manifest",
            ERROR,
            f"[project].description must be a string but is a "
            f"{type(py_desc_raw).__name__}; fix it by hand before re-running.",
            {"pyproject_description": str(py_desc_raw)},
        )

    resolved = _load_manifest_description(manifest_path, py_desc_raw)
    if isinstance(resolved, Check):
        return resolved
    mf_desc_raw, manifest, yaml = resolved

    py_desc = py_desc_raw.strip()
    mf_desc = (mf_desc_raw or "").strip()

    if py_desc == mf_desc:
        return Check(
            "description-matches-manifest",
            OK,
            "[project].description matches manifest.description.",
        )

    details = {
        "pyproject_description": py_desc,
        "manifest_description": mf_desc,
    }

    if description_source is None:
        return Check(
            "description-matches-manifest",
            NEEDS_INPUT,
            "[project].description does not match manifest.description. "
            "Re-run with --description-source={pyproject,manifest,delete} to "
            "resolve.",
            details,
        )

    # argparse restricts --description-source to these three choices, so no
    # safety branch for unknown values is needed here.
    dispatch = {
        "pyproject": lambda: _apply_desc_source_pyproject(
            report, manifest_path, manifest, yaml, py_desc, details, apply
        ),
        "manifest": lambda: _apply_desc_source_manifest(
            project, mf_desc, details, apply
        ),
        "delete": lambda: _apply_desc_source_delete(project, details, apply),
    }
    return dispatch[description_source]()


# ---------- build-system-present (report-only) -----------------------------


def check_build_system(doc: tomlkit.TOMLDocument) -> Check:
    """Report-only. Not auto-fixed because backend choice is editorial."""
    bs = doc.get("build-system")
    if bs and bs.get("requires") and bs.get("build-backend"):
        return Check(
            "build-system-present",
            OK,
            "[build-system] present with both `requires` and `build-backend`.",
        )
    missing = []
    if bs is None:
        missing.append("the whole [build-system] table")
    else:
        if not bs.get("requires"):
            missing.append("build-system.requires")
        if not bs.get("build-backend"):
            missing.append("build-system.build-backend")
    return Check(
        "build-system-present",
        REPORT_ONLY,
        f"Missing: {', '.join(missing)}. Without a valid [build-system], "
        f"`uv build` and `pip install .` fail. Add either the hatchling or "
        f"uv_build template (this skill does not auto-fix — backend choice "
        f'is editorial). Hatchling example: `requires = ["hatchling"]`, '
        f'`build-backend = "hatchling.build"`.',
        {"missing": missing},
    )


# ---------- default-pypi-index ---------------------------------------------
#
# On Google corp workstations, /etc/uv/uv.toml redirects the default index to
# a corp Airlock proxy that requires auth. We require every recipe to declare
# public PyPI as its default index so `uv sync` works without the developer
# having to authenticate to Airlock — see the block comment in the root
# pyproject.toml for full context.

_PYPI_URLS = frozenset(
    {
        "https://pypi.org/simple",
        "https://pypi.org/simple/",
    }
)


def _find_default_index(doc: tomlkit.TOMLDocument) -> tuple[Any, str | None]:
    """Return (entry, url) for the first [[tool.uv.index]] with default=true.

    Returns (None, None) if there is no default entry, or if `tool.uv.index`
    was declared as a single table instead of an array of tables (in that
    case the caller should detect the syntax mistake via an isinstance
    check first and emit an ERROR — this function refuses to iterate a
    non-list to avoid `AttributeError` on key-string `.get()` calls).

    Only the FIRST default entry is inspected — having two `default = true`
    entries is undefined at the tie-break level per uv docs.
    """
    tool = doc.get("tool")
    if tool is None:
        return (None, None)
    uv = tool.get("uv")
    if uv is None:
        return (None, None)
    index = uv.get("index")
    if not index or not isinstance(index, list):
        return (None, None)
    for entry in index:
        if entry.get("default") is True:
            return (entry, entry.get("url"))
    return (None, None)


def _url_is_public_pypi(url: str | None) -> bool:
    """True for `https://pypi.org/simple` and its trailing-slash variant."""
    if not url:
        return False
    return url.lower() in _PYPI_URLS


def _find_pypi_entry(doc: tomlkit.TOMLDocument) -> Any:
    """Return the first [[tool.uv.index]] entry that points at public PyPI.

    Ignores the `default` flag — used to promote an existing (non-default)
    PyPI entry rather than appending a redundant second PyPI block. Returns
    None if there is no such entry (or the index table is malformed).
    """
    tool = doc.get("tool")
    uv = tool.get("uv") if tool is not None else None
    index = uv.get("index") if uv is not None else None
    if not index or not isinstance(index, list):
        return None
    for entry in index:
        if _url_is_public_pypi(entry.get("url")):
            return entry
    return None


def _append_default_pypi_index(doc: tomlkit.TOMLDocument) -> None:
    """Append a `[[tool.uv.index]]` block declaring public PyPI as default.

    Placed under `[tool.uv]` (created if absent). Existing non-default
    `[[tool.uv.index]]` entries are preserved and remain higher-priority
    (per uv's index ordering).

    No leading comment is emitted — tomlkit attaches comments passed to a
    table INSIDE the table (after its header), which reads awkwardly.
    Recipes hand-edited by the scaffold-python-recipe template or the
    prepare-python-recipe skill get a single-line comment (`# Use public
    PyPI as the default index for this recipe.`); recipes fixed later by
    this auto-fix path just get the bare block. Both are equivalent
    functionally.
    """
    if "tool" not in doc:
        doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    if "uv" not in tool:
        tool["uv"] = tomlkit.table()
    uv = tool["uv"]
    if "index" not in uv:
        uv["index"] = tomlkit.aot()

    entry = tomlkit.table()
    entry["url"] = "https://pypi.org/simple/"
    entry["default"] = True
    uv["index"].append(entry)


def check_default_pypi_index(doc: tomlkit.TOMLDocument, apply: bool) -> Check:
    """Ensure a [[tool.uv.index]] with default=true points at public PyPI.

    Auto-fix: appends the block if no default index is declared at all.
    Report-only: if a default IS declared but points somewhere other than
    public PyPI (custom private index, TestPyPI, mirror). The skill does
    NOT rewrite a user's intentional non-PyPI default.
    Error: if `[tool.uv.index]` was declared as a single table instead of
    an array of tables (bad TOML syntax) — refuse to auto-append (would
    create a syntax conflict) and tell the user to fix it by hand.
    """
    # Guard against `[tool.uv.index]` (single-bracket) syntax mistake:
    # tomlkit parses it as a Table (dict-like), not an AoT (list-like).
    # Iterating a Table yields its key names (strings), and `.get()` on a
    # string is AttributeError. Detect this up-front and emit a clean ERROR
    # so the user sees the fix instead of a stack trace.
    tool = doc.get("tool")
    uv = tool.get("uv") if tool is not None else None
    raw_index = uv.get("index") if uv is not None else None
    if raw_index is not None and not isinstance(raw_index, list):
        return Check(
            "default-pypi-index",
            ERROR,
            "`[tool.uv.index]` is declared as a single table but uv requires "
            "an array of tables. Use double brackets: `[[tool.uv.index]]` "
            '(with `url = "https://pypi.org/simple/"` and `default = true`). '
            "Fix by hand — auto-rewriting would risk producing a TOML syntax "
            "conflict.",
        )

    entry, url = _find_default_index(doc)

    if entry is not None and _url_is_public_pypi(url):
        return Check(
            "default-pypi-index",
            OK,
            f"[[tool.uv.index]] with default=true points at public PyPI "
            f"('{url}').",
        )

    if entry is not None:
        # A default index exists but it isn't PyPI. Don't clobber it.
        return Check(
            "default-pypi-index",
            REPORT_ONLY,
            f"[[tool.uv.index]] has default=true but url='{url}' is not "
            f"public PyPI ('https://pypi.org/simple/'). If this is a "
            f"deliberate mirror or private index, ignore. Otherwise, change "
            f"the url to 'https://pypi.org/simple/' (this skill does not "
            f"auto-rewrite non-PyPI defaults).",
            {"current_url": url},
        )

    # No default index declared at all. If a public-PyPI entry already exists
    # (just not marked default), promote it rather than appending a redundant
    # second PyPI block. Only append a fresh block when there's no PyPI entry.
    existing = _find_pypi_entry(doc)
    if existing is not None:
        verb = "Set" if apply else "Would set"
        if apply:
            existing["default"] = True
        return Check(
            "default-pypi-index",
            FIXED if apply else WOULD_FIX,
            f"{verb} `default = true` on the existing public-PyPI "
            f"`[[tool.uv.index]]` entry (url='{existing.get('url')}') rather "
            f"than appending a duplicate. Required so `uv sync` works on "
            f"Google corp workstations without Airlock auth.",
            {"promoted_url": existing.get("url")},
        )

    verb = "Added" if apply else "Would add"
    if apply:
        _append_default_pypi_index(doc)
    return Check(
        "default-pypi-index",
        FIXED if apply else WOULD_FIX,
        f'{verb} `[[tool.uv.index]]` with `url = "https://pypi.org/simple/"` '
        f"and `default = true`. Required so `uv sync` works on Google corp "
        f"workstations without Airlock auth.",
        {"added_url": "https://pypi.org/simple/"},
    )


# ---------- Orchestration -------------------------------------------------


def _run_check(check_id: str, fn: Any) -> Check:
    """Run one check, isolating its exceptions into a scoped ERROR Check.

    Without this, an exception in any single check (e.g. a malformed field
    that slips past the up-front schema guard) would propagate out of run()
    and collapse the whole report to one generic `unhandled-exception`,
    discarding every other rule's otherwise-valid result. This skill exists
    to fix up messy recipes, so one bad field must degrade gracefully — the
    offending rule reports an error and the rest still run.
    """
    try:
        return fn()
    except Exception as e:  # deliberately broad — scoped into a Check below
        return Check(
            check_id,
            ERROR,
            f"Check '{check_id}' could not run: {type(e).__name__}: {e}. "
            f"This usually means a malformed field in pyproject.toml or "
            f"manifest.yaml — fix it by hand (or report a bug if the field "
            f"looks valid).",
        )


def _persist_changes(
    report: Report, pyproject_path: Path, doc: tomlkit.TOMLDocument
) -> None:
    """Write pyproject.toml, then any staged manifest.yaml, in that order.

    pyproject.toml goes first; the deferred manifest write (staged by the
    --description-source=pyproject fix) is flushed only after pyproject.toml
    is safely on disk, so the two files move together instead of
    manifest-first. On failure, an accurately-worded ERROR Check is appended
    naming exactly what did and did not reach disk.
    """
    try:
        with open(pyproject_path, "w") as f:
            f.write(tomlkit.dumps(doc))
    except OSError as e:
        # Nothing has touched disk yet — the manifest write was deferred.
        report.add(
            Check(
                "persist-pyproject",
                ERROR,
                f"Failed to write {pyproject_path}: {e}. No files were "
                f"modified (any staged manifest.yaml update was skipped too); "
                f"the recipe is unchanged on disk.",
            )
        )
        return

    if report.pending_manifest is not None:
        manifest_path, manifest_data, manifest_yaml = report.pending_manifest
        try:
            with open(manifest_path, "w") as f:
                manifest_yaml.dump(manifest_data, f)
        except OSError as e:
            report.add(
                Check(
                    "persist-manifest",
                    ERROR,
                    f"pyproject.toml was updated, but writing {manifest_path} "
                    f"failed: {e}. The two files may now be inconsistent — "
                    f"re-run once the write problem is resolved to reconcile "
                    f"them.",
                )
            )
            return

    report.note(
        "Run `uv sync` in the recipe to pick up any pyproject.toml "
        "dependency changes."
    )


def run(
    recipe_dir: Path,
    dry_run: bool,
    description_source: str | None,
) -> Report:
    mode = "dry-run" if dry_run else "apply"
    report = Report(recipe_dir=str(recipe_dir), mode=mode)

    # Guard rail: refuse to operate on a git repository root. The repo root
    # has its own pyproject.toml (the root ruff config, the corp-Airlock index
    # block, etc.) that this skill must never mutate. A recipe directory never
    # contains a `.git` entry; a temp/fixture directory used for testing
    # doesn't either — so this rejects only the genuinely dangerous case
    # without the brittleness of hard-coding `core/python/` / `contrib/`
    # path prefixes (which would also break fixture-based testing).
    if (recipe_dir / ".git").exists():
        report.add(
            Check(
                "recipe-directory",
                ERROR,
                f"{recipe_dir} looks like a git repository root (it contains "
                f"a .git entry), not a recipe. Point --recipe-dir at a recipe "
                f"root under core/python/<name>/ or contrib/<name>/.",
            )
        )
        return report

    pyproject_path = recipe_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        report.add(
            Check(
                "recipe-directory",
                ERROR,
                f"No pyproject.toml at {pyproject_path}. Point --recipe-dir "
                f"at the root of a Python recipe.",
            )
        )
        return report

    try:
        with open(pyproject_path) as f:
            doc = tomlkit.parse(f.read())
    except (OSError, tomlkit.exceptions.TOMLKitError) as e:
        report.add(
            Check(
                "recipe-directory",
                ERROR,
                f"Failed to read/parse {pyproject_path}: {e}",
            )
        )
        return report

    # Validate that top-level keys we touch are actually tables — a recipe
    # that writes `project = "foo"` or similar would otherwise crash our
    # `.get()` calls with AttributeError. Only ONE guard, up-front; every
    # check function below may assume these are tables (or absent).
    for key in ("project", "tool", "build-system"):
        val = doc.get(key)
        if val is not None and not hasattr(val, "get"):
            report.add(
                Check(
                    "pyproject-schema",
                    ERROR,
                    f"[{key}] in pyproject.toml is not a table (got "
                    f"{type(val).__name__}); fix by hand before re-running.",
                )
            )
            return report

    apply = not dry_run
    # Each check is isolated: a crash in one becomes a scoped ERROR Check for
    # that rule only, so the other five still produce results.
    report.add(
        _run_check(
            "no-local-ruff-config",
            lambda: check_no_local_ruff_config(pyproject_path, doc, apply),
        )
    )
    report.add(
        _run_check(
            "python-version-floor",
            lambda: check_python_version_floor(pyproject_path, doc, apply),
        )
    )
    report.add(
        _run_check(
            "project-name-matches-folder",
            lambda: check_project_name_matches_folder(
                recipe_dir, pyproject_path, doc, apply
            ),
        )
    )
    report.add(
        _run_check(
            "description-matches-manifest",
            lambda: check_description_matches_manifest(
                recipe_dir,
                pyproject_path,
                doc,
                description_source,
                apply,
                report,
            ),
        )
    )
    report.add(
        _run_check("build-system-present", lambda: check_build_system(doc))
    )
    report.add(
        _run_check(
            "default-pypi-index",
            lambda: check_default_pypi_index(doc, apply),
        )
    )

    # Persist edits — only in apply mode, and only if at least one auto-fix
    # actually changed something.
    if apply and any(c.status == FIXED for c in report.checks):
        _persist_changes(report, pyproject_path, doc)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align a recipe's pyproject.toml with repo standards."
    )
    parser.add_argument(
        "--recipe-dir",
        required=True,
        type=Path,
        help="Path to the recipe root (e.g. core/python/foo).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without modifying any files.",
    )
    parser.add_argument(
        "--description-source",
        choices=["pyproject", "manifest", "delete"],
        default=None,
        help=(
            "How to resolve a B2-desc mismatch. 'pyproject' overwrites "
            "manifest with pyproject value; 'manifest' overwrites pyproject "
            "with manifest value; 'delete' removes [project].description so "
            "manifest becomes the single source of truth."
        ),
    )
    args = parser.parse_args()

    if not args.recipe_dir.is_dir():
        print(
            f"Error: --recipe-dir {args.recipe_dir} is not a directory.",
            file=sys.stderr,
        )
        return 2

    # Top-level safety net: any unforeseen exception is surfaced as one
    # ERROR check with a stable JSON envelope, so the calling agent can
    # render it just like a normal check outcome instead of parsing a stack
    # trace from stderr.
    try:
        report = run(
            recipe_dir=args.recipe_dir,
            dry_run=args.dry_run,
            description_source=args.description_source,
        )
    except Exception as e:  # final safety net for the CLI
        report = Report(
            recipe_dir=str(args.recipe_dir),
            mode="dry-run" if args.dry_run else "apply",
        )
        report.add(
            Check(
                "unhandled-exception",
                ERROR,
                f"Unhandled {type(e).__name__}: {e}. This is a bug in the "
                f"align-recipe-pyproject skill; please report it.",
            )
        )
    print(report.to_json())

    if not args.dry_run:
        # Apply mode: exit nonzero if anything remains unfixed.
        unresolved = [
            c
            for c in report.checks
            if c.status in {NEEDS_INPUT, ERROR, REPORT_ONLY, WOULD_FIX}
        ]
        return 1 if unresolved else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
