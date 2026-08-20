"""
Make a Python recipe deployable: generate the serving files and configure them.

A deployable recipe is one that can be packaged into a container and run as a
service. This script writes the files that requires and patches the recipe's
metadata to match. It does NOT build an image, deploy anything, or provision
infrastructure.

The standard it implements lives in `.github/policy.yml` under `deployability:`
(minimum google-adk version, required dependencies, required files, the legacy
app_utils file list). Change the standard there, not here.

CHECKS, in the order they run
-----------------------------
Gates run first and can stop the whole script — there is no point generating
files for a recipe that cannot host them.

  - agent-package
        Locate the recipe's agent package (the directory holding agent.py).
        Everything downstream is written relative to it. ERROR if not found
        or if the recipe has no `root_agent`.

  - adk-version-floor  [GATE]
        [project].dependencies must carry a google-adk requirement whose
        specifier ACCEPTS `deployability.min_google_adk`.

        The floor is not arbitrary: the serving stack needs
        `a2a-sdk[http-server]>=1.0`, and google-adk's `a2a` extra caps a2a-sdk
        at <0.4 through 2.4.0 — only 2.5.0 widened it to <2.

        A recipe whose specifier excludes the floor (a `<2.0.0` ceiling, an
        `==1.31.0` pin) returns NEEDS_INPUT and the script STOPS. Crossing an
        ADK major is a code migration, not a metadata rewrite: widening the
        ceiling and re-locking would yield a recipe importing a major it was
        never ported to, a failure invisible here, in the lockfile, and in CI,
        surfacing only when someone runs the recipe. Same reasoning as
        align-recipe-pyproject's "DELIBERATELY NOT IMPLEMENTED HERE".

  - legacy-app-utils  [GATE]
        `<pkg>/app_utils/` must not contain the old ASP-era generation
        (telemetry.py, typing.py, deploy.py, memory_config.py). Filenames do
        not collide with the new set, but the two wire telemetry and services
        differently: the old fast_api_app.py imports the old modules and
        defines its own /feedback route, so overwriting the entrypoint either
        orphans them or double-wires telemetry. NEEDS_INPUT, script STOPS.

  - backing-infra  [GATE, advisory]
        Detects an existing infra/ or terraform/ directory, or imports of
        backing services this script cannot provision (Cloud SQL, Pub/Sub,
        a Vertex AI Search datastore, a vector index).

        This does not stop generation — it decides the OUTCOME. A recipe that
        needs infra can still be containerized; it just is not one-click
        deployable, so `manifest.deployable` is left alone. See "two outcomes"
        below.

  - serving-files
        The five files from `deployability.required_files`, copied from
        resources/templates/ with `__AGENT_PACKAGE__` / `__PROJECT_NAME__`
        substituted. An existing file is never overwritten without --overwrite.

  - dockerignore
        A real .dockerignore. Not cosmetic: every recipe in this repo has an
        in-tree .venv/, and `.env` must never be baked into an image.

  - required-dependencies
        Adds any missing entry from `deployability.required_dependencies`.

        For a package the recipe ALREADY declares, the two halves of the
        requirement are treated differently:
          * VERSION BOUND — left exactly as written, and reported. A recipe
            may carry a deliberate tighter pin, and silently widening it is
            how you break something that was working.
          * EXTRAS — merged in (`google-adk` -> `google-adk[gcp,otel-gcp]`,
            same bound). The generated serving code imports what those extras
            install, so omitting them ships a recipe that cannot start. An
            extra only pulls in more of that package's own optional
            dependencies; it cannot move the recipe onto a different version.

  - hatch-wheel-packages
        [tool.hatch.build.targets.wheel] must list the agent package, or
        `uv sync` fails.

  - app-object
        agent.py must define `app = App(root_agent=root_agent, name="<pkg>")`
        alongside root_agent. Appended (with its import) when absent.

  - manifest-deployable
        Sets `deployable: true` in manifest.yaml — but ONLY when the
        backing-infra gate found nothing. See below.

TWO OUTCOMES, NEVER CONFLATED
-----------------------------
`deployable` in .github/schemas/manifest-schema.json means "can be deployed
with one click". So this script reports one of:

  deployable     Image builds and boots, and no bespoke infrastructure is
                 needed. `manifest.deployable` is set to true.
  containerized  Everything above, but the recipe needs backing infra a human
                 must provision. `manifest.deployable` is NOT touched.

Setting the flag on a recipe that still needs hand-written terraform puts a
false claim in the manifest, which is worse than leaving it unset.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
  - Build or run a container. Out of scope by design; the owner's CI builds
    the image via Cloud Build and pushes to Artifact Registry.
  - Migrate agent code across an ADK major (see adk-version-floor).
  - Merge a legacy app_utils generation (see legacy-app-utils).
  - Write terraform.
  - Remove [tool.ruff*] or fix requires-python — that is
    align-recipe-pyproject's job, and duplicating it guarantees drift.
  - Complete .env.example — that is extract-python-environment-variables.
  - Run `uv lock` or ruff. The calling skill does that after this script, so
    a failure is attributable to the right step.

It reports which of those the owner still needs, rather than doing them.

A NOTE ON WHAT GENERATING a2a.py PROVES
---------------------------------------
Nothing, on its own. These files were designed assuming the project was
scaffolded by agents-cli, and a file named a2a.py does not make an agent
behave correctly over A2A. This script copies and configures; it does not
certify. The report says so, and so must the skill.
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomlkit
import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
from ruamel.yaml import YAML

# ---------------------------------------------------------------------------
# Status vocabulary — identical to align-recipe-pyproject so a caller can
# render both skills' reports with one code path.
# ---------------------------------------------------------------------------
CLEAN = "clean"  # already satisfies the rule
WOULD_FIX = "would_fix"  # dry-run: a fix is available and would be applied
FIXED = "fixed"  # apply mode: the fix was applied
NEEDS_INPUT = "needs_input"  # cannot proceed without a human decision
REPORT_ONLY = "report_only"  # informational, no auto-fix
ERROR = "error"  # unexpected failure

# Outcomes (see module docstring).
OUTCOME_DEPLOYABLE = "deployable"
OUTCOME_CONTAINERIZED = "containerized"
OUTCOME_BLOCKED = "blocked"

# Template placeholders. Both are valid Python identifiers on purpose, so the
# vendored .py templates parse and can be linted in place rather than only
# after substitution.
PLACEHOLDER_PACKAGE = "__AGENT_PACKAGE__"
PLACEHOLDER_PROJECT = "__PROJECT_NAME__"

# Modules whose presence means the recipe talks to a backing service this
# script cannot provision. Matched against the recipe's import statements.
BACKING_SERVICE_IMPORTS = {
    "google.cloud.sql": "Cloud SQL",
    "google.cloud.pubsub": "Pub/Sub",
    "google.cloud.pubsub_v1": "Pub/Sub",
    "google.cloud.discoveryengine": "Vertex AI Search datastore",
    "google.cloud.alloydb": "AlloyDB",
    "asyncpg": "PostgreSQL",
    "psycopg": "PostgreSQL",
    "psycopg2": "PostgreSQL",
    "pymysql": "MySQL",
    "sqlalchemy": "a SQL database",
}

# Environment variables that likewise imply provisioned infrastructure.
BACKING_SERVICE_ENV_VARS = {
    "INSTANCE_CONNECTION_NAME": "Cloud SQL",
    "SESSION_DB_URL": "a SQL session store",
    "TASK_DB_URL": "a SQL task store",
    "DATASTORE_ID": "a Vertex AI Search datastore",
    "VECTOR_INDEX_ID": "a vector index",
    "INDEX_ENDPOINT_ID": "a vector index endpoint",
}

DOCKERIGNORE = """\
# Build context exclusions. Keeping .venv/ out is not optional: every recipe
# in this repo has one in-tree, and copying it into the image both bloats it
# and can shadow the container's own site-packages.
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Never bake credentials into an image. .env.example is safe and stays.
.env

.git/
.gitignore
.DS_Store
"""


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
    outcome: str = OUTCOME_BLOCKED
    agent_package: str | None = None
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def todo(self, text: str) -> None:
        self.todos.append(text)

    def to_json(self) -> str:
        return json.dumps(
            {
                "recipe_dir": self.recipe_dir,
                "mode": self.mode,
                "outcome": self.outcome,
                "agent_package": self.agent_package,
                "checks": [asdict(c) for c in self.checks],
                "files_written": self.files_written,
                "todos": self.todos,
                "notes": self.notes,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def find_repo_root(start: Path) -> Path | None:
    """Walk up from `start` looking for the directory holding .github/policy.yml."""
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / ".github" / "policy.yml").is_file():
            return candidate
    return None


def load_policy(repo_root: Path) -> dict[str, Any]:
    """Read the `deployability:` section of .github/policy.yml.

    Read rather than hardcoded so the standard (ADK floor, dependency list,
    file list) can move without a code change — that was an explicit design
    requirement, not a convenience.
    """
    yaml = YAML(typ="safe")
    with open(repo_root / ".github" / "policy.yml", encoding="utf-8") as f:
        data = yaml.load(f)
    policy = (data or {}).get("deployability")
    if not policy:
        raise RuntimeError(
            "`deployability:` section missing from .github/policy.yml — the "
            "skill cannot know what the standard requires."
        )
    return policy


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
    "test",
    "frontend",
    "deployment",
    "infra",
    "terraform",
}


def _walk_py_files(root: Path):
    """Yield .py files under `root`, skipping build output and virtualenvs."""
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        yield path


def find_agent_package(recipe_dir: Path) -> tuple[Path, Path] | None:
    """Return (agent_py_path, package_dir) for the shallowest agent.py.

    The package directory is agent.py's parent — `app`, `horizon`,
    `expense_agent`. Shallowest wins so a sub-agent nested under the real
    package (e.g. `app/subagents/agent.py`) never shadows it; alphabetical
    tie-break keeps the choice deterministic.
    """
    candidates = []
    for path in _walk_py_files(recipe_dir):
        if path.name == "agent.py":
            depth = len(path.relative_to(recipe_dir).parts)
            candidates.append((depth, str(path), path))
    if not candidates:
        return None
    candidates.sort()
    agent_py = candidates[0][2]
    return agent_py, agent_py.parent


def module_has_name(tree: ast.Module, name: str) -> bool:
    """True if `name` is bound at module level by an assignment or a def."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return True
    return False


def imported_modules(tree: ast.Module) -> set[str]:
    """Every dotted module path imported anywhere in the file."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def check_adk_version_floor(
    deps: list[str], min_adk: str, migration_is_manual: bool
) -> Check:
    """Verify the recipe's google-adk specifier accepts the required floor."""
    floor = Version(min_adk)
    for raw in deps:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower() != "google-adk":
            continue

        # prereleases=True so an alpha floor like `>=2.0.0a0` is judged on
        # what it admits, not silently treated as excluding everything.
        if req.specifier.contains(floor, prereleases=True):
            return Check(
                id="adk-version-floor",
                status=CLEAN,
                message=f"google-adk specifier accepts {min_adk}.",
                details={"requirement": raw, "min_google_adk": min_adk},
            )

        if not migration_is_manual:
            return Check(
                id="adk-version-floor",
                status=REPORT_ONLY,
                message=(
                    f"google-adk specifier `{raw}` excludes {min_adk}, and "
                    "policy allows an automatic bump — not implemented."
                ),
                details={"requirement": raw, "min_google_adk": min_adk},
            )

        return Check(
            id="adk-version-floor",
            status=NEEDS_INPUT,
            message=(
                f"google-adk is pinned `{raw}`, which excludes {min_adk}. The "
                "serving stack needs a2a-sdk>=1.0, and only google-adk 2.5.0+ "
                "permits it (2.0.0-2.4.0 cap a2a-sdk at <0.4). Crossing an ADK "
                "major is a code migration, not a pin edit — widening the "
                "ceiling here would produce a recipe importing a major it was "
                "never ported to. Migrate the agent first, then re-run."
            ),
            details={"requirement": raw, "min_google_adk": min_adk},
        )

    return Check(
        id="adk-version-floor",
        status=NEEDS_INPUT,
        message=(
            "No google-adk requirement found in [project].dependencies. Add "
            f"one accepting >={min_adk} before making this recipe deployable."
        ),
        details={"min_google_adk": min_adk},
    )


def locked_version(lock_path: Path, package: str) -> Version | None:
    """Return the version of `package` currently pinned in uv.lock, if any."""
    if not lock_path.is_file():
        return None
    try:
        data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None
    for entry in data.get("package", []):
        if entry.get("name") == package:
            try:
                return Version(str(entry["version"]))
            except (KeyError, InvalidVersion):
                return None
    return None


def check_adk_locked_version(lock_path: Path, min_adk: str) -> Check:
    """Judge the ADK migration risk on what the recipe RESOLVES to today.

    The declared specifier is the wrong signal on its own. A recipe pinning
    `google-adk>=1.0.0` accepts 2.6 on paper, so a specifier-only check passes
    it — but if its lockfile says 1.28.0, that recipe has only ever run on ADK
    1.x, and adding the serving dependencies silently drags it across a major
    version. That is a code migration wearing a metadata disguise, which is
    exactly the failure align-recipe-pyproject warns about.

    Conversely a loose specifier already locked to 2.6.2 (financial-advisor)
    is genuinely fine, and blocking it on the specifier alone would be wrong.

    So: compare MAJORS. Crossing one stops the run; a minor bump inside the
    same major is reported but allowed.
    """
    floor = Version(min_adk)
    have = locked_version(lock_path, "google-adk")

    if have is None:
        return Check(
            id="adk-locked-version",
            status=REPORT_ONLY,
            message=(
                "No uv.lock, or google-adk absent from it — cannot tell which "
                "ADK version this recipe actually runs on. Verify the agent "
                f"works on {min_adk} before relying on the generated files."
            ),
        )

    if have.major < floor.major:
        return Check(
            id="adk-locked-version",
            status=NEEDS_INPUT,
            message=(
                f"uv.lock resolves google-adk to {have}, but deployability "
                f"needs {min_adk}. The declared specifier permits the newer "
                "version, so re-locking WOULD silently cross a major — and the "
                "agent code has only ever run against "
                f"{have.major}.x. Port the agent to {floor.major}.x first, "
                "re-lock, then re-run. This script only rewrites metadata; it "
                "cannot migrate code."
            ),
            details={"locked": str(have), "min_google_adk": min_adk},
        )

    if have < floor:
        return Check(
            id="adk-locked-version",
            status=REPORT_ONLY,
            message=(
                f"uv.lock resolves google-adk to {have}; re-locking will raise "
                f"it to at least {min_adk}. Same major, so this is a minor "
                "upgrade rather than a migration — but exercise the agent "
                "afterwards."
            ),
            details={"locked": str(have), "min_google_adk": min_adk},
        )

    return Check(
        id="adk-locked-version",
        status=CLEAN,
        message=f"uv.lock already resolves google-adk to {have}.",
        details={"locked": str(have)},
    )


def check_already_deployable(recipe_dir: Path, package_dir: Path) -> Check:
    """Warn when the recipe already serves, by its own arrangement.

    `long-horizon-harness` is the case this exists for: it ships a Dockerfile,
    `deployable: true`, and a bespoke ~400-line `fast_api_app.py` that wires
    its own A2A routes from a `horizon/a2a/` package. Nothing here is
    "missing", but because it has no `app_utils/`, a naive run would happily
    add three modules that the recipe's own entrypoint never imports — dead
    code that looks load-bearing.

    Advisory, not a gate: a recipe may legitimately want to migrate onto the
    standard layout. But the owner should choose that, not discover it.
    """
    has_dockerfile = (recipe_dir / "Dockerfile").is_file()
    has_entrypoint = (package_dir / "fast_api_app.py").is_file()
    if not (has_dockerfile and has_entrypoint):
        return Check(
            id="already-deployable",
            status=CLEAN,
            message="Recipe does not already ship a container entrypoint.",
        )
    return Check(
        id="already-deployable",
        status=REPORT_ONLY,
        message=(
            "This recipe ALREADY has a Dockerfile and a "
            f"{package_dir.name}/fast_api_app.py, so it serves by its own "
            "arrangement. Neither will be replaced without --overwrite, but "
            "any app_utils/ modules generated alongside a bespoke entrypoint "
            "will be dead code unless someone wires them in. Confirm the "
            "owner actually wants to migrate onto the standard layout before "
            "applying."
        ),
        details={"dockerfile": has_dockerfile, "entrypoint": has_entrypoint},
    )


def check_legacy_app_utils(package_dir: Path, legacy_files: list[str]) -> Check:
    """Stop if the recipe carries the old ASP-era app_utils generation."""
    app_utils = package_dir / "app_utils"
    if not app_utils.is_dir():
        return Check(
            id="legacy-app-utils",
            status=CLEAN,
            message="No existing app_utils/ — nothing to collide with.",
        )
    present = sorted(f for f in legacy_files if (app_utils / f).is_file())
    if not present:
        return Check(
            id="legacy-app-utils",
            status=CLEAN,
            message="app_utils/ exists but carries no legacy-generation files.",
        )
    return Check(
        id="legacy-app-utils",
        status=NEEDS_INPUT,
        message=(
            f"{package_dir.name}/app_utils/ carries the old generation "
            f"({', '.join(present)}). Those wire telemetry and services "
            "differently from services.py, and the existing fast_api_app.py "
            "imports them. Generating over the top would orphan them or "
            "double-wire telemetry. A human needs to decide how to migrate."
        ),
        details={"legacy_files_present": present},
    )


def check_backing_infra(recipe_dir: Path, package_dir: Path) -> Check:
    """Detect infrastructure this script cannot provision.

    Advisory: it selects the OUTCOME (containerized vs deployable) rather than
    stopping the run.
    """
    reasons: list[str] = []

    for dirname in ("infra", "terraform"):
        if (recipe_dir / dirname).is_dir():
            reasons.append(f"has a {dirname}/ directory")

    found_services: set[str] = set()
    for path in _walk_py_files(package_dir):
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
        except (SyntaxError, UnicodeDecodeError):
            continue
        for module in imported_modules(tree):
            for prefix, label in BACKING_SERVICE_IMPORTS.items():
                if module == prefix or module.startswith(prefix + "."):
                    found_services.add(label)
    for label in sorted(found_services):
        reasons.append(f"imports a client for {label}")

    env_example = recipe_dir / ".env.example"
    if env_example.is_file():
        text = env_example.read_text(encoding="utf-8")
        for var, label in BACKING_SERVICE_ENV_VARS.items():
            if re.search(rf"^\s*{re.escape(var)}\s*=", text, re.MULTILINE):
                reasons.append(f"declares {var}, implying {label}")

    if not reasons:
        return Check(
            id="backing-infra",
            status=CLEAN,
            message="No backing infrastructure detected — one-click deployable.",
        )
    return Check(
        id="backing-infra",
        status=REPORT_ONLY,
        message=(
            "Recipe needs infrastructure this skill cannot provision: "
            + "; ".join(reasons)
            + ". It can still be containerized, but it is not one-click "
            "deployable, so manifest.deployable is left unset."
        ),
        details={"reasons": reasons},
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def render_template(text: str, package: str, project: str) -> str:
    return text.replace(PLACEHOLDER_PACKAGE, package).replace(
        PLACEHOLDER_PROJECT, project
    )


def inject_data_dir_copies(
    dockerfile: str, package: str, data_dirs: list[str]
) -> str:
    """Add `COPY ./<dir> ./<dir>` lines for runtime data the agent reads.

    Inserted immediately after the agent-package COPY so they land before
    `uv sync`, matching the layer ordering of every hand-written Dockerfile in
    the repo. A missing data directory does not fail the build — it fails at
    request time — which is exactly why they are confirmed by a human rather
    than guessed.
    """
    if not data_dirs:
        return dockerfile
    anchor = f"COPY ./{package} ./{package}"
    if anchor not in dockerfile:
        return dockerfile
    extra = "\n".join(f"COPY ./{d} ./{d}" for d in data_dirs)
    return dockerfile.replace(anchor, f"{anchor}\n\n{extra}", 1)


def set_python_base_image(dockerfile: str, python_version: str | None) -> str:
    """Rewrite `FROM python:X-slim` to the recipe's own floor.

    The vendored template hardcodes 3.12. Recipes in this repo target 3.11,
    3.12 and 3.13, and building on a different minor than the recipe declares
    is a real source of resolution failures.
    """
    if not python_version:
        return dockerfile
    return re.sub(
        r"^FROM python:[\d.]+-slim$",
        f"FROM python:{python_version}-slim",
        dockerfile,
        count=1,
        flags=re.MULTILINE,
    )


def python_floor_from_requires(requires_python: str | None) -> str | None:
    """Extract `3.11` from a specifier like `>=3.11,<3.14`."""
    if not requires_python:
        return None
    match = re.search(r">=\s*(\d+\.\d+)", requires_python)
    return match.group(1) if match else None


def generate_serving_files(
    *,
    templates_dir: Path,
    recipe_dir: Path,
    package_dir: Path,
    project_name: str,
    python_version: str | None,
    data_dirs: list[str],
    apply: bool,
    overwrite: bool,
    report: Report,
) -> None:
    """Write the five required files plus .dockerignore."""
    package = package_dir.name
    jobs: list[tuple[Path, Path]] = [
        (templates_dir / "Dockerfile", recipe_dir / "Dockerfile"),
        (templates_dir / "fast_api_app.py", package_dir / "fast_api_app.py"),
        (
            templates_dir / "app_utils" / "a2a.py",
            package_dir / "app_utils" / "a2a.py",
        ),
        (
            templates_dir / "app_utils" / "services.py",
            package_dir / "app_utils" / "services.py",
        ),
        (
            templates_dir / "app_utils" / "reasoning_engine_adapter.py",
            package_dir / "app_utils" / "reasoning_engine_adapter.py",
        ),
    ]

    for src, dst in jobs:
        rel = dst.relative_to(recipe_dir)
        if dst.exists() and not overwrite:
            report.add(
                Check(
                    id=f"file:{rel}",
                    status=REPORT_ONLY,
                    message=(
                        f"{rel} already exists — left untouched. Re-run with "
                        "--overwrite to replace it, after checking what it does."
                    ),
                )
            )
            continue

        content = render_template(
            src.read_text(encoding="utf-8"), package, project_name
        )
        if dst.name == "Dockerfile":
            content = set_python_base_image(content, python_version)
            content = inject_data_dir_copies(content, package, data_dirs)

        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")
            report.files_written.append(str(rel))
        report.add(
            Check(
                id=f"file:{rel}",
                status=FIXED if apply else WOULD_FIX,
                message=f"{'Wrote' if apply else 'Would write'} {rel}.",
            )
        )

    dockerignore = recipe_dir / ".dockerignore"
    if dockerignore.exists() and not overwrite:
        report.add(
            Check(
                id="file:.dockerignore",
                status=REPORT_ONLY,
                message=(
                    ".dockerignore already exists — left untouched. Re-run "
                    "with --overwrite to replace it."
                ),
            )
        )
    else:
        if apply:
            dockerignore.write_text(DOCKERIGNORE, encoding="utf-8")
            report.files_written.append(".dockerignore")
        report.add(
            Check(
                id="file:.dockerignore",
                status=FIXED if apply else WOULD_FIX,
                message=(
                    f"{'Wrote' if apply else 'Would write'} .dockerignore "
                    "(excludes .venv/, .env, .git/)."
                ),
            )
        )


# ---------------------------------------------------------------------------
# pyproject.toml patching
# ---------------------------------------------------------------------------


def _with_extras(req: Requirement, extras: list[str]) -> str:
    """Re-render `req` carrying `extras`, preserving everything else.

    Rebuilt from the parsed parts rather than string-patched so a marker or a
    URL survives: `foo>=1; python_version<"3.12"` must not lose its marker on
    the way through.
    """
    spec = str(req.specifier)
    url = f" @ {req.url}" if req.url else ""
    marker = f" ; {req.marker}" if req.marker else ""
    return f"{req.name}[{','.join(extras)}]{url}{spec}{marker}"


def _replace_requirement(deps: Any, old: str, new: str) -> None:
    """Swap one entry of the dependencies array in place.

    Index assignment keeps the entry's position (and tomlkit's surrounding
    formatting) rather than removing and appending, which would silently
    reorder a hand-curated, commented dependency list.
    """
    for i, raw in enumerate(deps):
        if str(raw) == old:
            deps[i] = new
            return


def patch_dependencies(
    doc: tomlkit.TOMLDocument, required: list[str], apply: bool
) -> Check:
    """Add missing required dependencies and merge in any missing extras.

    Version specifiers the recipe already declares are never rewritten — see
    the extras comment inside for why those two edits carry different risk.
    """
    project = doc.get("project")
    if project is None:
        return Check(
            id="required-dependencies",
            status=ERROR,
            message="pyproject.toml has no [project] table.",
        )
    deps = project.get("dependencies")
    if deps is None:
        deps = tomlkit.array()
        deps.multiline(True)
        project["dependencies"] = deps

    # Map normalised name -> the recipe's OWN requirement string. Reporting
    # the recipe's actual spec rather than the policy's is the whole point of
    # the "left alone" message: telling an owner that
    # `google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0` is already present, when the
    # recipe really pins `>=2.2.0`, hides exactly the mismatch they were being
    # asked to check.
    existing: dict[str, str] = {}
    for raw in deps:
        try:
            existing[Requirement(str(raw)).name.lower()] = str(raw)
        except InvalidRequirement:
            continue

    missing: list[str] = []
    kept: list[dict[str, Any]] = []
    for spec in required:
        req = Requirement(spec)
        name = req.name.lower()
        if name not in existing:
            missing.append(spec)
            continue
        # Version compatibility is not the only thing that matters. Extras
        # carry real code: `google-adk[gcp,otel-gcp]` is what installs the
        # OTel exporters and GCP clients the generated services.py and
        # fast_api_app.py import. A recipe pinning bare `google-adk>=2.2.0`
        # satisfies the VERSION floor and still cannot run the serving stack,
        # so leaving it alone would knowingly emit a recipe that fails to
        # import.
        #
        # Extras are therefore MERGED IN, while the version specifier is left
        # exactly as the recipe wrote it. The two are not equivalent risks:
        # adding an extra only ever pulls in more of the same package's own
        # optional dependencies, whereas rewriting a version bound overrides a
        # deliberate pin and can move the recipe onto code it was never tested
        # against. So `google-adk>=2.2.0` becomes
        # `google-adk[gcp,otel-gcp]>=2.2.0` — never `>=2.6.0`.
        try:
            have = Requirement(existing[name])
            lacking = sorted(req.extras - have.extras)
        except InvalidRequirement:
            have, lacking = None, []
        merged = None
        if lacking and have is not None:
            merged = _with_extras(have, sorted(have.extras | req.extras))
            if apply:
                _replace_requirement(deps, existing[name], merged)
        kept.append(
            {
                "required": spec,
                "recipe_has": existing[name],
                "missing_extras": lacking,
                "rewritten_to": merged,
            }
        )

    kept_note = ""
    unchanged = [k for k in kept if not k["rewritten_to"]]
    if unchanged:
        pairs = "; ".join(
            f"{k['recipe_has']} (required: {k['required']})" for k in unchanged
        )
        kept_note = (
            f" Left {len(unchanged)} existing requirement(s) alone — {pairs}. "
            "A recipe may carry a deliberate tighter pin, so confirm each "
            "still admits the required version."
        )
    extras_gaps = [k for k in kept if k["rewritten_to"]]
    if extras_gaps:
        detail = "; ".join(
            f"{k['recipe_has']} -> {k['rewritten_to']}" for k in extras_gaps
        )
        kept_note += (
            f" {'Added' if apply else 'Would add'} missing extras — {detail}. "
            "Version bounds were NOT touched; only the extras the generated "
            "serving code imports were merged in."
        )

    if not missing:
        return Check(
            id="required-dependencies",
            status=CLEAN,
            message=(
                "All required serving dependencies are already declared."
                + kept_note
            ),
            details={"kept": kept},
        )

    if apply:
        for spec in missing:
            deps.append(spec)
        deps.multiline(True)

    return Check(
        id="required-dependencies",
        status=FIXED if apply else WOULD_FIX,
        message=(
            f"{'Added' if apply else 'Would add'} {len(missing)} required "
            f"dependency/ies: {', '.join(missing)}." + kept_note
        ),
        details={"added": missing, "kept": kept},
    )


def patch_hatch_packages(
    doc: tomlkit.TOMLDocument, package: str, apply: bool
) -> Check:
    """Ensure [tool.hatch.build.targets.wheel].packages lists the agent package."""
    tool = doc.get("tool")
    build_backend = (doc.get("build-system") or {}).get("build-backend", "")
    if "hatchling" not in str(build_backend):
        return Check(
            id="hatch-wheel-packages",
            status=REPORT_ONLY,
            message=(
                f"Build backend is `{build_backend or 'unset'}`, not hatchling — "
                "skipping the hatch wheel-packages check. Confirm the recipe's "
                f"backend installs the `{package}` package, or `uv sync` will "
                "fail in the image."
            ),
            details={"build_backend": str(build_backend)},
        )

    wheel = None
    if tool is not None:
        wheel = (
            tool.get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel")
        )
    if wheel is not None and package in [
        str(p) for p in wheel.get("packages", [])
    ]:
        return Check(
            id="hatch-wheel-packages",
            status=CLEAN,
            message=f"hatch wheel packages already includes `{package}`.",
        )

    if apply:
        if tool is None:
            tool = tomlkit.table(is_super_table=True)
            doc["tool"] = tool
        node: Any = tool
        for key in ("hatch", "build", "targets"):
            if key not in node:
                node[key] = tomlkit.table(is_super_table=True)
            node = node[key]
        if "wheel" not in node:
            node["wheel"] = tomlkit.table()
        wheel_tbl = node["wheel"]
        packages = wheel_tbl.get("packages")
        if packages is None:
            packages = tomlkit.array()
            wheel_tbl["packages"] = packages
        if package not in [str(p) for p in packages]:
            packages.append(package)

    return Check(
        id="hatch-wheel-packages",
        status=FIXED if apply else WOULD_FIX,
        message=(
            f"{'Added' if apply else 'Would add'} `{package}` to "
            "[tool.hatch.build.targets.wheel].packages — uv sync fails without it."
        ),
    )


# ---------------------------------------------------------------------------
# agent.py patching
# ---------------------------------------------------------------------------


def patch_app_object(agent_py: Path, package: str, apply: bool) -> Check:
    """Ensure agent.py defines `app = App(root_agent=root_agent, name=...)`.

    Appended at end of file rather than inserted, because `root_agent` must
    already be bound. The import goes after the last top-level import so the
    file keeps a conventional shape and ruff's import rules stay happy.
    """
    try:
        source = agent_py.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(agent_py))
    except (SyntaxError, UnicodeDecodeError) as e:
        return Check(
            id="app-object",
            status=ERROR,
            message=f"Could not parse {agent_py.name}: {e}",
        )

    if not module_has_name(tree, "root_agent"):
        return Check(
            id="app-object",
            status=NEEDS_INPUT,
            message=(
                f"{agent_py.name} defines no module-level `root_agent`, so an "
                "`App` cannot be wired to it. The serving entrypoint imports "
                "both by name."
            ),
        )

    if module_has_name(tree, "app"):
        return Check(
            id="app-object",
            status=CLEAN,
            message=f"{agent_py.name} already defines a module-level `app`.",
        )

    lines = source.splitlines()
    needs_import = "google.adk.apps" not in imported_modules(tree)

    if apply:
        if needs_import:
            last_import_line = 0
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    last_import_line = max(
                        last_import_line, node.end_lineno or 0
                    )
            lines.insert(last_import_line, "from google.adk.apps import App")
        body = "\n".join(lines).rstrip("\n")
        body += f'\n\napp = App(root_agent=root_agent, name="{package}")\n'
        agent_py.write_text(body, encoding="utf-8")

    return Check(
        id="app-object",
        status=FIXED if apply else WOULD_FIX,
        message=(
            f"{'Added' if apply else 'Would add'} "
            f'`app = App(root_agent=root_agent, name="{package}")` to '
            f"{agent_py.name}"
            + (" (plus the App import)." if needs_import else ".")
        ),
    )


# ---------------------------------------------------------------------------
# manifest.yaml
# ---------------------------------------------------------------------------


def patch_manifest_deployable(
    manifest_path: Path, infra_clean: bool, apply: bool
) -> Check:
    """Set `deployable: true`, but only when no backing infra was detected."""
    if not infra_clean:
        return Check(
            id="manifest-deployable",
            status=REPORT_ONLY,
            message=(
                "Left manifest.deployable unset: the recipe needs backing "
                "infrastructure, so it is containerized but not one-click "
                "deployable. Claiming otherwise would put a false flag in the "
                "manifest."
            ),
        )
    if not manifest_path.is_file():
        return Check(
            id="manifest-deployable",
            status=REPORT_ONLY,
            message="No manifest.yaml found — nothing to flag.",
        )

    text = manifest_path.read_text(encoding="utf-8")

    # Read with ruamel, but WRITE textually. A round-trip dump reflows the
    # whole document — it re-indents every sequence to its own default and
    # hard-wraps long scalars — turning "add one boolean key" into a 30-line
    # diff that silently restyles a file the owner did not ask us to touch.
    # Adding a top-level scalar is a one-line edit, so make it one.
    yaml = YAML()
    yaml.preserve_quotes = True
    try:
        data = yaml.load(text) or {}
    except Exception as e:
        return Check(
            id="manifest-deployable",
            status=ERROR,
            message=f"Failed to parse manifest.yaml: {e}",
        )

    if data.get("deployable") is True:
        return Check(
            id="manifest-deployable",
            status=CLEAN,
            message="manifest.deployable is already true.",
        )

    lines = text.splitlines()
    existing = next(
        (i for i, ln in enumerate(lines) if re.match(r"^deployable\s*:", ln)),
        None,
    )
    if existing is not None:
        # Present but false/commented-out value — replace in place so the key
        # keeps its position and any trailing comment context around it.
        new_lines = list(lines)
        new_lines[existing] = "deployable: true"
    else:
        # Insert after `language:` (where the schema examples put it), else
        # after `type:`, else append. Anchoring beats appending because a
        # manifest often ends inside a nested block, where a top-level key
        # appended at EOF reads as belonging to that block.
        anchor = next(
            (
                i
                for key in ("language", "type", "status")
                for i, ln in enumerate(lines)
                if re.match(rf"^{key}\s*:", ln)
            ),
            None,
        )
        new_lines = list(lines)
        new_lines.insert(
            len(lines) if anchor is None else anchor + 1, "deployable: true"
        )

    if apply:
        manifest_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return Check(
        id="manifest-deployable",
        status=FIXED if apply else WOULD_FIX,
        message=f"{'Set' if apply else 'Would set'} manifest.deployable = true.",
    )


# ---------------------------------------------------------------------------
# agents-cli-manifest.yaml
# ---------------------------------------------------------------------------

# Fields DELIBERATELY omitted, and why omission is more correct than a value.
# agents-cli's ProjectConfig.from_dict gives every field a default, so nothing
# here is required — the question is only which values would be TRUE.
#
#   acli_version   No agents-cli scaffold produced this recipe, so there is no
#                  honest value. Absence is handled explicitly: check_cli_version
#                  returns early on a falsy version and scaffold_older_than
#                  returns False, so both callers fall back to generic guidance.
#                  A FABRICATED version is actively harmful — it makes the CLI
#                  print "Upgrade the project: agents-cli scaffold upgrade" for
#                  a project that was never scaffolded and cannot be upgraded.
#   generated_at   Never read by from_dict, and would assert a generation event
#                  that did not happen.
#   base_template  Only consumed by upgrade/enhance (metadata_to_cli_args), which
#                  do not apply here. It already defaults to "adk".
#   description    Never read by from_dict; manifest.yaml already carries it.
_ACLI_MANIFEST = """\
# agents-cli project manifest (agents-cli 1.x reads this file).
#
# Written by the make-python-recipe-deployable repo skill — NOT by an
# agents-cli scaffold. `acli_version`, `generated_at` and `base_template` are
# deliberately omitted rather than invented: no scaffold produced this recipe,
# and a fabricated version makes agents-cli offer a `scaffold upgrade` that
# cannot work. agents-cli treats an absent version as "unknown" and falls back
# to generic guidance, which is the accurate outcome here.
#
# `agents-cli deploy` uses this file as the project-root marker and reads
# `deployment_target` to choose how to deploy.
name: "{name}"
agent_directory: "{agent_directory}"
language: "python"
region: "{region}"
create_params:
  deployment_target: "cloud_run"
  session_type: "in_memory"
  is_a2a: true
  cicd_runner: "skip"
  agent_gateway: false
  agent_guidance_filename: "{guidance}"
"""


def detect_guidance_filename(recipe_dir: Path) -> str:
    """Which agent-guidance file the recipe actually ships.

    Every core/ recipe is required to carry AGENTS.md (policy.yml
    required_files.by_root.core), while agents-cli's own default is GEMINI.md.
    Reporting the wrong one would send a reader to a file that is not there.
    """
    for candidate in ("AGENTS.md", "GEMINI.md", "CLAUDE.md"):
        if (recipe_dir / candidate).is_file():
            return candidate
    return "AGENTS.md"


def write_agents_cli_manifest(
    recipe_dir: Path,
    *,
    project_name: str,
    agent_directory: str,
    region: str,
    apply: bool,
) -> Check:
    """Write agents-cli-manifest.yaml with only fields that are actually true."""
    path = recipe_dir / "agents-cli-manifest.yaml"
    if path.is_file():
        return Check(
            id="agents-cli-manifest",
            status=REPORT_ONLY,
            message=(
                "agents-cli-manifest.yaml already exists — left untouched. If it "
                "predates the current agents-cli (check `acli_version`), its "
                "create_params may not describe what this skill just generated."
            ),
        )

    content = _ACLI_MANIFEST.format(
        name=project_name,
        agent_directory=agent_directory,
        region=region,
        guidance=detect_guidance_filename(recipe_dir),
    )
    if apply:
        path.write_text(content, encoding="utf-8")

    return Check(
        id="agents-cli-manifest",
        status=FIXED if apply else WOULD_FIX,
        message=(
            f"{'Wrote' if apply else 'Would write'} agents-cli-manifest.yaml "
            f"(region {region}). This is what lets `agents-cli deploy` find the "
            "project root and pick a deployment target. acli_version / "
            "generated_at / base_template are omitted on purpose — see the "
            "file's header."
        ),
        details={"region": region},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    *,
    recipe_dir: Path,
    apply: bool,
    overwrite: bool,
    data_dirs: list[str],
    region: str,
) -> Report:
    report = Report(
        recipe_dir=str(recipe_dir), mode="apply" if apply else "dry-run"
    )

    repo_root = find_repo_root(recipe_dir)
    if repo_root is None:
        report.add(
            Check(
                id="policy",
                status=ERROR,
                message="Could not locate .github/policy.yml above the recipe.",
            )
        )
        return report
    policy = load_policy(repo_root)

    templates_dir = (
        Path(__file__).resolve().parent.parent / "resources" / "templates"
    )

    # --- agent package -----------------------------------------------------
    found = find_agent_package(recipe_dir)
    if found is None:
        report.add(
            Check(
                id="agent-package",
                status=ERROR,
                message=(
                    "No agent.py found. This does not look like a Python ADK "
                    "recipe, or the agent lives somewhere unexpected."
                ),
            )
        )
        return report
    agent_py, package_dir = found
    report.agent_package = package_dir.name
    report.add(
        Check(
            id="agent-package",
            status=CLEAN,
            message=f"Agent package is `{package_dir.name}` ({agent_py.relative_to(recipe_dir)}).",
            details={"agent_file": str(agent_py.relative_to(recipe_dir))},
        )
    )

    # --- pyproject ---------------------------------------------------------
    pyproject_path = recipe_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        report.add(
            Check(
                id="pyproject",
                status=ERROR,
                message="No pyproject.toml — cannot check dependencies or build config.",
            )
        )
        return report
    doc = tomlkit.parse(pyproject_path.read_text(encoding="utf-8"))
    project_tbl = doc.get("project") or {}
    deps = [str(d) for d in project_tbl.get("dependencies", [])]
    project_name = str(project_tbl.get("name") or recipe_dir.name)
    python_version = python_floor_from_requires(
        str(project_tbl.get("requires-python") or "")
    )

    # --- GATE: ADK floor ---------------------------------------------------
    adk_check = check_adk_version_floor(
        deps,
        str(policy["min_google_adk"]),
        bool(policy.get("adk_major_migration_is_manual", True)),
    )
    report.add(adk_check)
    if adk_check.status == NEEDS_INPUT:
        report.outcome = OUTCOME_BLOCKED
        report.note(
            "Stopped before generating anything. Nothing on disk was changed."
        )
        return report

    # --- GATE: what the recipe actually resolves to today ------------------
    locked_check = check_adk_locked_version(
        recipe_dir / "uv.lock", str(policy["min_google_adk"])
    )
    report.add(locked_check)
    if locked_check.status == NEEDS_INPUT:
        report.outcome = OUTCOME_BLOCKED
        report.note(
            "Stopped before generating anything. Nothing on disk was changed."
        )
        return report

    # --- GATE: legacy app_utils -------------------------------------------
    legacy_check = check_legacy_app_utils(
        package_dir, list(policy.get("legacy_app_utils_files", []))
    )
    report.add(legacy_check)
    if legacy_check.status == NEEDS_INPUT:
        report.outcome = OUTCOME_BLOCKED
        report.note(
            "Stopped before generating anything. Nothing on disk was changed."
        )
        return report

    # --- advisory: recipe already serves by its own arrangement ------------
    report.add(check_already_deployable(recipe_dir, package_dir))

    # --- GATE (advisory): backing infra ------------------------------------
    infra_check = check_backing_infra(recipe_dir, package_dir)
    report.add(infra_check)
    infra_clean = infra_check.status == CLEAN
    report.outcome = (
        OUTCOME_DEPLOYABLE if infra_clean else OUTCOME_CONTAINERIZED
    )

    # --- generate ----------------------------------------------------------
    generate_serving_files(
        templates_dir=templates_dir,
        recipe_dir=recipe_dir,
        package_dir=package_dir,
        project_name=project_name,
        python_version=python_version,
        data_dirs=data_dirs,
        apply=apply,
        overwrite=overwrite,
        report=report,
    )

    # --- patch -------------------------------------------------------------
    deps_check = patch_dependencies(
        doc, list(policy.get("required_dependencies", [])), apply
    )
    report.add(deps_check)
    for kept in deps_check.details.get("kept", []):
        if kept.get("missing_extras"):
            report.todo(
                f"Add the [{','.join(kept['missing_extras'])}] extra(s) to "
                f"`{kept['recipe_has']}` — the generated serving code imports "
                "what they install. Left alone here because widening an "
                "existing requirement changes resolution for the whole recipe."
            )
    report.add(patch_hatch_packages(doc, package_dir.name, apply))
    if apply:
        pyproject_path.write_text(tomlkit.dumps(doc), encoding="utf-8")

    report.add(patch_app_object(agent_py, package_dir.name, apply))
    report.add(
        patch_manifest_deployable(
            recipe_dir / "manifest.yaml", infra_clean, apply
        )
    )

    if policy.get("emit_agents_cli_manifest"):
        report.add(
            write_agents_cli_manifest(
                recipe_dir,
                project_name=project_name,
                agent_directory=package_dir.name,
                region=region,
                apply=apply,
            )
        )

    # --- follow-up the owner must do --------------------------------------
    report.todo(
        "Run `uv lock --python 3.11` in the recipe — dependencies changed and "
        "the lockfile is now stale."
    )
    report.todo(
        "Run ruff format + check from the REPO ROOT so the root config wins."
    )
    report.todo(
        "Declare any new environment variables in .env.example (the "
        "extract-python-environment-variables skill does this)."
    )
    if not infra_clean:
        report.todo(
            "Provision the backing infrastructure and write its terraform — "
            "this skill cannot, and manifest.deployable stays unset until it "
            "exists."
        )
    report.note(
        "Generating a2a.py does not prove the recipe supports A2A. These files "
        "assume an agents-cli-scaffolded project; they were copied and "
        "configured, not verified. Exercise the agent over A2A before claiming "
        "support."
    )
    report.note(
        "reasoning_engine_adapter.py is included because the Recipe "
        "Deployability doc lists it, but agents-cli ships it only under the "
        "agent_runtime target. Worth confirming it belongs in a cloud_run "
        "recipe."
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and configure the files a Python recipe needs to be "
            "deployable. Does not build images or deploy."
        )
    )
    parser.add_argument(
        "--recipe-dir",
        required=True,
        type=Path,
        help="Path to the recipe root (e.g. contrib/python/foo).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this the script only reports.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace serving files that already exist. Off by default: an "
            "existing fast_api_app.py is usually bespoke."
        ),
    )
    parser.add_argument(
        "--data-dirs",
        default="",
        help=(
            "Comma-separated runtime data directories to COPY into the image "
            "(e.g. 'assets,sample_data'). Confirmed by a human — a missing one "
            "fails at request time, not build time."
        ),
    )
    parser.add_argument(
        "--region",
        default="us-east1",
        help=(
            "GCP region recorded in agents-cli-manifest.yaml. Nothing in a "
            "recipe declares one, so this is a deployment decision the owner "
            "makes; us-east1 matches agents-cli's own default."
        ),
    )
    args = parser.parse_args()

    if not args.recipe_dir.is_dir():
        print(
            f"Error: --recipe-dir {args.recipe_dir} is not a directory.",
            file=sys.stderr,
        )
        return 2

    data_dirs = [d.strip() for d in args.data_dirs.split(",") if d.strip()]

    # Any unforeseen exception is surfaced as one ERROR check inside the normal
    # JSON envelope, so the calling agent renders it like any other outcome
    # instead of parsing a stack trace off stderr.
    try:
        report = run(
            recipe_dir=args.recipe_dir,
            apply=args.apply,
            overwrite=args.overwrite,
            data_dirs=data_dirs,
            region=args.region,
        )
    except Exception as e:  # final safety net for the CLI
        report = Report(
            recipe_dir=str(args.recipe_dir),
            mode="apply" if args.apply else "dry-run",
        )
        report.add(
            Check(
                id="internal", status=ERROR, message=f"{type(e).__name__}: {e}"
            )
        )

    print(report.to_json())

    statuses = {c.status for c in report.checks}
    if ERROR in statuses:
        return 2
    if NEEDS_INPUT in statuses:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
