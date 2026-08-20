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
"""Tests for make_deployable.py.

The emphasis is on the GATES and on not-clobbering behaviour. Those are where
a bug is expensive: a gate that fails open silently crosses an ADK major or
overwrites a bespoke entrypoint, and neither shows up until someone runs the
recipe.
"""

import textwrap
from pathlib import Path

import make_deployable as md
import pytest

MIN_ADK = "2.6.0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def recipe(tmp_path: Path) -> Path:
    """A minimal recipe that should sail through every gate."""
    write(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "demo-recipe"
        requires-python = ">=3.11,<3.14"
        dependencies = [
            "google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0",
        ]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["app"]
        """,
    )
    write(
        tmp_path / "app" / "agent.py",
        """
        from google.adk.agents import Agent

        root_agent = Agent(name="demo")
        """,
    )
    write(
        tmp_path / "uv.lock",
        '[[package]]\nname = "google-adk"\nversion = "2.6.2"\n',
    )
    write(tmp_path / "manifest.yaml", "type: standalone\nlanguage: python\n")
    write(tmp_path / "README.md", "# demo\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_finds_shallowest_agent_package(tmp_path: Path):
    """A nested sub-agent must never shadow the real package."""
    write(tmp_path / "app" / "agent.py", "root_agent = 1\n")
    write(tmp_path / "app" / "subagents" / "agent.py", "root_agent = 2\n")
    agent_py, package_dir = md.find_agent_package(tmp_path)
    assert package_dir.name == "app"
    assert agent_py == tmp_path / "app" / "agent.py"


def test_ignores_agent_py_inside_venv(tmp_path: Path):
    """Every recipe here has an in-tree .venv full of other agents' code."""
    write(tmp_path / ".venv" / "lib" / "agent.py", "root_agent = 1\n")
    write(tmp_path / "horizon" / "agent.py", "root_agent = 1\n")
    _, package_dir = md.find_agent_package(tmp_path)
    assert package_dir.name == "horizon"


# ---------------------------------------------------------------------------
# Gate: declared specifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0",
        "google-adk>=2.0.0,<3.0.0",  # ceiling still admits 2.6
        "google-adk>=1.0.0",  # loose, admits everything
        "google-adk>=2.0.0a0",  # prerelease floor
    ],
)
def test_specifier_accepting_floor_is_clean(spec):
    check = md.check_adk_version_floor([spec], MIN_ADK, True)
    assert check.status == md.CLEAN


@pytest.mark.parametrize(
    "spec",
    [
        "google-adk>=1.15.0,<2.0.0",  # cross-session-memory
        "google-adk==1.31.0",  # oauth-user-consent-flow
        "google-adk<2",
    ],
)
def test_specifier_excluding_floor_blocks(spec):
    check = md.check_adk_version_floor([spec], MIN_ADK, True)
    assert check.status == md.NEEDS_INPUT


def test_missing_adk_dependency_blocks():
    check = md.check_adk_version_floor(["fastapi>=0.1"], MIN_ADK, True)
    assert check.status == md.NEEDS_INPUT


# ---------------------------------------------------------------------------
# Gate: locked version
#
# This is the gate that catches what the specifier cannot. A recipe pinning
# `google-adk>=1.0.0` passes the specifier check on paper while its lockfile
# says 1.28.0 — re-locking would cross a major silently.
# ---------------------------------------------------------------------------


def test_locked_on_older_major_blocks(tmp_path: Path):
    lock = write(
        tmp_path / "uv.lock",
        '[[package]]\nname = "google-adk"\nversion = "1.28.0"\n',
    )
    check = md.check_adk_locked_version(lock, MIN_ADK)
    assert check.status == md.NEEDS_INPUT
    assert "1.28.0" in check.message


def test_locked_below_floor_same_major_is_allowed(tmp_path: Path):
    """rag-agent-search sits at 2.3.0 — a minor bump, not a migration."""
    lock = write(
        tmp_path / "uv.lock",
        '[[package]]\nname = "google-adk"\nversion = "2.3.0"\n',
    )
    check = md.check_adk_locked_version(lock, MIN_ADK)
    assert check.status == md.REPORT_ONLY


def test_locked_at_or_above_floor_is_clean(tmp_path: Path):
    lock = write(
        tmp_path / "uv.lock",
        '[[package]]\nname = "google-adk"\nversion = "2.6.2"\n',
    )
    assert md.check_adk_locked_version(lock, MIN_ADK).status == md.CLEAN


def test_no_lockfile_reports_rather_than_guessing(tmp_path: Path):
    check = md.check_adk_locked_version(tmp_path / "uv.lock", MIN_ADK)
    assert check.status == md.REPORT_ONLY


# ---------------------------------------------------------------------------
# Gate: legacy app_utils
# ---------------------------------------------------------------------------


LEGACY = ["telemetry.py", "typing.py", "deploy.py", "memory_config.py"]


def test_legacy_app_utils_blocks(tmp_path: Path):
    write(tmp_path / "app" / "app_utils" / "telemetry.py", "x = 1\n")
    check = md.check_legacy_app_utils(tmp_path / "app", LEGACY)
    assert check.status == md.NEEDS_INPUT
    assert "telemetry.py" in check.details["legacy_files_present"]


def test_no_app_utils_is_clean(tmp_path: Path):
    (tmp_path / "app").mkdir()
    assert (
        md.check_legacy_app_utils(tmp_path / "app", LEGACY).status == md.CLEAN
    )


def test_new_generation_app_utils_is_clean(tmp_path: Path):
    """a2a.py/services.py are ours — re-running must not trip its own output."""
    write(tmp_path / "app" / "app_utils" / "a2a.py", "x = 1\n")
    write(tmp_path / "app" / "app_utils" / "services.py", "x = 1\n")
    assert (
        md.check_legacy_app_utils(tmp_path / "app", LEGACY).status == md.CLEAN
    )


# ---------------------------------------------------------------------------
# Gate: backing infra (selects the outcome, does not stop the run)
# ---------------------------------------------------------------------------


def test_infra_directory_downgrades_to_containerized(tmp_path: Path):
    (tmp_path / "app").mkdir()
    (tmp_path / "infra").mkdir()
    check = md.check_backing_infra(tmp_path, tmp_path / "app")
    assert check.status == md.REPORT_ONLY


def test_backing_service_import_detected(tmp_path: Path):
    write(tmp_path / "app" / "db.py", "import asyncpg\n")
    check = md.check_backing_infra(tmp_path, tmp_path / "app")
    assert check.status == md.REPORT_ONLY
    assert any("PostgreSQL" in r for r in check.details["reasons"])


def test_plain_recipe_is_deployable(tmp_path: Path):
    write(tmp_path / "app" / "agent.py", "root_agent = 1\n")
    assert md.check_backing_infra(tmp_path, tmp_path / "app").status == md.CLEAN


# ---------------------------------------------------------------------------
# Dependency patching
# ---------------------------------------------------------------------------


def _doc(deps: list[str]):
    import tomlkit

    body = ",\n".join(f'    "{d}"' for d in deps)
    return tomlkit.parse(
        f'[project]\nname = "x"\ndependencies = [\n{body}\n]\n'
    )


def test_missing_extras_are_merged_but_version_is_not_touched():
    """product-search pins bare `google-adk>=2.2.0`. The extras carry the
    OTel/GCP code the generated serving files import, so they must be added —
    but the version bound is the owner's deliberate choice and must survive.
    """
    doc = _doc(["google-adk>=2.2.0"])
    check = md.patch_dependencies(
        doc, ["google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0"], apply=True
    )
    kept = check.details["kept"][0]
    assert kept["missing_extras"] == ["gcp", "otel-gcp"]
    # The recipe's own pin must be reported, never the policy's.
    assert kept["recipe_has"] == "google-adk>=2.2.0"
    assert "google-adk[gcp,otel-gcp]>=2.2.0" in str(doc)
    # The policy's floor must NOT have been imposed on the recipe.
    assert ">=2.6.0" not in str(doc)


def test_extras_merge_preserves_existing_extras_and_markers():
    doc = _doc(["google-adk[eval]>=2.2.0 ; python_version < '3.13'"])
    md.patch_dependencies(
        doc, ["google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0"], apply=True
    )
    out = str(doc)
    assert "eval" in out and "gcp" in out and "otel-gcp" in out
    assert "python_version < " in out  # marker survived the rewrite
    assert ">=2.2.0" in out


def test_requirement_keeps_its_position_in_the_list():
    """Rewriting must not reorder a hand-curated dependency list."""
    doc = _doc(["aaa>=1", "google-adk>=2.2.0", "zzz>=1"])
    md.patch_dependencies(
        doc, ["google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0"], apply=True
    )
    deps = [str(d) for d in doc["project"]["dependencies"]]
    assert deps[0] == "aaa>=1"
    assert deps[1].startswith("google-adk[")
    assert deps[2] == "zzz>=1"


def test_requirement_already_carrying_extras_is_untouched():
    doc = _doc(["google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0"])
    before = str(doc)
    check = md.patch_dependencies(
        doc, ["google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0"], apply=True
    )
    assert check.status == md.CLEAN
    assert str(doc) == before


def test_existing_requirement_is_never_widened():
    doc = _doc(["aiohttp>=3.0"])
    md.patch_dependencies(doc, ["aiohttp>=3.13.4"], apply=True)
    assert "aiohttp>=3.0" in str(doc)
    assert "aiohttp>=3.13.4" not in str(doc)


def test_missing_dependency_is_added():
    doc = _doc(["google-adk>=2.6.0"])
    check = md.patch_dependencies(doc, ["gcsfs>=2024.11.0"], apply=True)
    assert check.status == md.FIXED
    assert "gcsfs>=2024.11.0" in str(doc)


# ---------------------------------------------------------------------------
# agent.py patching
# ---------------------------------------------------------------------------


def test_app_object_appended_with_import(tmp_path: Path):
    agent = write(
        tmp_path / "agent.py",
        """
        from google.adk.agents import Agent

        root_agent = Agent(name="d")
        """,
    )
    check = md.patch_app_object(agent, "app", apply=True)
    assert check.status == md.FIXED
    text = agent.read_text()
    assert "from google.adk.apps import App" in text
    assert 'app = App(root_agent=root_agent, name="app")' in text
    # Must remain parseable, and the App must come after root_agent.
    assert text.index("root_agent =") < text.index("app = App(")


def test_existing_app_is_left_alone(tmp_path: Path):
    agent = write(
        tmp_path / "agent.py",
        """
        from google.adk.apps import App

        root_agent = 1
        app = App(root_agent=root_agent, name="x")
        """,
    )
    before = agent.read_text()
    assert md.patch_app_object(agent, "app", apply=True).status == md.CLEAN
    assert agent.read_text() == before


def test_missing_root_agent_blocks(tmp_path: Path):
    agent = write(tmp_path / "agent.py", "something_else = 1\n")
    assert (
        md.patch_app_object(agent, "app", apply=True).status == md.NEEDS_INPUT
    )


# ---------------------------------------------------------------------------
# manifest.yaml — one-line edit, no reflow
# ---------------------------------------------------------------------------


def test_manifest_edit_is_surgical(tmp_path: Path):
    """A ruamel round-trip re-indents every sequence and wraps long scalars,
    turning a one-key addition into a whole-file restyle."""
    manifest = write(
        tmp_path / "manifest.yaml",
        """
        type: standalone
        language: python
        description: A very long description that a YAML dumper would happily hard-wrap onto a second line if given the chance.
        tags:
          - retail
          - search
        """,
    )
    before = manifest.read_text().splitlines()
    md.patch_manifest_deployable(manifest, infra_clean=True, apply=True)
    after = manifest.read_text().splitlines()

    assert "deployable: true" in after
    assert len(after) == len(before) + 1
    # Every original line survives byte-for-byte.
    for line in before:
        assert line in after


def test_manifest_untouched_when_infra_needed(tmp_path: Path):
    manifest = write(tmp_path / "manifest.yaml", "type: standalone\n")
    before = manifest.read_text()
    check = md.patch_manifest_deployable(
        manifest, infra_clean=False, apply=True
    )
    assert check.status == md.REPORT_ONLY
    assert manifest.read_text() == before


# ---------------------------------------------------------------------------
# Dockerfile rendering
# ---------------------------------------------------------------------------


def test_python_version_comes_from_the_recipe():
    """The vendored template hardcodes 3.12; recipes target 3.11-3.13."""
    out = md.set_python_base_image("FROM python:3.12-slim\n", "3.11")
    assert out == "FROM python:3.11-slim\n"


@pytest.mark.parametrize(
    ("requires", "expected"),
    [
        (">=3.11,<3.14", "3.11"),
        (">=3.12", "3.12"),
        ("", None),
        (None, None),
    ],
)
def test_python_floor_parsing(requires, expected):
    assert md.python_floor_from_requires(requires) == expected


def test_data_dirs_copied_before_uv_sync(tmp_path: Path):
    """Layer order matters: data must land before `uv sync`."""
    dockerfile = "COPY ./app ./app\n\nRUN uv sync --frozen\n"
    out = md.inject_data_dir_copies(
        dockerfile, "app", ["assets", "sample_data"]
    )
    assert out.index("COPY ./assets ./assets") < out.index("RUN uv sync")
    assert "COPY ./sample_data ./sample_data" in out


def test_no_data_dirs_leaves_dockerfile_untouched():
    dockerfile = "COPY ./app ./app\n"
    assert md.inject_data_dir_copies(dockerfile, "app", []) == dockerfile


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(recipe: Path, monkeypatch):
    monkeypatch.setattr(md, "load_policy", lambda _root: _policy())
    monkeypatch.setattr(md, "find_repo_root", lambda _p: recipe)
    report = md.run(
        recipe_dir=recipe,
        apply=False,
        overwrite=False,
        data_dirs=[],
        region="us-east1",
    )
    assert report.outcome == md.OUTCOME_DEPLOYABLE
    assert report.files_written == []
    assert not (recipe / "Dockerfile").exists()


def test_apply_writes_every_required_file(recipe: Path, monkeypatch):
    monkeypatch.setattr(md, "load_policy", lambda _root: _policy())
    monkeypatch.setattr(md, "find_repo_root", lambda _p: recipe)
    md.run(
        recipe_dir=recipe,
        apply=True,
        overwrite=False,
        data_dirs=[],
        region="us-east1",
    )
    for rel in [
        "Dockerfile",
        ".dockerignore",
        "app/fast_api_app.py",
        "app/app_utils/a2a.py",
        "app/app_utils/services.py",
        "app/app_utils/reasoning_engine_adapter.py",
    ]:
        assert (recipe / rel).is_file(), rel
    # Placeholders must be fully substituted.
    text = (recipe / "app" / "fast_api_app.py").read_text()
    assert md.PLACEHOLDER_PACKAGE not in text
    assert md.PLACEHOLDER_PROJECT not in text
    assert "from app.app_utils import services" in text


def test_existing_serving_file_not_clobbered(recipe: Path, monkeypatch):
    monkeypatch.setattr(md, "load_policy", lambda _root: _policy())
    monkeypatch.setattr(md, "find_repo_root", lambda _p: recipe)
    bespoke = write(recipe / "app" / "fast_api_app.py", "# hand written\n")
    md.run(
        recipe_dir=recipe,
        apply=True,
        overwrite=False,
        data_dirs=[],
        region="us-east1",
    )
    assert bespoke.read_text() == "# hand written\n"


def test_blocked_run_writes_nothing(recipe: Path, monkeypatch):
    """A gate must stop before ANY file is written — a half-converted recipe
    is worse than an unconverted one."""
    monkeypatch.setattr(md, "load_policy", lambda _root: _policy())
    monkeypatch.setattr(md, "find_repo_root", lambda _p: recipe)
    write(recipe / "app" / "app_utils" / "telemetry.py", "x = 1\n")
    report = md.run(
        recipe_dir=recipe,
        apply=True,
        overwrite=False,
        data_dirs=[],
        region="us-east1",
    )
    assert report.outcome == md.OUTCOME_BLOCKED
    assert report.files_written == []
    assert not (recipe / "Dockerfile").exists()


def _policy() -> dict:
    return {
        "min_google_adk": MIN_ADK,
        "adk_major_migration_is_manual": True,
        "required_dependencies": [
            "a2a-sdk[http-server]>=1.0,<2",
            "gcsfs>=2024.11.0",
            "google-adk[gcp,otel-gcp]>=2.6.0,<3.0.0",
        ],
        "legacy_app_utils_files": LEGACY,
        "emit_agents_cli_manifest": True,
    }


# ---------------------------------------------------------------------------
# agents-cli-manifest.yaml
#
# The file is functional: agents-cli uses it as the project-root marker and
# reads create_params.deployment_target. So it must be both PRESENT and
# free of invented provenance.
# ---------------------------------------------------------------------------


def test_acli_manifest_omits_fabricated_provenance(tmp_path: Path):
    """A fake acli_version makes the CLI tell the owner to run
    `agents-cli scaffold upgrade` on a project that was never scaffolded."""
    check = md.write_agents_cli_manifest(
        tmp_path,
        project_name="demo",
        agent_directory="app",
        region="us-east1",
        apply=True,
    )
    assert check.status == md.FIXED
    text = (tmp_path / "agents-cli-manifest.yaml").read_text()
    for fabricated in ("acli_version", "generated_at", "base_template"):
        assert f"\n{fabricated}:" not in text


def test_acli_manifest_is_loadable_by_agents_cli_shape(tmp_path: Path):
    """Mirror ProjectConfig.from_dict: the values agents-cli actually reads."""
    from ruamel.yaml import YAML

    md.write_agents_cli_manifest(
        tmp_path,
        project_name="demo",
        agent_directory="horizon",
        region="us-central1",
        apply=True,
    )
    data = YAML(typ="safe").load(
        (tmp_path / "agents-cli-manifest.yaml").read_text()
    )
    assert data["name"] == "demo"
    assert data["agent_directory"] == "horizon"
    assert data["region"] == "us-central1"
    assert data["language"] == "python"
    assert data["create_params"]["deployment_target"] == "cloud_run"
    assert data["create_params"]["is_a2a"] is True


def test_acli_manifest_never_clobbers_an_existing_one(tmp_path: Path):
    existing = write(tmp_path / "agents-cli-manifest.yaml", "name: mine\n")
    check = md.write_agents_cli_manifest(
        tmp_path,
        project_name="other",
        agent_directory="app",
        region="us-east1",
        apply=True,
    )
    assert check.status == md.REPORT_ONLY
    assert existing.read_text() == "name: mine\n"


def test_guidance_filename_follows_what_the_recipe_ships(tmp_path: Path):
    assert md.detect_guidance_filename(tmp_path) == "AGENTS.md"
    write(tmp_path / "GEMINI.md", "x")
    assert md.detect_guidance_filename(tmp_path) == "GEMINI.md"
    write(tmp_path / "AGENTS.md", "x")
    assert md.detect_guidance_filename(tmp_path) == "AGENTS.md"
