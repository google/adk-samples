# Copyright 2025 Google LLC
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

"""Live-tier environment for tests/integration (Q19 repair, stage 3a).

The root conftest pins FAKE env at session start (unit/e2e isolation) and
imports app.config at collection time. The live tier needs the REAL
environment from app/.env — re-applied per test, with app.config reloaded so
module-level values match, and the import-time baseline restored afterwards
so fast-tier tests in the same process never see live config.

Storage policy (Phase 16 open item 2): live tier is LOCAL-FIRST —
GCS_BUCKET is force-unset even when the developer's app/.env sets it.
"""

import importlib
import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

import app.config as config_module
from tests._config_baseline import restore_config_baseline
from tests.conftest import _copy_main_db_to_temp

ENV_FILE = Path(__file__).resolve().parents[2] / "app" / ".env"

# Keys force-REMOVED from the environment in the live tier.
# Local-first storage: never GCS in the live tier (open item 2).
LIVE_ENV_UNSET = ("GCS_BUCKET",)


def resolve_live_env() -> dict[str, str]:
    """Validated live-tier env mapping from app/.env (shared: fixture + CLI).

    Returns the exact key/value pairs to SET; callers must also delete the
    LIVE_ENV_UNSET keys. Raises RuntimeError on a missing/unusable app/.env.
    """
    if not ENV_FILE.exists():
        raise RuntimeError(
            "app/.env missing — live tier requires real credentials"
        )
    real = {k: v for k, v in dotenv_values(ENV_FILE).items() if v}
    # GOOGLE_CLOUD_PROJECT is not surfaced as an app.config attribute (the
    # google-genai/Vertex SDK clients read it straight from os.environ), so
    # validate it here rather than on config_module (verified against
    # app/config.py — no such attribute exists).
    project = real.get("GOOGLE_CLOUD_PROJECT")
    if not project or project == "test-project":
        raise RuntimeError(
            "app/.env must set a real GOOGLE_CLOUD_PROJECT for the live tier"
        )
    env = dict(real)
    # Gemini 3.x models need the global endpoint (CLAUDE.md gotcha).
    env["GOOGLE_CLOUD_LOCATION"] = real.get("GOOGLE_CLOUD_LOCATION", "global")
    env["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    for key in LIVE_ENV_UNSET:
        env.pop(key, None)
    return env


def apply_live_env_to_process() -> None:
    """os.environ-level bootstrap for non-pytest entry points (recorder CLI).

    Same env as the live_real_environment fixture, but applied directly to
    the process with no undo — only for one-shot CLI runs.
    """
    os.environ.update(resolve_live_env())
    for key in LIVE_ENV_UNSET:
        os.environ.pop(key, None)
    importlib.reload(config_module)
    assert config_module.GCS_BUCKET is None


@pytest.fixture(autouse=True)
def live_real_environment(monkeypatch):
    """Real Vertex env from app/.env for the duration of one test."""
    if not ENV_FILE.exists():
        pytest.skip("app/.env missing — live tier requires real credentials")
    try:
        real = resolve_live_env()
    except RuntimeError as e:
        pytest.fail(str(e))
    for key, value in real.items():
        monkeypatch.setenv(key, value)
    for key in LIVE_ENV_UNSET:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(config_module)
    assert config_module.GCS_BUCKET is None
    yield
    monkeypatch.undo()
    restore_config_baseline()


@pytest.fixture(autouse=True)
def isolated_live_db(live_real_environment, monkeypatch):
    """Live agent runs execute REAL tools that mutate the DB — use a copy."""
    db_path = _copy_main_db_to_temp()
    monkeypatch.setattr(config_module, "DB_PATH", db_path)
    yield db_path
    try:
        os.unlink(db_path)
    except OSError:
        pass
