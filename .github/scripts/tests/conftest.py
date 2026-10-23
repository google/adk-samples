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
"""Make .github/scripts importable from its tests.

The scripts are invoked by CI as standalone files (`python
.github/scripts/foo.py`), not as an installed package, so there is no
importable path to them by default. This shim mirrors the one each skill
keeps in .agents/skills/*/tests/conftest.py.

Only this directory is added to sys.path. A test needing a module from
elsewhere in the repo — such as the drift check against the
align-recipe-pyproject skill — gets it from the `align_pyproject` fixture
below, which loads from an explicit file path without mutating global import
state.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent
REPO_ROOT = SCRIPTS_DIR.parent.parent

sys.path.insert(0, str(SCRIPTS_DIR))

ALIGN_PYPROJECT_PATH = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "align-recipe-pyproject"
    / "scripts"
    / "align_pyproject.py"
)


def _load_module_by_path(name: str, path: Path) -> ModuleType:
    """Import a module from an explicit file path.

    Used instead of appending another directory to sys.path, for two reasons:

    1. It keeps the import local to the one test that needs it. Adding a
       skill's scripts/ directory globally would let any test here import it
       by accident, and would put two directories with independently-chosen
       module names on one path — a collision waiting to happen as skills
       are added.
    2. It respects the skills' "self-contained, portable bundle" property.
       Reaching into one becomes a deliberate, visible act rather than an
       ambient side effect of collecting these tests.

    The module is NOT registered in sys.modules, so nothing else can pick it
    up implicitly.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def align_pyproject() -> ModuleType:
    """The align-recipe-pyproject skill's script.

    check_recipe_pyproject.py validates four rules that align_pyproject.py
    auto-fixes; both files carry a "MUST stay semantically in sync" note. The
    drift tests turn that note into an executed assertion, which needs both
    modules loaded at once.
    """
    if not ALIGN_PYPROJECT_PATH.is_file():
        pytest.skip(f"{ALIGN_PYPROJECT_PATH} not present")
    return _load_module_by_path("align_pyproject", ALIGN_PYPROJECT_PATH)
