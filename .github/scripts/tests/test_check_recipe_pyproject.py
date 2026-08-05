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
"""Unit tests for the project-name rule in check_recipe_pyproject.py.

This script gates real PRs but had no test coverage. These tests pin the
`skills/<vertical>/<solution>` naming behaviour so a future refactor cannot
silently reintroduce the basename-only rule, which would force two verticals
that ship a same-named solution to declare the same distribution name.

The rule is implemented twice on purpose — the read-only validator here and
the comment-preserving auto-fixer in
.agents/skills/align-recipe-pyproject/scripts/align_pyproject.py. The two
implementations are cross-checked at the bottom of this file.
"""

from pathlib import Path

import check_recipe_pyproject as m
import pytest

# Where actions/checkout puts the repo on a GitHub runner. The workflow hands
# the script a repo-relative path, but a developer debugging by hand may pass
# an absolute one, so both must derive the same name.
CI_CHECKOUT = "/home/runner/work/adk-samples/adk-samples"


def _records(capsys) -> list[tuple[str, str]]:
    """Parse emitted `KIND::path::message` lines into (kind, message)."""
    out = capsys.readouterr().out.strip().splitlines()
    return [(ln.split("::", 2)[0], ln.split("::", 2)[2]) for ln in out if ln]


# ---------------------------------------------------------------------------
# expected_project_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Language-namespaced roots — basename wins.
        ("core/python/deep-search", "deep-search"),
        ("contrib/python/financial-advisor", "financial-advisor"),
        # Legacy flat layout still present under core/.
        ("core/rag-vector-search", "rag-vector-search"),
        # Vertical-namespaced root — the vertical is joined in.
        ("skills/retail/product-search", "retail-product-search"),
        ("skills/hr/onboarding", "hr-onboarding"),
        ("skills/finance/month-end-close", "finance-month-end-close"),
        # Absolute paths behave identically: only the last three segments are
        # inspected, so a CI checkout prefix cannot change the answer.
        (
            f"{CI_CHECKOUT}/skills/retail/product-search",
            "retail-product-search",
        ),
        (f"{CI_CHECKOUT}/core/python/deep-search", "deep-search"),
        # A bare directory name has no root to inspect.
        ("product-search", "product-search"),
        # Deeper than <root>/<namespace>/<solution>: the third-from-last
        # segment is not a namespaced root, so this falls back to basename.
        # tools/validate_placement.py rejects this layout anyway.
        ("skills/retail/deep/nested", "nested"),
    ],
)
def test_expected_project_name(path, expected):
    assert m.expected_project_name(Path(path)) == expected


def test_two_verticals_sharing_a_solution_name_do_not_collide():
    """The reason the rule exists: basenames are not unique under skills/."""
    retail = m.expected_project_name(Path("skills/retail/product-search"))
    grocery = m.expected_project_name(Path("skills/grocery/product-search"))
    assert retail != grocery


# ---------------------------------------------------------------------------
# check_name
# ---------------------------------------------------------------------------


def test_skill_with_vertical_prefixed_name_passes(capsys):
    m.check_name(
        {"name": "retail-product-search"},
        Path("skills/retail/product-search/pyproject.toml"),
        Path("skills/retail/product-search"),
    )
    ((kind, msg),) = _records(capsys)
    assert kind == "PASS"
    assert "retail-product-search" in msg


def test_skill_with_bare_basename_fails_and_explains_why(capsys):
    m.check_name(
        {"name": "product-search"},
        Path("skills/retail/product-search/pyproject.toml"),
        Path("skills/retail/product-search"),
    )
    ((kind, msg),) = _records(capsys)
    assert kind == "FAIL"
    assert "retail-product-search" in msg
    # The contributor should not have to go read the script to understand it.
    assert "<vertical>-<solution>" in msg


def test_core_recipe_message_omits_the_vertical_explanation(capsys):
    """core/ and contrib/ messages stay as terse as they were pre-change."""
    m.check_name(
        {"name": "wrong"},
        Path("core/python/deep-search/pyproject.toml"),
        Path("core/python/deep-search"),
    )
    ((kind, msg),) = _records(capsys)
    assert kind == "FAIL"
    assert "deep-search" in msg
    assert "vertical" not in msg


def test_core_recipe_with_matching_basename_passes(capsys):
    m.check_name(
        {"name": "deep-search"},
        Path("core/python/deep-search/pyproject.toml"),
        Path("core/python/deep-search"),
    )
    ((kind, _),) = _records(capsys)
    assert kind == "PASS"


def test_missing_name_reports_the_expected_value(capsys):
    m.check_name(
        {},
        Path("skills/retail/product-search/pyproject.toml"),
        Path("skills/retail/product-search"),
    )
    ((kind, msg),) = _records(capsys)
    assert kind == "FAIL"
    assert "is missing" in msg
    assert "retail-product-search" in msg


# ---------------------------------------------------------------------------
# Cross-implementation consistency
#
# check_recipe_pyproject.py (validate) and align_pyproject.py (auto-fix) each
# implement this rule independently and both carry a "keep in sync" note. A
# drift between them is worse than either being wrong alone: the skill would
# cheerfully rewrite [project].name to a value CI then rejects.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "core/python/deep-search",
        "contrib/python/financial-advisor",
        "core/rag-vector-search",
        "skills/retail/product-search",
        "skills/hr/onboarding",
        "product-search",
        "skills/retail/deep/nested",
    ],
)
def test_validator_and_autofixer_agree(align_pyproject, path):
    assert m.expected_project_name(
        Path(path)
    ) == align_pyproject.expected_project_name(Path(path))


def test_validator_and_autofixer_share_the_same_namespaced_roots(
    align_pyproject,
):
    """The two could agree on every sampled path and still diverge on a root
    only one of them knows about."""
    assert m.NAMESPACED_ROOTS == align_pyproject.NAMESPACED_ROOTS
