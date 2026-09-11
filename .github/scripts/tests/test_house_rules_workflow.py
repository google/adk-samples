"""Pin the four properties that make the house-rules lane safe.

It is the only AI-review workflow that checks out the pull request's own code.
That is a deliberate trade, and it holds only while all four of these are true:

  1. the job that sees PR code holds no secret and no write token
  2. nothing from the PR is executed, imported or installed
  3. the checker comes from the base checkout, not the PR's copy
  4. posting happens in a job that never saw the PR's code

Each is one edit away from being false, and none of them would fail loudly:
a workflow that adds `uv sync` in the wrong directory works perfectly right up
until someone points a hostile PR at it. So they are asserted here rather than
described in a comment nobody re-reads.
"""

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "ai-pr-review-house-rules.yml"
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW.exists(), f"{WORKFLOW.name} is gone; delete these tests too"
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _steps(job: dict) -> list[dict]:
    return [s for s in job.get("steps", []) if isinstance(s, dict)]


def _run_scripts(job: dict) -> str:
    return "\n".join(str(s.get("run", "")) for s in _steps(job))


# ------------------------------------------------------- 1. no credentials


def test_the_checking_job_cannot_post(workflow):
    perms = workflow["jobs"]["check"]["permissions"]
    assert perms.get("pull-requests") == "read", (
        "the job that reads PR code must not be able to write to the PR"
    )
    assert perms.get("contents") == "read"
    assert "issues" not in perms, "issues: write would let it comment"


def test_the_checking_job_holds_no_secret(raw):
    """No `secrets.` reference anywhere in the check job.

    Checked as text over the job's span rather than per-step: a secret can
    enter through `env:`, `with:`, or an action input, and enumerating the
    places it could hide is how one gets missed.
    """
    check_span = raw.split("jobs:", 1)[1].split("\n  post:", 1)[0]
    leaked = re.findall(r"secrets\.[A-Za-z_]+", check_span)
    assert not leaked, (
        f"the check job references {sorted(set(leaked))}. It sees untrusted "
        "code; it must hold nothing worth stealing."
    )


def test_the_top_level_grants_nothing(workflow):
    assert workflow["permissions"] == {}, (
        "a top-level grant is handed to every job, including the one that "
        "reads the pull request's code"
    )


# --------------------------------------------- 2. nothing from the PR runs

# Anything that would execute, import or install code out of the checkout.
FORBIDDEN_IN_CHECK = (
    "uv sync",
    "uv run",
    "uv pip install -r",
    "pip install -e",
    "pip install -r",
    "poetry install",
    "npm install",
    "npm ci",
    "make ",
    "docker build",
    "pytest",
)


def test_the_check_job_never_runs_anything_from_the_pr(workflow):
    script = _run_scripts(workflow["jobs"]["check"])
    for forbidden in FORBIDDEN_IN_CHECK:
        assert forbidden not in script, (
            f"{forbidden!r} appears in the check job. It has a checkout of "
            "untrusted code; running a build or a test out of it hands a "
            "hostile PR arbitrary execution."
        )


def test_the_only_install_is_pinned_and_from_pypi(workflow):
    """PyYAML by exact version, from the index — never from the PR's own
    pyproject.toml, lockfile or vendored wheels."""
    script = _run_scripts(workflow["jobs"]["check"])
    installs = re.findall(r"pip install[^\n]*", script)
    assert len(installs) == 1, f"expected exactly one install, got {installs}"
    assert re.search(r'"pyyaml==\d+\.\d+(\.\d+)?"', installs[0]), (
        f"the install is not a pinned PyYAML: {installs[0]!r}"
    )
    assert "pr-head" not in installs[0]


def test_the_pr_checkout_carries_no_credentials(workflow):
    for step in _steps(workflow["jobs"]["check"]):
        if "actions/checkout" in str(step.get("uses", "")):
            assert step["with"]["persist-credentials"] is False, (
                "a token left in .git/config is readable by anything that "
                "runs in that directory"
            )


# ------------------------------------------ 3. the checker is the base copy


def test_the_checker_comes_from_the_base_checkout(workflow):
    script = _run_scripts(workflow["jobs"]["check"])
    assert "--checker base/.agents/skills/github-pr-review" in script, (
        "the checker must be the base branch's copy. Reading it out of "
        "pr-head/ would let a PR rewrite the script that reviews it."
    )
    assert "--checker pr-head" not in script


def test_the_pr_tree_is_only_ever_the_subject(workflow):
    """`pr-head` may be passed as data (--repo-root) and never invoked."""
    script = _run_scripts(workflow["jobs"]["check"])
    for match in re.findall(r"python3?\s+(\S+)", script):
        assert not match.startswith("pr-head"), (
            f"running {match!r} executes code from the pull request"
        )


def test_both_checkouts_are_separate_directories(workflow):
    paths = [
        step["with"]["path"]
        for step in _steps(workflow["jobs"]["check"])
        if "actions/checkout" in str(step.get("uses", ""))
    ]
    assert paths == ["base", "pr-head"], (
        f"expected a base and a pr-head checkout, got {paths}. Checking the "
        "PR out over the base would substitute the PR's checker for ours."
    )


# ---------------------------------------------------- 4. posting is split


def test_posting_happens_in_a_job_with_no_checkout(workflow):
    post = workflow["jobs"]["post"]
    assert post["permissions"].get("pull-requests") == "write"
    for step in _steps(post):
        assert "actions/checkout" not in str(step.get("uses", "")), (
            "the posting job holds a write token; it must never have the "
            "pull request's code in front of it"
        )


def test_the_post_job_waits_for_the_check_job(workflow):
    assert workflow["jobs"]["post"]["needs"] == ["check"]


def test_a_dry_run_posts_nothing(workflow):
    script = _run_scripts(workflow["jobs"]["post"])
    assert "if [[ \"${DRY_RUN}\" == 'true' ]]; then" in script
    body = script.split("DRY_RUN", 1)[1]
    assert body.index("exit 0") < body.index("--method POST"), (
        "the dry-run branch must return before anything is posted"
    )
