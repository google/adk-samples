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
"""Pin the security properties of the two agy-driven workflows.

These workflows run a model over input an anonymous contributor wrote — a
fork PR's diff, an issue body — on a runner that holds a Google Cloud
credential and, in one job, a token that can write to the repository. What
keeps that safe is a handful of small facts spread across two YAML files:
an empty tool allowlist, a job that cannot write, a job that cannot reach
the cloud.

Every one of those facts is a line someone can plausibly "fix" later for a
good-sounding local reason — a review that failed on a denied tool, a job
that would be simpler merged back into one. None of them is covered by
ruff, by actionlint, or by any test of the scripts themselves, so without
this file the entire remediation for b/555419958 can be undone silently and
every check stays green.

The assertions are deliberately about PROPERTIES, not about exact text, so
ordinary edits to these workflows do not trip them.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"

PR_REVIEW = WORKFLOWS / "_ai-pr-review-core.yml"
ISSUE_TRIAGE = WORKFLOWS / "_ai-issue-triage-core.yml"

# Both workflows that hand an untrusted string to an agy agent.
AGENT_WORKFLOWS = [PR_REVIEW, ISSUE_TRIAGE]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps(path: Path, job: str) -> list[dict]:
    return _load(path)["jobs"][job]["steps"]


def _code(script: str) -> str:
    """A `run:` block with its comment lines removed.

    These workflows are heavily commented and the comments discuss the very
    things these tests search for — "the agy call", "cat agy_result.json".
    Matching against raw text finds the prose and reports a command that is
    not there, so every search below runs on code only.
    """
    return "\n".join(
        line
        for line in script.splitlines()
        if not line.lstrip().startswith("#")
    )


def _run_blocks(path: Path, job: str | None = None) -> str:
    """Every `run:` script in the workflow (or one job), comments stripped.

    `steps` is read defensively because a job that calls a reusable workflow
    has `uses:` and no steps at all. Such a job contributes no shell, and it
    should not turn an invariant check into a KeyError that says nothing
    about the invariant.
    """
    doc = _load(path)
    jobs = [doc["jobs"][job]] if job else list(doc["jobs"].values())
    return "\n".join(
        _code(step["run"])
        for j in jobs
        for step in j.get("steps") or []
        if isinstance(step.get("run"), str)
    )


def _agy_settings(path: Path) -> dict:
    """The settings.json the workflow writes for agy, parsed.

    Pulled out of the heredoc and JSON-parsed rather than matched as text, so
    reformatting the block does not break the test but changing what it GRANTS
    does. Found by searching every job rather than by naming one, so renaming
    or moving the job does not quietly stop this from checking anything.
    """
    script = next(
        step["run"]
        for job in _load(path)["jobs"].values()
        for step in job.get("steps") or []
        if "antigravity-cli/settings.json" in str(step.get("run", ""))
    )
    body = script.split("<<'SETTINGS_EOF'", 1)[1].split("SETTINGS_EOF", 1)[0]
    return json.loads(body)


# --------------------------------------------------------------------------
# The tool grant — the actual boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", AGENT_WORKFLOWS, ids=lambda p: p.name)
def test_the_agy_tool_allowlist_is_empty(path):
    """The one that matters.

    b/555419958 was reachable because this list held `read_file(*)`,
    `write_file(*)` and `command(cat)` for a step whose own comment said it
    needed no tool. $GOOGLE_APPLICATION_CREDENTIALS is an absolute path
    outside the workspace and the file at it is on its own enough to
    impersonate the service account, so ANY read primitive here is a
    credential disclosure waiting for someone to write the right diff.

    If a review starts failing with "produced no response", fix the prompt.
    Do not make this list non-empty.
    """
    assert _agy_settings(path)["permissions"]["allow"] == []


@pytest.mark.parametrize("path", AGENT_WORKFLOWS, ids=lambda p: p.name)
def test_the_agent_result_is_never_echoed_whole_to_the_log(path):
    """A public repository's workflow log is world-readable, and live.

    Printing the raw response puts model text — derived from attacker-written
    input — on a public channel that GitHub's secret masking does not cover,
    because the credential on this runner is not a registered secret.
    """
    assert "cat agy_result.json" not in _run_blocks(path)


# --------------------------------------------------------------------------
# The job split — no runner holds both halves
# --------------------------------------------------------------------------

WRITE_SCOPES = {"write", "admin"}


def _writes(permissions) -> set[str]:
    if not isinstance(permissions, dict):
        # `write-all`, or inherited — either way, not what we want here.
        return {str(permissions)}
    return {k for k, v in permissions.items() if v in WRITE_SCOPES}


def test_the_job_that_runs_the_model_cannot_write():
    """`review` is the job an injected diff could plausibly reach.

    `id-token: write` is the exception and is not a repository write: it
    mints the OIDC token for Workload Identity Federation. Everything that
    mutates the pull request belongs to `post`.
    """
    assert _writes(_load(PR_REVIEW)["jobs"]["review"]["permissions"]) == {
        "id-token"
    }


def test_the_job_that_can_write_holds_no_cloud_credential():
    """`post` must not authenticate to Google Cloud or run a model.

    Checked structurally — no `id-token`, no auth action, no agy — rather
    than by name, so merging the model call back into this job trips it
    however it is spelled.
    """
    post = _load(PR_REVIEW)["jobs"]["post"]
    assert "id-token" not in (post["permissions"] or {})

    assert not [
        s
        for s in post["steps"]
        if "google-github-actions/auth" in s.get("uses", "")
    ]
    assert "agy " not in _run_blocks(PR_REVIEW, "post")


def test_repo_code_run_after_the_agent_is_integrity_checked():
    """The one step that still executes the checkout in the agent's own job.

    Splitting the posting job off removed the report's "write to a file the
    pipeline executes seconds later" vector from `post`, but `build_review`
    runs `.github/scripts/post_review_comments.py` in the SAME job as the
    agent and after it. With an empty tool allowlist nothing can write there;
    this pins the check that catches it if something can.

    `--porcelain` and not `git diff`: python puts a script's directory on
    sys.path, so a NEW untracked `.github/scripts/json.py` shadows the stdlib
    without modifying any tracked file, and `git diff` would not see it.
    """
    # Comments stripped first: the block above the gate explains itself by
    # naming post_review_comments.py, and partitioning on the raw text would
    # split at the prose rather than at the invocation.
    script = _code(
        next(
            s["run"]
            for s in _steps(PR_REVIEW, "review")
            if s.get("id") == "build_review"
        )
    )
    gate, _, invocation = script.partition("post_review_comments.py")

    assert "git status --porcelain -- .github/scripts" in gate, (
        "the integrity check must run BEFORE the script it protects"
    )
    assert "exit 1" in gate
    assert invocation, "build_review no longer invokes the script"


def test_the_job_that_can_write_takes_no_checkout():
    """So there is no script on its disk for a written file to become.

    The report's code-execution half turned on the agent writing to
    `.github/scripts/post_review_comments.py`, which the next step executed.
    That half was real: `write_file` was tested against this agy build and is
    NOT confined to the agent's working directory — it wrote to an absolute
    path outside it. A posting job with no checkout is what makes the vector
    structurally unavailable rather than merely denied.
    """
    assert not [
        s
        for s in _steps(PR_REVIEW, "post")
        if "actions/checkout" in s.get("uses", "")
    ]


def test_the_model_and_the_posting_step_are_in_different_jobs():
    """The property the two tests above are each half of."""
    jobs = _load(PR_REVIEW)["jobs"]
    runs_agy = {n for n in jobs if "agy -p" in _run_blocks(PR_REVIEW, n)}
    posts = {
        n
        for n in jobs
        if "--method POST" in _run_blocks(PR_REVIEW, n)
        and "/reviews" in _run_blocks(PR_REVIEW, n)
    }

    # Both non-empty first: an assertion that the two sets are disjoint is
    # trivially satisfied if a rename made one of them empty, which would
    # turn this test green at exactly the moment it stopped checking.
    assert runs_agy, "no job runs agy — has the invocation been renamed?"
    assert posts, "no job posts a review — has the endpoint changed?"
    assert runs_agy.isdisjoint(posts)


# --------------------------------------------------------------------------
# The job graph
#
# Splitting one job into three is the largest behavioural change here, and its
# correctness lives entirely in `needs:` and two `if:` expressions. Nothing
# else in the suite touches them, and the failure they produce is silent: a
# review that is built and then never posted looks exactly like a clean PR.
# These pin the wiring, not the wording.
# --------------------------------------------------------------------------


def _needs(job: str) -> set[str]:
    n = _load(PR_REVIEW)["jobs"][job].get("needs") or []
    return {n} if isinstance(n, str) else set(n)


def test_the_jobs_are_wired_in_order():
    assert _needs("review") == set(), "review starts the chain"
    assert _needs("post") == {"review"}
    assert _needs("notify") == {"review", "post"}


def test_post_runs_for_both_reasons_it_exists_and_no_others():
    """Oversize PRs and real findings. A clean PR must not spin a runner.

    Both operands matter. Dropping `skip` silently stops the too-large-to-diff
    comment ever being posted, since the job that works that out holds a
    read-only token and cannot say it.
    """
    cond = " ".join(_load(PR_REVIEW)["jobs"]["post"]["if"].split())
    assert "needs.review.outputs.skip == 'true'" in cond
    assert "needs.review.outputs.has_payload == 'true'" in cond
    assert "||" in cond, "the two reasons are alternatives, not both required"


def test_notify_fires_on_either_job_failing():
    """`always()` plus explicit results, not a bare `failure()`.

    When `review` fails, `post` is SKIPPED rather than failed. A condition
    written only against `post` would stay silent on exactly the failure the
    contributor most needs told about.
    """
    cond = " ".join(_load(PR_REVIEW)["jobs"]["notify"]["if"].split())
    assert "always()" in cond
    assert "needs.review.result == 'failure'" in cond
    assert "needs.post.result == 'failure'" in cond


def test_every_output_post_and_notify_consume_is_actually_produced():
    """A typo in an output name is an empty string, not an error.

    `needs.review.outputs.has_paylod` would evaluate to '' forever, `post`
    would never run, and every review would vanish with all checks green.
    """
    doc = _load(PR_REVIEW)
    produced = set(doc["jobs"]["review"].get("outputs") or {})

    # Walked, not serialised. `yaml.dump` wraps at 80 columns, so a long
    # enough expression would be folded mid-token and the regex would quietly
    # stop seeing a consumed output — this test would then pass by failing to
    # look, which is the exact failure it exists to prevent elsewhere.
    consumed: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, str):
            consumed.update(
                re.findall(r"needs\.review\.outputs\.([A-Za-z0-9_]+)", node)
            )
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for job in ("post", "notify"):
        walk(doc["jobs"][job])

    assert consumed, (
        "no job reads the review job's outputs — is it still split?"
    )
    assert consumed <= produced, (
        f"consumed but never produced: {consumed - produced}"
    )


def test_the_payload_artifact_name_matches_between_upload_and_download():
    """Different names fail at download time, on every single review.

    Paired against the upload that carries the PAYLOAD specifically, not
    against every upload in the job. `review` also uploads a dry-run bundle,
    and a download pointed at that would still be a name that exists — so a
    subset check alone would wave through a swap that breaks every real run.
    The payload upload is identified by its gate, which is the same
    `has_payload` condition the download uses.
    """

    def gated_on_payload(step):
        return "has_payload" in str(step.get("if", ""))

    uploads = {
        s["with"]["name"]
        for s in _steps(PR_REVIEW, "review")
        if "actions/upload-artifact" in s.get("uses", "")
        and gated_on_payload(s)
    }
    downloads = {
        s["with"]["name"]
        for s in _steps(PR_REVIEW, "post")
        if "actions/download-artifact" in s.get("uses", "")
    }

    assert len(uploads) == 1, f"expected one payload upload, got {uploads}"
    assert downloads == uploads, (
        f"post downloads {downloads}, review uploads {uploads}"
    )


# --------------------------------------------------------------------------
# Caller/callee permission ceiling
# --------------------------------------------------------------------------

# GitHub's ordering for a permission scope. A caller must grant at least what
# the called workflow asks for; a reusable workflow can narrow, never widen.
LEVELS = {"none": 0, "read": 1, "write": 2}


def _effective(workflow: dict, job: dict) -> dict:
    """What a calling job actually holds: its own block, else the file's."""
    perms = job.get("permissions")
    if perms is None:
        perms = workflow.get("permissions")
    return perms if isinstance(perms, dict) else {}


def test_no_job_asks_for_more_than_every_caller_grants():
    """A reusable workflow cannot widen its caller's grant.

    Ask for a scope the caller withheld and the run dies before any step,
    with "The workflow is requesting 'x: write', but is only allowed
    'x: none'" — on every pull request, in all four lanes at once. Nothing
    else catches this: actionlint validates each file alone, and the two
    sides live in six different files.

    This is not hypothetical bookkeeping. While splitting these jobs I
    considered giving `post` an extra scope for `download-artifact` before
    establishing it needed none; had it been one the lanes do not grant, it
    would have taken the whole reviewer down.
    """
    callers: dict[str, list[tuple[str, dict]]] = {}
    for path in WORKFLOWS.glob("*.yml"):
        doc = _load(path)
        for job_name, job in (doc.get("jobs") or {}).items():
            uses = str(job.get("uses") or "")
            if uses.startswith("./.github/workflows/"):
                callee = uses.rsplit("/", maxsplit=1)[-1]
                callers.setdefault(callee, []).append(
                    (f"{path.name}:{job_name}", _effective(doc, job))
                )

    for callee_name in (PR_REVIEW.name, ISSUE_TRIAGE.name):
        assert callers.get(callee_name), f"no caller found for {callee_name}"
        callee = _load(WORKFLOWS / callee_name)

        for job_name, job in callee["jobs"].items():
            # A job with no block of its own inherits the callee file's
            # workflow-level grant, and that inherited set is what the caller
            # has to cover. Skipping such a job would leave the issue-triage
            # core — whose single job declares nothing — entirely unguarded.
            requested = _effective(callee, job)
            for scope, level in requested.items():
                for caller_id, granted in callers[callee_name]:
                    have = LEVELS.get(str(granted.get(scope, "none")), 0)
                    want = LEVELS.get(str(level), 0)
                    assert have >= want, (
                        f"{callee_name}:{job_name} asks for {scope}:{level}, "
                        f"but {caller_id} grants {scope}:"
                        f"{granted.get(scope, 'none')}"
                    )


# --------------------------------------------------------------------------
# The credential scan
# --------------------------------------------------------------------------


def test_the_response_is_scanned_against_real_credential_material():
    """Not a keyword list, and not the whole ADC file either.

    A keyword list is one rewording away from useless. The whole ADC file is
    the opposite failure: `.type` is "external_account" and `.token_url` is
    the public STS endpoint, and this repository is full of Google Cloud
    recipes whose diffs legitimately contain both — which the reviewer then
    quotes verbatim into `window`. Scanning for those would fail honest PRs
    and cry security incident. Only the secret-bearing fields belong here.
    """
    script = next(
        s["run"]
        for s in _steps(PR_REVIEW, "review")
        if isinstance(s.get("run"), str) and "NEEDLES=" in s["run"]
    )
    assert "ACTIONS_ID_TOKEN_REQUEST_TOKEN" in script
    assert ".credential_source" in script

    # The config fields that must NOT become needles.
    jq_filter = script.split("jq -r '[", 1)[1].split("]", 1)[0]
    for public in (".type", ".token_url", ".audience", ".quota_project_id"):
        assert public not in jq_filter, f"{public} is not secret"


def test_a_response_carrying_the_credential_fails_the_job():
    """The scan must stop the run, not filter and carry on.

    If the response contains this runner's credential, the tool allowlist did
    not hold, and nothing downstream is trustworthy enough to sort the good
    findings from the bad.
    """
    script = next(
        s["run"]
        for s in _steps(PR_REVIEW, "review")
        if isinstance(s.get("run"), str) and "grep -qFf" in s["run"]
    )
    scan = script.split("grep -qFf", 1)[1]
    assert "exit 1" in scan


# --------------------------------------------------------------------------
# The prompt's untrusted-input guard
#
# Hygiene, not a boundary — but the gap it closes is the one the report led
# with, and a silent regression would put the file back where it started.
# --------------------------------------------------------------------------


def test_the_diff_is_marked_untrusted_in_the_prompt():
    """The section itself, not a mention of it.

    A substring check for "## Untrusted input" passes on the cross-reference
    inside the PR Context header — `(ALL UNTRUSTED — see "## Untrusted
    input")` — so deleting the actual section would leave it green. Matching
    a whole line is what distinguishes the heading from the pointer to it.
    """
    script = next(
        s["run"]
        for s in _steps(PR_REVIEW, "review")
        if isinstance(s.get("run"), str) and "## PR Context" in s["run"]
    )
    headings = {line.strip() for line in script.splitlines()}
    assert "## Untrusted input" in headings, (
        "the prompt's untrusted-input section is gone (a reference to it in "
        "another line does not count)"
    )

    diff_header = next(
        line for line in script.splitlines() if "Complete unified diff" in line
    )
    assert "UNTRUSTED" in diff_header
