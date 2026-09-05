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
"""Pin the security and correctness properties of recipe-images.yml.

This workflow builds container images from Dockerfiles that community
contributors write, and is intended to push them to a PUBLIC registry under
Google's name.

What these tests are NOT for: generic GitHub Actions anti-patterns. An
org-level required workflow runs zizmor over every workflow here — it is not
a file in this repository, which makes it easy to miss when grepping — and it
already covers unpinned actions, over-broad permissions and the like.

What they ARE for is the repository-specific properties zizmor cannot know:
that publishing is gated on three named variables, that the build runs before
authentication, that the trigger filters agree with publish_matrix.py's path
constants, that images are tagged by SHA alone. Each is a line someone could
plausibly "fix" later for a good-sounding local reason, and ruff does not read
YAML.

The assertions are about PROPERTIES rather than exact text, so ordinary edits
to the workflow do not trip them.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "recipe-images.yml"


@pytest.fixture(scope="module")
def doc() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def triggers(doc: dict) -> dict:
    # PyYAML parses the unquoted key `on:` as the boolean True, since YAML 1.1
    # treats `on` as a truthy scalar. Accept whichever key survives so this
    # does not break if the workflow ever quotes it.
    return doc.get("on") or doc[True]


def _steps(doc: dict, job: str) -> list[dict]:
    return doc["jobs"][job]["steps"]


def _run_blocks(doc: dict) -> list[tuple[str, str]]:
    out = []
    for job_name, job in doc["jobs"].items():
        for step in job.get("steps") or []:
            if "run" in step:
                out.append((f"{job_name}:{step.get('name', '?')}", step["run"]))
    return out


def test_the_workflow_parses():
    assert yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Fork safety
# --------------------------------------------------------------------------


def test_it_never_uses_pull_request_target(triggers):
    """pull_request_target runs with repository credentials in scope of code
    the contributor wrote. For a workflow that builds contributor Dockerfiles
    it is the single most dangerous trigger available."""
    assert "pull_request_target" not in triggers


def test_it_builds_on_pull_request(triggers):
    """A first build discovered on main is a build nobody is watching."""
    assert "pull_request" in triggers


def test_push_is_restricted_to_main(triggers):
    assert triggers["push"]["branches"] == ["main"]


# --------------------------------------------------------------------------
# Publishing is gated
# --------------------------------------------------------------------------


def test_every_push_step_is_gated_on_the_publish_decision(doc):
    """Nothing may reach the registry unless the mode step said so.

    This is what keeps the workflow inert while the registry variables are
    unset, and what keeps a pull request — including one from a fork — from
    publishing anything.
    """
    for step in _steps(doc, "build"):
        run = step.get("run", "")
        uses = str(step.get("uses", ""))
        touches_registry = (
            "docker push" in run
            or "docker login" in run
            or "google-github-actions/auth" in uses
        )
        if touches_registry:
            assert "steps.mode.outputs.publish == 'true'" in step.get(
                "if", ""
            ), f"unguarded registry step: {step.get('name')}"


def test_the_publish_decision_requires_all_three_registry_vars(doc):
    """A half-configured registry must not half-publish."""
    mode = next(s for s in _steps(doc, "build") if s.get("id") == "mode")
    for var in (
        "RECIPE_IMAGE_REGISTRY",
        "RECIPE_IMAGE_WIF_PROVIDER",
        "RECIPE_IMAGE_SA",
    ):
        assert var in str(mode.get("env", {})), f"{var} not consulted"
    assert "PUBLISH=false" in mode["run"]


def test_the_publish_decision_requires_a_push_to_main(doc):
    mode = next(s for s in _steps(doc, "build") if s.get("id") == "mode")
    assert "refs/heads/main" in mode["run"]
    assert '"$EVENT_NAME" != "push"' in mode["run"]


# --------------------------------------------------------------------------
# Credential hygiene
# --------------------------------------------------------------------------


def test_the_build_happens_before_authentication(doc):
    """A credential acquired before the build is a credential the build
    context could capture. Checkout still comes first, because auth writes
    into the workspace that a later checkout would clobber."""
    names = [s.get("name", "") for s in _steps(doc, "build")]
    order = {n: i for i, n in enumerate(names)}
    build = next(i for n, i in order.items() if n.startswith("Build image"))
    auth = next(i for n, i in order.items() if n.startswith("Authenticate"))
    checkout = next(i for n, i in order.items() if n.startswith("Checkout"))
    assert checkout < build < auth


def test_auth_writes_no_credential_file(doc):
    """A credential FILE in the workspace can be swept into a layer by a
    COPY. A short-lived token in the step environment cannot."""
    auth = next(
        s
        for s in _steps(doc, "build")
        if "google-github-actions/auth" in str(s.get("uses", ""))
    )
    assert auth["with"]["create_credentials_file"] is False
    assert auth["with"]["token_format"] == "access_token"


def test_the_registry_token_is_passed_on_stdin(doc):
    """Not as a `docker login -p` argument, where it would appear in a
    process listing and in any `set -x` trace."""
    push = next(
        s for s in _steps(doc, "build") if "docker push" in s.get("run", "")
    )
    assert "--password-stdin" in push["run"]
    assert "-p " not in push["run"]


def test_checkout_never_persists_credentials(doc):
    for job_name, job in doc["jobs"].items():
        for step in job.get("steps") or []:
            if "actions/checkout" in str(step.get("uses", "")):
                assert (
                    step.get("with", {}).get("persist-credentials") is False
                ), f"{job_name} checkout persists credentials"


# --------------------------------------------------------------------------
# Repository conventions
# --------------------------------------------------------------------------


def test_no_github_expression_reaches_a_run_block(doc):
    """The repo-wide rule: dynamic values arrive through `env:`, so nothing
    a contributor controls can be substituted into shell source."""
    for where, run in _run_blocks(doc):
        assert "${{" not in run, f"{where} interpolates an expression"


def test_every_action_is_pinned_to_a_sha(doc):
    sha = re.compile(r"^[^@]+@[0-9a-f]{40}$")
    for job in doc["jobs"].values():
        for step in job.get("steps") or []:
            uses = step.get("uses")
            if uses:
                assert sha.match(uses), f"{uses} is not pinned to a full SHA"


def test_workflow_level_permissions_are_empty(doc):
    """Declared per job instead, so `detect` and `gate` do not inherit the
    id-token grant that only `build` needs."""
    assert doc["permissions"] == {}


def test_id_token_is_granted_only_to_the_build_job(doc):
    for name, job in doc["jobs"].items():
        perms = job.get("permissions") or {}
        if name == "build":
            assert perms.get("id-token") == "write"
        else:
            assert "id-token" not in perms, f"{name} should not mint tokens"


def test_every_job_has_a_timeout(doc):
    for name, job in doc["jobs"].items():
        assert job.get("timeout-minutes"), f"{name} has no timeout"


def test_the_matrix_does_not_fail_fast(doc):
    """One broken image must not hide the state of the others."""
    assert doc["jobs"]["build"]["strategy"]["fail-fast"] is False


def test_a_main_build_is_not_cancelled_by_a_later_push(doc):
    """Cancelling the run that publishes could leave some images pushed and
    others not. Superseding a PR build is free; superseding this is not."""
    assert (
        doc["concurrency"]["cancel-in-progress"]
        == "${{ github.event_name == 'pull_request' }}"
    )


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_the_gate_always_runs(doc):
    """A skipped required check is often treated as passing by branch
    protection, which is the silent-green failure this guards against."""
    gate = doc["jobs"]["gate"]
    assert gate["if"] == "always()"
    assert set(gate["needs"]) == {"detect", "build"}


def test_the_gate_passes_when_no_image_was_affected(doc):
    """`skipped` on build is the normal outcome for most pull requests."""
    run = doc["jobs"]["gate"]["steps"][0]["run"]
    assert "skipped)" in run


def test_the_gate_fails_when_detect_fails(doc):
    """Otherwise an invalid declaration produces a green run that built
    nothing."""
    run = doc["jobs"]["gate"]["steps"][0]["run"]
    assert '"$DETECT_RESULT" != "success"' in run
    assert "exit 1" in run


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_the_matrix_comes_from_the_detect_job(doc):
    build = doc["jobs"]["build"]
    assert build["needs"] == "detect"
    assert (
        build["strategy"]["matrix"]["entry"]
        == "${{ fromJson(needs.detect.outputs.matrix) }}"
    )
    assert build["if"] == "needs.detect.outputs.count != '0'"


def test_the_detect_job_declares_the_outputs_build_consumes(doc):
    outputs = doc["jobs"]["detect"]["outputs"]
    assert "matrix" in outputs
    assert "count" in outputs


def test_the_build_uses_the_declared_context_and_dockerfile(doc):
    """The matrix carries these per image; hardcoding either here would
    silently ignore the declaration."""
    step = next(
        s for s in _steps(doc, "build") if "docker build" in s.get("run", "")
    )
    env = step["env"]
    assert env["DOCKERFILE"] == "${{ matrix.entry.dockerfile }}"
    assert env["CONTEXT"] == "${{ matrix.entry.context }}"
    assert env["PLATFORMS"] == "${{ matrix.entry.platforms }}"


def test_images_are_tagged_by_commit_sha_only(doc):
    """No moving tag. What a consumer should pin to is still an open
    question, and a public tag published once is hard to withdraw."""
    push = next(
        s for s in _steps(doc, "build") if "docker push" in s.get("run", "")
    )
    assert "$COMMIT_SHA" in push["run"]
    assert ":latest" not in push["run"]


def test_the_workflow_path_matches_the_scripts_global_rebuild_list():
    """publish_matrix.GLOBAL_REBUILD_PATHS names this file literally.

    Rename the workflow without updating that tuple and a change to it stops
    triggering rebuilds, silently.
    """
    import publish_matrix

    rel = WORKFLOW.relative_to(REPO_ROOT).as_posix()
    assert rel in publish_matrix.GLOBAL_REBUILD_PATHS


def _matches_a_trigger_path(candidate: str, patterns: list[str]) -> bool:
    """Would `candidate` satisfy one of a trigger's `paths` globs?

    Only the two glob shapes this workflow uses are handled — a literal path
    and a `dir/**` prefix. Anything else is deliberately not guessed at.
    """
    for pattern in patterns:
        if pattern == candidate:
            return True
        if pattern.endswith("/**") and candidate.startswith(pattern[:-2]):
            return True
    return False


@pytest.mark.parametrize("trigger", ["pull_request", "push"])
def test_every_global_rebuild_path_also_triggers_the_workflow(
    triggers: dict, trigger: str
):
    """The two lists have to agree or a rebuild rule can never fire.

    GLOBAL_REBUILD_PATHS decides which changes rebuild every image, but it is
    only consulted once the workflow is already running. A path listed there
    and absent from these filters is a rule that looks right in the script
    and never runs — the workflow is not triggered at all, so nothing reports
    anything, which is the worst shape a failure can take.
    """
    import publish_matrix

    patterns = triggers[trigger]["paths"]
    for path in publish_matrix.GLOBAL_REBUILD_PATHS:
        assert _matches_a_trigger_path(path, patterns), (
            f"{path} is in GLOBAL_REBUILD_PATHS but no `on.{trigger}.paths` "
            f"pattern matches it, so changing it would not start this "
            f"workflow at all"
        )


@pytest.mark.parametrize("trigger", ["pull_request", "push"])
def test_every_publishable_root_also_triggers_the_workflow(
    triggers: dict, trigger: str
):
    """Same coupling, for the roots recipes may be published from.

    Add a root to PUBLISHABLE_ROOTS without adding it here and images
    declared under it are never rebuilt when their recipe changes.
    """
    import publish_matrix

    patterns = triggers[trigger]["paths"]
    for root in publish_matrix.PUBLISHABLE_ROOTS:
        probe = f"{root}some/recipe/agent.py"
        assert _matches_a_trigger_path(probe, patterns), (
            f"{root} is a publishable root but no `on.{trigger}.paths` "
            f"pattern matches paths under it"
        )


def test_the_two_triggers_filter_on_the_same_paths(triggers: dict):
    """A path that builds on a PR but not on merge — or the reverse — means
    the thing verified before merge is not the thing published after it."""
    assert triggers["pull_request"]["paths"] == triggers["push"]["paths"]


def test_dispatch_offers_no_control_that_does_nothing(triggers: dict):
    """A manual run has no diff base, so it always builds everything.

    A `build_all` input would be a checkbox users could untick to no effect.
    Asserting its absence rather than the exact input set, so a future input
    that genuinely does something is not blocked by this test.
    """
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert "build_all" not in inputs
    assert "image" in inputs


def test_the_diff_does_not_c_quote_non_ascii_paths(doc: dict):
    """Without core.quotePath=false git emits `"caf\\303\\251.md"`.

    A quoted path matches no recipe prefix, so the image owning it is
    silently not rebuilt — the exact silent-miss this pipeline exists to
    avoid. Verified against real git, not assumed.
    """
    step = next(s for s in _steps(doc, "detect") if s.get("id") == "detect")
    diff_lines = [
        line
        for line in step["run"].splitlines()
        if "git " in line
        and "diff --name-only" in line
        and not line.strip().startswith("#")
    ]
    assert diff_lines, "the diff line moved or changed shape"
    for line in diff_lines:
        assert "core.quotePath=false" in line, (
            f"non-ASCII paths would be C-quoted and silently skipped: {line}"
        )


def test_the_pull_request_diff_fetch_is_not_shallow(doc: dict):
    """`--depth` here makes the repository shallow and breaks merge-base.

    The checkout already uses fetch-depth: 0, so this fetch only refreshes a
    ref that is already present. Adding a depth introduces a shallow
    boundary, after which `git merge-base` and `git diff` against the base
    can fail — reproduced on a real clone, not theorised.
    """
    step = next(s for s in _steps(doc, "detect") if s.get("id") == "detect")
    fetch_lines = [
        line
        for line in step["run"].splitlines()
        if "git fetch" in line and not line.strip().startswith("#")
    ]
    assert fetch_lines, "the PR path no longer fetches its base ref"
    for line in fetch_lines:
        assert "--depth" not in line, f"shallow fetch reintroduced: {line}"


def test_the_detect_checkout_takes_full_history(doc: dict):
    """merge-base needs it, and it is what makes the fetch above safe."""
    checkout = next(
        s
        for s in _steps(doc, "detect")
        if "actions/checkout" in str(s.get("uses", ""))
    )
    assert checkout["with"]["fetch-depth"] == 0


# --------------------------------------------------------------------------
# Failure reporting
# --------------------------------------------------------------------------


def test_a_failed_build_still_fails_its_job(doc: dict):
    """`continue-on-error` on the build step exists only so the log can be
    captured and classified. Without a step restoring the failure, the leg
    reports success, the gate aggregates three successes, and a required
    check goes green over an image that did not build."""
    steps = _steps(doc, "build")
    build = next(s for s in steps if s.get("id") == "build")
    assert build["continue-on-error"] is True

    restorer = next(
        s
        for s in steps
        if "steps.build.outcome == 'failure'" in str(s.get("if", ""))
    )
    assert "exit 1" in restorer["run"]
    # After classify and upload, or the evidence is lost before anything
    # records it.
    names = [s.get("name", "") for s in steps]
    assert names.index(restorer["name"]) > names.index("Upload result")


def test_every_leg_uploads_a_result_even_when_it_passed(doc: dict):
    """A recipe absent from the results cannot be told apart from one that
    passed, which is how a recovery goes unnoticed and its tracking issue
    stays open forever."""
    for name in ("Classify the result", "Upload result"):
        step = next(s for s in _steps(doc, "build") if s.get("name") == name)
        assert step["if"] == "always()", f"{name} does not always run"


def test_the_report_job_only_runs_on_a_merge_to_main(doc: dict):
    """On a pull request the author is already looking at the failing check;
    a bot comment repeating it is noise."""
    condition = doc["jobs"]["report"]["if"]
    assert "github.event_name == 'push'" in condition
    assert "refs/heads/main" in condition
    assert "!cancelled()" in condition


def test_the_report_job_is_the_only_writer(doc: dict):
    """It writes to the tracker and takes no part in building, so nothing a
    contributor's Dockerfile does happens in a job holding these scopes."""
    for name, job in doc["jobs"].items():
        perms = job.get("permissions") or {}
        writes = {
            k for k, v in perms.items() if v == "write" and k != "id-token"
        }
        if name == "report":
            assert writes == {"issues", "pull-requests"}
        else:
            assert not writes, f"{name} can write {writes}"


def test_the_report_job_does_not_build_anything(doc: dict):
    runs = " ".join(s.get("run", "") for s in _steps(doc, "report"))
    assert "docker build" not in runs
    assert "docker push" not in runs


def test_the_report_job_reads_the_results_the_build_job_wrote(doc: dict):
    """The artifact name is the contract between the two jobs; a mismatch
    means the reporter silently finds nothing and says nothing."""
    upload = next(
        s
        for s in _steps(doc, "build")
        if "upload-artifact" in str(s.get("uses", ""))
    )
    download = next(
        s
        for s in _steps(doc, "report")
        if "download-artifact" in str(s.get("uses", ""))
    )
    produced = upload["with"]["name"]
    pattern = download["with"]["pattern"]
    assert pattern.endswith("*")
    assert produced.startswith(pattern[:-1]), (
        f"upload writes {produced!r}, download looks for {pattern!r}"
    )


def test_the_gate_does_not_depend_on_the_reporter(doc: dict):
    """Branch protection must not turn red because commenting failed, nor
    green because it succeeded. The two answer different questions."""
    assert "report" not in doc["jobs"]["gate"]["needs"]


def test_the_reporter_does_not_run_when_no_image_was_built(doc: dict):
    """Most merges that trigger this workflow affect no declared image, and
    then `build` is skipped and no artifacts exist. Without this guard the
    reporter goes red on an ordinary healthy merge, and a channel that cries
    wolf gets muted along with its real reports."""
    assert "needs.build.result != 'skipped'" in doc["jobs"]["report"]["if"]


def test_the_build_job_name_matches_what_the_reporter_parses(doc: dict):
    """The job name is the only link between a historical run and an image.

    image_build_report strips BUILD_JOB_PREFIX off each job name to work out
    which image it built, which is how "did this fail last time it was
    built" is answered. Rename the job and that parsing silently returns
    nothing: no repeats are ever detected, and no tracking issue is ever
    opened again.
    """
    import image_build_report

    name = doc["jobs"]["build"]["name"]
    prefix = image_build_report.BUILD_JOB_PREFIX

    assert name.startswith(prefix), (
        f"build job is named {name!r} but the reporter strips {prefix!r}"
    )
    # What remains must be the image expression, or the parsed value is not
    # an image name at all.
    assert name[len(prefix) :].strip() == "${{ matrix.entry.image }}"
