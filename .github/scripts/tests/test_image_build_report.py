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
"""Unit tests for image_build_report.py.

The classifier decides whether a human gets told their change broke
something. Both mistakes it can make are tested here, because they are not
equally expensive:

  * calling infrastructure trouble a recipe failure blames an author for a
    registry timeout, and a channel that does that once gets ignored
    afterwards, including when it is right;
  * calling a recipe failure infrastructure loses one report, which the next
    run recovers.

So the bias is deliberate and asserted: every path to a verdict without a log
to read must land on `infra`, and `fail` must require log text that matched
no infrastructure signature.
"""

import json
from pathlib import Path

import image_build_report as m
import pytest


@pytest.fixture(autouse=True)
def repo_env(monkeypatch):
    """Run every test as CI does, with GITHUB_REPOSITORY set.

    cmd_report refuses to write when it is absent, because the module-level
    fallback would otherwise aim comments and issues at google/adk-samples
    from anyone's laptop. Tests that exercise the writing paths therefore
    have to supply it; test_writing_without_an_explicit_repo_is_refused
    covers the opposite case deliberately.
    """
    monkeypatch.setenv("GITHUB_REPOSITORY", "google/adk-samples")


# --------------------------------------------------------------------------
# The evidence rule
# --------------------------------------------------------------------------


def test_a_successful_build_passes():
    assert m.classify("success", "anything at all")[0] == m.PASS


@pytest.mark.parametrize("outcome", ["", "skipped", "cancelled"])
def test_a_build_that_never_ran_is_never_the_recipes_fault(outcome: str):
    """Checkout, the mode step or the runner died first. Nothing was learned
    about the recipe, so nothing may be said about it."""
    verdict, detail = m.classify(outcome, "")

    assert verdict == m.INFRA
    assert "did not run" in detail


@pytest.mark.parametrize("log", ["", "   \n  \n", None])
def test_a_failure_with_no_log_is_infra(log):
    """docker creates the log before it starts, so an empty one means it
    never got far enough to say anything."""
    verdict, detail = m.classify("failure", log)

    assert verdict == m.INFRA
    assert "without producing any output" in detail


# --------------------------------------------------------------------------
# Infrastructure signatures — the failures that must never be blamed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("log", "expected"),
    [
        (
            "failed to do request: Head https://registry-1.docker.io/v2/: "
            "received unexpected HTTP status: 503 Service Unavailable",
            "registry",
        ),
        (
            "toomanyrequests: You have reached your pull rate limit.",
            "rate limit",
        ),
        ("read tcp 10.1.0.4:52134: connection reset by peer", "network"),
        ("dial tcp: i/o timeout", "network"),
        (
            "dial tcp: lookup registry-1.docker.io: no such host",
            "DNS",
        ),
        ("tls handshake timeout", "TLS"),
        (
            "ERROR: failed to solve: failed to resolve source metadata for "
            "docker.io/library/python:3.12-slim",
            "base image",
        ),
        ("pull access denied, repository does not exist", "base image"),
        ("error pulling image configuration: download failed", "base image"),
        ("write /var/lib/docker/tmp: no space left on device", "disk"),
        ("fatal error: runtime: cannot allocate memory", "memory"),
        (
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
            "docker daemon",
        ),
        ("unauthorized: authentication failed", "authentication"),
        (
            'denied: Permission "artifactregistry.repositories.uploadArtifacts" denied',
            "authentication",
        ),
    ],
)
def test_infrastructure_failures_are_not_blamed_on_the_recipe(
    log: str, expected: str
):
    verdict, detail = m.classify("failure", log)

    assert verdict == m.INFRA, f"{log!r} was blamed on the recipe"
    assert expected.lower() in detail.lower()


# --------------------------------------------------------------------------
# Genuine recipe failures
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "log",
    [
        'ERROR: failed to solve: process "/bin/sh -c uv sync --frozen" '
        "did not complete successfully: exit code: 1",
        "COPY failed: file not found in build context: stat app: no such "
        "file or directory",
        "Dockerfile:14\n--------------------\nERROR: invalid instruction",
        "executor failed running [/bin/sh -c pip install -r req.txt]: "
        "exit code 2",
    ],
)
def test_real_build_failures_are_reported(log: str):
    verdict, detail = m.classify("failure", log)

    assert verdict == m.FAIL
    assert detail


def test_the_detail_prefers_the_line_naming_the_failure():
    log = (
        "#8 [4/6] RUN uv sync --frozen\n"
        "#8 0.4 error: no lockfile\n"
        'ERROR: failed to solve: process "/bin/sh -c uv sync --frozen" '
        "did not complete successfully: exit code: 1\n"
        "some trailing noise\n"
    )

    _, detail = m.classify("failure", log)

    assert "did not complete successfully" in detail


def test_an_infra_signature_anywhere_in_the_log_wins():
    """A recipe-looking error preceded by a registry failure is still infra.

    The build fails at the first problem; a later line that looks like the
    recipe's fault is usually a consequence of the earlier one.
    """
    log = (
        "failed to resolve source metadata for docker.io/library/python\n"
        'ERROR: failed to solve: process "/bin/sh -c true" did not '
        "complete successfully: exit code: 1\n"
    )

    assert m.classify("failure", log)[0] == m.INFRA


def test_log_tail_is_bounded():
    log = "\n".join(f"line {i}" for i in range(500))

    tail = m.log_tail(log)

    assert len(tail.splitlines()) == m.LOG_TAIL_LINES
    assert "line 499" in tail
    assert "line 0" not in tail


# --------------------------------------------------------------------------
# The classify subcommand
# --------------------------------------------------------------------------


def test_classify_writes_a_result_file(tmp_path: Path):
    log = tmp_path / "build.log"
    log.write_text("no space left on device", encoding="utf-8")
    out = tmp_path / "result.json"

    rc = m.main(
        [
            "classify",
            "--image",
            "demo",
            "--dockerfile",
            "core/python/demo/Dockerfile",
            "--context",
            "core/python/demo",
            "--outcome",
            "failure",
            "--log",
            str(log),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    result = json.loads(out.read_text())
    assert result == {
        "image": "demo",
        "dockerfile": "core/python/demo/Dockerfile",
        "context": "core/python/demo",
        "outcome": "infra",
        "detail": "the runner ran out of disk",
        "tail": "",
    }


def test_classify_carries_a_tail_only_for_real_failures(tmp_path: Path):
    log = tmp_path / "build.log"
    log.write_text(
        "ERROR: failed to solve: process did not complete successfully",
        encoding="utf-8",
    )
    out = tmp_path / "result.json"

    m.main(
        [
            "classify",
            "--image",
            "demo",
            "--outcome",
            "failure",
            "--log",
            str(log),
            "--out",
            str(out),
        ]
    )

    assert json.loads(out.read_text())["tail"]


def test_classify_survives_a_missing_log(tmp_path: Path):
    """A missing log is evidence of nothing, not a crash."""
    out = tmp_path / "result.json"

    rc = m.main(
        [
            "classify",
            "--image",
            "demo",
            "--outcome",
            "failure",
            "--log",
            str(tmp_path / "nope.log"),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    assert json.loads(out.read_text())["outcome"] == "infra"


# --------------------------------------------------------------------------
# The report subcommand
# --------------------------------------------------------------------------


def _result(tmp_path: Path, index: int, **fields) -> None:
    entry = {
        "image": "demo",
        "dockerfile": "core/python/demo/Dockerfile",
        "context": "core/python/demo",
        "outcome": "pass",
        "detail": "",
        "tail": "",
    }
    entry.update(fields)
    d = tmp_path / "results" / f"image-result-{index}"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(entry), encoding="utf-8")


@pytest.fixture
def no_network(monkeypatch):
    """Every `gh` call recorded, none made."""
    calls: list[tuple[str, ...]] = []

    def fake_gh(*args: str, check: bool = True) -> str:
        calls.append(args)
        if args[:2] == ("issue", "list"):
            return "[]"
        if args[0] == "api" and "/pulls" in args[1]:
            return json.dumps([{"number": 4242}])
        if args[0] == "api" and "/runs" in args[1] and "jobs" not in args[1]:
            return json.dumps({"workflow_runs": []})
        return ""

    monkeypatch.setattr(m, "gh", fake_gh)
    return calls


def test_report_says_nothing_when_everything_passed(tmp_path: Path, no_network):
    _result(tmp_path, 0, outcome="pass")

    rc = m.main(["report", "--results", str(tmp_path / "results")])

    assert rc == 0
    assert not [c for c in no_network if c[:2] == ("issue", "comment")]


def test_report_stays_silent_on_infrastructure_failures(
    tmp_path: Path, no_network, capsys
):
    """The whole point. Nobody is told a registry had a bad minute."""
    _result(tmp_path, 0, outcome="infra", detail="registry rate limit")

    rc = m.main(["report", "--results", str(tmp_path / "results")])

    assert rc == 0
    assert not [c for c in no_network if c[:2] == ("issue", "comment")]
    assert "nothing reported to anyone" in capsys.readouterr().out


def test_report_comments_on_the_pull_request_for_a_real_failure(
    tmp_path: Path, no_network
):
    _result(tmp_path, 0, outcome="fail", detail="boom", tail="ERROR: boom")

    rc = m.main(
        [
            "report",
            "--results",
            str(tmp_path / "results"),
            "--sha",
            "abcdef1234567890",
            "--run-url",
            "https://run",
        ]
    )

    assert rc == 0
    comments = [c for c in no_network if c[:2] == ("issue", "comment")]
    assert len(comments) == 1
    assert comments[0][2] == "4242"
    body = comments[0][-1]
    assert "demo" in body and "ERROR: boom" in body


def test_no_issue_is_opened_on_a_first_failure(
    tmp_path: Path, no_network, capsys
):
    """One bad merge does not deserve a tracker item — the comment is on the
    pull request of the person who can fix it."""
    _result(tmp_path, 0, outcome="fail", tail="ERROR: boom")

    m.main(
        [
            "report",
            "--results",
            str(tmp_path / "results"),
            "--sha",
            "abcdef1234567890",
        ]
    )

    assert not [c for c in no_network if c[:2] == ("issue", "create")]
    assert "first failure" in capsys.readouterr().out


def test_an_issue_is_opened_when_the_same_image_fails_again(
    tmp_path: Path, no_network, monkeypatch
):
    _result(tmp_path, 0, outcome="fail", tail="ERROR: boom")
    monkeypatch.setattr(
        m,
        "images_that_failed_in_the_previous_run",
        lambda _run, _wanted=None: {"demo"},
    )

    m.main(
        [
            "report",
            "--results",
            str(tmp_path / "results"),
            "--sha",
            "abcdef1234567890",
        ]
    )

    created = [c for c in no_network if c[:2] == ("issue", "create")]
    assert len(created) == 1
    assert m.issue_title("demo") in created[0]


def test_a_repeat_with_an_existing_issue_comments_instead_of_duplicating(
    tmp_path: Path, monkeypatch
):
    _result(tmp_path, 0, outcome="fail", tail="ERROR: boom")
    calls: list[tuple[str, ...]] = []

    def fake_gh(*args: str, check: bool = True) -> str:
        calls.append(args)
        if args[:2] == ("issue", "list"):
            return json.dumps([{"number": 77, "title": m.issue_title("demo")}])
        if args[0] == "api" and "/pulls" in args[1]:
            return json.dumps([{"number": 1}])
        return ""

    monkeypatch.setattr(m, "gh", fake_gh)
    monkeypatch.setattr(
        m,
        "images_that_failed_in_the_previous_run",
        lambda _run, _wanted=None: {"demo"},
    )

    m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "a" * 40]
    )

    assert not [c for c in calls if c[:2] == ("issue", "create")]
    assert any(c[:3] == ("issue", "comment", "77") for c in calls)


def test_a_recovered_image_closes_its_issue(tmp_path: Path, monkeypatch):
    _result(tmp_path, 0, outcome="pass")
    calls: list[tuple[str, ...]] = []

    def fake_gh(*args: str, check: bool = True) -> str:
        calls.append(args)
        if args[:2] == ("issue", "list"):
            return json.dumps([{"number": 88, "title": m.issue_title("demo")}])
        return ""

    monkeypatch.setattr(m, "gh", fake_gh)

    m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "b" * 40]
    )

    assert any(c[:3] == ("issue", "close", "88") for c in calls)


def test_an_exact_title_match_is_required(tmp_path: Path, monkeypatch):
    """`demo` must not be handed the issue belonging to `demo-sandbox`,
    which would report one image's failure on another's thread."""
    _result(tmp_path, 0, outcome="pass")
    calls: list[tuple[str, ...]] = []

    def fake_gh(*args: str, check: bool = True) -> str:
        calls.append(args)
        if args[:2] == ("issue", "list"):
            return json.dumps(
                [{"number": 99, "title": m.issue_title("demo-sandbox")}]
            )
        return ""

    monkeypatch.setattr(m, "gh", fake_gh)

    m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "c" * 40]
    )

    assert not [c for c in calls if c[:2] == ("issue", "close")]


def test_no_results_is_an_error_not_a_clean_run(tmp_path: Path, capsys):
    """Every leg writes a result, including the passing ones, so none at all
    means collection broke — not that nothing failed."""
    (tmp_path / "results").mkdir()

    rc = m.main(["report", "--results", str(tmp_path / "results")])

    assert rc == 1
    assert "no results found" in capsys.readouterr().err


def test_a_direct_push_with_no_pull_request_does_not_crash(
    tmp_path: Path, monkeypatch, capsys
):
    _result(tmp_path, 0, outcome="fail", tail="ERROR: boom")
    monkeypatch.setattr(m, "gh", lambda *a, **k: "[]")

    rc = m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "d" * 40]
    )

    assert rc == 0
    assert "No pull request found" in capsys.readouterr().out


def test_dry_run_writes_nothing(tmp_path: Path, no_network, capsys):
    _result(tmp_path, 0, outcome="fail", tail="ERROR: boom")

    m.main(
        [
            "report",
            "--results",
            str(tmp_path / "results"),
            "--sha",
            "e" * 40,
            "--dry-run",
        ]
    )

    for verb in ("comment", "create", "close"):
        assert not [c for c in no_network if c[:2] == ("issue", verb)]
    assert "[dry-run]" in capsys.readouterr().out


def test_a_mixed_run_reports_only_the_real_failure(tmp_path: Path, no_network):
    # Distinctive names on purpose. An earlier version used "ok", which is a
    # substring of "broke" in the comment body, so the assertion failed for a
    # reason that had nothing to do with the code under test.
    _result(tmp_path, 0, image="healthy-one", outcome="pass")
    _result(tmp_path, 1, image="flaky-one", outcome="infra", detail="DNS")
    _result(tmp_path, 2, image="broken-one", outcome="fail", tail="E: boom")

    m.main(
        [
            "report",
            "--results",
            str(tmp_path / "results"),
            "--sha",
            "f" * 40,
        ]
    )

    body = next(c for c in no_network if c[:2] == ("issue", "comment"))[-1]
    assert "broken-one" in body
    assert "flaky-one" not in body
    assert "healthy-one" not in body


# --------------------------------------------------------------------------
# Previous-run lookup
# --------------------------------------------------------------------------


def test_previous_run_lookup_skips_the_current_run(monkeypatch):
    def fake_gh(*args: str, check: bool = True) -> str:
        if "/runs?" in args[1]:
            return json.dumps({"workflow_runs": [{"id": 100}, {"id": 99}]})
        return json.dumps(
            {"jobs": [{"name": "build demo", "conclusion": "failure"}]}
        )

    monkeypatch.setattr(m, "gh", fake_gh)

    assert m.images_that_failed_in_the_previous_run("100") == {"demo"}


def test_previous_run_lookup_is_empty_when_the_api_is_unreachable(
    monkeypatch,
):
    """Biases toward commenting without opening an issue, which the next
    failure corrects."""

    def boom(*args: str, check: bool = True) -> str:
        raise m.ReportError("network down")

    monkeypatch.setattr(m, "gh", boom)

    assert m.images_that_failed_in_the_previous_run("1") == set()


def test_previous_run_lookup_ignores_jobs_that_are_not_builds(monkeypatch):
    def fake_gh(*args: str, check: bool = True) -> str:
        if "/runs?" in args[1]:
            return json.dumps({"workflow_runs": [{"id": 1}]})
        return json.dumps(
            {
                "jobs": [
                    {"name": "Detect affected images", "conclusion": "failure"},
                    {"name": "build demo", "conclusion": "success"},
                ]
            }
        )

    monkeypatch.setattr(m, "gh", fake_gh)

    assert m.images_that_failed_in_the_previous_run("2") == set()


def test_repeat_looks_back_past_runs_that_did_not_build_the_image(
    monkeypatch,
):
    """The defect that made the issue path unreachable in practice.

    Builds are affected-only, so the common sequence is: the image fails,
    then several merges touch other recipes and never build it, then it
    fails again. Asking only about the immediately previous run answers
    "not failing" every time, so a repeat is never detected and no issue is
    ever opened.
    """
    runs = {"workflow_runs": [{"id": 4}, {"id": 3}, {"id": 2}]}
    jobs = {
        # The two most recent runs built a different recipe entirely.
        4: {"jobs": [{"name": "build other", "conclusion": "success"}]},
        3: {"jobs": [{"name": "build other", "conclusion": "success"}]},
        # The last run that actually built `demo` — and it failed.
        2: {"jobs": [{"name": "build demo", "conclusion": "failure"}]},
    }

    def fake_gh(*args: str, check: bool = True) -> str:
        if "/runs?" in args[1]:
            return json.dumps(runs)
        run_id = int(args[1].split("/runs/")[1].split("/")[0])
        return json.dumps(jobs[run_id])

    monkeypatch.setattr(m, "gh", fake_gh)

    assert m.images_that_failed_in_the_previous_run("5") == {"demo"}


def test_a_later_success_beats_an_earlier_failure(monkeypatch):
    """Only the MOST RECENT build of an image counts. A failure further back
    that has since been fixed must not reopen an issue."""
    runs = {"workflow_runs": [{"id": 9}, {"id": 8}]}
    jobs = {
        9: {"jobs": [{"name": "build demo", "conclusion": "success"}]},
        8: {"jobs": [{"name": "build demo", "conclusion": "failure"}]},
    }

    def fake_gh(*args: str, check: bool = True) -> str:
        if "/runs?" in args[1]:
            return json.dumps(runs)
        run_id = int(args[1].split("/runs/")[1].split("/")[0])
        return json.dumps(jobs[run_id])

    monkeypatch.setattr(m, "gh", fake_gh)

    assert m.images_that_failed_in_the_previous_run("10") == set()


def test_the_current_run_is_never_its_own_evidence(monkeypatch):
    runs = {"workflow_runs": [{"id": 7}]}

    def fake_gh(*args: str, check: bool = True) -> str:
        if "/runs?" in args[1]:
            return json.dumps(runs)
        return json.dumps(
            {"jobs": [{"name": "build demo", "conclusion": "failure"}]}
        )

    monkeypatch.setattr(m, "gh", fake_gh)

    assert m.images_that_failed_in_the_previous_run("7") == set()


def test_too_many_failures_opens_no_issues_at_all(
    tmp_path: Path, no_network, monkeypatch, capsys
):
    """The message used to say "commenting only" while the loop went on
    opening issues anyway. This many at once is one systemic cause, and N
    issues would each be wrong about what to fix."""
    for i in range(m.MAX_ISSUES_PER_RUN + 1):
        _result(tmp_path, i, image=f"img{i}", outcome="fail", tail="E: boom")
    monkeypatch.setattr(
        m,
        "images_that_failed_in_the_previous_run",
        lambda _run, _wanted=None: {
            f"img{i}" for i in range(m.MAX_ISSUES_PER_RUN + 1)
        },
    )

    m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "a" * 40]
    )

    assert not [c for c in no_network if c[:2] == ("issue", "create")]
    assert "no issues are being opened" in capsys.readouterr().err


def test_every_failing_image_gets_its_own_reproduce_command(
    tmp_path: Path, no_network
):
    """One command printed under a list of three reads as though it covers
    all of them, so a reader fixes one image and believes they are done."""
    _result(
        tmp_path,
        0,
        image="one",
        outcome="fail",
        tail="E: 1",
        dockerfile="core/python/one/Dockerfile",
        context="core/python/one",
    )
    _result(
        tmp_path,
        1,
        image="two",
        outcome="fail",
        tail="E: 2",
        dockerfile="core/python/two/Dockerfile",
        context="core/python/two",
    )

    m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "b" * 40]
    )

    body = next(c for c in no_network if c[:2] == ("issue", "comment"))[-1]
    assert "docker build -f core/python/one/Dockerfile core/python/one" in body
    assert "docker build -f core/python/two/Dockerfile core/python/two" in body


def test_a_corrupt_result_file_is_skipped_not_fatal(
    tmp_path: Path, no_network, capsys
):
    """A valid-JSON but wrong-shape artifact must not raise deep in the
    reporting logic, possibly after a comment has already been posted."""
    _result(tmp_path, 0, image="good", outcome="pass")
    bad = tmp_path / "results" / "image-result-9"
    bad.mkdir(parents=True)
    (bad / "result.json").write_text("[]", encoding="utf-8")

    rc = m.main(["report", "--results", str(tmp_path / "results")])

    assert rc == 0
    assert "not a usable result" in capsys.readouterr().err


def test_the_issue_index_is_fetched_once_per_run(tmp_path: Path, monkeypatch):
    """canary_issues documents why: a lookup inside the loop is n
    subprocesses and n chances to hit a rate limit, to answer a question one
    call already answers."""
    for i in range(3):
        _result(tmp_path, i, image=f"img{i}", outcome="pass")
    calls: list[tuple[str, ...]] = []

    def fake_gh(*args: str, check: bool = True) -> str:
        calls.append(args)
        return "[]" if args[:2] == ("issue", "list") else ""

    monkeypatch.setattr(m, "gh", fake_gh)

    m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "c" * 40]
    )

    assert len([c for c in calls if c[:2] == ("issue", "list")]) == 1


def test_the_history_walk_stops_once_every_image_is_answered(monkeypatch):
    """Each extra run costs an API call. In the common case the previous run
    built the same image and one call is enough."""
    runs = {"workflow_runs": [{"id": 3}, {"id": 2}, {"id": 1}]}
    fetched: list[int] = []

    def fake_gh(*args: str, check: bool = True) -> str:
        if "/runs?" in args[1]:
            return json.dumps(runs)
        run_id = int(args[1].split("/runs/")[1].split("/")[0])
        fetched.append(run_id)
        return json.dumps(
            {"jobs": [{"name": "build demo", "conclusion": "failure"}]}
        )

    monkeypatch.setattr(m, "gh", fake_gh)

    assert m.images_that_failed_in_the_previous_run("9", {"demo"}) == {"demo"}
    assert fetched == [3], f"walked further than needed: {fetched}"


def test_the_history_walk_keeps_going_until_it_finds_the_image(monkeypatch):
    runs = {"workflow_runs": [{"id": 3}, {"id": 2}]}
    jobs = {
        3: {"jobs": [{"name": "build other", "conclusion": "success"}]},
        2: {"jobs": [{"name": "build demo", "conclusion": "failure"}]},
    }

    def fake_gh(*args: str, check: bool = True) -> str:
        if "/runs?" in args[1]:
            return json.dumps(runs)
        return json.dumps(jobs[int(args[1].split("/runs/")[1].split("/")[0])])

    monkeypatch.setattr(m, "gh", fake_gh)

    assert m.images_that_failed_in_the_previous_run("9", {"demo"}) == {"demo"}


def test_an_infra_failure_leaves_an_existing_issue_alone(
    tmp_path: Path, monkeypatch
):
    """Infrastructure trouble says nothing about the recipe, so it must
    neither open, comment on, nor close a tracking issue."""
    _result(tmp_path, 0, image="demo", outcome="infra", detail="DNS failure")
    calls: list[tuple[str, ...]] = []

    def fake_gh(*args: str, check: bool = True) -> str:
        calls.append(args)
        if args[:2] == ("issue", "list"):
            return json.dumps([{"number": 5, "title": m.issue_title("demo")}])
        return ""

    monkeypatch.setattr(m, "gh", fake_gh)

    m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "d" * 40]
    )

    for verb in ("create", "comment", "close"):
        assert not [c for c in calls if c[:2] == ("issue", verb)]


def test_the_issue_index_is_fetched_once_even_with_both_passes_and_failures(
    tmp_path: Path, monkeypatch
):
    """The earlier version of this assertion only covered the all-pass case,
    where just one of the two call sites ran."""
    _result(tmp_path, 0, image="good", outcome="pass")
    _result(tmp_path, 1, image="bad", outcome="fail", tail="E: boom")
    calls: list[tuple[str, ...]] = []

    def fake_gh(*args: str, check: bool = True) -> str:
        calls.append(args)
        if args[:2] == ("issue", "list"):
            return "[]"
        if args[0] == "api" and "/pulls" in args[1]:
            return json.dumps([{"number": 1}])
        if args[0] == "api" and "/runs?" in args[1]:
            return json.dumps({"workflow_runs": []})
        return ""

    monkeypatch.setattr(m, "gh", fake_gh)

    m.main(
        ["report", "--results", str(tmp_path / "results"), "--sha", "e" * 40]
    )

    assert len([c for c in calls if c[:2] == ("issue", "list")]) == 1


def test_writing_without_an_explicit_repo_is_refused(
    tmp_path: Path, monkeypatch, capsys
):
    """Run this on a laptop with GITHUB_REPOSITORY unset and the fallback
    would post comments and open issues on google/adk-samples."""
    _result(tmp_path, 0, outcome="fail", tail="E: boom")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr(m, "gh", lambda *a, **k: "[]")

    rc = m.main(["report", "--results", str(tmp_path / "results")])

    assert rc == 1
    assert "GITHUB_REPOSITORY is not set" in capsys.readouterr().err


def test_dry_run_needs_no_repo(tmp_path: Path, monkeypatch, capsys):
    """Reading and previewing are safe without it; only writing is not."""
    _result(tmp_path, 0, outcome="fail", tail="E: boom")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr(m, "gh", lambda *a, **k: "[]")

    assert (
        m.main(["report", "--results", str(tmp_path / "results"), "--dry-run"])
        == 0
    )
