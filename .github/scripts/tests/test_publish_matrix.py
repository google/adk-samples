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
"""Unit tests for publish_matrix.py.

This matrix decides which container images are built and pushed to a PUBLIC
registry under Google's name, so the tests here pin the refusals rather than
the happy path. Two failures are worth more than the rest combined:

  * a duplicate image name, where the second entry silently replaces the
    first at the same public address and the run still reports success;
  * an empty matrix, where a workflow that builds nothing is indistinguishable
    from one that is working.

The final test validates the REAL policy.yml against the REAL repository, so
moving or deleting a declared Dockerfile fails here — on every PR, through
tools-tests.yml — instead of minutes into a build on a runner.
"""

import json
import re
import shlex
from pathlib import Path
from typing import Any

import publish_matrix as m
import pytest
import yaml

# Derived here independently of publish_matrix.REPO_ROOT, and deliberately
# so: this file sits one level deeper than the module, so the two walk a
# different number of parents to reach the same place. Reusing the module's
# value would make the guard tests below agree with it by construction —
# including when it is wrong. test_repo_root_derivations_agree pins that they
# match, which is the part worth asserting.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _write_policy(
    tmp_path: Path,
    images: list[dict[str, Any]] | Any,
    platforms: Any = ("linux/amd64",),
    *,
    omit_publish: bool = False,
    omit_deployability: bool = False,
    omit_platforms: bool = False,
) -> Path:
    """Write a minimal .github/policy.yml into a throwaway repo root."""
    publish: dict[str, Any] = {"images": images}
    if not omit_platforms:
        publish["platforms"] = (
            list(platforms) if isinstance(platforms, tuple) else platforms
        )

    if omit_deployability:
        doc: dict[str, Any] = {"recipe_naming": {}}
    elif omit_publish:
        doc = {"deployability": {"min_google_adk": "2.6.0"}}
    else:
        doc = {"deployability": {"publish": publish}}

    github = tmp_path / ".github"
    github.mkdir(parents=True, exist_ok=True)
    (github / "policy.yml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    return tmp_path


def _scaffold(
    tmp_path: Path,
    recipe: str = "core/python/demo",
    dockerfile: str = "Dockerfile",
    context: str = ".",
    *,
    manifest: bool = True,
) -> None:
    """Create the recipe directory a declaration points at."""
    recipe_dir = tmp_path / recipe
    recipe_dir.mkdir(parents=True, exist_ok=True)
    if manifest:
        (recipe_dir / "manifest.yaml").write_text(
            "language: python\nstatus: active\n", encoding="utf-8"
        )
    (recipe_dir / context).mkdir(parents=True, exist_ok=True)
    df = recipe_dir / dockerfile
    df.parent.mkdir(parents=True, exist_ok=True)
    df.write_text("FROM python:3.12-slim\n", encoding="utf-8")


def _entry(**overrides: Any) -> dict[str, Any]:
    base = {
        "recipe": "core/python/demo",
        "dockerfile": "Dockerfile",
        "context": ".",
        "image": "demo",
        "serves_http": True,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_valid_declaration_yields_repo_relative_paths(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()])

    matrix = m.build_matrix(repo_root=tmp_path)

    assert matrix == [
        {
            "recipe": "core/python/demo",
            "image": "demo",
            "dockerfile": "core/python/demo/Dockerfile",
            "context": "core/python/demo",
            "serves_http": True,
            "platforms": "linux/amd64",
        }
    ]


def test_nested_dockerfile_and_context(tmp_path: Path):
    """The sandbox-runtime shape: both nested, one inside the other."""
    _scaffold(
        tmp_path,
        dockerfile="horizon/sandbox/runtime/Dockerfile",
        context="horizon/sandbox/runtime",
    )
    _write_policy(
        tmp_path,
        [
            _entry(
                dockerfile="horizon/sandbox/runtime/Dockerfile",
                context="horizon/sandbox/runtime",
                image="demo-sandbox-runtime",
            )
        ],
    )

    (entry,) = m.build_matrix(repo_root=tmp_path)

    assert entry["dockerfile"] == (
        "core/python/demo/horizon/sandbox/runtime/Dockerfile"
    )
    assert entry["context"] == "core/python/demo/horizon/sandbox/runtime"


def test_nested_dockerfile_with_recipe_root_context(tmp_path: Path):
    """The multiformat shape: Dockerfile in a subdirectory, context at root.

    This is the case that makes `context` a declared field rather than
    something inferred from the Dockerfile's location.
    """
    _scaffold(tmp_path, dockerfile="services/api/Dockerfile", context=".")
    _write_policy(
        tmp_path,
        [_entry(dockerfile="services/api/Dockerfile", context=".")],
    )

    (entry,) = m.build_matrix(repo_root=tmp_path)

    assert entry["dockerfile"] == "core/python/demo/services/api/Dockerfile"
    assert entry["context"] == "core/python/demo"


def test_multiple_images_from_one_recipe(tmp_path: Path):
    _scaffold(tmp_path)
    _scaffold(tmp_path, dockerfile="sub/Dockerfile", context="sub")
    _write_policy(
        tmp_path,
        [
            _entry(),
            _entry(
                dockerfile="sub/Dockerfile", context="sub", image="demo-sub"
            ),
        ],
    )

    matrix = m.build_matrix(repo_root=tmp_path)

    assert [e["image"] for e in matrix] == ["demo", "demo-sub"]


def test_serves_http_false_is_preserved(tmp_path: Path):
    """A publishable image need not be a server; the KFP base image is not."""
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(serves_http=False)])

    (entry,) = m.build_matrix(repo_root=tmp_path)

    assert entry["serves_http"] is False


# --------------------------------------------------------------------------
# The policy file itself
# --------------------------------------------------------------------------


def test_missing_policy_file(tmp_path: Path):
    with pytest.raises(m.PublishMatrixError, match="not found"):
        m.build_matrix(repo_root=tmp_path)


def test_invalid_yaml(tmp_path: Path):
    github = tmp_path / ".github"
    github.mkdir()
    (github / "policy.yml").write_text("deployability: [unclosed\n")

    with pytest.raises(m.PublishMatrixError, match="not valid YAML"):
        m.build_matrix(repo_root=tmp_path)


def test_unreadable_policy_reports_cleanly(tmp_path: Path):
    """An I/O failure must not surface as a traceback in a workflow log.

    A directory where policy.yml should be stands in for the whole OSError
    family (permission denied, I/O error): it is the one case reproducible
    without depending on the test process's privileges, since a root-owned
    CI runner can read a chmod 000 file anyway.
    """
    (tmp_path / ".github" / "policy.yml").mkdir(parents=True)

    with pytest.raises(m.PublishMatrixError, match="cannot read"):
        m.build_matrix(repo_root=tmp_path)


def test_missing_deployability_section(tmp_path: Path):
    _write_policy(tmp_path, [], omit_deployability=True)

    with pytest.raises(m.PublishMatrixError, match="no `deployability`"):
        m.build_matrix(repo_root=tmp_path)


def test_missing_publish_section(tmp_path: Path):
    _write_policy(tmp_path, [], omit_publish=True)

    with pytest.raises(
        m.PublishMatrixError, match=r"no `deployability\.publish`"
    ):
        m.build_matrix(repo_root=tmp_path)


def test_images_must_be_a_list(tmp_path: Path):
    _write_policy(tmp_path, {"not": "a list"})

    with pytest.raises(m.PublishMatrixError, match="must be a list"):
        m.build_matrix(repo_root=tmp_path)


def test_images_key_absent(tmp_path: Path):
    """Distinguished from a wrong type, because the fix differs."""
    github = tmp_path / ".github"
    github.mkdir()
    (github / "policy.yml").write_text(
        yaml.safe_dump(
            {"deployability": {"publish": {"platforms": ["linux/amd64"]}}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(m.PublishMatrixError, match="images` is missing"):
        m.build_matrix(repo_root=tmp_path)


# --------------------------------------------------------------------------
# platforms
# --------------------------------------------------------------------------


def test_platforms_missing_is_refused(tmp_path: Path):
    """Defaulting would silently build for the runner, not the target."""
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()], omit_platforms=True)

    with pytest.raises(m.PublishMatrixError, match="platforms` is missing"):
        m.build_matrix(repo_root=tmp_path)


def test_platforms_empty_is_refused(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()], platforms=[])

    with pytest.raises(m.PublishMatrixError, match="non-empty list"):
        m.build_matrix(repo_root=tmp_path)


def test_platforms_non_string_entry_is_refused(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()], platforms=[42])

    with pytest.raises(m.PublishMatrixError, match="non-string entry"):
        m.build_matrix(repo_root=tmp_path)


def test_multiple_platforms_are_comma_joined(tmp_path: Path):
    """Supported, but see _platforms(): it obliges the workflow to use
    buildx and to push straight to the registry, because a multi-platform
    build cannot be loaded into the local daemon."""
    _scaffold(tmp_path)
    _write_policy(
        tmp_path, [_entry()], platforms=["linux/amd64", "linux/arm64/v8"]
    )

    (entry,) = m.build_matrix(repo_root=tmp_path)

    assert entry["platforms"] == "linux/amd64,linux/arm64/v8"


@pytest.mark.parametrize(
    "value",
    [
        "linux/amd46/extra/parts",
        "linux",
        "/amd64",
        "linux/",
        "Linux/AMD64",
        "linux amd64",
        "   ",
        "",
    ],
)
def test_malformed_platform_is_refused(tmp_path: Path, value: str):
    """A typo here costs a runner slot and an opaque docker error."""
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()], platforms=[value])

    with pytest.raises(m.PublishMatrixError, match="is not a platform"):
        m.build_matrix(repo_root=tmp_path)


def test_platform_whitespace_is_stripped(tmp_path: Path):
    """A stray space survives the join into `--platform`, which docker
    rejects."""
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()], platforms=["  linux/amd64  "])

    (entry,) = m.build_matrix(repo_root=tmp_path)

    assert entry["platforms"] == "linux/amd64"


def test_duplicate_platform_is_refused(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(
        tmp_path, [_entry()], platforms=["linux/amd64", "linux/amd64"]
    )

    with pytest.raises(m.PublishMatrixError, match="twice"):
        m.build_matrix(repo_root=tmp_path)


# --------------------------------------------------------------------------
# Entry shape
# --------------------------------------------------------------------------


def test_entry_must_be_a_mapping(tmp_path: Path):
    _write_policy(tmp_path, ["core/python/demo"])

    with pytest.raises(m.PublishMatrixError, match="is not a mapping"):
        m.build_matrix(repo_root=tmp_path)


def test_entry_fixture_covers_every_required_key():
    """Keeps _entry() honest as REQUIRED_KEYS grows.

    test_missing_required_key deletes each required key from _entry() in
    turn, so a key added to the module but not to the fixture would surface
    as a KeyError inside the test rather than as the missing coverage it is.
    """
    assert set(_entry()) == m.REQUIRED_KEYS


@pytest.mark.parametrize("key", sorted(m.REQUIRED_KEYS))
def test_missing_required_key(tmp_path: Path, key: str):
    _scaffold(tmp_path)
    entry = _entry()
    del entry[key]
    _write_policy(tmp_path, [entry])

    with pytest.raises(m.PublishMatrixError, match=f"missing required.*{key}"):
        m.build_matrix(repo_root=tmp_path)


def test_unknown_key_is_refused(tmp_path: Path):
    """A misspelled key would otherwise be accepted and quietly ignored."""
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(dockefile="Dockerfile")])

    with pytest.raises(m.PublishMatrixError, match=r"unknown key.*dockefile"):
        m.build_matrix(repo_root=tmp_path)


@pytest.mark.parametrize(
    "name", ["Demo", "-demo", "demo/sub", "demo:tag", "", "DEMO"]
)
def test_invalid_image_names(tmp_path: Path, name: str):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(image=name)])

    with pytest.raises(m.PublishMatrixError, match="`image` must be"):
        m.build_matrix(repo_root=tmp_path)


def test_serves_http_must_be_boolean(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(serves_http="yes")])

    with pytest.raises(m.PublishMatrixError, match="must be true or false"):
        m.build_matrix(repo_root=tmp_path)


# --------------------------------------------------------------------------
# Paths and containment
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recipe",
    [
        "python/agents/demo",  # pre-migration legacy tree
        "java/agents/demo",  # pre-migration legacy tree
        "skills/retail/demo",  # out of scope for deployability
        "docs/demo",
    ],
)
def test_recipe_outside_publishable_roots(tmp_path: Path, recipe: str):
    _scaffold(tmp_path, recipe=recipe)
    _write_policy(tmp_path, [_entry(recipe=recipe)])

    with pytest.raises(m.PublishMatrixError, match="must be under"):
        m.build_matrix(repo_root=tmp_path)


def test_missing_recipe_directory(tmp_path: Path):
    _write_policy(tmp_path, [_entry()])

    with pytest.raises(m.PublishMatrixError, match="does not exist"):
        m.build_matrix(repo_root=tmp_path)


def test_recipe_without_manifest_is_not_a_recipe(tmp_path: Path):
    _scaffold(tmp_path, manifest=False)
    _write_policy(tmp_path, [_entry()])

    with pytest.raises(m.PublishMatrixError, match=r"no manifest\.yaml"):
        m.build_matrix(repo_root=tmp_path)


def test_missing_dockerfile(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(dockerfile="Dockerfile.nope")])

    with pytest.raises(m.PublishMatrixError, match="no Dockerfile at"):
        m.build_matrix(repo_root=tmp_path)


def test_context_must_be_a_directory(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(context="Dockerfile")])

    with pytest.raises(m.PublishMatrixError, match="is not a directory"):
        m.build_matrix(repo_root=tmp_path)


def test_dockerfile_outside_context_is_refused(tmp_path: Path):
    """Every COPY resolves against the context, so this pairing cannot work."""
    _scaffold(tmp_path)
    (tmp_path / "core/python/demo/sub").mkdir(parents=True, exist_ok=True)
    _write_policy(tmp_path, [_entry(dockerfile="Dockerfile", context="sub")])

    with pytest.raises(m.PublishMatrixError, match="outside its build context"):
        m.build_matrix(repo_root=tmp_path)


@pytest.mark.parametrize("field", ["dockerfile", "context"])
def test_absolute_paths_are_refused(tmp_path: Path, field: str):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(**{field: "/etc"})])

    with pytest.raises(m.PublishMatrixError, match="absolute path"):
        m.build_matrix(repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dockerfile", "../Dockerfile"),
        ("context", ".."),
        ("recipe", "core/../../etc"),
    ],
)
def test_parent_escapes_are_refused(tmp_path: Path, field: str, value: str):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(**{field: value})])

    with pytest.raises(
        m.PublishMatrixError, match=r"must stay inside|must be under"
    ):
        m.build_matrix(repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dockerfile", "Docker file"),
        ("dockerfile", "Dockerfile;rm -rf /"),
        ("dockerfile", "$(id)/Dockerfile"),
        ("dockerfile", "Dockerfile`whoami`"),
        ("context", "sub dir"),
        ("context", "sub&dir"),
        ("recipe", "core/python/de mo"),
        ("recipe", "core/python/demo$X"),
    ],
)
def test_shell_unsafe_path_characters_are_refused(
    tmp_path: Path, field: str, value: str
):
    """These values are pasted into the build command by the workflow.

    POSIX allows almost any byte in a filename, so a directory really can be
    called `demo;rm -rf /`. Constraining the declaration is more reliable
    than hoping every downstream use site quotes correctly.
    """
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(**{field: value})])

    with pytest.raises(
        m.PublishMatrixError, match=r"may only contain|must be under"
    ):
        m.build_matrix(repo_root=tmp_path)


# --------------------------------------------------------------------------
# Symlink containment
#
# The textual checks cannot see symlinks, and a reviewer reading
# `context: data` in policy.yml cannot tell that `data` links to /etc. These
# are the checks that do not depend on anyone noticing.
# --------------------------------------------------------------------------


def test_symlinked_context_out_of_repo_is_refused(tmp_path: Path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir()
    (outside / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    _scaffold(tmp_path)
    (tmp_path / "core/python/demo/data").symlink_to(
        outside, target_is_directory=True
    )
    _write_policy(
        tmp_path,
        [_entry(dockerfile="data/Dockerfile", context="data")],
    )

    with pytest.raises(m.PublishMatrixError, match="outside the repository"):
        m.build_matrix(repo_root=tmp_path)


def test_symlinked_dockerfile_out_of_repo_is_refused(tmp_path: Path):
    outside = tmp_path.parent / f"outside-df-{tmp_path.name}"
    outside.mkdir()
    (outside / "evil").write_text("FROM scratch\n", encoding="utf-8")

    _scaffold(tmp_path)
    (tmp_path / "core/python/demo/Dockerfile.link").symlink_to(outside / "evil")
    _write_policy(tmp_path, [_entry(dockerfile="Dockerfile.link")])

    with pytest.raises(m.PublishMatrixError, match="outside the repository"):
        m.build_matrix(repo_root=tmp_path)


def test_symlinked_recipe_out_of_repo_is_refused(tmp_path: Path):
    outside = tmp_path.parent / f"outside-recipe-{tmp_path.name}"
    (outside / "core/python/demo").mkdir(parents=True)
    (outside / "core/python/demo/manifest.yaml").write_text(
        "language: python\n", encoding="utf-8"
    )
    (outside / "core/python/demo/Dockerfile").write_text(
        "FROM scratch\n", encoding="utf-8"
    )

    (tmp_path / "core/python").mkdir(parents=True)
    (tmp_path / "core/python/demo").symlink_to(
        outside / "core/python/demo", target_is_directory=True
    )
    _write_policy(tmp_path, [_entry()])

    with pytest.raises(m.PublishMatrixError, match="outside the repository"):
        m.build_matrix(repo_root=tmp_path)


def test_symlink_inside_the_repo_is_allowed(tmp_path: Path):
    """Containment, not a blanket ban — an in-tree link is legitimate."""
    _scaffold(tmp_path)
    real = tmp_path / "core/python/demo/real"
    real.mkdir()
    (real / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (tmp_path / "core/python/demo/aliased").symlink_to(
        real, target_is_directory=True
    )
    _write_policy(
        tmp_path,
        [_entry(dockerfile="aliased/Dockerfile", context="aliased")],
    )

    (entry,) = m.build_matrix(repo_root=tmp_path)

    assert entry["context"] == "core/python/demo/aliased"


# --------------------------------------------------------------------------
# Duplicates — the failures that would otherwise be silent
# --------------------------------------------------------------------------


def test_duplicate_image_name_is_refused(tmp_path: Path):
    """Two entries, one public address: the second would replace the first."""
    _scaffold(tmp_path)
    _scaffold(tmp_path, recipe="contrib/python/demo")
    _write_policy(
        tmp_path,
        [_entry(), _entry(recipe="contrib/python/demo")],
    )

    with pytest.raises(m.PublishMatrixError, match="duplicate image name"):
        m.build_matrix(repo_root=tmp_path)


def test_same_dockerfile_twice_is_refused(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry(), _entry(image="demo-again")])

    with pytest.raises(m.PublishMatrixError, match="already published as"):
        m.build_matrix(repo_root=tmp_path)


# --------------------------------------------------------------------------
# Change detection
# --------------------------------------------------------------------------


def _two_images(tmp_path: Path) -> list[dict[str, Any]]:
    _scaffold(tmp_path)
    _scaffold(tmp_path, recipe="contrib/python/other")
    _write_policy(
        tmp_path,
        [_entry(), _entry(recipe="contrib/python/other", image="other")],
    )
    return m.build_matrix(repo_root=tmp_path)


def test_affected_by_selects_only_the_touched_recipe(tmp_path: Path):
    entries = _two_images(tmp_path)

    affected = m.affected_by(entries, ["core/python/demo/agent.py"])

    assert [e["image"] for e in affected] == ["demo"]


def test_affected_by_ignores_unrelated_changes(tmp_path: Path):
    entries = _two_images(tmp_path)

    assert m.affected_by(entries, ["README.md", "docs/thing.md"]) == []


@pytest.mark.parametrize("path", m.GLOBAL_REBUILD_PATHS)
def test_global_paths_rebuild_everything(tmp_path: Path, path: str):
    """The declaration and its tooling can change any image."""
    entries = _two_images(tmp_path)

    affected = m.affected_by(entries, [path])

    assert [e["image"] for e in affected] == ["demo", "other"]


def test_affected_by_preserves_declaration_order(tmp_path: Path):
    """Job names stay stable between runs, which matters when reading a
    matrix in the Actions UI."""
    entries = _two_images(tmp_path)

    affected = m.affected_by(
        entries, ["contrib/python/other/x.py", "core/python/demo/y.py"]
    )

    assert [e["image"] for e in affected] == ["demo", "other"]


def test_affected_by_tolerates_leading_dot_slash_and_blanks(tmp_path: Path):
    entries = _two_images(tmp_path)

    affected = m.affected_by(
        entries, ["./core/python/demo/agent.py", "", "   "]
    )

    assert [e["image"] for e in affected] == ["demo"]


def test_a_prefix_match_is_not_a_recipe_match(tmp_path: Path):
    """`core/python/demo-extra` must not count as a change to `demo`.

    Matching the bare recipe string rather than `recipe + "/"` would make
    every sibling whose name starts the same way look affected.
    """
    entries = _two_images(tmp_path)

    assert m.affected_by(entries, ["core/python/demo-extra/agent.py"]) == []


def test_the_recipe_directory_itself_is_not_a_change(tmp_path: Path):
    entries = _two_images(tmp_path)

    assert m.affected_by(entries, ["core/python/demo"]) == []


def test_changed_from_file_filters_the_matrix(tmp_path, monkeypatch, capsys):
    _two_images(tmp_path)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    changed = tmp_path / "changed.txt"
    changed.write_text("core/python/demo/agent.py\n", encoding="utf-8")

    assert m.main(["--changed-from", str(changed)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [e["image"] for e in payload] == ["demo"]


def test_changed_from_empty_result_is_success_not_an_error(
    tmp_path, monkeypatch, capsys
):
    """Nothing affected is the normal outcome for most pull requests.

    It must stay distinguishable from "nothing declared", which is a broken
    repository — the two are told apart by the exit code.
    """
    _two_images(tmp_path)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    changed = tmp_path / "changed.txt"
    changed.write_text("README.md\n", encoding="utf-8")

    assert m.main(["--changed-from", str(changed)]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == []
    assert "0 image(s) affected" in captured.err


def test_changed_from_still_validates_untouched_entries(
    tmp_path, monkeypatch, capsys
):
    """A broken entry fails the run even when the diff does not touch it.

    policy.yml is either valid or it is not. Letting an unrelated PR sail
    past a broken entry leaves it for whoever next touches that recipe.
    """
    _scaffold(tmp_path)
    _write_policy(
        tmp_path,
        [_entry(), _entry(recipe="contrib/python/ghost", image="ghost")],
    )
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    changed = tmp_path / "changed.txt"
    changed.write_text("README.md\n", encoding="utf-8")

    assert m.main(["--changed-from", str(changed)]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_changed_from_missing_file_is_an_error(tmp_path, monkeypatch, capsys):
    _two_images(tmp_path)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m.main(["--changed-from", str(tmp_path / "nope.txt")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_c_quoted_paths_warn_loudly_instead_of_being_skipped_silently(
    tmp_path: Path, capsys
):
    """Git C-quotes non-ASCII paths unless core.quotePath=false is set.

    Such a path matches no recipe prefix. If the workflow ever loses that
    flag, this warning is what turns an invisible missed rebuild into
    something a reader can see in the log.
    """
    entries = _two_images(tmp_path)

    affected = m.affected_by(entries, ['"core/python/demo/caf\\303\\251.md"'])

    assert affected == []
    err = capsys.readouterr().err
    assert "C-quoted" in err
    assert "core.quotePath=false" in err


def test_ordinary_paths_produce_no_quoting_warning(tmp_path: Path, capsys):
    entries = _two_images(tmp_path)

    m.affected_by(entries, ["core/python/demo/agent.py"])

    assert "C-quoted" not in capsys.readouterr().err


def test_global_rebuild_paths_exist_in_the_repo():
    """A stale entry here would silently stop triggering rebuilds.

    Each path is compared literally against `git diff` output, so a renamed
    file leaves a rule that can never match — and nothing else would notice.
    """
    for path in m.GLOBAL_REBUILD_PATHS:
        assert (REPO_ROOT / path).is_file(), f"{path} no longer exists"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_image_filter_selects_one(tmp_path: Path):
    _scaffold(tmp_path)
    _scaffold(tmp_path, dockerfile="sub/Dockerfile", context="sub")
    _write_policy(
        tmp_path,
        [
            _entry(),
            _entry(
                dockerfile="sub/Dockerfile", context="sub", image="demo-sub"
            ),
        ],
    )

    matrix = m.build_matrix(repo_root=tmp_path, only="demo-sub")

    assert [e["image"] for e in matrix] == ["demo-sub"]


def test_unknown_image_filter_is_an_error_not_an_empty_matrix(tmp_path: Path):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()])

    with pytest.raises(m.PublishMatrixError, match="is not a declared image"):
        m.build_matrix(repo_root=tmp_path, only="typo")


def test_main_emits_strict_json(tmp_path, monkeypatch, capsys):
    """stdout must be JSON, parsed with a JSON parser — not a YAML one.

    The consumer is `fromJson()` in a GitHub Actions workflow, which accepts
    strict JSON and nothing else. Asserting with yaml.safe_load would pass on
    output `fromJson` rejects, since YAML is a superset of JSON: it happily
    reads `[{image: demo, serves_http: True}]`, which json.loads refuses.
    That would leave this script's entire contract with the workflow untested.
    """
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()])
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m.main([]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload == [
        {
            "recipe": "core/python/demo",
            "image": "demo",
            "dockerfile": "core/python/demo/Dockerfile",
            "context": "core/python/demo",
            "serves_http": True,
            "platforms": "linux/amd64",
        }
    ]


def test_emitted_matrix_is_json_serialisable_scalars_only(tmp_path):
    """Every value must survive the round trip a matrix makes.

    GitHub expands matrix entries into the job context, so a nested object or
    a non-JSON scalar would arrive at the workflow as something it cannot
    interpolate.
    """
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()])

    for entry in m.build_matrix(repo_root=tmp_path):
        for key, value in entry.items():
            assert isinstance(value, (str, bool)), (
                f"{key} is {type(value).__name__}, which a matrix cannot carry"
            )


def test_main_validate_prints_summary_and_no_json(
    tmp_path, monkeypatch, capsys
):
    _scaffold(tmp_path)
    _write_policy(tmp_path, [_entry()])
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m.main(["--validate"]) == 0

    out = capsys.readouterr().out
    assert "1 image(s) declared and valid" in out
    assert "core/python/demo/Dockerfile" in out
    assert not out.lstrip().startswith("[")


def test_main_refuses_an_empty_matrix(tmp_path, monkeypatch, capsys):
    """A workflow that builds nothing must not report success."""
    _write_policy(tmp_path, [])
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m.main([]) == 1
    assert "empty matrix" in capsys.readouterr().err


def test_main_returns_1_on_error(tmp_path, monkeypatch, capsys):
    _write_policy(tmp_path, [_entry()])  # recipe never scaffolded
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m.main([]) == 1
    assert "does not exist" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Guard: the real declaration against the real repository
# --------------------------------------------------------------------------


def test_missing_pyyaml_reports_cleanly_instead_of_killing_the_import(
    tmp_path: Path, monkeypatch
):
    """The module must stay importable when PyYAML is absent.

    Exiting at import time would take the whole test session down during
    collection, and leave any other consumer unable to import the module to
    ask it anything at all.
    """
    monkeypatch.setattr(m, "yaml", None)

    with pytest.raises(m.PublishMatrixError, match="PyYAML is not installed"):
        m.build_matrix(repo_root=tmp_path)


def test_repo_root_derivations_agree():
    """The module and this file compute the repo root independently.

    They walk a different number of parents from different depths, so this
    asserts the two agree rather than assuming it — which is the reason the
    duplication is kept rather than replaced by `m.REPO_ROOT`.
    """
    assert REPO_ROOT == m.REPO_ROOT


def test_real_policy_declaration_is_valid():
    """Every declared image must still resolve on disk.

    This is the test that earns the file. `.github/scripts/tests/` is in the
    root pytest testpaths, so moving or deleting a declared Dockerfile turns
    this red on the PR that does it, rather than on a runner later.
    """
    matrix = m.build_matrix(repo_root=REPO_ROOT)

    assert matrix, "no images declared in policy.yml"
    for entry in matrix:
        assert (REPO_ROOT / entry["dockerfile"]).is_file()
        assert (REPO_ROOT / entry["context"]).is_dir()


def _copy_sources(dockerfile: Path) -> list[str]:
    """The context-relative COPY sources in a Dockerfile.

    Deliberately conservative — it returns only what can be resolved with
    certainty, because a false failure here blocks a legitimate declaration:

      * `COPY --from=<stage>` reads from an earlier build stage, not the
        context, so it is not a context question at all.
      * a source containing `$` is substituted from an ARG or ENV at build
        time and cannot be resolved by reading the file.
      * the JSON-array form, `COPY ["src", "dest"]`, which shlex would split
        into `['["src",', '"dest"]']` — quoted fragments that resolve to
        nothing and would be reported as missing files. Valid Dockerfile
        syntax, unused by any image declared today, and a false failure is
        exactly what this parser must not produce.
    """
    text = re.sub(r"\\\s*\n", " ", dockerfile.read_text(encoding="utf-8"))
    sources: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not re.match(r"(?i)^copy\s", stripped):
            continue
        try:
            parts = shlex.split(stripped)[1:]
        except ValueError:
            # Unbalanced quoting. The Dockerfile would not build either way,
            # and guessing at its intent here helps nobody.
            continue
        if not parts or parts[0].startswith("["):
            continue
        if any(p.startswith("--from=") for p in parts):
            continue
        args = [p for p in parts if not p.startswith("--")]
        # The last argument is the destination inside the image.
        sources.extend(s for s in args[:-1] if "$" not in s)
    return sources


def test_real_policy_copy_sources_resolve_in_context():
    """Every COPY source must exist inside the context declared for it.

    This is the one claim the declaration makes that nothing else can check.
    A wrong `context` satisfies every validation rule in publish_matrix.py —
    the directory exists, the Dockerfile is inside it — and then fails deep
    in `docker build` with a COPY error that names a path the reader cannot
    find in policy.yml. Checking it here turns that into a failing test on
    the PR that mis-declares it.

    Lives in the tests rather than in publish_matrix.py on purpose. Dockerfile
    parsing has enough edge cases that a false positive is a real risk, and a
    false positive in a test is an afternoon's annoyance while a false
    positive in the validator blocks publishing outright.
    """
    for entry in m.build_matrix(repo_root=REPO_ROOT):
        context = REPO_ROOT / entry["context"]
        for src in _copy_sources(REPO_ROOT / entry["dockerfile"]):
            if src.endswith("*"):
                # The `uv.lock*` idiom: a glob that deliberately tolerates
                # matching nothing, so absence is not an error.
                continue
            assert (context / src).exists(), (
                f"{entry['image']}: the Dockerfile COPYs {src!r}, which does "
                f"not exist in its declared context {entry['context']!r}"
            )


def test_copy_source_parser_skips_what_it_cannot_resolve(tmp_path: Path):
    """The parser's exclusions are load-bearing; pin them."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM scratch AS build\n"
        "COPY --chown=1000:1000 ./app ./app\n"
        "COPY --from=build /out /out\n"
        "COPY ${BUILD_DIR}/thing /thing\n"
        'COPY ["json form.txt", "/dest/"]\n'
        'COPY "unbalanced /dest/\n'
        "COPY a.txt \\\n    b.txt /dest/\n",
        encoding="utf-8",
    )

    assert _copy_sources(dockerfile) == ["./app", "a.txt", "b.txt"]


def test_real_policy_image_names_follow_the_convention():
    """Names start with the recipe's directory basename.

    The convention is what keeps a public pull address predictable from the
    recipe path. A secondary image adds a suffix; nothing else is allowed to
    drift, because renaming after publication breaks whoever pulled it.
    """
    for entry in m.build_matrix(repo_root=REPO_ROOT):
        basename = entry["recipe"].rsplit("/", 1)[-1]
        assert entry["image"].startswith(basename), (
            f"{entry['image']!r} does not start with the recipe basename "
            f"{basename!r}"
        )
