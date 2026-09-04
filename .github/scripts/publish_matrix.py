#!/usr/bin/env python3
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
"""
Build the job matrix for the recipe image publishing workflow.

Reads `deployability.publish` from .github/policy.yml, validates every
declared image against the filesystem, and emits one entry per image as a
JSON array on stdout:

    [{"recipe": "core/python/long-horizon-harness",
      "image": "long-horizon-harness",
      "dockerfile": "core/python/long-horizon-harness/Dockerfile",
      "context": "core/python/long-horizon-harness",
      "serves_http": true,
      "platforms": "linux/amd64"}, ...]

`dockerfile` and `context` come back REPO-RELATIVE, already joined onto the
recipe, because that is what `docker build -f <dockerfile> <context>` wants
from a workspace checkout. The policy file stores them relative to the recipe
because that is what is readable when editing it; resolving the difference
here rather than in shell keeps the workflow free of path arithmetic.

Why an allowlist rather than a scan
-----------------------------------
`docker build` runs every RUN instruction it is given, and these images are
published publicly. Deriving the set from "a Dockerfile exists" would let a
contributor add one and have it built and pushed under Google's name without
an admin ever agreeing. It would also pick up Dockerfiles that cannot be
published as they stand: several in this repo re-resolve dependencies at
build time instead of installing from a committed lockfile, and one is a
template full of `{{PLACEHOLDER}}` that does not build at all.

Why validation is strict
------------------------
Every rule below turns a silent, expensive failure into a loud, cheap one.
A typo'd path fails the build minutes in, on a runner, with a message about a
missing file rather than about the policy entry that named it. A duplicate
image name is worse than a failure: two entries push to the same address and
the second silently replaces the first, so a green run publishes an image
nobody asked for. Unknown keys are rejected for the same reason — `dockefile`
would otherwise be accepted and ignored, and the entry would build the wrong
thing while looking correct in review.

An empty matrix exits non-zero. A workflow that builds nothing and reports
success is indistinguishable from one that is working, which is how a broken
selector survives for months.

Usage:
    publish_matrix.py                    # every declared image
    publish_matrix.py --image long-horizon-harness
    publish_matrix.py --validate         # check only, human-readable output

Invoke in CI as:
    uv run --no-project --with pyyaml python3 .github/scripts/publish_matrix.py

`--no-project` because this script needs the standard library plus yaml and
nothing else; without it uv resolves and builds the root project for no
reason. Matches how stale-sweep.yml invokes load_policy.py.

This script does NOT use load_policy.py. That helper prints scalars and
lists, and `publish.images` is a list of dicts, which it cannot represent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without pyyaml
    # Recorded, not acted on. Calling sys.exit() here would make the module
    # unimportable rather than merely unusable, so anything that imports it —
    # the test suite included — would die during collection with a SystemExit
    # instead of a readable failure. The missing dependency is reported by
    # load_publish_block(), the first function that actually needs it.
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Recipes may only be published from the curated and community trees.
#
# This one rule is also what excludes everything else: the pre-migration
# python/agents/ and java/agents/ trees, and skills/, which policy.yml
# already places out of scope for deployability. Naming the two roots that
# ARE allowed keeps that exclusion from drifting as directories come and go.
PUBLISHABLE_ROOTS = ("core/", "contrib/")

# Artifact Registry image names: lowercase alphanumerics, dashes,
# underscores and dots, starting with an alphanumeric. Deliberately tighter
# than the registry strictly requires — these names are public and permanent
# in practice, so the uppercase and slash-bearing forms a registry would
# tolerate are refused rather than debated later.
#
# The tightness is also load-bearing downstream. These names are pasted into
# an image reference in the build workflow, so a name containing whitespace,
# a quote or a shell metacharacter would be an injection point. The character
# class here makes every emitted name shell-safe by construction, which is
# cheaper and more reliable than quoting correctly at each use site.
IMAGE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# `os/arch`, optionally `os/arch/variant` — linux/amd64, linux/arm64/v8.
PLATFORM_RE = re.compile(r"^[a-z0-9]+/[a-z0-9]+(?:/[a-z0-9]+)?$")

# A change to any of these rebuilds EVERY declared image, not just one.
#
# The declaration and the code that reads it decide what gets built and how,
# so a change to either can alter an image without touching the recipe it
# comes from. The workflow file is here for the same reason: it owns the
# build arguments. Rebuilding everything is the cheap, obviously-correct
# answer for a handful of images, and the alternative — reasoning about
# which policy edit affects which entry — is the kind of cleverness that
# quietly stops rebuilding something.
GLOBAL_REBUILD_PATHS = (
    ".github/policy.yml",
    ".github/scripts/publish_matrix.py",
    ".github/workflows/recipe-images.yml",
)

# The characters a recipe, Dockerfile or context path may contain.
#
# Same reasoning as IMAGE_NAME_RE above, and for the same reason: these
# values are pasted into `docker build -f <dockerfile> <context>` by the
# build workflow. A path holding a space, a quote, a semicolon or a `$`
# would be a command-injection point at that use site, and the filesystem
# is perfectly willing to hold such a directory — POSIX permits almost
# anything but NUL and `/` in a name. Constraining the input is more
# reliable than hoping every downstream use site quotes correctly.
#
# Uppercase is allowed because filenames need it (`Dockerfile`, `README.md`)
# even though recipe directories are lowercase by convention.
PATH_CHARS_RE = re.compile(r"^[A-Za-z0-9._/-]+$")

REQUIRED_KEYS = frozenset(
    {"recipe", "dockerfile", "context", "image", "serves_http"}
)


class PublishMatrixError(RuntimeError):
    """The matrix cannot be built as declared.

    Always fatal. A publishing workflow that cannot say which images to build
    must not fall back to building some of them.
    """


def _policy_path(repo_root: Path) -> Path:
    return repo_root / ".github" / "policy.yml"


def load_publish_block(repo_root: Path) -> dict[str, Any]:
    """Read `deployability.publish` from policy.yml.

    Raises rather than returning a default for a missing block: the caller is
    a workflow whose entire purpose is publishing these images, so an absent
    declaration is a broken repository, not an empty one.
    """
    if yaml is None:
        raise PublishMatrixError(
            "PyYAML is not installed. Run via "
            "`uv run --no-project --with pyyaml python3 ...`."
        )

    path = _policy_path(repo_root)
    try:
        with open(path, "rb") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise PublishMatrixError(f"{path} not found") from exc
    except OSError as exc:
        # Permission denied, a directory where the file should be, an I/O
        # error. Reported as a policy failure rather than a traceback: the
        # caller is a workflow, and a stack trace in its log says nothing a
        # reader can act on.
        raise PublishMatrixError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PublishMatrixError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise PublishMatrixError(f"{path} does not contain a YAML mapping")

    deployability = data.get("deployability")
    if not isinstance(deployability, dict):
        raise PublishMatrixError(
            f"{path} has no `deployability` section; "
            f"the publish declaration lives under it"
        )

    publish = deployability.get("publish")
    if not isinstance(publish, dict):
        raise PublishMatrixError(
            f"{path} has no `deployability.publish` section"
        )
    return publish


def _platforms(publish: dict[str, Any]) -> str:
    """The `--platform` value for `docker build`, as a comma-joined string.

    Carried on every matrix entry rather than fetched separately by the
    workflow. One lookup, one shape, and no way for the two to disagree.

    MORE THAN ONE PLATFORM IS NOT FREE. `docker build --platform a,b` is
    rejected by the classic builder outright, and under buildx a multi-
    platform build cannot be loaded into the local daemon — it has to go
    straight to a registry with `--push` (or to `--output`). So a second
    entry here is not a one-line change: it obliges the consuming workflow
    to use buildx and to stop doing anything that needs the image locally,
    such as probing it before publishing.

    Validated rather than trusted because the failure is remote and late: a
    typo'd `linux/amd46` costs a runner slot and a confusing docker error,
    when it can be caught here for nothing.
    """
    raw = publish.get("platforms")
    if raw is None:
        raise PublishMatrixError(
            "`deployability.publish.platforms` is missing. Declare it "
            "explicitly — defaulting would silently build for the runner's "
            "architecture, which is not necessarily the deployment target's."
        )
    if not isinstance(raw, list) or not raw:
        raise PublishMatrixError(
            "`deployability.publish.platforms` must be a non-empty list"
        )

    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise PublishMatrixError(
                f"`deployability.publish.platforms` contains a non-string "
                f"entry: {item!r}"
            )
        # Stripped before use: a stray space survives the join and reaches
        # `docker build --platform " linux/amd64"`, which docker rejects.
        # An entry that is ONLY whitespace falls through to the shape check
        # below, which names the real problem — calling it "non-string" when
        # it is a string sends the reader looking for the wrong mistake.
        value = item.strip()
        if not PLATFORM_RE.match(value):
            raise PublishMatrixError(
                f"`deployability.publish.platforms` entry {item!r} is not a "
                f"platform. Expected os/arch, optionally with a variant — "
                f"for example linux/amd64 or linux/arm64/v8."
            )
        if value in cleaned:
            raise PublishMatrixError(
                f"`deployability.publish.platforms` lists {value!r} twice"
            )
        cleaned.append(value)
    return ",".join(cleaned)


def _check_relative(value: str, field: str, image: str) -> None:
    """Reject absolute paths, parent escapes and unsafe characters.

    The first two are containment failures with the same consequence: a build
    context or Dockerfile outside the recipe it claims to belong to, which
    puts files the recipe does not own into an image published under its name.

    The third is a downstream concern — see PATH_CHARS_RE. It is enforced
    here, at the single point every path field passes through, rather than at
    each place a path is later used.
    """
    if Path(value).is_absolute():
        raise PublishMatrixError(
            f"image {image!r}: `{field}` must be relative to the recipe, "
            f"got the absolute path {value!r}"
        )
    if ".." in Path(value).parts:
        raise PublishMatrixError(
            f"image {image!r}: `{field}` must stay inside the recipe, "
            f"got {value!r}"
        )
    if not PATH_CHARS_RE.match(value):
        raise PublishMatrixError(
            f"image {image!r}: `{field}` may only contain letters, digits, "
            f"dot, dash, underscore and slash — got {value!r}. These paths "
            f"are interpolated into the build command, so a character the "
            f"shell treats specially is refused rather than quoted."
        )


def _check_inside_repo(
    resolved: Path, repo_root: Path, field: str, image: str, shown: str
) -> None:
    """Refuse a path that resolves outside the repository.

    The textual checks above cannot see symlinks. A recipe may contain one
    pointing anywhere on the host, and a reviewer reading `context: data` in
    policy.yml has no way to tell that `data` is a link to /etc — which is
    precisely why "an admin approved it" is not the safeguard it appears to
    be. Resolving both sides and comparing is the check that does not depend
    on anyone noticing.

    repo_root is resolved by the caller as well, so a repository that itself
    lives under a symlink (/tmp on macOS is the everyday case) compares
    like with like instead of failing for everyone.
    """
    if not resolved.is_relative_to(repo_root):
        raise PublishMatrixError(
            f"image {image!r}: `{field}` {shown!r} resolves to {resolved}, "
            f"outside the repository at {repo_root}. A symlink leading out "
            f"of the tree would copy host files into a public image."
        )


def _validate_entry(
    raw: Any, index: int, repo_root: Path, platforms: str
) -> dict[str, Any]:
    """Validate one declared image and return its normalised matrix entry."""
    where = f"`deployability.publish.images[{index}]`"

    if not isinstance(raw, dict):
        raise PublishMatrixError(f"{where} is not a mapping")

    keys = set(raw)
    missing = REQUIRED_KEYS - keys
    if missing:
        raise PublishMatrixError(
            f"{where} is missing required key(s): {', '.join(sorted(missing))}"
        )
    # Fail closed on anything unrecognised. A misspelled key would otherwise
    # be accepted and ignored, and the entry would build something other than
    # what its author read back in review.
    unknown = keys - REQUIRED_KEYS
    if unknown:
        raise PublishMatrixError(
            f"{where} has unknown key(s): {', '.join(sorted(unknown))}. "
            f"Allowed keys: {', '.join(sorted(REQUIRED_KEYS))}"
        )

    image = raw["image"]
    if not isinstance(image, str) or not IMAGE_NAME_RE.match(image):
        raise PublishMatrixError(
            f"{where}: `image` must be a lowercase name matching "
            f"{IMAGE_NAME_RE.pattern}, got {image!r}"
        )

    serves_http = raw["serves_http"]
    if not isinstance(serves_http, bool):
        raise PublishMatrixError(
            f"image {image!r}: `serves_http` must be true or false, "
            f"got {serves_http!r}"
        )

    recipe = raw["recipe"]
    if not isinstance(recipe, str) or not recipe:
        raise PublishMatrixError(
            f"image {image!r}: `recipe` must be a non-empty string"
        )
    recipe = recipe.rstrip("/")
    if not recipe.startswith(PUBLISHABLE_ROOTS):
        raise PublishMatrixError(
            f"image {image!r}: `recipe` must be under one of "
            f"{', '.join(PUBLISHABLE_ROOTS)} — got {recipe!r}"
        )
    _check_relative(recipe, "recipe", image)

    # Resolved once here and reused, so every containment comparison below is
    # made against the same real location.
    repo_real = repo_root.resolve()
    recipe_dir = repo_root / recipe
    if not recipe_dir.is_dir():
        raise PublishMatrixError(
            f"image {image!r}: recipe directory {recipe!r} does not exist"
        )
    _check_inside_repo(recipe_dir.resolve(), repo_real, "recipe", image, recipe)
    if not (recipe_dir / "manifest.yaml").is_file():
        raise PublishMatrixError(
            f"image {image!r}: {recipe!r} has no manifest.yaml, so it is not "
            f"a recipe"
        )

    dockerfile = raw["dockerfile"]
    if not isinstance(dockerfile, str) or not dockerfile:
        raise PublishMatrixError(
            f"image {image!r}: `dockerfile` must be a non-empty string"
        )
    _check_relative(dockerfile, "dockerfile", image)
    dockerfile_path = recipe_dir / dockerfile
    if not dockerfile_path.is_file():
        raise PublishMatrixError(
            f"image {image!r}: no Dockerfile at {recipe}/{dockerfile}"
        )
    _check_inside_repo(
        dockerfile_path.resolve(), repo_real, "dockerfile", image, dockerfile
    )

    context = raw["context"]
    if not isinstance(context, str) or not context:
        raise PublishMatrixError(
            f"image {image!r}: `context` must be a non-empty string "
            f'(use "." for the recipe root)'
        )
    _check_relative(context, "context", image)
    context_path = recipe_dir / context
    if not context_path.is_dir():
        raise PublishMatrixError(
            f"image {image!r}: build context {recipe}/{context} is not a "
            f"directory"
        )
    _check_inside_repo(
        context_path.resolve(), repo_real, "context", image, context
    )

    # `docker build -f` accepts a Dockerfile outside the context, but every
    # COPY in it resolves against the context, so the pairing is almost
    # always a mistake rather than an intention. Caught here because the
    # symptom otherwise appears as a confusing COPY failure mid-build.
    if not dockerfile_path.resolve().is_relative_to(context_path.resolve()):
        raise PublishMatrixError(
            f"image {image!r}: the Dockerfile {recipe}/{dockerfile} is "
            f"outside its build context {recipe}/{context}, so nothing it "
            f"COPYs can resolve"
        )

    return {
        "recipe": recipe,
        "image": image,
        # Repo-relative and normalised, ready for `docker build -f X Y`.
        # pathlib drops "." components, so a context of "." collapses to the
        # recipe directory on its own.
        "dockerfile": (Path(recipe) / dockerfile).as_posix(),
        "context": (Path(recipe) / context).as_posix(),
        "serves_http": serves_http,
        "platforms": platforms,
    }


def affected_by(
    entries: list[dict[str, Any]], changed: list[str]
) -> list[dict[str, Any]]:
    """The subset of `entries` a set of changed files could have altered.

    An image is affected when a changed file lies inside its recipe. The
    build context is always inside the recipe (the validator enforces it), so
    recipe containment is the whole test — no separate context check is
    needed, and adding one would only invite the two to disagree.

    A change to anything in GLOBAL_REBUILD_PATHS affects every image.

    Order is preserved from the declaration so job names stay stable between
    runs, which matters for reading a matrix in the Actions UI.
    """
    # removeprefix, NOT lstrip("./"): lstrip strips a SET of characters, so
    # it would turn `.github/policy.yml` into `github/policy.yml` and quietly
    # stop every global rebuild rule from ever matching.
    normalised = [c.strip().removeprefix("./") for c in changed if c.strip()]

    # A leading quote means the caller handed us git's C-quoted form, which
    # git emits for any path containing a non-ASCII byte unless
    # `core.quotePath=false` is set. Such a path matches no recipe prefix, so
    # the image owning it would be skipped — silently, which is the one
    # outcome worth shouting about.
    #
    # A warning rather than an error: the other paths in the same list were
    # still classified correctly, and refusing to build anything would turn a
    # partial miss into a total one.
    quoted = [c for c in normalised if c.startswith('"')]
    if quoted:
        print(
            f"WARNING: {len(quoted)} changed path(s) arrived C-quoted, e.g. "
            f"{quoted[0]}. The caller is missing `-c core.quotePath=false`; "
            f"images owning those paths will NOT be rebuilt.",
            file=sys.stderr,
        )

    if any(c in GLOBAL_REBUILD_PATHS for c in normalised):
        return list(entries)

    affected = []
    for entry in entries:
        prefix = entry["recipe"].rstrip("/") + "/"
        if any(c.startswith(prefix) for c in normalised):
            affected.append(entry)
    return affected


def build_matrix(
    repo_root: Path | None = None, only: str | None = None
) -> list[dict[str, Any]]:
    """Validate the whole publish block and return the job matrix."""
    root = REPO_ROOT if repo_root is None else repo_root
    publish = load_publish_block(root)
    platforms = _platforms(publish)

    images = publish.get("images")
    if images is None:
        raise PublishMatrixError(
            "`deployability.publish.images` is missing. Declare the images "
            "to publish, or remove the `publish` block entirely."
        )
    if not isinstance(images, list):
        raise PublishMatrixError(
            f"`deployability.publish.images` must be a list, got "
            f"{type(images).__name__}"
        )

    entries: list[dict[str, Any]] = []
    seen_names: dict[str, int] = {}
    seen_builds: dict[tuple[str, str], str] = {}

    for index, raw in enumerate(images):
        entry = _validate_entry(raw, index, root, platforms)

        name = entry["image"]
        if name in seen_names:
            raise PublishMatrixError(
                f"duplicate image name {name!r} at images[{index}] and "
                f"images[{seen_names[name]}]. Names become public pull "
                f"addresses, so two entries sharing one would push to the "
                f"same place and the second would replace the first."
            )
        seen_names[name] = index

        build = (entry["dockerfile"], entry["context"])
        if build in seen_builds:
            raise PublishMatrixError(
                f"{entry['dockerfile']} is already published as "
                f"{seen_builds[build]!r}; declaring it again as {name!r} "
                f"would build one image twice under two names"
            )
        seen_builds[build] = name

        entries.append(entry)

    if only is not None:
        matched = [e for e in entries if e["image"] == only]
        if not matched:
            # Taking `--image` on trust would produce an empty matrix, which
            # the caller reports as "nothing to build" rather than as the
            # typo it is.
            known = ", ".join(sorted(seen_names)) or "(none declared)"
            raise PublishMatrixError(
                f"{only!r} is not a declared image. Known images: {known}"
            )
        return matched

    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the recipe image publishing matrix."
    )
    parser.add_argument(
        "--image",
        help="Limit the matrix to one declared image name.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the declaration and print a summary; emit no JSON.",
    )
    parser.add_argument(
        "--changed-from",
        metavar="FILE",
        help=(
            "Limit the matrix to images affected by the changed paths listed "
            "in FILE, one per line. An empty result is legitimate here and "
            "emits [] with exit 0."
        ),
    )
    args = parser.parse_args(argv)

    try:
        matrix = build_matrix(only=args.image)
    except PublishMatrixError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # The whole declaration is validated FIRST, above, and only then filtered.
    # A broken entry therefore fails the run even when the change under test
    # does not touch it — which is the point: policy.yml is either valid or it
    # is not, and letting an unrelated PR sail past a broken entry would leave
    # it to be discovered by whoever next touches that recipe.
    if not matrix:
        print(
            "error: no images are declared in `deployability.publish.images`."
            " Refusing to report success on an empty matrix — a workflow that"
            " builds nothing looks exactly like one that is working.",
            file=sys.stderr,
        )
        return 1

    if args.changed_from is not None:
        try:
            text = Path(args.changed_from).read_text(encoding="utf-8")
        except OSError as exc:
            print(
                f"error: cannot read {args.changed_from}: {exc}",
                file=sys.stderr,
            )
            return 1
        changed = text.splitlines()
        matrix = affected_by(matrix, changed)
        # Deliberately NOT an error when empty. Unlike an empty declaration,
        # "this change touched no image" is the normal outcome for most pull
        # requests. The two are distinguished by exit code: 1 above, 0 here.
        #
        # On stderr so it cannot contaminate the JSON on stdout.
        print(
            f"{len(matrix)} image(s) affected by "
            f"{len(changed)} changed path(s)",
            file=sys.stderr,
        )

    if args.validate:
        if args.changed_from is not None:
            print(f"{len(matrix)} image(s) affected and valid:")
        else:
            print(f"{len(matrix)} image(s) declared and valid:")
        for entry in matrix:
            serves = "http" if entry["serves_http"] else "no-server"
            print(
                f"  {entry['image']:<40} {entry['dockerfile']} "
                f"(context: {entry['context']}, {serves})"
            )
        return 0

    json.dump(matrix, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
