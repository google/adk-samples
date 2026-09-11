#!/usr/bin/env python3
"""Run the deterministic house-rules checker over the recipes a PR touches.

Used by .github/workflows/ai-pr-review-house-rules.yml. This is the fifth
review lane and the only one with no model in it.

The other four lanes ask a model to judge repository conventions from prose
injected into its prompt, with no tools and no checkout. Most of those
conventions are not judgement calls at all -- a `[tool.ruff]` table either is
in the file or is not -- and a model that gets one wrong produces the single
most expensive comment this system can make: a confident, specific, false
"this will fail CI". `.agents/skills/github-pr-review/scripts/check_house_rules.py`
already decides those rules by reading the files, so this lane runs it and
turns its findings into the same shape the model lanes emit.

Two properties make that safe to run against a pull request's own code:

  NOTHING FROM THE PR IS EXECUTED.  The checker parses. It reads text, walks
  directories, calls tomllib and `ast.parse`, and runs `git ls-files`. It
  never imports, execs, or installs anything out of the tree under review, so
  a hostile recipe gets no more privilege than a hostile text file.

  THE CHECKER ITSELF COMES FROM THE BASE BRANCH.  --checker points at the base
  checkout, never at the PR's copy. Otherwise a PR could rewrite the script
  that reviews it, which is the same reason the workflow injects the rules
  from the base branch rather than the head.

Usage:
  python3 house_rules_lane.py \
    --checker <base>/.agents/skills/github-pr-review/scripts/check_house_rules.py \
    --repo-root <head checkout> \
    --changed-files changed.txt \
    --out findings.json

Exit codes:
  0  findings written (possibly an empty list)
  2  CI fault -- the checker could not be loaded or run
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ci_message import (
    EXIT_OK,
    guard,
    infra_fault,
    report_infra_fault,
)

CHECKER = "house_rules_lane.py"

# A recipe root is <area>/<language>/<name> or skills/<vertical>/<solution>.
# Derived from the changed paths rather than by walking the tree: a PR that
# edits one recipe must not collect findings about the other four hundred.
#
# The lookahead is load-bearing. Without it `contrib/python/README.md` — a file
# that lives BESIDE the recipes, not in one — matches as a recipe called
# README.md, and the lane then reports a missing pyproject.toml, a missing
# README and a missing runnability test against a path that is not a recipe.
RECIPE_ROOT = re.compile(
    r"^(?:(?:core|contrib)/[^/]+/[^/]+|skills/[^/]+/[^/]+)(?=/)"
)


def recipe_roots(changed: list[str]) -> list[str]:
    """Every recipe directory the PR touches, in a stable order."""
    roots = set()
    for path in changed:
        match = RECIPE_ROOT.match(path.strip())
        if match:
            roots.add(match.group(0))
    return sorted(roots)


def load_checker(path: Path):
    """Import check_house_rules.py from an explicit path.

    Imported rather than shelled out to: the module keeps per-run state (the
    changed-file set, the git-tracked cache, the skipped-rule list) that has to
    be reset between recipes, and doing that in-process is both cheaper and
    honest about the coupling.
    """
    spec = importlib.util.spec_from_file_location("check_house_rules", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_reviewer_finding(finding: dict) -> dict:
    """Translate a checker finding into the shape the posting script reads.

    `window` is deliberately left out. The posting script treats a window as
    the model's claim about what the source says and rejects a finding whose
    claim is wrong -- that check exists because a model that invents a finding
    invents the source too. A deterministic checker read the file, so there is
    no claim to audit, and supplying a window would only add a way for this
    lane to fail.
    """
    body = str(finding.get("what") or "").strip()
    evidence = str(finding.get("evidence") or "").strip()
    if evidence and evidence not in body:
        # The citation is what makes a rule comment checkable rather than
        # arbitrary: a file the author can open beats a rule id they cannot.
        body = f"{body} ({evidence})"
    return {
        "path": finding.get("path"),
        "line": finding.get("line") or 1,
        "body": body,
        "verify_steps": finding.get("verify_steps") or "",
        "_rule": finding.get("rule"),
        "_ci": finding.get("ci"),
    }


def run_checker(module, repo_root: str, recipe: str, changed: set[str] | None):
    """Findings for one recipe, with the module's per-run state reset first."""
    module.SKIPPED = []
    module.FILTERED = []
    module.CHANGED = changed
    module.NEW_RECIPE = bool(changed) and any(
        c.startswith(recipe + "/")
        and c.endswith(("manifest.yaml", "pyproject.toml"))
        for c in changed
    )

    name = recipe.rsplit("/", 1)[-1]
    schema = str(Path(repo_root) / ".github/schemas/manifest-schema.json")
    pyproject = module.load_toml(
        str(Path(repo_root) / recipe / "pyproject.toml")
    )
    pyproject_name = (pyproject or {}).get("project", {}).get("name", "")

    out: list[dict] = []
    module.check_pyproject(out, repo_root, recipe, name)
    module.check_uv_lock(out, repo_root, recipe, name, pyproject_name)
    module.check_dotenv_bootstrap(out, repo_root, recipe)
    module.check_manifest(out, repo_root, recipe, schema)
    module.check_readme(out, repo_root, recipe)
    module.check_layout(out, repo_root, recipe, name)
    module.check_text_wide(out, repo_root, recipe)
    module.check_env_defaults(out, repo_root, recipe)
    module.check_license_headers(out, repo_root, recipe)
    module.check_pr_shape(out, repo_root, recipe)
    return out, list(module.SKIPPED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic house-rule findings for a pull request."
    )
    parser.add_argument(
        "--checker",
        required=True,
        type=Path,
        help="check_house_rules.py from the BASE checkout, never the PR's copy",
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        type=Path,
        help="checkout of the PR head; read as data, never executed",
    )
    parser.add_argument(
        "--changed-files",
        required=True,
        type=Path,
        help="one PR-changed path per line",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="findings destination"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        changed = [
            line.strip()
            for line in args.changed_files.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
    except OSError as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"cannot read {args.changed_files}: {exc}")
        )

    roots = recipe_roots(changed)
    if not roots:
        print("No recipe directories in this PR; nothing for this lane to do.")
        args.out.write_text("[]", encoding="utf-8")
        return EXIT_OK

    try:
        module = load_checker(args.checker)
    except Exception as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"cannot load {args.checker}: {exc}")
        )

    findings: list[dict] = []
    for recipe in roots:
        if not (args.repo_root / recipe).is_dir():
            # Deleted, or renamed away. Nothing to check and nothing wrong.
            continue
        try:
            raw, skipped = run_checker(
                module, str(args.repo_root), recipe, set(changed)
            )
        except Exception as exc:
            # One unreadable recipe must not cost the review every finding in
            # the others. A plain line, not an annotation: nothing a
            # contributor reading their PR can act on, and ci_message owns the
            # annotations a contributor does see.
            print(f"  house-rules check failed for {recipe}: {exc}")
            continue
        print(f"{recipe}: {len(raw)} finding(s)")
        # A rule that could not be evaluated is not a rule that passed.
        for rule, why in skipped:
            print(f"  not checked — {rule}: {why}")
        findings.extend(to_reviewer_finding(f) for f in raw)

    # CI-failing findings first: an author acts on "this blocks the build" and
    # may never act on a convention nit.
    findings.sort(key=lambda f: (f.get("_ci") != "fail", f.get("path") or ""))

    args.out.write_text(json.dumps(findings, indent=1), encoding="utf-8")
    print(f"{len(findings)} finding(s) across {len(roots)} recipe(s).")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(guard(CHECKER, main))
