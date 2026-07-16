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
Guardrail for prepare-python-recipe Phase 0: verify the recipe's folder
name matches the two rules that .github/workflows/python-validate-recipe.yml
enforces so the pipeline halts LOCALLY on a bad name instead of running to
completion and shipping files under a path CI will reject.

Rules enforced (mirrored from python-validate-recipe.yml, Check 1):
  1. Regex: ^[a-z][a-z-]*$  — lowercase letters and hyphens only, must
     start with a letter. Rejects underscores, uppercase, digits, and
     symbols.
  2. Length: <= --max-length characters (default 30, source of truth:
     .github/policy.yml `recipe_naming.max_folder_name_length`). The
     caller is expected to look up the live value and pass it in; the
     default is a safety net for standalone use only.

On violation, prints one line per broken rule PLUS a suggested compliant
name derived from the current one (lowercase, `_` → `-`, drop other
characters, collapse `--`, strip trailing `-`, truncate at hyphen
boundary when possible so we don't cut mid-word). The suggestion is
ADVISORY — the script never renames anything on disk. Exits 1.

On pass, prints one PASS line and exits 0.

Usage:
  python check_folder_name.py --recipe-dir <path> [--max-length N]

Stdlib-only so it can be invoked with `uv run --no-project python3` and
never pulls in transitive deps. If the caller wants to source the max
length from `.github/policy.yml` at runtime, they should call
`.github/scripts/load_policy.py recipe_naming.max_folder_name_length`
and pass the result via --max-length — that keeps this script decoupled
from the policy file's format.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VALID_NAME_RE = re.compile(r"^[a-z][a-z-]*$")
DEFAULT_MAX_LENGTH = 30


def suggest_compliant_name(name: str, max_len: int) -> str:
    """Derive a name that satisfies both rules from a possibly-bad input.

    Transformations, in order:
      1. Lowercase everything.
      2. Replace ``_`` with ``-`` (the most common ADK-recipe author habit).
      3. Drop any character that isn't in ``[a-z-]`` (digits, punctuation,
         accented letters, whitespace, etc.).
      4. Collapse runs of ``-`` into a single ``-``.
      5. Strip leading/trailing ``-``.
      6. If longer than ``max_len``, truncate — preferring to cut at a
         hyphen boundary within the second half of the string so the
         suggestion doesn't end mid-word ("airflow-version-upg" is
         uglier than "airflow-version" for the same violation).
      7. Strip any trailing ``-`` left over from truncation.

    Returns the suggested name, or "" when no salvageable suggestion exists
    (empty after transformations, or starts with a non-letter and there is
    no letter anywhere to promote to the front). Callers should render the
    empty-string case as "no automatic suggestion; please choose a name
    manually."
    """
    normalized = name.lower().replace("_", "-")
    filtered = "".join(c for c in normalized if c == "-" or "a" <= c <= "z")
    while "--" in filtered:
        filtered = filtered.replace("--", "-")
    filtered = filtered.strip("-")

    if len(filtered) > max_len:
        truncated = filtered[:max_len]
        last_hyphen = truncated.rfind("-")
        # Only cut at hyphen if it doesn't discard more than half of what
        # fit — otherwise the mid-word cut is closer to the user's intent.
        if last_hyphen >= max_len // 2:
            truncated = truncated[:last_hyphen]
        filtered = truncated.rstrip("-")

    if not filtered or not ("a" <= filtered[0] <= "z"):
        return ""
    return filtered


def check(folder_name: str, max_len: int) -> list[str]:
    """Return a list of violation descriptions; empty when the name is OK."""
    violations: list[str] = []
    if not VALID_NAME_RE.match(folder_name):
        # Enumerate the concrete offending characters/positions so the
        # user sees exactly what to fix, not just "bad regex".
        offenders = sorted(
            {c for c in folder_name if not (c == "-" or "a" <= c <= "z")}
        )
        detail_bits: list[str] = []
        if folder_name and not ("a" <= folder_name[0] <= "z"):
            detail_bits.append(
                f"starts with '{folder_name[0]}' (must start with a lowercase letter)"
            )
        if offenders:
            offender_str = ", ".join(f"'{c}'" for c in offenders)
            detail_bits.append(f"disallowed characters: {offender_str}")
        detail = (
            "; ".join(detail_bits)
            if detail_bits
            else "does not match ^[a-z][a-z-]*$"
        )
        violations.append(
            f"Folder name '{folder_name}' fails regex ^[a-z][a-z-]*$ — {detail}."
        )
    if len(folder_name) > max_len:
        violations.append(
            f"Folder name '{folder_name}' is {len(folder_name)} characters, "
            f"exceeds max {max_len} (source: .github/policy.yml "
            "recipe_naming.max_folder_name_length)."
        )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a recipe folder name matches CI's rules."
    )
    parser.add_argument(
        "--recipe-dir",
        required=True,
        help="Path to the recipe root; the folder basename is what gets checked.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=(
            f"Max allowed name length (default {DEFAULT_MAX_LENGTH}). Live "
            "value lives in .github/policy.yml under "
            "recipe_naming.max_folder_name_length — pass it explicitly to "
            "stay in sync with CI."
        ),
    )
    args = parser.parse_args(argv)

    recipe_path = Path(args.recipe_dir)
    folder_name = recipe_path.name

    violations = check(folder_name, args.max_length)

    if not violations:
        print(
            f"[PASS] Folder name '{folder_name}' is compliant "
            f"(matches ^[a-z][a-z-]*$, length {len(folder_name)} <= "
            f"{args.max_length})."
        )
        return 0

    print("[FAIL] Recipe folder name violates the CI naming rules.\n")
    for v in violations:
        print(f"  - {v}")

    suggestion = suggest_compliant_name(folder_name, args.max_length)
    print()
    if suggestion:
        print(
            f"Suggested compliant name: '{suggestion}'\n"
            f"  (derived from '{folder_name}' by lowercasing, replacing '_' "
            "with '-', dropping disallowed characters, and truncating on a "
            "hyphen boundary — advisory only, tweak as you like.)"
        )
    else:
        print(
            "No automatic suggestion — please choose a name that matches "
            f"^[a-z][a-z-]*$ and is at most {args.max_length} chars long."
        )

    print(
        "\nRename the directory manually and re-run the pipeline. The "
        "prepare-python-recipe skill NEVER renames recipe directories on "
        "its own — that's your call. Example:\n"
        f"  git mv '{recipe_path}' '{recipe_path.parent / (suggestion or 'YOUR-CHOSEN-NAME')}'"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
