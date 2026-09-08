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
"""Process the AI issue responder's output and determine ticket actions.

Used by .github/workflows/_ai-issue-response-core.yml. Parses the JSON output
from Antigravity CLI (agy), validates the selected core functional option
(clarify, simple solution, detailed solution, acknowledge and assign), resolves
developer assignment based on repository ownership paths when Option 4 is
selected, and prepares the response comment.

Routing rules for Option 4:
  Catch-all (docs, CI workflows, root configs, unmatched): @happyhuman
  /core/python/**      -> @eliasecchig
  /core/go/**          -> @tklopfenstein
  /core/java/**        -> @eliasecchig
  /core/typescript/**  -> @happyhuman
  /core/kotlin/**      -> @happyhuman
  /contrib/python/**   -> @happyhuman
  /contrib/go/**       -> @tklopfenstein
  /contrib/java/**     -> @happyhuman
  /contrib/typescript/** -> @happyhuman
  /contrib/kotlin/**   -> @happyhuman
  /skills/**           -> @happyhuman

Usage:
  python3 process_issue_response.py \\
    --result agy_result.json \\
    --issue-number 123 \\
    --comment-out comment.md \\
    --assignee-out assignee.txt \\
    [--github-output "$GITHUB_OUTPUT"]

Exit codes:
  0  success (comment and optional assignee written)
  2  CI fault (unreadable result or execution crash)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ci_message import (
    EXIT_OK,
    guard,
    infra_fault,
    report_infra_fault,
)

CHECKER = "process_issue_response.py"


class Option(IntEnum):
    """The 4 core functional options for issue triage response."""

    CLARIFY = 1
    SIMPLE_SOLUTION = 2
    DETAILED_SOLUTION = 3
    ACKNOWLEDGE_AND_ASSIGN = 4


ROUTING_RULES: list[tuple[str, str]] = [
    # Core directory assignments
    ("core/python", "eliasecchig"),
    ("core/go", "tklopfenstein"),
    ("core/java", "eliasecchig"),
    ("core/typescript", "happyhuman"),
    ("core/kotlin", "happyhuman"),
    # Contrib directory assignments
    ("contrib/python", "happyhuman"),
    ("contrib/go", "tklopfenstein"),
    ("contrib/java", "happyhuman"),
    ("contrib/typescript", "happyhuman"),
    ("contrib/kotlin", "happyhuman"),
    # Skills directory assignments
    ("skills", "happyhuman"),
]

DEFAULT_ASSIGNEE = "happyhuman"

# Known developer usernames (without @ prefix)
VALID_ASSIGNEES = {"eliasecchig", "tklopfenstein", "happyhuman"}

FENCED_JSON = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def normalize_path(path: str) -> str:
    """Normalize a repository path for routing comparison."""
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("/"):
        cleaned = cleaned[1:]
    while cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    return cleaned.lower()


def resolve_assignee_from_path(path: str | None) -> str:
    """Resolve the assigned developer username based on path ownership."""
    if not path:
        return DEFAULT_ASSIGNEE

    normalized = normalize_path(path)
    if not normalized:
        return DEFAULT_ASSIGNEE

    for prefix, assignee in ROUTING_RULES:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return assignee

    return DEFAULT_ASSIGNEE


def resolve_assignee_from_text(text: str | None) -> str:
    """Scan text for mentioned paths or language indicators."""
    if not text:
        return DEFAULT_ASSIGNEE

    lower = text.lower()
    for prefix, assignee in ROUTING_RULES:
        # Match "/core/python" or "core/python"
        if f"/{prefix}" in lower or prefix in lower:
            return assignee

    return DEFAULT_ASSIGNEE


def parse_option(raw_option: Any) -> Option:
    """Parse raw option value into an Option enum member."""
    if isinstance(raw_option, int) and raw_option in (1, 2, 3, 4):
        return Option(raw_option)

    if isinstance(raw_option, str):
        cleaned = raw_option.strip().lower()
        if cleaned in (
            "1",
            "option 1",
            "option_1",
            "option1",
            "clarify",
            "option_1_clarify",
        ):
            return Option.CLARIFY
        if cleaned in (
            "2",
            "option 2",
            "option_2",
            "option2",
            "simple",
            "quick",
            "simple_solution",
            "quick_solution",
            "option_2_quick_solution",
        ):
            return Option.SIMPLE_SOLUTION
        if cleaned in (
            "3",
            "option 3",
            "option_3",
            "option3",
            "detailed",
            "detailed_solution",
            "step_by_step",
            "option_3_detailed_solution",
        ):
            return Option.DETAILED_SOLUTION
        if cleaned in (
            "4",
            "option 4",
            "option_4",
            "option4",
            "acknowledge",
            "assign",
            "acknowledge_and_assign",
            "option_4_acknowledge_and_assign",
        ):
            return Option.ACKNOWLEDGE_AND_ASSIGN

        for num in (1, 2, 3, 4):
            if str(num) in cleaned:
                return Option(num)

    return Option.ACKNOWLEDGE_AND_ASSIGN


def extract_decision_json(raw_text: str) -> dict[str, Any]:
    """Extract and parse the JSON decision from agy output."""
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty output from AI agent.")

    # Check if raw_text is the agy envelope JSON: {"status": ..., "response": ...}
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            if "response" in parsed and isinstance(parsed["response"], str):
                inner = parsed["response"].strip()
                # If inner has markdown fences, extract JSON from them
                match = FENCED_JSON.search(inner)
                if match:
                    return json.loads(match.group(1))
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass
            elif "option" in parsed or "response" in parsed:
                return parsed
    except json.JSONDecodeError:
        pass

    # Check for markdown code fence in raw text
    match = FENCED_JSON.search(raw_text)
    if match:
        return json.loads(match.group(1))

    # Try direct parse
    return json.loads(raw_text)


def process_response(
    decision: dict[str, Any],
    *,
    fallback_text: str | None = None,
) -> tuple[Option, str, str | None]:
    """Process decision dict and return (option, response_body, assignee)."""
    raw_opt = decision.get("option")
    option = parse_option(raw_opt)

    response_body = str(decision.get("response") or "").strip()
    if not response_body:
        response_body = (
            "Thank you for opening this issue! We have received your report "
            "and are looking into it."
        )

    path = decision.get("path")
    if path is not None:
        path = str(path).strip()

    raw_assignee = decision.get("assignee")
    if raw_assignee is not None:
        raw_assignee = str(raw_assignee).strip().lstrip("@").lower()

    if option == Option.ACKNOWLEDGE_AND_ASSIGN:
        # Determine assignee based on path routing
        if path:
            assignee = resolve_assignee_from_path(path)
        elif raw_assignee and raw_assignee in VALID_ASSIGNEES:
            assignee = raw_assignee
        elif fallback_text:
            assignee = resolve_assignee_from_text(fallback_text)
        else:
            assignee = DEFAULT_ASSIGNEE
    else:
        assignee = None

    return option, response_body, assignee


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process AI issue responder decision and routing."
    )
    parser.add_argument(
        "--result",
        required=True,
        type=Path,
        help="Path to agy_result.json output file",
    )
    parser.add_argument(
        "--issue-number",
        required=True,
        type=int,
        help="The GitHub issue number",
    )
    parser.add_argument(
        "--comment-out",
        required=True,
        type=Path,
        help="Destination path for markdown comment file",
    )
    parser.add_argument(
        "--assignee-out",
        required=True,
        type=Path,
        help="Destination path for assignee text file",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Path to $GITHUB_OUTPUT file",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()

    try:
        raw_content = args.result.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"cannot read {args.result}: {exc}")
        )

    try:
        decision = extract_decision_json(raw_content)
    except Exception as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"failed to parse JSON from agy output: {exc}")
        )

    option, response_body, assignee = process_response(decision)

    try:
        args.comment_out.write_text(response_body + "\n", encoding="utf-8")
    except OSError as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"cannot write {args.comment_out}: {exc}")
        )

    try:
        assignee_text = (assignee or "") + "\n"
        args.assignee_out.write_text(assignee_text, encoding="utf-8")
    except OSError as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"cannot write {args.assignee_out}: {exc}")
        )

    print(
        f"Issue #{args.issue_number}: selected Option {option.value} ({option.name})"
    )
    if assignee:
        print(f"Assigned to developer: @{assignee}")
    else:
        print("No assignee specified (Options 1-3).")

    if args.github_output:
        outputs = [
            f"option={option.value}",
            f"option_name={option.name}",
            f"assignee={assignee or ''}",
            f"has_assignee={'true' if assignee else 'false'}",
        ]
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(outputs) + "\n")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(guard(CHECKER, main))
