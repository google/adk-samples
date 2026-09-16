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
Verify that changes to the .github/ directory are made only by repository admins.

Files under .github/ (CI workflows, issue templates, automation scripts, and
repository policy configuration) control repository-wide infrastructure and
security. Non-admin pull requests that touch any file under .github/ are
rejected with actionable diagnostic messages.

Reads changed file paths from stdin (one per line) or from --changed-files,
and inspects the PR author's repository permission level.

Usage:
    git diff --name-only origin/main...HEAD | \\
        uv run python tools/check_github_dir_changes.py \\
            --author "<login>" \\
            --author-association "<association>" \\
            --repo "<owner>/<repo>"

Exit codes:
    0  no .github/ changes, or author is a repository administrator
    1  unauthorized changes to .github/ by a non-admin
    2  CI fault / unhandled exception
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Stdlib only + repo tools/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci_message import (
    EXIT_OK,
    Diagnostic,
    Doc,
    Severity,
    guard,
    report,
)


def find_github_files(changed_files: list[str]) -> list[str]:
    """Return all unique file paths sitting under the .github/ directory.

    Matches whole path components so '.github/workflows/foo.yml' matches, but
    a hypothetical '.github-archive/bar.txt' does not.
    """
    github_files: list[str] = []
    seen: set[str] = set()

    for raw in changed_files:
        path = raw.strip()
        if not path:
            continue
        # Normalize leading './' or '/'
        normalized = path
        if normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized.lstrip("/")
        parts = normalized.split("/")
        if parts and parts[0] == ".github":
            if normalized not in seen:
                seen.add(normalized)
                github_files.append(normalized)

    return github_files


def check_is_admin(
    author: str,
    repo: str | None = None,
    author_association: str | None = None,
    token: str | None = None,
    is_admin_override: bool | None = None,
) -> bool:
    """Determine whether the PR author is a repository administrator.

    Resolution order:
      1. Explicit override (if provided, e.g. for testing).
      2. author_association == 'OWNER' (repository / org owner).
      3. GitHub REST API collaborator permission check.
      4. Fallback to `gh` CLI if installed.
      5. Default to False (fail-closed for security).
    """
    if is_admin_override is not None:
        return is_admin_override

    if author_association == "OWNER":
        return True

    if not repo or not author:
        return False

    tok = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    # 1. Try direct GitHub REST API
    api_url = (
        f"https://api.github.com/repos/{repo}/collaborators/{author}/permission"
    )
    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "adk-samples-ci")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            permission = data.get("permission")
            role_name = data.get("role_name")
            permissions = data.get("permissions", {})
            return bool(
                permission == "admin"
                or role_name == "admin"
                or (
                    isinstance(permissions, dict)
                    and permissions.get("admin") is True
                )
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # User is not a collaborator with special access
            return False
    except Exception:
        # Fall through to CLI check
        pass

    # 2. Fallback: try gh CLI
    if shutil.which("gh"):
        try:
            cmd = [
                "gh",
                "api",
                f"/repos/{repo}/collaborators/{author}/permission",
            ]
            res = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=10
            )
            if res.returncode == 0:
                data = json.loads(res.stdout)
                permission = data.get("permission")
                role_name = data.get("role_name")
                permissions = data.get("permissions", {})
                return bool(
                    permission == "admin"
                    or role_name == "admin"
                    or (
                        isinstance(permissions, dict)
                        and permissions.get("admin") is True
                    )
                )
        except Exception:
            pass

    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify .github/ modifications are made only by repo admins."
    )
    parser.add_argument(
        "--author",
        default="",
        help="PR author GitHub username (login)",
    )
    parser.add_argument(
        "--author-association",
        default="",
        help="PR author association (e.g. OWNER, MEMBER, COLLABORATOR)",
    )
    parser.add_argument(
        "--repo",
        default="",
        help="Repository in owner/name format",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub API token (defaults to GITHUB_TOKEN or GH_TOKEN env var)",
    )
    parser.add_argument(
        "--changed-files",
        type=Path,
        default=None,
        help="Path to a file listing changed files (one per line). Reads stdin if omitted.",
    )
    parser.add_argument(
        "--is-admin",
        dest="is_admin_override",
        default=None,
        type=lambda v: v.lower() in ("true", "1", "yes"),
        help="Explicit boolean override for admin check (primarily for testing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.changed_files:
        changed_files = args.changed_files.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    else:
        changed_files = sys.stdin.read().splitlines()

    github_files = find_github_files(changed_files)
    if not github_files:
        print("[PASS] No files under .github/ modified in this PR.")
        return EXIT_OK

    is_admin = check_is_admin(
        author=args.author,
        repo=args.repo,
        author_association=args.author_association,
        token=args.token,
        is_admin_override=args.is_admin_override,
    )

    if is_admin:
        author_desc = args.author or "Author"
        print(
            f"[PASS] .github/ changes authorized: PR author '{author_desc}' is a repository administrator."
        )
        return EXIT_OK

    # Non-admin user modified files under .github/
    author_desc = args.author or "unknown user"
    repo_desc = args.repo or "this repository"

    diagnostics = [
        Diagnostic(
            check="github-dir-admin-only",
            what=f"'{file_path}' is under .github/, which can only be modified by repository administrators.",
            why=(
                f"Only repository administrators are permitted to create, modify, or delete files in the "
                f".github/ directory (including GitHub Actions workflows, issue templates, scripts, and repository policy configuration). "
                f"PR author '{author_desc}' does not have administrator permissions on {repo_desc}."
            ),
            how=(
                "Revert any additions, modifications, or deletions under .github/ in your pull request:\n"
                "  git checkout origin/main -- .github/\n"
                "If CI workflow or repository configuration changes are needed, please open an issue or ask a repository administrator."
            ),
            doc=Doc.GITHUB_DIR_ADMIN,
            file=file_path,
            severity=Severity.ERROR,
        )
        for file_path in github_files
    ]

    count = len(diagnostics)
    noun = "file" if count == 1 else "files"
    return report(
        diagnostics,
        header=f"{count} unauthorized {noun} modified under .github/",
        passed_message="No unauthorized changes to .github/ directory.",
        next_step=(
            "Revert changes to .github/ files in your branch, then push again.\n"
            "Only repository administrators are permitted to modify .github/."
        ),
    )


if __name__ == "__main__":
    sys.exit(guard("check_github_dir_changes.py", main))
