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

"""Shell command parsing for the permission gate.

Decides the auto-scope prefix shown on the approval card and splits chained
commands so a benign segment can't smuggle a gated one (``ls && bq rm``).
"""

from __future__ import annotations

import re

_WRAPPER_RE = re.compile(
    r"""^(?:bash|sh|zsh)\s+-c\s+(['"])(.*)\1\s*$""", re.DOTALL
)
_REDIRECTION_RE = re.compile(r"(>>|>|<)")
_BARE_WORD_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_SKIP_PREFIXES = frozenset({"sudo", "env"})


def strip_wrapper(command: str) -> str:
    """Unwrap a single ``bash -c '...'`` / ``sh -c "..."`` shell wrapper."""
    m = _WRAPPER_RE.match(command.strip())
    return m.group(2).strip() if m else command.strip()


def split_segments(command: str) -> list[str]:
    """Split on top-level ``&&`` / ``||`` / ``|`` / ``;`` / bare ``&`` / newline
    into trimmed, non-empty parts.

    Quote-aware: operators inside single/double quotes are literal, so a quoted
    interpreter body (``python3 -c "a; b | c"``) stays one segment instead of
    being shredded into a fake prefix per source line. A lone ``&`` (background)
    splits, but not an fd-dup/redirect ``&`` (``2>&1``, ``&>out``) that sits
    next to a ``<``/``>``/``&``.
    """
    parts: list[str] = []
    buf: list[str] = []
    in_single = in_double = False
    i, n = 0, len(command)

    def flush() -> None:
        seg = "".join(buf).strip()
        if seg:
            parts.append(seg)
        buf.clear()

    while i < n:
        ch = command[i]
        if in_single:
            buf.append(ch)
            in_single = ch != "'"
            i += 1
        elif in_double:
            if ch == "\\" and i + 1 < n:
                buf.append(ch + command[i + 1])
                i += 2
            else:
                buf.append(ch)
                in_double = ch != '"'
                i += 1
        elif ch == "\\" and i + 1 < n:
            buf.append(ch + command[i + 1])
            i += 2
        elif ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
        elif ch == '"':
            in_double = True
            buf.append(ch)
            i += 1
        elif ch in ";\n":
            flush()
            i += 1
        elif ch == "&":
            # Slice form (not command[i-1]) so i==0 yields '' — a leading `&` has
            # no adjacent fd-dup char and stays a bare-background split.
            prev, nxt = command[i - 1 : i], command[i + 1 : i + 2]
            if nxt == "&":
                flush()
                i += 2
            elif prev not in ("<", ">", "&") and nxt not in ("<", ">", "&"):
                flush()  # bare background &
                i += 1
            else:  # fd-dup / &>file redirect
                buf.append(ch)
                i += 1
        elif ch == "|":
            flush()
            i += 2 if command[i + 1 : i + 2] == "|" else 1
        else:
            buf.append(ch)
            i += 1
    flush()
    return parts


def command_prefix(segment: str) -> str:
    """Binary + first bare-word subcommand token (the auto-scope prefix)."""
    tokens = segment.split()
    i = 0
    # Skip leading VAR=val assignments and sudo/env wrappers.
    while i < len(tokens) and ("=" in tokens[i] or tokens[i] in _SKIP_PREFIXES):
        i += 1
    if i >= len(tokens):
        return segment.strip()
    binary = tokens[i]
    nxt = tokens[i + 1] if i + 1 < len(tokens) else None
    # Only a plain-name binary takes a bare-word subcommand; a path-like binary
    # (e.g. ``./run.sh``) has positional args, not subcommands.
    if (
        _BARE_WORD_RE.match(binary)
        and nxt is not None
        and _BARE_WORD_RE.match(nxt)
    ):
        return f"{binary} {nxt}"
    return binary


# fd-dup (`2>&1`, `1>&2`, `>&2`, `2>&-`) duplicates/closes a descriptor — it
# opens no file, so it's not a write worth gating. `&>file` is a file redirect
# (different order) and is intentionally not matched here.
_FD_DUP_RE = re.compile(r"\d*[<>]&(?:\d+|-)")
# Redirecting to /dev/null discards output — it's a sink, not a file write, so it
# doesn't warrant a prompt (`2>/dev/null`, `>/dev/null`, `&>/dev/null`, `>>/dev/null`).
# The target must end at a shell boundary, so a real path like `/dev/null2` or
# `/dev/null.txt` stays matched as a redirect (gated), not treated as the sink.
_DEVNULL_RE = re.compile(r"(?:\d*|&)>>?\s*/dev/null(?=\s|$)")


def has_redirection(segment: str) -> bool:
    """True if the segment redirects to/from a file (fd-dups like ``2>&1`` and the
    ``/dev/null`` sink don't count)."""
    cleaned = _DEVNULL_RE.sub("", _FD_DUP_RE.sub("", segment))
    return _REDIRECTION_RE.search(cleaned) is not None


_CMD_SUBSTITUTION_RE = re.compile(r"\$\(|`|<\(|>\(")


def has_command_substitution(segment: str) -> bool:
    """True if the segment embeds a nested command via $(...), backticks, or <()/>()."""
    return _CMD_SUBSTITUTION_RE.search(segment) is not None


__all__ = [
    "command_prefix",
    "has_command_substitution",
    "has_redirection",
    "split_segments",
    "strip_wrapper",
]
