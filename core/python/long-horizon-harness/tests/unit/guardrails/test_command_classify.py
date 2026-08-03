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

"""Unit tests for shell command classification used by the permission gate."""

from __future__ import annotations

import pytest

from horizon.guardrails.command_classify import (
    command_prefix,
    has_command_substitution,
    has_redirection,
    split_segments,
    strip_wrapper,
)


@pytest.mark.parametrize(
    "command,expected",
    [
        ("bash -c 'bq rm -t x.y'", "bq rm -t x.y"),
        ('sh -c "ls -la"', "ls -la"),
        ("ls -la", "ls -la"),
    ],
)
def test_strip_wrapper(command, expected):
    assert strip_wrapper(command) == expected


@pytest.mark.parametrize(
    "command,expected",
    [
        ("ls && bq rm x", ["ls", "bq rm x"]),
        ("a | b | c", ["a", "b", "c"]),
        ("x; y || z", ["x", "y", "z"]),
        ("single", ["single"]),
        ("ls & bq rm x", ["ls", "bq rm x"]),  # bare & (background)
        ("ls\nbq rm x", ["ls", "bq rm x"]),  # newline
        ("a && b", ["a", "b"]),  # && still splits to 2, not on the inner &
        ("cmd 2>&1", ["cmd 2>&1"]),  # fd-dup & is not a separator
        ("gws x 2>&1 | head", ["gws x 2>&1", "head"]),
        ("cmd &>out", ["cmd &>out"]),  # &> redirect, not background
        # Operators inside quotes are literal — a quoted body stays one segment.
        ('echo "a; b | c"', ['echo "a; b | c"']),
        ("echo 'a && b'", ["echo 'a && b'"]),
        ('python3 -c "x; y\nz"', ['python3 -c "x; y\nz"']),  # multiline -c body
        # ...but operators OUTSIDE the quotes still split.
        ('python3 -c "x; y" ; ls', ['python3 -c "x; y"', "ls"]),
    ],
)
def test_split_segments(command, expected):
    assert split_segments(command) == expected


@pytest.mark.parametrize(
    "segment,expected",
    [
        ("ls $(bq rm x)", True),
        ("echo `bq rm x`", True),
        ("diff <(a) <(b)", True),
        ("ls -la", False),
    ],
)
def test_has_command_substitution(segment, expected):
    assert has_command_substitution(segment) is expected


@pytest.mark.parametrize(
    "segment,expected",
    [
        ("bq rm -t x.y", "bq rm"),
        ("git push origin main", "git push"),
        ("npm install foo", "npm install"),
        ("python train.py", "python"),  # train.py is not a bare word → stop
        ("ls", "ls"),
        ("FOO=bar bq rm x", "bq rm"),  # skip env assignment
        ("sudo systemctl restart x", "systemctl restart"),  # skip sudo
        ("./run.sh arg", "./run.sh"),  # binary with no bare-word subcmd
    ],
)
def test_command_prefix(segment, expected):
    assert command_prefix(segment) == expected


@pytest.mark.parametrize(
    "segment,expected",
    [
        ("echo hi > out.txt", True),
        ("cat a >> b", True),
        ("ls -la", False),
        ("cmd 2> err.log", True),
        ("cmd 1>&2", False),  # fd-dup, not a file write
        ("cmd 2>&1", False),
        (
            "cmd 2>&1 > out.txt",
            True,
        ),  # fd-dup + real redirect → still redirects
        ("cmd &>out", True),  # &> opens a file
        ("cmd 2>/dev/null", False),  # /dev/null is a sink, not a file write
        ("cmd >/dev/null", False),
        ("cmd > /dev/null", False),  # spaced
        ("cmd >>/dev/null 2>&1", False),  # append-to-null + fd-dup
        ("cmd &>/dev/null", False),
        (
            "cmd 2>/dev/null > out.txt",
            True,
        ),  # null sink + real redirect → still redirects
        (
            "cmd > /dev/null2",
            True,
        ),  # a real file that merely starts with /dev/null
        ("cmd > /dev/null.txt", True),  # a sibling file in /dev, not the sink
        ("cmd > /dev/null/../etc/foo", True),  # traversal off the device path
    ],
)
def test_has_redirection(segment, expected):
    assert has_redirection(segment) is expected
