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

from __future__ import annotations

# -----------------------------------------------------------------------------
# slug + filename
# -----------------------------------------------------------------------------


def test_slug_lowercases_and_dashes():
    from horizon.tools.todos import slugify

    assert slugify("Write the Test!") == "write-the-test"


def test_slug_collapses_and_trims_separators():
    from horizon.tools.todos import slugify

    assert slugify("  Run   pytest -v  ") == "run-pytest-v"


def test_slug_is_capped_and_never_empty():
    from horizon.tools.todos import slugify

    assert slugify("x" * 200) == "x" * 60
    assert slugify("!!!") == "todo"


def test_filename_for_uses_padded_index_prefix():
    from horizon.tools.todos import filename_for

    assert filename_for(0, "do it") == "001-do-it.md"
    assert filename_for(11, "another task") == "012-another-task.md"


# -----------------------------------------------------------------------------
# frontmatter serialize / parse round-trip
# -----------------------------------------------------------------------------


def test_serialize_emits_frontmatter_then_body():
    from horizon.tools.todos import serialize_todo

    text = serialize_todo(
        {
            "id": "do-it",
            "content": "do it",
            "status": "pending",
            "priority": "high",
        }
    )
    assert text.startswith("---\n")
    assert "status: pending" in text
    assert "priority: high" in text
    assert "id: do-it" in text
    assert text.rstrip().endswith("do it")


def test_serialize_omits_absent_priority():
    from horizon.tools.todos import serialize_todo

    text = serialize_todo({"id": "x", "content": "x", "status": "pending"})
    assert "priority:" not in text


def test_parse_reads_frontmatter_and_body():
    from horizon.tools.todos import parse_todo

    text = (
        "---\nid: do-it\nstatus: in_progress\npriority: low\n---\ndo it now\n"
    )
    todo = parse_todo(text)
    assert todo == {
        "id": "do-it",
        "content": "do it now",
        "status": "in_progress",
        "priority": "low",
    }


def test_parse_round_trips_serialize():
    from horizon.tools.todos import parse_todo, serialize_todo

    original = {"id": "ship", "content": "ship it", "status": "completed"}
    assert parse_todo(serialize_todo(original)) == original


def test_parse_missing_frontmatter_defaults_pending():
    from horizon.tools.todos import parse_todo

    todo = parse_todo("just a body line")
    assert todo["status"] == "pending"
    assert todo["content"] == "just a body line"


# -----------------------------------------------------------------------------
# render
# -----------------------------------------------------------------------------


def test_render_empty_is_empty_string():
    from horizon.tools.todos import render_todos

    assert render_todos([]) == ""
    assert render_todos(None) == ""


def test_render_header_and_status_markers():
    from horizon.tools.todos import render_todos

    out = render_todos(
        [
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "in_progress"},
            {"content": "c", "status": "pending"},
        ]
    )
    lines = out.splitlines()
    assert lines[0] == "# Todos"
    assert "[x] a" in out
    assert "[~] b" in out
    assert "[ ] c" in out


def test_render_flags_only_high_priority():
    from horizon.tools.todos import render_todos

    out = render_todos(
        [
            {"content": "urgent", "status": "pending", "priority": "high"},
            {"content": "normal", "status": "pending", "priority": "medium"},
        ]
    )
    assert "urgent (high)" in out
    assert "(medium)" not in out


def test_render_multiple_in_progress_adds_caution():
    from horizon.tools.todos import render_todos

    out = render_todos(
        [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ]
    )
    assert "caution" in out.lower()
    assert "in_progress" in out


def test_render_single_in_progress_has_no_caution():
    from horizon.tools.todos import render_todos

    out = render_todos([{"content": "a", "status": "in_progress"}])
    assert "caution" not in out.lower()
