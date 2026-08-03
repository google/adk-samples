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

"""Lightweight fake objects shared across unit and smoke test files."""

from __future__ import annotations


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeActions:
    def __init__(self) -> None:
        self.skip_summarization = False


class FakeToolContext:
    """Minimal fake tool context for permission_guard and related tests."""

    def __init__(self, *, tool_confirmation=None) -> None:
        self.state: dict = {}
        self.actions = _FakeActions()
        self.tool_confirmation = tool_confirmation
        self.function_call_id = "fc-1"
        self.agent_name = "root"
        self._requested = None

    def request_confirmation(self, *, hint=None, payload=None) -> None:
        self._requested = {"hint": hint, "payload": payload}
