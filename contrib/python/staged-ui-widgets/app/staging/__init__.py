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
"""Staged widget lifecycle: tools stage, one flush emits.

Tools use ``stage_widget`` / ``revive_widget`` / ``suppress_widget``. The
agent's ``after_agent_callback`` uses ``emit_staged_widgets``. Nothing else
needs importing.
"""

from .lifecycle import (
    EmissionOutcome,
    blocked_emissions,
    emit_staged_widgets,
    log_flush,
)
from .spec import WIDGET_SPECS, StagedWidgetSpec, all_specs, spec_for
from .state import (
    clear_staged,
    revive_widget,
    stage_widget,
    suppress_widget,
)

__all__ = [
    "WIDGET_SPECS",
    "EmissionOutcome",
    "StagedWidgetSpec",
    "all_specs",
    "blocked_emissions",
    "clear_staged",
    "emit_staged_widgets",
    "log_flush",
    "revive_widget",
    "spec_for",
    "stage_widget",
    "suppress_widget",
]
