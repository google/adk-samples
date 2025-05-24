# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from starter_template.tools.example_tool import tool1_name
from starter_template.sub_agents.sub_agent1_name.tools.example_tool2 import tool2_name


# ----- Define basic unit tests for individual tools -----


def test_tool1_name():
    result = tool1_name(name="User")
    assert result == "Hello User!"


def test_tool2_name():
    result = tool2_name(name="User 2")
    assert result == "Hello User 2!"
