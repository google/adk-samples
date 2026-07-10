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
"""Unit tests for the custom weather and time tools."""

from app.agent import get_weather


def test_get_weather_san_francisco() -> None:
    """Tests that get_weather returns foggy weather for San Francisco."""
    result = get_weather("San Francisco")
    assert "60 degrees and foggy" in result


def test_get_weather_other_city() -> None:
    """Tests that get_weather returns sunny weather for other cities."""
    result = get_weather("London")
    assert "90 degrees and sunny" in result
