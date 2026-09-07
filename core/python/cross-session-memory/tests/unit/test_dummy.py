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
Unit tests for business logic, data processing, and core components.
"""

from app.agent import get_current_time, get_weather


def test_get_weather_sf() -> None:
    assert "60" in get_weather("San Francisco")
    assert "60" in get_weather("sf")


def test_get_weather_default() -> None:
    assert "90" in get_weather("Austin")


def test_get_current_time_sf() -> None:
    result = get_current_time("San Francisco")
    assert "America/Los_Angeles" in result or "P" in result


def test_get_current_time_unknown() -> None:
    result = get_current_time("Tokyo")
    assert "Sorry" in result
