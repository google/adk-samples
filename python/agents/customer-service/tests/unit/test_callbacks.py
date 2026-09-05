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

from customer_service.shared_libraries.callbacks import lowercase_value


def test_lowercase_value_recurses_through_nested_dicts():
    value = {
        "name": "JOHN DOE",
        "email": "John@Example.COM",
        "nested": {"City": "NEW YORK"},
    }

    assert lowercase_value(value) == {
        "name": "john doe",
        "email": "john@example.com",
        "nested": {"city": "new york"},
    }


def test_lowercase_value_handles_lists_and_tuples():
    assert lowercase_value(["ABC", {"Key": "Value"}]) == [
        "abc",
        {"key": "value"},
    ]
    assert lowercase_value(("ABC", "DEF")) == ("abc", "def")


def test_lowercase_value_leaves_non_string_scalars_unchanged():
    assert lowercase_value(42) == 42
    assert lowercase_value(True) is True
