# Copyright 2025 Google LLC
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

"""Tests for shared/utils.py - Parallel execution utilities."""

from workflows.shared.utils import predict_parallel


class TestPredictParallel:
    """Tests for predict_parallel function."""

    def test_basic_execution(self):
        """Basic parallel execution should work."""
        inputs = [1, 2, 3, 4, 5]
        results = predict_parallel(
            inputs,
            lambda x: x * 2,
            max_workers=2,
            show_progress_bar=False,
        )
        assert results == [2, 4, 6, 8, 10]

    def test_with_different_workers(self):
        """Different worker counts should produce same results."""
        inputs = list(range(10))

        results_2 = predict_parallel(
            inputs, lambda x: x + 1, max_workers=2, show_progress_bar=False
        )
        results_4 = predict_parallel(
            inputs, lambda x: x + 1, max_workers=4, show_progress_bar=False
        )

        assert results_2 == results_4 == list(range(1, 11))

    def test_empty_input(self):
        """Empty input should return empty list."""
        results = predict_parallel(
            [],
            lambda x: x,
            show_progress_bar=False,
        )
        assert results == []

    def test_single_item(self):
        """Single item should be processed correctly."""
        results = predict_parallel(
            ["hello"],
            lambda x: x.upper(),
            show_progress_bar=False,
        )
        assert results == ["HELLO"]

    def test_preserves_order(self):
        """Results should preserve input order."""
        inputs = [3, 1, 4, 1, 5, 9, 2, 6]
        results = predict_parallel(
            inputs,
            lambda x: x * 10,
            max_workers=4,
            show_progress_bar=False,
        )
        assert results == [30, 10, 40, 10, 50, 90, 20, 60]

    def test_with_complex_function(self):
        """Complex transformation function should work."""

        def complex_transform(item):
            return {
                "original": item,
                "squared": item**2,
                "label": f"item_{item}",
            }

        inputs = [1, 2, 3]
        results = predict_parallel(
            inputs,
            complex_transform,
            show_progress_bar=False,
        )

        assert len(results) == 3
        assert results[0] == {"original": 1, "squared": 1, "label": "item_1"}
        assert results[1] == {"original": 2, "squared": 4, "label": "item_2"}
        assert results[2] == {"original": 3, "squared": 9, "label": "item_3"}

    def test_with_progress_bar(self):
        """Progress bar option should not affect results."""
        inputs = [1, 2, 3]
        results = predict_parallel(
            inputs,
            lambda x: x * 2,
            max_workers=2,
            show_progress_bar=True,
        )
        assert results == [2, 4, 6]
