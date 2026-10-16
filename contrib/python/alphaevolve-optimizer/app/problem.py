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
"""The optimization problem: one-dimensional bin packing.

This module is the only problem-specific part of the recipe. To evolve a
different program, replace the problem statement, the benchmark instances,
the baseline program and `score_outputs` with your own, keeping the same
function signatures.

Candidates must define `pack(items, capacity)` returning a list of bins, where
each bin is a list of *indices* into `items`. Every index must appear exactly
once and no bin may exceed `capacity`. The score is the total number of bins
used across all benchmark instances (lower is better).
"""

import math
import random
from typing import Any

ENTRY_POINT = "pack"

PROBLEM_STATEMENT = """\
Write a Python function `pack(items: list[int], capacity: int) -> list[list[int]]`
for one-dimensional bin packing.

- `items` are positive integer sizes, each <= capacity.
- Return a list of bins. Each bin is a list of INDICES into `items`.
- Every index 0..len(items)-1 must appear in exactly one bin.
- The sum of item sizes in each bin must be <= capacity.
- Use only the Python standard library. No I/O, no network, no randomness
  without a fixed seed. Each call must finish well under one second.

Objective: minimize the TOTAL number of bins over all benchmark instances.
"""

BASELINE_PROGRAM = '''\
def pack(items, capacity):
    """First-fit: put each item in the first bin with enough room."""
    bins = []
    loads = []
    for i, size in enumerate(items):
        for b, load in enumerate(loads):
            if load + size <= capacity:
                bins[b].append(i)
                loads[b] += size
                break
        else:
            bins.append([i])
            loads.append(size)
    return bins
'''


def _make_instance(
    seed: int, n: int, capacity: int, low: int, high: int
) -> dict[str, Any]:
    rng = random.Random(seed)  # noqa: S311 -- deterministic benchmark, not crypto
    items = [rng.randint(low, high) for _ in range(n)]
    return {"name": f"seed{seed}_n{n}", "capacity": capacity, "items": items}


# Fixed, deterministic benchmark. A mix of uniform and "hard" instances where
# item sizes cluster around a third of the capacity, which defeats greedy
# heuristics.
INSTANCES: list[dict[str, Any]] = [
    _make_instance(seed=1, n=120, capacity=100, low=1, high=100),
    _make_instance(seed=2, n=120, capacity=100, low=20, high=60),
    _make_instance(seed=3, n=150, capacity=150, low=40, high=70),
    _make_instance(seed=4, n=200, capacity=100, low=25, high=50),
    _make_instance(seed=5, n=90, capacity=1000, low=250, high=500),
]


def lower_bound() -> int:
    """Sum over instances of ceil(total size / capacity); no packing can beat it."""
    return sum(
        math.ceil(sum(inst["items"]) / inst["capacity"]) for inst in INSTANCES
    )


def _validate_packing(bins: Any, items: list[int], capacity: int) -> str | None:
    """Returns an error message, or None if `bins` is a valid packing."""
    if not isinstance(bins, list):
        return f"expected a list of bins, got {type(bins).__name__}"
    seen: set[int] = set()
    for b, contents in enumerate(bins):
        if not isinstance(contents, list):
            return f"bin {b} is {type(contents).__name__}, expected list"
        load = 0
        for idx in contents:
            if not isinstance(idx, int) or isinstance(idx, bool):
                return f"bin {b} contains non-integer index {idx!r}"
            if not 0 <= idx < len(items):
                return f"bin {b} contains out-of-range index {idx}"
            if idx in seen:
                return f"index {idx} is packed more than once"
            seen.add(idx)
            load += items[idx]
        if load > capacity:
            return f"bin {b} load {load} exceeds capacity {capacity}"
    if len(seen) != len(items):
        missing = sorted(set(range(len(items))) - seen)[:5]
        return f"{len(items) - len(seen)} items not packed (e.g. indices {missing})"
    return None


def score_outputs(outputs: list[Any]) -> dict[str, Any]:
    """Scores candidate outputs, one per benchmark instance.

    Returns:
        A dict with `valid` (bool), `score` (total bins, lower is better, or
        None when invalid), `per_instance` bin counts and an `error` message.
    """
    if len(outputs) != len(INSTANCES):
        return {
            "valid": False,
            "score": None,
            "error": f"expected {len(INSTANCES)} outputs, got {len(outputs)}",
        }
    per_instance = {}
    for inst, bins in zip(INSTANCES, outputs, strict=True):
        error = _validate_packing(bins, inst["items"], inst["capacity"])
        if error:
            return {
                "valid": False,
                "score": None,
                "error": f"instance {inst['name']}: {error}",
            }
        per_instance[inst["name"]] = len(bins)
    return {
        "valid": True,
        "score": sum(per_instance.values()),
        "per_instance": per_instance,
        "error": None,
    }
