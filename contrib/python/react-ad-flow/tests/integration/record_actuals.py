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

"""CLI recorder: dump actual live trajectories + answers for an eval set.

Usage:
    python -m tests.integration.record_actuals <eval_set.json> <out.json>

Runs the live eval harness (inference once per case + 3-dimension scoring)
under the SAME env bootstrap as tests/integration/conftest.py (real app/.env,
GCS_BUCKET force-unset, isolated DB copy) and writes each case's actual tool
calls, final answer, and dimension scores to <out.json>. Used by the stage-4
calibration tasks (Tasks 6-10) and the Task 15 trace export.
"""

import asyncio
import os
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m tests.integration.record_actuals"
            " <eval_set.json> <out.json>",
            file=sys.stderr,
        )
        return 2
    eval_set_path, out_path = argv

    # Same env bootstrap as the integration conftest (real env, local-first).
    from tests.integration.conftest import apply_live_env_to_process

    apply_live_env_to_process()

    # Same DB isolation as the isolated_live_db fixture — live agent runs
    # execute REAL tools that mutate the DB (e.g. create_campaign).
    import app.config as config_module
    from tests.conftest import _copy_main_db_to_temp

    db_path = _copy_main_db_to_temp()
    config_module.DB_PATH = db_path

    from tests.integration.eval_harness import run_eval_set

    try:
        outcomes = asyncio.run(run_eval_set(eval_set_path, record_to=out_path))
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass

    for outcome in outcomes:
        status = "ok" if outcome.inference_ok else "INFERENCE FAILED"
        dims = " ".join(
            f"{name}={d.score}({'pass' if d.passed else 'fail'})"
            for name, d in outcome.dimensions.items()
        )
        print(f"{outcome.eval_id}: {status} {dims}".rstrip())
        if outcome.error_message:
            print(f"  error: {outcome.error_message}")
    print(f"recorded {len(outcomes)} case(s) -> {out_path}")
    return 0 if all(o.inference_ok for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
