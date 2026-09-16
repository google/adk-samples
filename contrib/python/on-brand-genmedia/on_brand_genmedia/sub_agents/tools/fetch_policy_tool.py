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

from pathlib import Path


def get_policy() -> dict[str, str]:
    # Construct the full path to the policy file
    policy_file_path = Path(__file__).resolve().parents[2] / "policy.json"

    with open(policy_file_path, encoding="utf-8") as file:
        policy_text = file.read()
    return {"policy_text": policy_text}
