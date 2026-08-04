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

"""Seed the env from .env.example so tests never require a real .env."""

import os
import pathlib

from dotenv import dotenv_values

# Runs before sibling test modules import financial_advisor, whose agents
# read MODEL_NAME at import time and fail pydantic validation if it is unset.
for key, value in dotenv_values(
    pathlib.Path(__file__).parent / ".env.example"
).items():
    if value is not None:
        os.environ.setdefault(key, value)
