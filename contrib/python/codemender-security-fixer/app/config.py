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
"""Reads configuration from the environment. Defaults live in .env.example."""

import os


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


def require(name: str) -> str:
    """Returns the value of `name`, which may be empty but must be set."""
    value = os.getenv(name)
    if value is None:
        raise ConfigError(
            f"{name} is not set. Copy .env.example to .env (or set it in the"
            " service environment)."
        )
    return value


def require_number(name: str, kind: type[int] | type[float]) -> int | float:
    """Returns `name` parsed as `kind` (int or float)."""
    raw = require(name)
    try:
        return kind(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def optional(name: str) -> str | None:
    """Returns the value of `name`, or None when it is unset or empty."""
    return os.getenv(name) or None
