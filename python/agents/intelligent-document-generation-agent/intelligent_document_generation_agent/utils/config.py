# Copyright 2026 Google LLC
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

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives at the sample root, alongside pyproject.toml.
project_root = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = project_root / ".env"

load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)

# The ADK / google.genai client reads GOOGLE_CLOUD_LOCATION from the env at
# request time. Force it to "global" so all Vertex AI traffic from the agent
# targets the global endpoint, regardless of what .env says. The .env value
# remains the source of truth for the Cloud Run endpoint deploy script.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"


class Settings(BaseSettings):
    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str
    WORKER_MODEL: str
    ADK_STAGING_BUCKET: str
    ADK_OUTPUT_BUCKET: str
    REASONING_ENGINE: str
    CONVERSION_SERVICE_URL: str
    PROJECT_SERVICE_ACCOUNT: str

    # Pydantic V2 way to specify .env file and other configurations
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env or environment
    )


def get_settings() -> Settings:
    return Settings()


# Instantiate settings for easy import, or use the get_settings() function
settings = get_settings()
