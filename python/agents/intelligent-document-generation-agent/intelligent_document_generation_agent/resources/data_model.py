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

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExtractedInformation(BaseModel):
    """
    You are an expert at extracting information from documents.
    You should extract the following generic fields:
    """

    project_name: str = Field(
        alias="Project Name", description="Name of the project or initiative"
    )
    project_purpose: str = Field(
        alias="Project Purpose", description="Purpose or main goals of the project"
    )
    target_audience: str = Field(
        alias="Target Audience", description="Target audience or users of the project"
    )
    key_features: List[str] = Field(
        alias="Key Features", description="List of key features or capabilities"
    )
    technologies_used: List[str] = Field(
        alias="Technologies Used",
        description="List of technologies, frameworks, or tools mentioned",
    )
    data_handled: List[str] = Field(
        alias="Data Handled",
        description="Types of data processed or handled by the system",
    )
    security_measures: Optional[str] = Field(
        None,
        alias="Security Measures",
        description="Any security or privacy measures mentioned",
    )
    external_integrations: Optional[List[str]] = Field(
        None,
        alias="External Integrations",
        description="Any third-party or external integrations",
    )

    model_config = ConfigDict(populate_by_name=True)
