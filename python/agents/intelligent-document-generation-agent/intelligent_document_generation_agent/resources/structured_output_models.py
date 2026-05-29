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

from typing import List

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    field: str = Field(
        ..., description="The specific field from the document template."
    )
    reasoning: str = Field(
        ..., description="The step-by-step reasoning used to arrive at the answer."
    )
    value: str = Field(..., description="The comprehensive answer for the field.")
    source_data: str = Field(
        ..., description="Relevant snippets from the Source Data JSON object."
    )
    documents_source_data: str = Field(
        ...,
        description='Citations from source documents in the format: <document name> page (<page number>): "<Exact text from PDF>".',
    )
    explanation: str = Field(
        ...,
        description="A detailed explanation of why the source text justifies the answer in the 'Value' column.",
    )


class GeneratedDocument(BaseModel):
    items: List[DocumentItem] = Field(
        ..., description="A list of all completed items for the document."
    )
