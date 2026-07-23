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

import glob
import os


def save_content_to_file(content: str, filename: str) -> dict:
    """Saves text content (such as a blog post or social media post) to a file."""
    with open(filename, "w") as f:
        f.write(content)
    return {"status": "success"}


def analyze_codebase(directory: str) -> dict:
    """Analyzes the codebase in the given directory."""
    files = glob.glob(os.path.join(directory, "**"), recursive=True)
    codebase_context = ""
    for file in files:
        if os.path.isfile(file):
            codebase_context += f"""- **{file}**:"""
            try:
                with open(file, encoding="utf-8") as f:
                    codebase_context += f.read()
            except UnicodeDecodeError:
                with open(file, encoding="latin-1") as f:
                    codebase_context += f.read()
    return {"codebase_context": codebase_context}
