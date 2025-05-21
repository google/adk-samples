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

"""Prompt definition for extract_page_data_agent in FOMC Research Agent"""

PROMPT = """
Your job is to extract important data from a web page.

 <PAGE_CONTENTS>
 {page_contents}
 </PAGE_CONTENTS>

<INSTRUCTIONS>
The contents of the web page are provided above in the 'page_contents' section.
The data fields needed are provided in the 'data_to_extract' section of the user
input.

Read the contents of the web page and extract the pieces of data requested.
Don't use any other HTML parser, just examine the HTML yourself and extract the
information.

First, use the store_state tool to store the extracted data in the ToolContext.

Second, return the information you found to the user in JSON format.
 </INSTRUCTIONS>

"""
