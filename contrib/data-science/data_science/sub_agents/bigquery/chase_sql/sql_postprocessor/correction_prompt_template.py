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

"""Prompt template for making any corrections to the translation of SQL."""

CORRECTION_PROMPT_TEMPLATE_V1_0 = """
You are an expert in multiple databases and SQL dialects.
You are given a SQL query that is formatted for the SQL dialect:
{sql_dialect}

The SQL query is:
{sql_query}
{schema_insert}
This SQL query could have the following errors:
{errors}

Please correct the SQL query to make sure it is formatted correctly for the SQL dialect:
{sql_dialect}

DO not change any table or column names in the query. However, you may qualify column names with table names.
Do not change any literals in the query.
You may *only* rewrite the query so that it is formatted correctly for the specified SQL dialect.
Do not return any other information other than the corrected SQL query.

Corrected SQL query:
"""
