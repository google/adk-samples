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

"""This file contains the tools used by the AlloyDB agent."""

import logging
import os
import re

from google.adk.tools import ToolContext
from google.genai import Client
from google.genai.types import HttpOptions
from toolbox_core import ToolboxSyncClient, auth_methods

from data_science.utils.utils import get_env_var

from ...utils.utils import USER_AGENT

ALLOYDB_TOOLSET = os.getenv("ALLOYDB_TOOLSET", "postgres-database-tools")
# The agent connects to AlloyDB using the MCP Toolbox for Databases
# By default it connects to localhost; change this environment variable
# for a remote deployment.
MCP_TOOLBOX_HOST = os.getenv("MCP_TOOLBOX_HOST", "localhost")
MCP_TOOLBOX_PORT = os.getenv("MCP_TOOLBOX_PORT", "5000")

# MAX_NUM_ROWS = 80

vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT", None)
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
http_options = HttpOptions(headers={"user-agent": USER_AGENT})
llm_client = Client(
    vertexai=True,
    project=vertex_project,
    location=location,
    http_options=http_options,
)

database_settings = None
toolbox_client = None
toolbox_toolset = None

logger = logging.getLogger(__name__)


def get_toolbox_client():
    """Get MCP Toolbox client."""
    global toolbox_client
    if toolbox_client is None:
        logger.info(
            "Connecting to MCP Toolbox at %s:%s",
            MCP_TOOLBOX_HOST,
            MCP_TOOLBOX_PORT,
        )
        if MCP_TOOLBOX_HOST in ["localhost", "127.0.0.1"]:
            toolbox_url = f"http://{MCP_TOOLBOX_HOST}:{MCP_TOOLBOX_PORT}"
            logger.info("Connecting to local MCP Toolbox at %s", toolbox_url)
            toolbox_client = ToolboxSyncClient(toolbox_url)
        else:
            toolbox_url = f"https://{MCP_TOOLBOX_HOST}"
            if MCP_TOOLBOX_PORT != "":
                toolbox_url += f":{MCP_TOOLBOX_PORT}"
            logger.info("Connecting to remote MCP Toolbox at %s", toolbox_url)
            auth_token_provider = auth_methods.aget_google_id_token(toolbox_url)
            toolbox_client = ToolboxSyncClient(
                toolbox_url,
                client_headers={"Authorization": auth_token_provider},
            )
    return toolbox_client


def get_toolbox_toolset():
    """Get MCP Toolbox toolset."""
    global toolbox_toolset
    if toolbox_toolset is None:
        toolbox_toolset = get_toolbox_client().load_toolset(ALLOYDB_TOOLSET)
    return toolbox_toolset


def get_database_settings():
    """Get database settings."""
    global database_settings
    if database_settings is None:
        database_settings = update_database_settings()
    return database_settings


def get_schema():
    get_schema_tool = get_toolbox_client().load_tool("list_tables")
    schema = get_schema_tool(
        schema_names=get_env_var("ALLOYDB_SCHEMA_NAME"), table_names=""
    )
    return schema


def update_database_settings():
    """Update database settings."""
    global database_settings

    schema = get_schema()

    database_settings = {
        "project_id": get_env_var("ALLOYDB_PROJECT_ID"),
        "database": get_env_var("ALLOYDB_DATABASE"),
        "schema_name": get_env_var("ALLOYDB_SCHEMA_NAME"),
        "schema": schema,
    }
    return database_settings


def alloydb_nl2sql(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Generates an initial SQL query from a natural language question.

    Args:
        question (str): Natural language question.
        tool_context (ToolContext): The tool context to use for generating the
          SQL query.

    Returns:
        str: An SQL statement to answer this question.
    """

    prompt_template = """
You are an AlloyDB and PostgreSQL expert tasked with translating a user's
questions about AlloyDB or Postgres tables into a valid SQL query in the
PostgreSQL dialect. Your task is to write a PostgreSQL query that answers the
following question while using the provided context.

**Guidelines:**

- **Table Referencing:** Tables should be referred to using the table name
  enclosed in double quotes (") e.g. "table_name". Table names are case
  sensitive.
- **Joins:** Join as few tables as possible. When joining tables, ensure all
  join columns are the same data type. Analyze the database and the table schema
  provided to understand the relationships between columns and tables.
- **Aggregations:**  Use all non-aggregated columns from the "SELECT" statement
  in the `GROUP BY` clause.
- **SQL Syntax:** Return syntactically and semantically correct SQL for
  Postgres / AlloyDb with proper relation mapping (i.e., project_id, owner,
  table, and column relation). Use SQL `AS` statement to assign a new name
  temporarily to a table column or even a table wherever needed. Always enclose
  subqueries and union queries in parentheses.
- **Column Usage:** Use *ONLY* the column names (column_name) mentioned in the
  Table Schema. Do *NOT* use any other column names. Associate `column_name`
  mentioned in the Table Schema only to the `table_name` specified under Table
  Schema.
- **FILTERS:** You should write the query effectively  to reduce and minimize the
  total rows to be returned. For example, you can use filters (like `WHERE`,
  `HAVING`, etc. (like 'COUNT', 'SUM', etc.) in the SQL query.

**Schema:**

The database structure is defined by the following table schemas (possibly with
sample rows):

```
{SCHEMA}
```

**Natural language question:**

```
{QUESTION}
```

**Think Step-by-Step:** Carefully consider the schema, question, guidelines, and
  best practices outlined above to generate the correct PostgreSQL.

   """

    schema = tool_context.state["database_settings"]["alloydb"]["schema"]

    prompt = prompt_template.format(
        #        MAX_NUM_ROWS=MAX_NUM_ROWS,
        SCHEMA=schema,
        QUESTION=question,
    )

    response = llm_client.models.generate_content(
        model=os.getenv("BASELINE_NL2SQL_MODEL", ""),
        contents=prompt,
        config={"temperature": 0.1},
    )

    sql = response.text or ""
    if sql:
        sql = sql.replace("```sql", "").replace("```", "").strip()

    logger.debug("sql: %s", sql)

    tool_context.state["sql_query"] = sql

    return sql


def run_alloydb_query(
    sql_string: str,
    tool_context: ToolContext,
) -> dict:
    """
    Runs an AlloyDB SQL query.

    This function validates the provided SQL string, then runs it against
    AlloyDB and returns the results.

    It performs the following steps:

    1. **SQL Cleanup:**  Preprocesses the SQL string using a `cleanup_sql`
    function
    2. **DML/DDL Restriction:**  Rejects any SQL queries containing DML or DDL
       statements (e.g., UPDATE, DELETE, INSERT, CREATE, ALTER) to ensure
       read-only operations.
    3. **Syntax and Execution:** Sends the cleaned SQL to BigQuery for
       execution and retrieves the results.

    Args:
        sql_string (str): The SQL query string to validate.
        tool_context (ToolContext): The tool context to use for validation.

    Returns:
        A dict with two keys:
            query_results (list): A list of {key, value} dicts for each element
                in the result set.
            error_message (str): A message indicating the query outcome.
                This includes:
                  - "Valid SQL. Results: ..." if the query is valid and returns
                    data.
                  - "Query executed successfully (no results)." if
                    the query is valid but returns no data.
                  - "Invalid SQL: ..." if the query is invalid, along with the
                    error message from BigQuery.
                  - "Query error: ..." if another error occurs, including any
                  error message.
    """

    def cleanup_sql(sql_string):
        """Processes the SQL string to get a printable, valid SQL string."""

        # 1. Remove backslashes escaping double quotes
        sql_string = sql_string.replace('\\"', '"')

        # 2. Remove newlines
        sql_string = sql_string.replace("\\\n", " ")
        sql_string = sql_string.replace("\n", " ")

        # 3. Replace escaped single quotes
        sql_string = sql_string.replace("\\'", "'")

        # 4. Replace escaped newlines (those not preceded by a backslash)
        # sql_string = sql_string.replace("\\n", "\n")

        # 5. Add limit clause if not present
        # if "limit" not in sql_string.lower():
        #    sql_string = sql_string + " limit " + str(MAX_NUM_ROWS)

        return sql_string

    logger.debug("Executing SQL: %s", sql_string)
    sql_string = cleanup_sql(sql_string)
    logger.debug("Validating SQL (after cleanup): %s", sql_string)

    final_result = {"query_result": "", "error_message": ""}

    # Disallow DML and DDL
    if re.search(
        r"(?i)(update|delete|drop|insert|create|alter|truncate|merge)",
        sql_string,
    ):
        final_result["error_message"] = (
            "Invalid SQL: Contains disallowed DML/DDL operations."
        )
        return final_result

    try:
        execute_sql_tool = get_toolbox_client().load_tool("execute_sql")
        logger.debug("Sending SQL query: %s", sql_string)
        results = execute_sql_tool(sql_string)
        logger.debug("Received results: %s", results)

        if results:  # Check if query returned data
            final_result["query_result"] = results
            tool_context.state["alloydb_query_result"] = results

        else:
            final_result["error_message"] = (
                "Valid SQL. Query executed successfully (no results)."
            )

    except (
        # Catch generic BQ exceptions  # pylint: disable=broad-exception-caught
        Exception
    ) as e:
        final_result["error_message"] = f"Query error: {e}"

    logger.debug("run_alloydb_query final_result: %s", final_result)

    return final_result
