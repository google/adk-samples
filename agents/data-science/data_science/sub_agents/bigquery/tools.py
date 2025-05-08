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

"""This file contains the tools used by the database agent."""

import datetime
import logging
import os
import re

from data_science.utils.utils import get_env_var
from google.adk.tools import ToolContext
from google.cloud import bigquery
from google.genai import Client
from vertexai.language_models import TextEmbeddingModel

from .chase_sql import chase_constants

# Assume that `BQ_PROJECT_ID` is set in the environment. See the
# `data_agent` README for more details.
project = os.getenv("BQ_PROJECT_ID", None)
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
llm_client = Client(vertexai=True, project=project, location=location)

MAX_NUM_ROWS = 80
SCHEMA_EMBEDDING_MODEL_NAME = os.getenv("SCHEMA_EMBEDDING_MODEL_NAME", "text-embedding-004")
MAX_SCHEMA_RESULTS = 20
BQ_RAG_DATASET_ID = os.getenv("BQ_RAG_DATASET_ID")

database_settings = None
bq_client = None


def get_bq_client():
    """Get BigQuery client."""
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=get_env_var("BQ_PROJECT_ID"))
    return bq_client


def get_column_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of texts."""
    model = TextEmbeddingModel.from_pretrained(SCHEMA_EMBEDDING_MODEL_NAME)
    embeddings = model.get_embeddings(texts)
    return [embedding.values for embedding in embeddings]

def get_relevant_schema_from_embeddings(question: str, project_id: str, data_dataset_id: str, rag_dataset_id: str) -> str:
    """Retrieves relevant schema details (tables and columns) based on vector similarity to the question."""
    client = get_bq_client()
    question_embedding = get_column_embeddings([question])[0]

    # Ensure BQ_RAG_DATASET_ID is available, otherwise this function cannot proceed.
    if not rag_dataset_id:
        print("Error: BQ_RAG_DATASET_ID is not set. Cannot query schema embeddings.")
        return "" # Return empty string or handle error as appropriate

    # Query to find relevant columns from the schema_embeddings table in the RAG dataset
    query = f"""
    WITH QuestionEmbedding AS (
        SELECT {question_embedding} AS q_embedding
    ),
    SchemaEmbeddingsWithNorm AS (
        SELECT
            table_name,
            column_name,
            column_description,
            data_type,
            embedding,
            (SELECT SQRT(SUM(value * value)) FROM UNNEST(embedding) AS value) AS norm_v,
            (SELECT SQRT(SUM(value * value)) FROM UNNEST(q_embedding) AS value) AS norm_q
        FROM
            `{project_id}.{rag_dataset_id}.schema_embeddings`, QuestionEmbedding
    )
    SELECT
        se.table_name,
        se.column_name,
        se.column_description,
        se.data_type,
        (SELECT SUM(ve * qe) FROM UNNEST(se.embedding) AS ve WITH OFFSET i JOIN UNNEST(q_embedding) AS qe WITH OFFSET j ON i = j) / SAFE_DIVIDE(norm_v * norm_q, 1) AS similarity
    FROM SchemaEmbeddingsWithNorm se, QuestionEmbedding
    ORDER BY similarity DESC
    LIMIT {MAX_SCHEMA_RESULTS}
    """

    try:
        query_job = client.query(query)
        results = query_job.result()
    except Exception as e:
        print(f"Error querying schema embeddings: {e}")
        return "" # Return empty string if error occurs

    ddl_statements = ""
    tables_processed = set()

    table_columns = {}
    for row in results:
        if row.table_name not in table_columns:
            table_columns[row.table_name] = []
        table_columns[row.table_name].append(row)

    for table_name, cols in table_columns.items():
        if table_name in tables_processed:
            continue

        table_ref = client.dataset(data_dataset_id).table(table_name)
        try:
            table_obj = client.get_table(table_ref)
        except Exception as e:
            print(f"Could not get table {table_name}: {e}")
            continue
        
        if table_obj.table_type != "TABLE":
            continue

        ddl_statement = f"CREATE OR REPLACE TABLE `{project_id}.{data_dataset_id}.{table_name}` (\n"
        relevant_columns_in_table = {col.column_name for col in cols}

        for field in table_obj.schema:
            if field.name in relevant_columns_in_table:
                ddl_statement += f"  `{field.name}` {field.field_type}"
                if field.mode == "REPEATED":
                    ddl_statement += " ARRAY"
                col_description = next((c.column_description for c in cols if c.column_name == field.name and c.column_description), field.description)
                if col_description:
                    ddl_statement += f" COMMENT '{col_description}'"
                ddl_statement += ",\n"
        
        if not ddl_statement.endswith("(\n"):
            ddl_statement = ddl_statement[:-2] + "\n);\n\n"
            ddl_statements += ddl_statement
            tables_processed.add(table_name)

    return ddl_statements

def get_database_settings():
    """Get database settings."""
    global database_settings
    if database_settings is None:
        database_settings = update_database_settings()
    return database_settings


def update_database_settings():
    """Update database settings."""
    global database_settings
    # Determine the RAG dataset ID to use. Fallback to BQ_DATASET_ID if BQ_RAG_DATASET_ID is not set.
    # This is a fallback for cases where BQ_RAG_DATASET_ID might not be configured,
    # though for RAG functionality, it should ideally always be set.
    rag_dataset_id_to_use = get_env_var("BQ_RAG_DATASET_ID", default_value=get_env_var("BQ_DATASET_ID"))

    ddl_schema = get_bigquery_schema(
        dataset_id=get_env_var("BQ_DATASET_ID"), # This is the data dataset
        client=get_bq_client(),
        project_id=get_env_var("BQ_PROJECT_ID"),
        rag_dataset_id=rag_dataset_id_to_use # Pass the RAG dataset ID
    )
    database_settings = {
        "bq_project_id": get_env_var("BQ_PROJECT_ID"),
        "bq_dataset_id": get_env_var("BQ_DATASET_ID"), # Data dataset
        "bq_rag_dataset_id": rag_dataset_id_to_use, # RAG dataset for schema embeddings
        "bq_ddl_schema": ddl_schema,
        # Include ChaseSQL-specific constants.
        **chase_constants.chase_sql_constants_dict,
    }
    return database_settings


def get_bigquery_schema(dataset_id, client=None, project_id=None, question: str = None, rag_dataset_id: str = None):
    """Retrieves schema and generates DDL with example values for a BigQuery dataset.
    If a question is provided, it retrieves only schema relevant to the question using embeddings from the rag_dataset_id.

    Args:
        dataset_id (str): The ID of the BigQuery DATASET containing the actual data.
        client (bigquery.Client): A BigQuery client.
        project_id (str): The ID of your Google Cloud Project.
        question (str, optional): The user's question to filter schema. Defaults to None.
        rag_dataset_id (str, optional): The ID of the BigQuery RAG DATASET containing schema embeddings. Defaults to None.

    Returns:
        str: A string containing the generated DDL statements.
    """
    if question and project_id and dataset_id and rag_dataset_id:
        print(f"Retrieving schema relevant to the question using embeddings from RAG dataset: {rag_dataset_id}...")
        return get_relevant_schema_from_embeddings(question, project_id, dataset_id, rag_dataset_id)

    if client is None:
        client = bigquery.Client(project=project_id)

    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)

    ddl_statements = ""

    for table in client.list_tables(dataset_ref):
        table_ref = dataset_ref.table(table.table_id)
        table_obj = client.get_table(table_ref)

        if table_obj.table_type != "TABLE":
            continue

        ddl_statement = f"CREATE OR REPLACE TABLE `{table_ref}` (\n"

        for field in table_obj.schema:
            ddl_statement += f"  `{field.name}` {field.field_type}"
            if field.mode == "REPEATED":
                ddl_statement += " ARRAY"
            if field.description:
                ddl_statement += f" COMMENT '{field.description}'"
            ddl_statement += ",\n"

        ddl_statement = ddl_statement[:-2] + "\n);\n\n"

        rows = client.list_rows(table_ref, max_results=5).to_dataframe()
        if not rows.empty:
            ddl_statement += f"-- Example values for table `{table_ref}`:\n"
            for _, row in rows.iterrows():
                ddl_statement += f"INSERT INTO `{table_ref}` VALUES\n"
                example_row_str = "("
                for value in row.values:
                    if isinstance(value, str):
                        example_row_str += f"'{value}',"
                    elif value is None:
                        example_row_str += "NULL,"
                    else:
                        example_row_str += f"{value},"
                example_row_str = (
                    example_row_str[:-1] + ");\n\n"
                )
                ddl_statement += example_row_str

        ddl_statements += ddl_statement

    return ddl_statements


def initial_bq_nl2sql(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Generates an initial SQL query from a natural language question.

    Args:
        question (str): Natural language question.
        tool_context (ToolContext): The tool context to use for generating the SQL
          query.

    Returns:
        str: An SQL statement to answer this question.
    """

    prompt_template = """
You are a BigQuery SQL expert tasked with answering user's questions about BigQuery tables by generating SQL queries in the GoogleSql dialect.  Your task is to write a Bigquery SQL query that answers the following question while using the provided context.

**Guidelines:**

- **Table Referencing:** Always use the full table name with the database prefix in the SQL statement.  Tables should be referred to using a fully qualified name with enclosed in backticks (`) e.g. `project_name.dataset_name.table_name`.  Table names are case sensitive.
- **Joins:** Join as few tables as possible. When joining tables, ensure all join columns are the same data type. Analyze the database and the table schema provided to understand the relationships between columns and tables.
- **Aggregations:**  Use all non-aggregated columns from the `SELECT` statement in the `GROUP BY` clause.
- **SQL Syntax:** Return syntactically and semantically correct SQL for BigQuery with proper relation mapping (i.e., project_id, owner, table, and column relation). Use SQL `AS` statement to assign a new name temporarily to a table column or even a table wherever needed. Always enclose subqueries and union queries in parentheses.
- **Column Usage:** Use *ONLY* the column names (column_name) mentioned in the Table Schema. Do *NOT* use any other column names. Associate `column_name` mentioned in the Table Schema only to the `table_name` specified under Table Schema.
- **FILTERS:** You should write query effectively  to reduce and minimize the total rows to be returned. For example, you can use filters (like `WHERE`, `HAVING`, etc. (like 'COUNT', 'SUM', etc.) in the SQL query.
- **LIMIT ROWS:**  The maximum number of rows returned should be less than {MAX_NUM_ROWS}.

**Schema:**

The database structure is defined by the following table schemas (possibly with sample rows):

```
{SCHEMA}
```

**Natural language question:**

```
{QUESTION}
```

**Think Step-by-Step:** Carefully consider the schema, question, guidelines, and best practices outlined above to generate the correct BigQuery SQL.

   """

    # Retrieve schema based on the question, if NL2SQL_METHOD is not BASELINE (i.e., CHASE or other RAG-based methods)
    nl2sql_method = os.getenv("NL2SQL_METHOD", "BASELINE")
    rag_dataset_id_for_nl2sql = tool_context.state["database_settings"].get("bq_rag_dataset_id", BQ_RAG_DATASET_ID) # Get RAG dataset from context or env

    if nl2sql_method != "BASELINE":
        ddl_schema = get_bigquery_schema(
            dataset_id=tool_context.state["database_settings"]["bq_dataset_id"], # Data dataset
            project_id=tool_context.state["database_settings"]["bq_project_id"],
            question=question,
            rag_dataset_id=rag_dataset_id_for_nl2sql # Pass the RAG dataset ID
        )
    else:
        # For BASELINE, if a question is present, still try to get RAG schema, else full schema.
        # This allows BASELINE to also benefit from RAG if a question is asked.
        if question and rag_dataset_id_for_nl2sql:
             ddl_schema = get_bigquery_schema(
                dataset_id=tool_context.state["database_settings"]["bq_dataset_id"],
                project_id=tool_context.state["database_settings"]["bq_project_id"],
                question=question,
                rag_dataset_id=rag_dataset_id_for_nl2sql
            )
        else:
            ddl_schema = tool_context.state["database_settings"]["bq_ddl_schema"]

    prompt = prompt_template.format(
        MAX_NUM_ROWS=MAX_NUM_ROWS, SCHEMA=ddl_schema, QUESTION=question
    )

    response = llm_client.models.generate_content(
        model=os.getenv("BASELINE_NL2SQL_MODEL"),
        contents=prompt,
        config={"temperature": 0.1},
    )

    sql = response.text
    if sql:
        sql = sql.replace("```sql", "").replace("```", "").strip()

    print("\n sql:", sql)

    tool_context.state["sql_query"] = sql

    return sql


def run_bigquery_validation(
    sql_string: str,
    tool_context: ToolContext,
) -> str:
    """Validates BigQuery SQL syntax and functionality.

    This function validates the provided SQL string by attempting to execute it
    against BigQuery in dry-run mode. It performs the following checks:

    1. **SQL Cleanup:**  Preprocesses the SQL string using a `cleanup_sql`
    function
    2. **DML/DDL Restriction:**  Rejects any SQL queries containing DML or DDL
       statements (e.g., UPDATE, DELETE, INSERT, CREATE, ALTER) to ensure
       read-only operations.
    3. **Syntax and Execution:** Sends the cleaned SQL to BigQuery for validation.
       If the query is syntactically correct and executable, it retrieves the
       results.
    4. **Result Analysis:**  Checks if the query produced any results. If so, it
       formats the first few rows of the result set for inspection.

    Args:
        sql_string (str): The SQL query string to validate.
        tool_context (ToolContext): The tool context to use for validation.

    Returns:
        str: A message indicating the validation outcome. This includes:
             - "Valid SQL. Results: ..." if the query is valid and returns data.
             - "Valid SQL. Query executed successfully (no results)." if the query
                is valid but returns no data.
             - "Invalid SQL: ..." if the query is invalid, along with the error
                message from BigQuery.
    """

    def cleanup_sql(sql_string):
        """Processes the SQL string to get a printable, valid SQL string."""

        # 1. Remove backslashes escaping double quotes
        sql_string = sql_string.replace('\\"', '"')

        # 2. Remove backslashes before newlines (the key fix for this issue)
        sql_string = sql_string.replace("\\\n", "\n")  # Corrected regex

        # 3. Replace escaped single quotes
        sql_string = sql_string.replace("\\'", "'")

        # 4. Replace escaped newlines (those not preceded by a backslash)
        sql_string = sql_string.replace("\\n", "\n")

        # 5. Add limit clause if not present
        if "limit" not in sql_string.lower():
            sql_string = sql_string + " limit " + str(MAX_NUM_ROWS)

        return sql_string

    logging.info("Validating SQL: %s", sql_string)
    sql_string = cleanup_sql(sql_string)
    logging.info("Validating SQL (after cleanup): %s", sql_string)

    final_result = {"query_result": None, "error_message": None}

    if re.search(
        r"(?i)(update|delete|drop|insert|create|alter|truncate|merge)", sql_string
    ):
        final_result["error_message"] = (
            "Invalid SQL: Contains disallowed DML/DDL operations."
        )
        return final_result

    try:
        query_job = get_bq_client().query(sql_string)
        results = query_job.result()

        if results.schema:
            rows = [
                {
                    key: (
                        value
                        if not isinstance(value, datetime.date)
                        else value.strftime("%Y-%m-%d")
                    )
                    for (key, value) in row.items()
                }
                for row in results
            ][
                :MAX_NUM_ROWS
            ]
            final_result["query_result"] = rows

            tool_context.state["query_result"] = rows

        else:
            final_result["error_message"] = (
                "Valid SQL. Query executed successfully (no results)."
            )

    except (
        Exception
    ) as e:
        final_result["error_message"] = f"Invalid SQL: {e}"

    print("\n run_bigquery_validation final_result: \n", final_result)

    return final_result
