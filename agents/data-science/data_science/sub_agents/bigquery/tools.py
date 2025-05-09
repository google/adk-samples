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
BQ_METADATA_RAG_CORPUS_ID = os.getenv("BQ_METADATA_RAG_CORPUS_ID")

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

def get_relevant_schema_from_embeddings(question: str, project_id: str, rag_corpus_id: str) -> str:
    """Retrieves relevant schema details (tables and columns) based on vector similarity to the question,
    querying a centralized RAG corpus that contains metadata for all configured datasets."""
    client = get_bq_client()
    question_embedding = get_column_embeddings([question])[0]

    if not rag_corpus_id:
        print("Error: BQ_METADATA_RAG_CORPUS_ID is not set. Cannot query schema embeddings.")
        return ""

    print(f"Querying RAG Corpus: {rag_corpus_id} for question: {question}")
    print(f"Fetching relevant DDL from RAG corpus {rag_corpus_id} based on the question.")
    return f"-- Placeholder: DDLs for tables relevant to '{question}' from RAG corpus '{rag_corpus_id}' would be listed here.\n"

def get_database_settings():
    """Get database settings."""
    global database_settings
    if database_settings is None:
        database_settings = update_database_settings()
    return database_settings


def update_database_settings():
    """Update database settings."""
    global database_settings
    
    project_id = get_env_var("BQ_PROJECT_ID")
    dataset_ids_str = get_env_var("BQ_DATASET_IDS")
    metadata_rag_corpus_id = get_env_var("BQ_METADATA_RAG_CORPUS_ID")

    if not dataset_ids_str:
        raise ValueError("BQ_DATASET_IDS environment variable is not set.")
    if not metadata_rag_corpus_id:
        print("Warning: BQ_METADATA_RAG_CORPUS_ID is not set. RAG-based schema retrieval will be limited.")
    
    dataset_ids = [ds_id.strip() for ds_id in dataset_ids_str.split(',')]

    ddl_overview = f"-- Schema for datasets ({', '.join(dataset_ids)}) is primarily retrieved dynamically via RAG from corpus: {metadata_rag_corpus_id}\n"

    database_settings = {
        "bq_project_id": project_id,
        "bq_dataset_ids": dataset_ids, # List of dataset IDs
        "bq_metadata_rag_corpus_id": metadata_rag_corpus_id, # Central RAG corpus for schema
        "bq_ddl_schema": ddl_overview, # Overview or placeholder
        **chase_constants.chase_sql_constants_dict,
    }
    return database_settings


def get_bigquery_schema(
    client=None, 
    project_id=None, 
    question: str = None, 
    rag_corpus_id: str = None,
    target_dataset_ids: list[str] = None 
    ):
    """
    Retrieves schema. If a question and rag_corpus_id are provided, it uses RAG.
    Otherwise, if target_dataset_ids are provided, it gets all tables for those.
    """
    if question and project_id and rag_corpus_id:
        print(f"Retrieving schema relevant to the question using RAG corpus: {rag_corpus_id}...")
        return get_relevant_schema_from_embeddings(question, project_id, rag_corpus_id)

    if not target_dataset_ids:
        return "-- No specific datasets provided for full schema dump and no question for RAG-based retrieval --\n"

    if client is None:
        client = get_bq_client()

    all_ddl_statements = f"-- Full DDL for datasets: {', '.join(target_dataset_ids)}\n"

    for dataset_id_str in target_dataset_ids:
        all_ddl_statements += f"-- Schema for dataset: {project_id}.{dataset_id_str}\n"
        dataset_ref = bigquery.DatasetReference(project_id, dataset_id_str)
        try:
            tables = client.list_tables(dataset_ref)
        except Exception as e:
            all_ddl_statements += f"-- Could not list tables for dataset {dataset_id_str}: {e}\n"
            continue
            
        for table in tables:
            table_ref = dataset_ref.table(table.table_id)
            try:
                table_obj = client.get_table(table_ref)
            except Exception as e:
                all_ddl_statements += f"-- Could not get table {table_ref} from dataset {dataset_id_str}: {e}\n"
                continue

            if table_obj.table_type != "TABLE":
                continue

            ddl_statement = f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id_str}.{table.table_id}` (\n"
            for field in table_obj.schema:
                ddl_statement += f"  `{field.name}` {field.field_type}"
                if field.mode == "REPEATED":
                    ddl_statement += " ARRAY"
                if field.description:
                    clean_description = str(field.description).replace("'", "''").replace("\\n", " ")
                    ddl_statement += f" COMMENT '{clean_description}'"
                ddl_statement += ",\n"
            
            if ddl_statement.endswith(",\n"):
                ddl_statement = ddl_statement.removesuffix(",\n") + "\n);\n\n"
            elif ddl_statement.endswith("(\n"): 
                ddl_statement = ddl_statement.removesuffix("(\n") + "();\n\n"
            # If ddl_statement doesn't end with either (e.g., it's empty or malformed from prior steps),
            # it will pass through this block unchanged. This relies on prior logic correctly
            # initializing ddl_statement and appending columns.

            try:
                rows_df = client.list_rows(table_ref, max_results=2).to_dataframe()
                if not rows_df.empty:
                    ddl_statement += f"-- Example values for table `{project_id}.{dataset_id_str}.{table.table_id}`:\n"
                    for _, row_val in rows_df.iterrows():
                        example_row_str = "INSERT INTO `{}.{}.{}` VALUES (".format(project_id, dataset_id_str, table.table_id)
                        values = []
                        for item in row_val.values:
                            if isinstance(item, str):
                                values.append(f"'{str(item).replace(\"'\", \"''\")}'")
                            elif item is None:
                                values.append("NULL")
                            else:
                                values.append(str(item))
                        example_row_str += ", ".join(values) + ");\n"
                        ddl_statement += example_row_str
                    ddl_statement += "\n"
            except Exception as e:
                ddl_statement += f"-- Could not fetch example rows for {table.table_id}: {e}\n"
            
            all_ddl_statements += ddl_statement
    return all_ddl_statements


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

    nl2sql_method = os.getenv("NL2SQL_METHOD", "BASELINE")
    current_db_settings = get_database_settings()
    metadata_rag_corpus_id_for_nl2sql = current_db_settings.get("bq_metadata_rag_corpus_id", BQ_METADATA_RAG_CORPUS_ID)

    if nl2sql_method != "BASELINE":
        if not metadata_rag_corpus_id_for_nl2sql:
            ddl_schema = "-- ERROR: BQ_METADATA_RAG_CORPUS_ID not configured for RAG schema retrieval. --\n"
        else:
            ddl_schema = get_bigquery_schema(
                project_id=current_db_settings["bq_project_id"],
                question=question,
                rag_corpus_id=metadata_rag_corpus_id_for_nl2sql
            )
    else:
        if question and metadata_rag_corpus_id_for_nl2sql:
            ddl_schema = get_bigquery_schema(
                project_id=current_db_settings["bq_project_id"],
                question=question,
                rag_corpus_id=metadata_rag_corpus_id_for_nl2sql
            )
        else:
            ddl_schema = current_db_settings.get("bq_ddl_schema", "-- Schema information is missing. --\n")
    
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
                {key: value for key, value in row.items()}
                for row in results
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
