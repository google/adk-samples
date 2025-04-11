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

"""Translator from SQLite to BigQuery."""

import re
from typing import Any, Final

import regex
import sqlglot
import sqlglot.optimizer

from ..llm_utils import GeminiModel  # pylint: disable=g-importing-member
from .correction_prompt_template import CORRECTION_PROMPT_TEMPLATE_V1_0  # pylint: disable=g-importing-member


ColumnSchemaType = tuple[str, str]
AllColumnsSchemaType = list[ColumnSchemaType]
TableSchemaType = tuple[str, AllColumnsSchemaType]
DDLSchemaType = list[TableSchemaType]

SQLGlotColumnsDictType = dict[str, str]
SQLGlotSchemaType = dict[str, Any]

BirdSampleType = dict[str, Any]


def _isinstance_list_of_str_tuples_lists(obj: Any) -> bool:
  """Checks if the object is a list of tuples or listsof strings."""
  return (
      isinstance(obj, list)
      and all([isinstance(v, (tuple, list)) for v in obj])
      and all([isinstance(v[0], str) and isinstance(v[1], str) for v in obj])
  )


def _isinstance_ddl_schema_type(obj: Any) -> bool:
  """Checks if the object is a DDL schema type."""
  # pylint: disable=g-complex-comprehension
  return (
      isinstance(obj, list)
      and all(
          # Every element is a tuple or list.
          [isinstance(v, (tuple, list)) for v in obj]
      )
      and all(
          # First element is a string (table name) and
          # second element is a list (of tuples or lists).
          [isinstance(v[0], str) and isinstance(v[1], list) for v in obj]
      )
      and all(
          # Every element of above list is a tuple or list of strings
          # (column name, column type)
          [_isinstance_list_of_str_tuples_lists(v[1]) for v in obj]
      )
  )
  # pylint: enable=g-complex-comprehension


def _isinstance_sqlglot_schema_type(obj: Any) -> bool:
  """Checks if the object is a SQLGlot schema type."""
  # pylint: disable=g-complex-comprehension
  return (
      isinstance(obj, dict)
      and all([isinstance(v, dict) for v in obj.values()])
      and all([isinstance(c, str) for d in obj.values() for c, _ in d.items()])
      and all([isinstance(t, str) for d in obj.values() for _, t in d.items()])
  )
  # pylint: enable=g-complex-comprehension


def _isinstance_bird_sample_type(obj: Any) -> bool:
  """Checks if the object is a SQLGlot schema type."""
  return isinstance(obj, dict) and not _isinstance_sqlglot_schema_type(obj)


class SqlTranslator:
  """Translator from SQLite to BigQuery.

  This class is used to translate SQL queries from an input SQL dialect like
  SQLite to an output SQL dialect like BigQuery. It uses the SQLGlot library as
  a tool to perform the translation.

  The translation is done by the following steps:
  1. (Optional) If there are errors in the input SQL query, the input SQL query
     is first modified by the LLM to address the errors.
  2. The input SQL query is then translated to a SQL query in the output SQL
     dialect by the tool.
  3. (Optional) If there are errors in the tool output SQL query, the tool
     output SQL query is modified by the LLM to address the errors.

  Class Attributes:
    INPUT_DIALECT: The input SQL dialect.
    OUTPUT_DIALECT: The output SQL dialect.

  Attributes:
    sql_query: The SQL query to translate.
    model: The model object, or the name of the model to use for the LLM.
    temperature: The temperature to use for the LLM.
    process_input_errors: True if any errors in the input SQL query should be
      processed by the LLM.
    process_tool_output_errors: True if any errors in the tool output SQL query
      should be processed by the LLM.
  """

  INPUT_DIALECT: Final[str] = "sqlite"
  OUTPUT_DIALECT: Final[str] = "bigquery"

  def __init__(
      self,
      model: str | GeminiModel = "gemini-1.5-flash-002",
      temperature: float = 0.5,
      process_input_errors: bool = False,
      process_tool_output_errors: bool = False,
  ):
    """Initializes the translator."""
    self._process_input_errors: bool = process_input_errors
    self._process_tool_output_errors: bool = process_tool_output_errors
    self._input_errors: str | None = None
    self._tool_output_errors: str | None = None
    self._temperature: float = temperature
    if isinstance(model, str):
      self._model = GeminiModel(model_name=model, temperature=self._temperature)
    else:
      self._model = model

  @classmethod
  def _parse_response(cls, text: str) -> str | None:
    """Extracts the SQL query from the response text."""
    pattern = r"```sql(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
      return match.group(1).strip()
    return None

  @classmethod
  def _apply_heuristics(cls, sql_query: str) -> str:
    """Applies heuristics to the SQL query."""
    if "''" in sql_query:
      sql_query = sql_query.replace("''", "\\'")
    return sql_query

  @classmethod
  def _extract_schema_from_ddl_statement(
      cls, ddl_statement: str
  ) -> TableSchemaType:
    """Extracts the schema from a single DDL statement."""
    # Split the DDL statement into table name and columns.
    # Match the following pattern:
    # CREATE [OR REPLACE] TABLE [`]<table_name>[`] (<all_columns>);
    splitter_pattern = (
        # CREATE [OR REPLACE] TABLE
        r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+"
        # Match the table name, optionally surrounded by backticks.
        r"(?:`)?(?P<table_name>[\w\d\-\_\.]+)(?:`)?\s*"
        # Match the column name as everything between the first and last
        # parentheses followed by a semicolon.
        r"\((?P<all_columns>.*)\);$"
    )
    split_match = regex.search(
        splitter_pattern,
        ddl_statement,
        flags=re.DOTALL | re.VERBOSE | re.MULTILINE,
    )
    if not split_match:
      return None, None

    table_name = split_match.group("table_name")
    all_columns = split_match.group("all_columns").strip()
    if not table_name or not all_columns:
      return None, None

    # Extract the columns from the DDL statement.
    # Match the following pattern:
    # <column_name> <column_type> [<ignored_text>]
    # [, <column_name> <column_type> [<ignored_text>]]*
    # Ignore any comments. Ignore any INSERT INTO statements. Ignore any
    # lines beginning with a parenthesis (these are example values).
    column_pattern = (
        # Ignore any comments.
        r"\s*--.*(*SKIP)(*FAIL)"
        # Ignore any INSERT INTO statements.
        r"|\s*INSERT\s+INTO.*(*SKIP)(*FAIL)"
        # Ignore any lines beginning with a parenthesis.
        r"|\s*\(.*(*SKIP)(*FAIL)"
        # Match the column name and type, optionally with backticks.
        r"|\s*(?:`)?\s*(?P<column_name>\w+)(?:`)?\s+(?P<column_type>\w+).*"
    )  # (?:,)?
    columns = regex.findall(column_pattern, all_columns, flags=re.VERBOSE)
    return table_name, columns

  @classmethod
  def extract_schema_from_ddls(cls, ddls: str) -> DDLSchemaType:
    """Extracts the schema from multiple DDL statements."""
    ddl_statements = ddls.split(";\n")
    ddl_statements = [ddl.strip() for ddl in ddl_statements if ddl.strip()]
    schema = []
    for ddl_statement in ddl_statements:
      if ddl_statement:
        ddl_statement = ddl_statement.strip() + ";"  # Add the semicolon back.
        table_name, columns = cls._extract_schema_from_ddl_statement(
            ddl_statement
        )
        if table_name and columns:
          schema.append((table_name, columns))
    return schema

  @classmethod
  def _get_schema_from_bird_sample(
      cls, sample: BirdSampleType
  ) -> dict[str, dict[str, str]]:
    """Returns the schema from the Bird dataset example."""
    col_types_map: dict[str, str] = {
        "text": "TEXT",
        "number": "FLOAT",
        "date": "DATE",
        "datetime": "DATETIME",
        "time": "TIME",
        "timestamp": "TIMESTAMP",
        "bool": "BOOL",
    }
    tables = sample["db_table_names"]
    table_ids = sample["db_column_names"]["table_id"][1:]
    column_names = sample["db_column_names"]["column_name"][1:]
    column_types = sample["db_column_types"][1:]
    column_types = [col_types_map[col_type] for col_type in column_types]
    assert len(column_names) == len(column_types)
    cols_and_types: list[tuple[str, str]] = list(
        zip(column_names, column_types)
    )
    tables_to_columns: dict[str, dict[str, str]] = {}
    for id_pos, table_id in enumerate(table_ids):
      if tables[table_id] in tables_to_columns.keys():
        tables_to_columns[tables[table_id]].update(
            dict([cols_and_types[id_pos]])
        )
      else:
        tables_to_columns[tables[table_id]] = dict([cols_and_types[id_pos]])
    return tables_to_columns

  @classmethod
  def _get_table_parts(
      cls, table_name: str
  ) -> tuple[str | None, str | None, str]:
    """Returns the table parts from the table name."""
    table_parts = table_name.split(".")
    if len(table_parts) == 3:
      return table_parts
    elif len(table_parts) == 2:
      return None, *table_parts
    elif len(table_parts) == 1:
      return None, None, *table_parts
    else:
      raise ValueError(f"Invalid table name: {table_name}")

  @classmethod
  def format_schema(cls, schema: DDLSchemaType) -> SQLGlotSchemaType:
    """Formats the DDL schema for use in SQLGlot."""
    schema_dict = {}
    catalog, db = None, None
    for table_name, columns in schema:
      catalog, db, table_name = cls._get_table_parts(table_name)
      schema_dict[table_name] = {}
      for column_name, column_type in columns:
        schema_dict[table_name][column_name] = column_type
    if db:
      schema_dict = {db: schema_dict}
    if catalog:
      schema_dict = {catalog: schema_dict}
    return schema_dict

  @classmethod
  def rewrite_schema_for_sqlglot(
      cls, schema: str | SQLGlotSchemaType | BirdSampleType
  ) -> SQLGlotSchemaType:
    """Rewrites the schema for use in SQLGlot."""
    schema_dict = None
    if schema:
      if isinstance(schema, str):
        schema = cls.extract_schema_from_ddls(schema)
        schema_dict = cls.format_schema(schema)
      elif _isinstance_sqlglot_schema_type(schema):
        schema_dict = schema
      elif _isinstance_bird_sample_type(schema):
        schema_dict = cls._get_schema_from_bird_sample(schema)
      elif _isinstance_ddl_schema_type(schema):
        schema_dict = cls.format_schema(schema)
      else:
        raise TypeError(f"Unsupported schema type: {type(schema)}")
    return schema_dict

  @classmethod
  def _check_for_errors(
      cls,
      sql_query: str,
      sql_dialect: str,
      db: str | None = None,
      catalog: str | None = None,
      schema_dict: SQLGlotSchemaType | None = None,
  ) -> tuple[str | None, str]:
    """Checks for errors in the SQL query.

    Args:
      sql_query: The SQL query to check for errors.
      sql_dialect: The SQL dialect of the SQL query.
      db: The database to use for the translation. This field is optional.
      catalog: The catalog to use for the translation. `catalog` is the SQLGlot
        term for the project ID. This field is optional.
      schema_dict: The DDL schema to use for the translation. The DDL format is
        in the SQLGlot format. This field is optional.

    Returns:
      tuple of the errors in the SQL query, or None if there are no errors, and
      the SQL query after optimization.
    """
    try:
      # First, try to parse the SQL query into a SQLGlot AST.
      sql_query_ast = sqlglot.parse_one(
          sql=sql_query,
          read=sql_dialect.lower(),
          error_level=sqlglot.ErrorLevel.IMMEDIATE,
      )
      # Then add the database and catalog information for each table to the AST.
      for table in sql_query_ast.find_all(sqlglot.exp.Table):
        table.set("catalog", sqlglot.exp.Identifier(this=catalog, quoted=True))
        table.set("db", sqlglot.exp.Identifier(this=db, quoted=True))
      # Then, try to optimize the SQL query.
      sql_query_ast = sqlglot.optimizer.optimize(
          sql_query_ast,
          dialect=sql_dialect.lower(),
          schema=schema_dict,
          db=db,
          catalog=catalog,
          error_level=sqlglot.ErrorLevel.IMMEDIATE,
      )
      sql_query = sql_query_ast.sql(sql_dialect.lower())
    except sqlglot.errors.SqlglotError as e:
      return str(e), sql_query
    return None, sql_query

  def _fix_errors(
      self,
      sql_query: str,
      sql_dialect: str,
      apply_heuristics: bool,
      db: str | None = None,
      catalog: str | None = None,
      ddl_schema: str | SQLGlotSchemaType | BirdSampleType | None = None,
      number_of_candidates: int = 1,
  ) -> str:
    """Fixes errors in the SQL query.

    Args:
      sql_query: The SQL query to fix.
      sql_dialect: The input SQL dialect.
      apply_heuristics: True if the heuristics should be applied.
      db: The database to use for the translation. This field is optional.
      catalog: The catalog to use for the translation. `catalog` is the SQLGlot
        term for the project ID. This field is optional.
      ddl_schema: The DDL schema to use for the translation. The DDL format can
        be the SQLGlot format, the DDL schema format, a Bird dataset example, or
        a string containing multiple DDL statements. This field is optional.
      number_of_candidates: The number of candidates to generate, default is 1.

    Returns:
      str: The fixed SQL query.
    """
    if apply_heuristics:
      sql_query = self._apply_heuristics(sql_query)
    # Reformat the schema if provided. This will remove any comments and
    # `INSERT INTO` statements.
    schema_dict = self.rewrite_schema_for_sqlglot(ddl_schema)
    errors_and_sql: tuple[str | None, str] = self._check_for_errors(
        sql_query=sql_query,
        sql_dialect=self.OUTPUT_DIALECT,
        db=db,
        catalog=catalog,
        schema_dict=schema_dict,
    )
    errors, sql_query = errors_and_sql
    responses = sql_query  # Default to the input SQL query after error check.
    if errors:
      print("Processing input errors")
      if schema_dict:
        # If the schema is provided, then insert it into the prompt.
        schema_insert = f"\nThe database schema is:\n{schema_dict}\n"
      else:
        schema_insert = "\n"
      prompt: str = CORRECTION_PROMPT_TEMPLATE_V1_0.format(
          sql_dialect=sql_dialect.lower(),
          errors=errors,
          sql_query=sql_query,
          schema_insert=schema_insert,
      )
      requests: list[str] = [prompt for _ in range(number_of_candidates)]
      responses: list[str] = self._model.call_parallel(
          requests, parser_func=self._parse_response
      )
      if responses:
        # We only use the first response. Therefore the `number_of_candidates`
        # parameter is not used.
        # pylint: disable=g-bad-todo
        # pylint: enable=g-bad-todo
        # First, find the first non-None response.
        responses = [r for r in responses if r is not None]
        if responses:
          # Then, return the first non-None response.
          responses = responses[0]
    return responses

  def translate(
      self,
      sql_query: str,
      db: str | None = None,
      catalog: str | None = None,
      ddl_schema: str | SQLGlotSchemaType | BirdSampleType | None = None,
  ) -> str:
    """Translates the SQL query to the output SQL dialect.

    Args:
      sql_query: The SQL query to translate.
      db: The database to use for the translation. This field is optional.
      catalog: The catalog to use for the translation. `catalog` is the SQLGlot
        term for the project ID. This field is optional.
      ddl_schema: The DDL schema to use for the translation. The DDL format can
        be the SQLGlot format or the DDL schema format. This field is optional.

    Returns:
      The translated SQL query.
    """
    print("****** sql_query at translator entry:", sql_query)
    if self._process_input_errors:
      sql_query = self._fix_errors(
          sql_query,
          db=db,
          catalog=catalog,
          sql_dialect=self.OUTPUT_DIALECT,
          ddl_schema=ddl_schema,
          apply_heuristics=True,
      )
    print("****** sql_query after fix_errors:", sql_query)
    sql_query = sqlglot.transpile(
        sql=sql_query,
        read=self.INPUT_DIALECT,
        write=self.OUTPUT_DIALECT,
        error_level=sqlglot.ErrorLevel.IMMEDIATE,
    )[
        0
    ]  # Transpile returns a list of strings.
    print("****** sql_query after transpile:", sql_query)
    if self._tool_output_errors:
      sql_query = self._fix_errors(
          sql_query,
          db=db,
          catalog=catalog,
          sql_dialect=self.OUTPUT_DIALECT,
          ddl_schema=ddl_schema,
          apply_heuristics=True,
      )

    sql_query = sql_query.strip().replace('"', "`")
    sql_query = self._apply_heuristics(sql_query)

    return sql_query
