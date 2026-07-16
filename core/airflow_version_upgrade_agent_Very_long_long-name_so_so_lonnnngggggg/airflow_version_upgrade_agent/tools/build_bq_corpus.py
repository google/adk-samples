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


import json
import logging
import os
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Define the path to the .env file
env_file_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_file_path)

output_schema = {
    "operator_path": "string",
    "source_version": "string",
    "target_version": "string",
    "is_deprecated": "boolean",
    "new_operator_path": "string",
    "parameter_changes": "string",
    "summary": "string",
    "code_example_before": "string",
    "code_example_after": "string",
    "source_urls": ["string"],
    "created_at": "timestamp",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@cache
def ensure_bq_table_exists(project_id: str, dataset_name: str, table_name: str):
    """Checks if the BigQuery table exists and creates it if it does not."""
    bq_client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    try:
        bq_client.get_table(table_id)
        logging.info(f"Table {table_id} already exists.")
    except NotFound:
        logging.warning(f"Table {table_id} not found. Proceeding to create it.")
        try:
            dataset = bigquery.Dataset(f"{project_id}.{dataset_name}")
            dataset.location = "US"
            bq_client.create_dataset(dataset, exists_ok=True)

            schema = [
                bigquery.SchemaField(
                    "operator_path", "STRING", mode="REQUIRED"
                ),
                bigquery.SchemaField(
                    "source_version", "STRING", mode="REQUIRED"
                ),
                bigquery.SchemaField(
                    "target_version", "STRING", mode="REQUIRED"
                ),
                bigquery.SchemaField("is_deprecated", "BOOLEAN"),
                bigquery.SchemaField("new_operator_path", "STRING"),
                bigquery.SchemaField("parameter_changes", "JSON"),
                bigquery.SchemaField("summary", "STRING"),
                bigquery.SchemaField("code_example_before", "STRING"),
                bigquery.SchemaField("code_example_after", "STRING"),
                bigquery.SchemaField("source_urls", "STRING", mode="REPEATED"),
                bigquery.SchemaField(
                    "created_at", "TIMESTAMP", mode="REQUIRED"
                ),
            ]
            table = bigquery.Table(table_id, schema=schema)
            bq_client.create_table(table)
            logging.info(f"Successfully created table {table_id}.")
        except Exception as e:
            logging.error(f"Error creating BigQuery dataset or table: {e}")
            raise Exception(
                f"Failed to create BigQuery infrastructure: {e}"
            ) from e


def generate_and_store_knowledge(
    research_results: list[dict],
    source_version: str,
    target_version: str,
    project_id: str,
) -> str:
    """
    Processes scraped text with an LLM and stores it in BigQuery as a data corpus.

    Args:
        research_results : List of results from web scrapper
        source_version (str): The source Airflow version for documentation research.
        target_version (str): The target Airflow version for documentation research.

    Returns:
        str: A message indicating the completion of the knowledge base update.
    """
    dataset_name = os.getenv("BIGQUERY_DATASET")
    table_name = os.getenv("BIGQUERY_TABLE")

    if not dataset_name or not table_name:
        return "ERROR: BIGQUERY_DATASET or BIGQUERY_TABLE missing from environment variables."

    table_id = f"{project_id}.{dataset_name}.{table_name}"
    bq_client = bigquery.Client(project=project_id)

    try:
        ensure_bq_table_exists(project_id, dataset_name, table_name)
    except Exception as e:
        return str(e)  # Return error to the agent

    client = genai.Client(
        vertexai=True, project=project_id, location="us-central1"
    )

    success_count = 0
    errors_list = []

    for result in research_results:
        operator = result["operator"]
        scraped_text = result["content"]
        source_urls = result["urls"]

        prompt = f"""
        You are an expert Airflow migration engineer. Your task is to analyze the provided text & URLs 
        and extract specific migration information for an Airflow operator.
        **Context:**
        - Operator to analyze: `{operator}`
        - Migrating from Airflow version: `{source_version}`
        - Migrating to Airflow version: `{target_version}`
        **Source Text & URLs to Analyze:**
        ```
        {scraped_text}
        ```
        **Instructions:**
        Respond ONLY with a single, valid JSON structure adhere to the following schema:
        ```json\n{json.dumps(output_schema, indent=2)}\n```
        """
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro", contents=prompt
            )
            json_str = (
                response.text.strip().replace("```json", "").replace("```", "")
            )
            structured_data = json.loads(json_str)

            row_to_insert = {
                "operator_path": operator,
                "source_version": source_version,
                "target_version": target_version,
                "is_deprecated": structured_data.get("is_deprecated"),
                "new_operator_path": structured_data.get("new_operator_path"),
                "parameter_changes": json.dumps(
                    structured_data.get("parameter_changes", {})
                ),
                "summary": structured_data.get("summary"),
                "code_example_before": structured_data.get(
                    "code_example_before"
                ),
                "code_example_after": structured_data.get("code_example_after"),
                "source_urls": source_urls,
                "created_at": datetime.now(UTC).isoformat(),
            }

            errors = bq_client.insert_rows_json(table_id, [row_to_insert])
            if errors:
                errors_list.append(f"BQ Error for {operator}: {errors}")
            else:
                success_count += 1
                logging.info(f"Successfully stored knowledge for {operator}")

        except Exception as e:
            errors_list.append(f"Failed processing {operator}: {e!s}")
            logging.error(
                f"Failed to process and store knowledge for {operator}: {e}"
            )

    if errors_list:
        return f"WARNING: Completed with errors. Inserted {success_count} operators. Errors: {errors_list[:2]}..."

    return f"Knowledge base update complete. Successfully stored {success_count} operators."


# For local testing
# if __name__ == '__main__':
#     from web_scrapper import research_operator_documentation
#     project_id = os.getenv("PROJECT_ID")

#     # Example usage
#     operator_path = ["airflow.contrib.operators.bigquery_operator.BigQueryOperator",
#         "airflow.operators.bash_operator.BashOperator",
#         "airflow.operators.data_plugins.DataprocsK8sOperator",
#         "airflow.operators.dummy_operator.DummyOperator",
#         "airflow.operators.python_operator.PythonOperator",
#         "airflow.operators.python_operator.PythonVirtualenvOperator",
#         "airflow.operators.slack_operator.SlackAPIPostOperator"]
#     source_version = '1.10'
#     target_version = '2.10.5'
#     scraped_content_dict = research_operator_documentation(operator_path, source_version, target_version, project_id)
#     return_msg = generate_and_store_knowledge(scraped_content_dict, source_version, target_version, project_id)
#     print(return_msg)
