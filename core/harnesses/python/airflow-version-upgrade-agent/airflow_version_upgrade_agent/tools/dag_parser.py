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


import ast
import logging

from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OperatorVisitor(ast.NodeVisitor):
    """
    AST visitor to find imported and instantiated Airflow operators/sensors.
    This version is specifically filtered to 'airflow.operators', 'airflow.providers',
    and 'airflow.contrib' modules.
    """

    ALLOWED_PREFIXES = (
        "airflow.operators",
        "airflow.providers",
        "airflow.contrib",
    )

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.import_map = {}
        self.instantiated_operators = set()

    def visit_Import(self, node: ast.Import):
        """
        Handles imports like:
        - import airflow.operators.bash
        - import airflow.providers.google.cloud.operators.bigquery as bq_ops
        """
        for alias in node.names:
            if alias.name.startswith(self.ALLOWED_PREFIXES):
                local_name = alias.asname or alias.name
                self.import_map[local_name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """
        Handles imports like:
        - from airflow.contrib.operators import bigquery_operator
        - from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
        """
        if node.module and node.module.startswith(self.ALLOWED_PREFIXES):
            for alias in node.names:
                local_name = alias.asname or alias.name
                self.import_map[local_name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def _is_valid_task(self, full_path: str) -> bool:
        """
        Helper to check if the class name follows Airflow's Operator or Sensor naming conventions.
        This prevents misidentifying helper classes within the operator modules.
        """
        if not full_path:
            return False
        class_name = full_path.rsplit(".", maxsplit=1)[-1]
        return class_name.endswith("Operator") or class_name.endswith("Sensor")

    def visit_Call(self, node: ast.Call):
        """
        Identifies calls to instantiated operators or sensors from the filtered import map.
        """
        full_path = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.import_map:
                full_path = self.import_map[func_name]
        elif isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            module_alias = node.func.value.id
            if module_alias in self.import_map:
                base_module_path = self.import_map[module_alias]
                operator_class_name = node.func.attr
                full_path = f"{base_module_path}.{operator_class_name}"

        # Apply the filter to ensure we only add valid tasks
        if full_path and self._is_valid_task(full_path):
            self.instantiated_operators.add(full_path)

        self.generic_visit(node)


def extract_operators_from_gcs(gcs_folder_uri: str) -> list[str]:
    """
    Parses all Python files in a given GCS folder to extract unique Airflow operators.

    Args:
        gcs_folder_uri (str): The GCS URI of the folder containing Python files.

    Returns:
        list[str]: A sorted list of unique Airflow operators found in the GCS folder.
    """
    unique_operators = set()

    if not gcs_folder_uri.startswith("gs://"):
        logger.error(
            f"Invalid GCS URI: '{gcs_folder_uri}'. It must start with 'gs://'."
        )
        return []

    try:
        bucket_name, prefix = gcs_folder_uri.replace("gs://", "").split("/", 1)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        storage_client = storage.Client()
        logger.info(
            f"Listing blobs in bucket '{bucket_name}' with prefix '{prefix}'"
        )
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    except Exception as e:
        logger.error(
            f"Failed to connect to or parse GCS URI '{gcs_folder_uri}': {e}",
            exc_info=True,
        )
        return []

    file_count = 0
    for blob in blobs:
        if blob.name.endswith("/") or not blob.name.endswith(".py"):
            continue

        file_count += 1
        gcs_path = f"gs://{bucket_name}/{blob.name}"
        logger.info(f"Parsing file: {gcs_path}")
        try:
            source_code = blob.download_as_bytes().decode("utf-8")
            tree = ast.parse(source_code, filename=gcs_path)
            visitor = OperatorVisitor(gcs_path)
            visitor.visit(tree)
            if visitor.instantiated_operators:
                logger.info(
                    f"Found operators in {gcs_path}: {visitor.instantiated_operators}"
                )
                unique_operators.update(visitor.instantiated_operators)
        except SyntaxError as e:
            logger.warning(
                f"Could not parse {gcs_path} due to a syntax error: {e}"
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while processing {gcs_path}: {e}",
                exc_info=True,
            )

    if file_count == 0:
        logger.warning(f"No Python (.py) files found in '{gcs_folder_uri}'.")

    return sorted(unique_operators)


# Local testing
# if __name__ == "__main__":
#     dags_gcs_path = "gs://<bucket-name>/dags/"

#     print(f"Searching for operators in: {dags_gcs_path}")
#     operators = extract_operators_from_gcs(dags_gcs_path)

#     if operators:
#         print("\n Unique Airflow Operators Found")
#         for op in operators:
#             print(op)
#     else:
#         print("\n No operators found or an error occurred. Check logs.")
