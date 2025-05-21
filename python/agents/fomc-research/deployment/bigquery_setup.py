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

"""BigQuery table creation script."""

import csv
from collections.abc import Sequence

from absl import app, flags
from google.cloud import bigquery
from google.cloud.exceptions import Conflict, GoogleCloudError, NotFound

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("dataset_id", None, "BigQuery dataset ID.")
flags.DEFINE_string("data_file", None, "Path to the data file.")
flags.DEFINE_string("location", "us-central1", "Location for the dataset.")
flags.mark_flags_as_required(["project_id", "dataset_id"])


def create_bigquery_dataset(
    client: bigquery.Client,
    dataset_id: str,
    location: str,
    description: str = None,
    exists_ok: bool = True,
) -> bigquery.Dataset:
    """Creates a new BigQuery dataset.

    Args:
        client: A BigQuery client object.
        dataset_id: The ID of the dataset to create.
        location: The location for the dataset (e.g., "US", "EU").
            Defaults to "US".
        description: An optional description for the dataset.
        exists_ok: If True, do not raise an exception if the dataset already
            exists.

    Returns:
        The newly created bigquery.Dataset object.

    Raises:
        google.cloud.exceptions.Conflict: If the dataset already exists and
            exists_ok is False.
        Exception: for any other error.
    """

    dataset_ref = bigquery.DatasetReference(client.project, dataset_id)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location
    if description:
        dataset.description = description

    try:
        dataset = client.create_dataset(dataset)
        print(f"Dataset {dataset.dataset_id} created in {dataset.location}.")
        return dataset
    except Conflict as e:
        if exists_ok:
            print(f"Dataset {dataset.dataset_id} already exists.")
            return client.get_dataset(dataset_ref)
        else:
            raise e


def create_bigquery_table(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    schema: list[bigquery.SchemaField],
    description: str = None,
    exists_ok: bool = False,
) -> bigquery.Table:
    """Creates a new BigQuery table.

    Args:
        client: A BigQuery client object.
        dataset_id: The ID of the dataset containing the table.
        table_id: The ID of the table to create.
        schema: A list of bigquery.SchemaField objects defining the table
            schema.
        description: An optional description for the table.
        exists_ok: If True, do not raise an exception if the table already
            exists.

    Returns:
        The newly created bigquery.Table object.

    Raises:
        ValueError: If the schema is empty.
        google.cloud.exceptions.Conflict: If the table already exists and
            exists_ok is False.
        google.cloud.exceptions.NotFound: If the dataset does not exist.
        Exception: for any other error.
    """

    if not schema:
        raise ValueError("Schema cannot be empty.")

    table_ref = bigquery.TableReference(
        bigquery.DatasetReference(client.project, dataset_id), table_id
    )
    table = bigquery.Table(table_ref, schema=schema)

    if description:
        table.description = description

    try:
        table = client.create_table(table)
        print(
            f"Table {table.project}.{table.dataset_id}.{table.table_id} "
            "created."
        )
        return table
    except Exception as e:  # pylint: disable=broad-exception-caught
        if isinstance(e, NotFound):
            raise NotFound(
                f"Dataset {dataset_id} not found in project {client.project}"
            ) from e
        if "Already Exists" in str(e) and exists_ok:
            print(
                f"Table {table.project}.{table.dataset_id}.{table.table_id} "
                "already exists."
            )
            return client.get_table(table_ref)
        else:
            # pylint: disable=broad-exception-raised
            raise Exception(f"Error creating table: {e}") from e


def insert_csv_to_bigquery(
    client: bigquery.Client,
    table: bigquery.Table,
    csv_filepath: str,
    write_disposition: str = "WRITE_APPEND",
) -> None:
    """
    Reads a CSV file and inserts its contents into a BigQuery table.

    Args:
        client: A BigQuery client object.
        table: A BigQuery table object.
        csv_filepath: The path to the CSV file.
        write_disposition: Specifies the action that occurs if the destination
            table already exists.
            Valid values are:
                - "WRITE_APPEND": Appends the data to the table.
                - "WRITE_TRUNCATE": Overwrites the table data.
                - "WRITE_EMPTY": Only writes if the table is empty.
            Defaults to "WRITE_APPEND".

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the write_disposition is invalid.
        google.cloud.exceptions.GoogleCloudError: If any other error occurs
            during the BigQuery operation.
    """

    if write_disposition not in [
        "WRITE_APPEND",
        "WRITE_TRUNCATE",
        "WRITE_EMPTY",
    ]:
        raise ValueError(
            f"Invalid write_disposition: {write_disposition}. "
            "Must be one of 'WRITE_APPEND', 'WRITE_TRUNCATE', or 'WRITE_EMPTY'."
        )

    try:
        with open(csv_filepath, "r", encoding="utf-8") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            rows_to_insert = list(csv_reader)

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}") from None

    if not rows_to_insert:
        print("CSV file is empty. Nothing to insert.")
        return

    errors = client.insert_rows_json(
        table, rows_to_insert, row_ids=[None] * len(rows_to_insert)
    )

    if errors:
        raise GoogleCloudError(
            f"Errors occurred while inserting rows: {errors}"
        )
    else:
        print(
            f"Successfully inserted {len(rows_to_insert)} rows into "
            f"{table.table_id}."
        )


def main(argv: Sequence[str]) -> None:  # pylint: disable=unused-argument

    # Define the table schema
    data_table_name = "timeseries_data"
    data_table_schema = [
        bigquery.SchemaField("timeseries_code", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("value", "FLOAT", mode="REQUIRED"),
    ]
    data_table_description = "Timeseries data"

    client = bigquery.Client(project=FLAGS.project_id)

    dataset = create_bigquery_dataset(
        client,
        FLAGS.dataset_id,
        FLAGS.location,
        description="Timeseries dataset",
    )

    data_table = create_bigquery_table(
        client,
        dataset.dataset_id,
        data_table_name,
        data_table_schema,
        data_table_description,
    )

    if FLAGS.data_file:
        insert_csv_to_bigquery(client, data_table, FLAGS.data_file)


if __name__ == "__main__":
    app.run(main)
