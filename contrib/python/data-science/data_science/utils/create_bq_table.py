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


import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

# Define the path to the .env file
env_file_path = Path(__file__).parent.parent.parent / ".env"
print(env_file_path)

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=env_file_path)


def load_csv_to_bigquery(
    data_project_id, dataset_name, table_name, csv_filepath
):
    """Loads a CSV file into a BigQuery table.

    Args:
        data_project_id: GCP Project for BQ data.
        dataset_name: The name of the BigQuery dataset.
        table_name: The name of the BigQuery table.
        csv_filepath: The path to the CSV file.
    """

    client = bigquery.Client(project=data_project_id)

    dataset_ref = client.dataset(dataset_name)
    table_ref = dataset_ref.table(table_name)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,  # Skip the header row
        autodetect=True,  # Automatically detect the schema
    )

    with open(csv_filepath, "rb") as source_file:
        job = client.load_table_from_file(
            source_file, table_ref, job_config=job_config
        )

    job.result()  # Wait for the job to complete

    print(f"Loaded {job.output_rows} rows into {dataset_name}.{table_name}")


def create_dataset_if_not_exists(
    compute_project_id, data_project_id, dataset_name
):
    """Creates a BigQuery dataset if it does not already exist.

    Args:
        compute_project_id: GCP Project for BQ compute.
        data_project_id: GQP Project for BQ data.
        dataset_name: The name of the BigQuery dataset.
    """
    client = bigquery.Client(project=compute_project_id)
    dataset_full_name = f"{data_project_id}.{dataset_name}"

    try:
        client.get_dataset(dataset_full_name)  # Make an API request.
        print(f"Dataset {dataset_full_name} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_full_name)
        dataset.location = "US"  # Set the location (e.g., "US", "EU")
        dataset = client.create_dataset(
            dataset, timeout=30
        )  # Make an API request.
        print(f"Created dataset {dataset_full_name}")


def main():
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    """Main function to load CSV files into BigQuery."""
    data_project_id = os.getenv("BQ_DATA_PROJECT_ID")
    compute_project_id = os.getenv("BQ_COMPUTE_PROJECT_ID")
    if not data_project_id:
        raise ValueError("BQ_DATA_PROJECT_ID environment variable not set.")
    if not compute_project_id:
        raise ValueError("BQ_COMPUTE_PROJECT_ID environment variable not set.")

    dataset_name = "forecasting_sticker_sales"
    train_csv_filepath = "data_science/utils/data/train.csv"
    test_csv_filepath = "data_science/utils/data/test.csv"

    # Create the dataset if it doesn't exist
    print("Creating dataset.")
    create_dataset_if_not_exists(
        compute_project_id, data_project_id, dataset_name
    )

    # Load the train data
    print("Loading train table.")
    load_csv_to_bigquery(
        data_project_id, dataset_name, "train", train_csv_filepath
    )

    # Load the test data
    print("Loading test table.")
    load_csv_to_bigquery(
        data_project_id, dataset_name, "test", test_csv_filepath
    )


if __name__ == "__main__":
    main()
