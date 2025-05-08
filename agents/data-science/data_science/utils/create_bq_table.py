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
from google.cloud import bigquery
from pathlib import Path
from dotenv import load_dotenv
from vertexai.language_models import TextEmbeddingModel

# Define the path to the .env file
env_file_path = Path(__file__).parent.parent.parent / ".env"
print(env_file_path)

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=env_file_path)

SCHEMA_EMBEDDING_MODEL_NAME = os.getenv("SCHEMA_EMBEDDING_MODEL_NAME", "text-embedding-004")
BQ_RAG_DATASET_ID = os.getenv("BQ_RAG_DATASET_ID") # New environment variable for RAG dataset


def get_column_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of texts."""
    model = TextEmbeddingModel.from_pretrained(SCHEMA_EMBEDDING_MODEL_NAME)
    embeddings = model.get_embeddings(texts)
    return [embedding.values for embedding in embeddings]


def load_csv_to_bigquery(project_id, dataset_name, table_name, csv_filepath):
    """Loads a CSV file into a BigQuery table.

    Args:
        project_id: The ID of the Google Cloud project.
        dataset_name: The name of the BigQuery dataset.
        table_name: The name of the BigQuery table.
        csv_filepath: The path to the CSV file.
    """

    client = bigquery.Client(project=project_id)

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


def create_dataset_if_not_exists(project_id, dataset_name):
    """Creates a BigQuery dataset if it does not already exist.

    Args:
        project_id: The ID of the Google Cloud project.
        dataset_name: The name of the BigQuery dataset.
    """
    client = bigquery.Client(project=project_id)
    dataset_id = f"{project_id}.{dataset_name}"

    try:
        client.get_dataset(dataset_id)  # Make an API request.
        print(f"Dataset {dataset_id} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"  # Set the location (e.g., "US", "EU")
        dataset = client.create_dataset(dataset, timeout=30)  # Make an API request.
        print(f"Created dataset {dataset_id}")


def create_schema_embeddings_table(project_id, data_dataset_name, rag_dataset_name):
    """Creates and populates the schema_embeddings table in the specified RAG dataset."""
    client = bigquery.Client(project=project_id)
    
    # Use the dedicated RAG dataset for the schema_embeddings table
    rag_dataset_ref = client.dataset(rag_dataset_name)
    schema_embeddings_table_name = "schema_embeddings"
    table_ref = rag_dataset_ref.table(schema_embeddings_table_name)

    # Define the schema for the schema_embeddings table
    schema = [
        bigquery.SchemaField("table_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("column_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("column_description", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("data_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"), # For storing vector embeddings
    ]

    try:
        client.get_table(table_ref)
        print(f"Table {schema_embeddings_table_name} already exists.")
        # Clear the table if it already exists to repopulate
        client.delete_table(table_ref)
        print(f"Table {schema_embeddings_table_name} cleared.")
    except Exception:
        print(f"Table {schema_embeddings_table_name} does not exist. Creating it now.")

    table = bigquery.Table(table_ref, schema=schema)
    client.create_table(table)
    print(f"Created table {project_id}.{rag_dataset_name}.{schema_embeddings_table_name}")

    rows_to_insert = []
    # Iterate through tables in the *data* dataset
    data_dataset_ref = client.dataset(data_dataset_name)
    tables = client.list_tables(data_dataset_ref)

    for table_item in tables:
        # No need to check for schema_embeddings_table_name here if rag_dataset_name is different from data_dataset_name
        # However, if they can be the same, this check is still valid.
        if table_item.table_id == schema_embeddings_table_name and data_dataset_name == rag_dataset_name:
            continue
        current_table_ref = data_dataset_ref.table(table_item.table_id)
        current_table_obj = client.get_table(current_table_ref)
        
        column_texts_for_embedding = []
        column_details = []

        for field in current_table_obj.schema:
            description = field.description if field.description else ""
            # Prepare text for embedding: include table name, column name, and description for better context
            text_for_embedding = f"Table: {table_item.table_id}, Column: {field.name}, Description: {description}, Data Type: {field.field_type}"
            column_texts_for_embedding.append(text_for_embedding)
            column_details.append({
                "table_name": table_item.table_id,
                "column_name": field.name,
                "column_description": description,
                "data_type": field.field_type,
            })

        if column_texts_for_embedding:
            embeddings = get_column_embeddings(column_texts_for_embedding)
            for i, detail in enumerate(column_details):
                rows_to_insert.append({
                    "table_name": detail["table_name"],
                    "column_name": detail["column_name"],
                    "column_description": detail["column_description"],
                    "data_type": detail["data_type"],
                    "embedding": embeddings[i]
                })

    if rows_to_insert:
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if not errors:
            print(f"Inserted {len(rows_to_insert)} rows into {schema_embeddings_table_name}")
        else:
            print(f"Errors encountered while inserting rows: {errors}")


def main():

    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    """Main function to load CSV files into BigQuery."""
    project_id = os.getenv("BQ_PROJECT_ID")
    if not project_id:
        raise ValueError("BQ_PROJECT_ID environment variable not set.")
    if not BQ_RAG_DATASET_ID:
        raise ValueError("BQ_RAG_DATASET_ID environment variable not set. This is required for the schema embeddings table.")

    data_dataset_name = "forecasting_sticker_sales" # Name of the dataset containing the actual data
    train_csv_filepath = "data_science/utils/data/train.csv"
    test_csv_filepath = "data_science/utils/data/test.csv"

    # Create the dataset if it doesn't exist
    print("Creating data dataset.")
    create_dataset_if_not_exists(project_id, data_dataset_name)

    # Create the RAG dataset if it doesn't exist
    print("Creating RAG dataset for schema embeddings.")
    create_dataset_if_not_exists(project_id, BQ_RAG_DATASET_ID)

    # Load the train data into the data dataset
    print("Loading train table.")
    load_csv_to_bigquery(project_id, data_dataset_name, "train", train_csv_filepath)

    # Load the test data into the data dataset
    print("Loading test table.")
    load_csv_to_bigquery(project_id, data_dataset_name, "test", test_csv_filepath)

    # Create and populate the schema_embeddings table in the RAG dataset, using schema from the data dataset
    print("Creating and populating schema_embeddings table.")
    create_schema_embeddings_table(project_id, data_dataset_name, BQ_RAG_DATASET_ID)


if __name__ == "__main__":
    main()
