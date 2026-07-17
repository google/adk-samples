import json
import os

from dotenv import load_dotenv
from google.cloud import bigquery

# Construct the path to the .env file in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

# Load the environment variables from the .env file
load_dotenv(dotenv_path)


GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")


def clean_sql_query(text):
    return (
        text.replace("\\n", " ")
        .replace("\n", " ")
        .replace("\\", "")
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )


def execute_bigquery_sql(sql: str) -> str:
    """Executes a BigQuery SQL query and returns the result as a JSON string."""
    print(f"Executing BigQuery SQL query: {sql}")
    cleaned_sql = clean_sql_query(sql)
    print(f"Cleaned SQL query: {cleaned_sql}")
    try:
        # The client uses the GOOGLE_CLOUD_PROJECT environment variable.
        client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)
        query_job = client.query(cleaned_sql)  # Make an API request.
        results = query_job.result()  # Wait for the job to complete.

        # Convert RowIterator to a list of dictionaries
        sql_results = [dict(row) for row in results]

        # Return the results as a JSON string.
        if not sql_results:
            return "Query returned no results."
        else:
            # Use json.dumps for proper JSON formatting, handle non-serializable
            # types like datetime
            return (
                json.dumps(sql_results, default=str)
                .replace("```sql", "")
                .replace("```", "")
            )
    except Exception as e:
        return f"Error executing BigQuery query: {e!s}"
