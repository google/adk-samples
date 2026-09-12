# Guide: Setting up PostgreSQL with `pgvector` on Google Cloud SQL

This guide provides step-by-step instructions for creating a PostgreSQL instance in Google Cloud SQL and enabling the `pgvector` extension for use with LlamaIndex.

## Prerequisites
- A Google Cloud Platform (GCP) project.
- Billing enabled for the project.
- Cloud SQL Admin API enabled.

---

## Step 1: Create a Cloud SQL Instance

You can create the instance using the Google Cloud Console or the `gcloud` CLI.

### Option A: Using the GCP Console
1.  Go to the **Cloud SQL** page in the Console.
2.  Click **Create Instance**.
3.  Choose **PostgreSQL**.
4.  Enter an **Instance ID** (e.g., `hazmat-vector-db`).
5.  Set a password for the `postgres` user.
6.  Select the **Database Version**: Choose **PostgreSQL 15** or **PostgreSQL 16** (both support `pgvector`).
7.  Select a **Region** close to your application.
8.  Choose **Configuration options**:
    *   **Zonal availability:** Single zone is fine for development.
    *   **Machine configuration:** Start with a small machine (e.g., `db-custom-1-3840` or shared core for dev).
    *   **Storage:** 10GB SSD is usually enough to start.
9.  Click **Create Instance**. It may take a few minutes to initialize.

### Option B: Using the `gcloud` CLI
Run the following command in terminal or Cloud Shell:
```bash
gcloud sql instances create hazmat-vector-db \
    --database-version=POSTGRES_15 \
    --tier=db-custom-1-3840 \
    --region=us-central1 \
    --storage-type=SSD \
    --storage-size=10
```

---

## Step 2: Create a Database and User

Once the instance is running, create a specific database for the project.

1.  In the Cloud SQL instance details page, go to the **Databases** tab.
2.  Click **Create Database** and name it (e.g., `hazmat_db`).
3.  Go to the **Users** tab and create a new user or use the default `postgres` user (set a strong password).

---

## Step 3: Connect and Enable the `pgvector` Extension

Cloud SQL supports `pgvector` without requiring OS-level installation. You just need to enable it via SQL.

### 1. Connect to the Database
You can connect using the **Cloud Shell** directly from the console:
1.  On the instance details page, click **Open Cloud Shell**.
2.  Run the connection command (GCP provides this in the console):
    ```bash
    gcloud sql connect hazmat-vector-db --user=postgres
    ```
3.  Enter the password when prompted.

### 2. Enable the Extension
Once connected to the `postgres` prompt, switch to your database and run the command:
```sql
\c hazmat_db;
CREATE EXTENSION IF NOT EXISTS vector;
```

To verify it is enabled, run:
```sql
\dx
```
You should see `vector` in the list of installed extensions.

---

## Step 4: Connecting from LlamaIndex

When connecting from your Python application, use the connection string pointing to your Cloud SQL instance.

### Local Development / External Access
If running outside of GCP, use the **Cloud SQL Auth Proxy** for safe connection.

Example connection string format for standard use:
```python
connection_string = "postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/hazmat_db"
```
Or using the Cloud SQL Python Connector:
```python
from google.cloud.sql.connector import Connector, IPTypes
import pg8000

# Initialize connector
connector = Connector()

def getconn():
    conn: pg8000.dbapi.Connection = connector.connect(
        "project-id:region:instance-id",
        "pg8000",
        user="postgres",
        password="password",
        db="hazmat_db",
        ip_type=IPTypes.PUBLIC
    )
    return conn
```

### Connecting from Cloud Run (Production)
For production deployment on Cloud Run, use the Cloud SQL integration which mounts the database via a Unix socket. This is more secure as it doesn't require public IP access.

```python
import os

cloudsql_instance = os.environ.get("CLOUDSQL_INSTANCE")
# Format: postgresql://user:pass@/dbname?host=/cloudsql/PROJECT_ID:REGION:INSTANCE_ID
connection_string = f"postgresql://postgres:password@/hazmat_db?host=/cloudsql/{cloudsql_instance}"
```

In this project, we fetch the base connection string from Secret Manager and dynamically update it to use the Unix socket if `CLOUDSQL_INSTANCE` is set (see `app/db.py`).

```

## Summary
You now have a PostgreSQL database on GCP ready to store vectors and metadata for the Hazmat Co-Pilot project!
