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

from datetime import datetime

from airflow import DAG
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.gcs_to_bq import (
    GoogleCloudStorageToBigQueryOperator,
)
from airflow.operators.dummy_operator import DummyOperator

default_args = {
    "owner": "migration_test",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
}

dag = DAG(
    "deprecated_dag_2",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
)

start_task = DummyOperator(
    task_id="start",
    dag=dag,
)

load_data = GoogleCloudStorageToBigQueryOperator(
    task_id="gcs_to_bq_load",
    bucket="my-source-bucket",
    source_objects=["data/sales.csv"],
    destination_project_dataset_table="my_project.my_dataset.sales_table",
    source_format="CSV",
    write_disposition="WRITE_TRUNCATE",
    dag=dag,
)

transform_data = BigQueryOperator(
    task_id="bq_transform",
    bql="SELECT * FROM `my_project.my_dataset.sales_table` LIMIT 100",
    use_legacy_sql=False,
    dag=dag,
)

start_task >> load_data >> transform_data
