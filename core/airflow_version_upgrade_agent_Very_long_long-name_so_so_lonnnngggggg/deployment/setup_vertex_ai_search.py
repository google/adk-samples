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
import os
import subprocess
import time

import requests


def get_access_token():
    try:
        token = (
            subprocess.check_output(
                ["gcloud", "auth", "application-default", "print-access-token"]
            )
            .decode("utf-8")
            .strip()
        )
        return token
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None


def get_project_id():
    project_id = os.environ.get("PROJECT_ID")
    if not project_id:
        try:
            project_id = (
                subprocess.check_output(
                    ["gcloud", "config", "get-value", "project"]
                )
                .decode("utf-8")
                .strip()
            )
        except Exception as e:
            print(f"Error getting project ID from gcloud: {e}")
            return None
    return project_id


def create_data_store(
    project_id,
    location,
    collection,
    data_store_id,
    display_name,
    content_config,
):
    token = get_access_token()
    if not token:
        return False

    url = f"https://discoveryengine.googleapis.com/v1/projects/{project_id}/locations/{location}/collections/{collection}/dataStores?dataStoreId={data_store_id}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-goog-user-project": project_id,
    }

    payload = {
        "displayName": display_name,
        "industryVertical": "GENERIC",
        "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
        "contentConfig": content_config,
    }

    print(f"Creating data store {data_store_id}...")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print(f"Data store {data_store_id} created successfully.")
        return True
    elif response.status_code == 409:
        print(f"Data store {data_store_id} already exists.")
        return True
    else:
        print(
            f"Error creating data store: {json.dumps(response.json(), indent=2)}"
        )
        return False


def add_target_site(
    project_id, location, collection, data_store_id, uri_pattern
):
    token = get_access_token()
    if not token:
        return False

    url = f"https://discoveryengine.googleapis.com/v1/projects/{project_id}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/siteSearchEngine/targetSites"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-goog-user-project": project_id,
    }

    payload = {"providedUriPattern": uri_pattern, "type": "INCLUDE"}

    print(f"Adding target site {uri_pattern} to {data_store_id}...")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code in [200, 409]:
        print(f"Target site {uri_pattern} added/exists successfully.")
        return True
    else:
        print(
            f"Error adding target site: {json.dumps(response.json(), indent=2)}"
        )
        return False


def create_app(
    project_id, location, collection, app_id, display_name, data_store_ids
):
    token = get_access_token()
    if not token:
        return False

    url = f"https://discoveryengine.googleapis.com/v1/projects/{project_id}/locations/{location}/collections/{collection}/engines?engineId={app_id}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-goog-user-project": project_id,
    }

    payload = {
        "displayName": display_name,
        "solutionType": "SOLUTION_TYPE_SEARCH",
        "industryVertical": "GENERIC",
        "dataStoreIds": data_store_ids,
        "searchEngineConfig": {
            "searchTier": "SEARCH_TIER_ENTERPRISE",
            "searchAddOns": ["SEARCH_ADD_ON_LLM"],
        },
    }

    print(f"Creating app {app_id}...")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print(f"App {app_id} created successfully.")
        return True
    elif response.status_code == 409:
        print(f"App {app_id} already exists.")
        return True
    else:
        print(f"Error creating app: {json.dumps(response.json(), indent=2)}")
        return False


def main():
    project_id = get_project_id()
    if not project_id:
        print(
            "Could not determine Project ID. Please set PROJECT_ID environment variable."
        )
        return

    print(f"Using Project ID: {project_id}")

    location = "global"
    collection = "default_collection"

    web_ds_id = os.environ.get("VERTEX_AI_WEB_DATASTORE_ID", "test-web-ds")
    bq_ds_id = os.environ.get("VERTEX_AI_BQ_DATASTORE_ID", "test-bq-ds")
    app_id = os.environ.get("VERTEX_AI_SEARCH_APP_ID", "test-migration-app")

    # 1. Create Website Data Store
    if create_data_store(
        project_id,
        location,
        collection,
        web_ds_id,
        "Airflow Web DS",
        "PUBLIC_WEBSITE",
    ):
        # Give the backend a brief moment to initialize the datastore
        time.sleep(2)
        urls = [
            "airflow.apache.org/docs/*",
            "stackoverflow.com/questions/tagged/airflow/*",
            "medium.com/*",
        ]
        for url in urls:
            add_target_site(project_id, location, collection, web_ds_id, url)

    # 2. Create Standalone BigQuery Data Store
    # Uses "NO_CONTENT" because it is a Structured Data store.
    # No import is triggered here.
    create_data_store(
        project_id,
        location,
        collection,
        bq_ds_id,
        "Airflow BQ DS",
        "NO_CONTENT",
    )

    # 3. Create App and link ONLY to the Web Data Store
    # Wait 5 seconds to ensure Vertex AI backend has fully registered the Data Stores
    time.sleep(5)
    create_app(
        project_id,
        location,
        collection,
        app_id,
        "Airflow Migration App",
        [web_ds_id],
    )


if __name__ == "__main__":
    main()
