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
from vertexai import rag


def check_bq_models(dataset_id: str) -> str:
    """Lists models in a BigQuery dataset and returns them as a string.

    Args:
        dataset_id: The ID of the BigQuery dataset (e.g., "project.dataset").

    Returns:
        A string representation of a list of dictionaries, where each dictionary
        contains the 'name' and 'type' of a model in the specified dataset.
        Returns an empty string "[]" if no models are found.
    """

    try:
        client = bigquery.Client()

        models = client.list_models(dataset_id)
        model_list = []  # Initialize as a list

        print(f"Models contained in '{dataset_id}':")
        for model in models:
            model_id = model.model_id
            model_type = model.model_type
            model_list.append({"name": model_id, "type": model_type})

        return str(model_list)

    except Exception as e:
        return f"An error occurred: {e!s}"


def rag_response(query: str) -> str:
    """Retrieves contextually relevant information from a RAG corpus.

    Args:
        query (str): The query string to search within the corpus.

    Returns:
        vertexai.rag.RagRetrievalQueryResponse: The response containing retrieved
        information from the corpus.
    """
    corpus_name = os.getenv("BQML_RAG_CORPUS_NAME")

    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=3,  # Optional
        filter=rag.Filter(vector_distance_threshold=0.5),  # Optional
    )
    response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=corpus_name,
            )
        ],
        text=query,
        rag_retrieval_config=rag_retrieval_config,
    )
    return str(response)
