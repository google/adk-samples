# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import re
import warnings
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def search_asset_bank(query: str) -> dict | None:
    # Helper to clean text and split CamelCase (e.g., 'TrueBlue' -> 'true blue')
    def preprocess(text: str) -> str:
        # Insert space before capital letters if they are preceded by a lowercase letter
        text = re.sub(r"(?<!^)(?=[A-Z])", " ", text)
        return text.lower()

    # Prepare document strings from the JSON data
    # Assets can be stored in GCS or any other storage system like Digital Assets Management (DAM)
    # For now, we are using a local JSON file for demonstration purposes
    dataset_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "brand_assets_metadata.json"
    )

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    documents = []
    for item in dataset:
        # Create a rich text representation of the item
        doc_text = f"{item['name']} {item['description']} {item['primary_subject_color_name']} {item['primary_subject_type']}"
        documents.append(preprocess(doc_text))

    # Preprocess the query
    processed_query = preprocess(query)

    # Use TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit on documents and transform both documents and query
    tfidf_matrix = vectorizer.fit_transform([*documents, processed_query])

    # Calculate Cosine Similarity between the query (last vector) and all documents
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    if cosine_sim.size == 0 or cosine_sim.max() <= 0.0:
        logger.info(f"No relevant asset match found for query: {query}")
        return None

    # Find the index of the highest score
    best_match_index = int(cosine_sim.argmax())

    logger.info(f"Best match found: {dataset[best_match_index]}")

    return dataset[best_match_index]
