# Product Search - Evaluation Guide

Comprehensive evaluation patterns for measuring product search agent quality, relevance, and user experience.

---

## Overview

This reference provides templates for Layer 6 (Evaluation) of the product search agent. It covers:
1. Search relevance metrics (NDCG, MRR, Precision, Recall)
2. Response quality evaluation (coherence, groundedness, safety)
3. Multimodal search evaluation
4. RAG faithfulness testing
5. Audio experience evaluation

---

## Architecture

```
Evaluation Pipeline:
1. Create evalsets (ground truth data)
2. Run agent on test queries
3. Calculate metrics
4. Write results to BigQuery
5. Gate deployment on thresholds
```

**Integration with CI/CD**:
- Build gate: NDCG@10 must exceed 0.6
- Failed evaluations block deployment
- Results tracked in BigQuery for trend analysis

---

## Script 1: eval_search.py

**Purpose**: Measure search relevance using standard IR metrics

**Location**: `tests/eval/eval_search.py`

### Template

```python
#!/usr/bin/env python3
"""Evaluate product search relevance using NDCG, MRR, Precision, and Recall.

Usage:
    python eval_search.py \
        --project-id YOUR_PROJECT \
        --evalset tests/eval/evalsets/search_relevance.jsonl \
        --gate-threshold 0.6
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from google.cloud import bigquery
from sklearn.metrics import ndcg_score

# Import your search agent
from app.agent import root_agent
from app.retrievers import search_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchEvaluator:
    """Evaluate product search relevance."""

    def __init__(self, project_id: str, collection_path: str):
        self.project_id = project_id
        self.collection_path = collection_path
        self.bq_client = bigquery.Client(project=project_id)

    def load_evalset(self, evalset_path: str) -> List[Dict]:
        """
        Load evaluation dataset.

        Format (JSONL):
        {
          "query": "wireless headphones under $200",
          "relevant_products": ["prod-123", "prod-456", "prod-789"],
          "relevance_scores": [3, 2, 1],  # 3=highly relevant, 2=relevant, 1=somewhat relevant
          "search_type": "text"
        }
        """
        evalset = []
        with open(evalset_path, 'r') as f:
            for line in f:
                evalset.append(json.loads(line))

        logger.info(f"Loaded {len(evalset)} eval queries from {evalset_path}")
        return evalset

    def run_search(self, query: str, top_k: int = 10) -> List[str]:
        """Run product search and return product IDs."""
        # Use your agent's search functionality
        results = search_collection(
            query=query,
            collection_path=self.collection_path,
            top_k=top_k
        )

        # Extract product IDs from results
        # Adjust based on your result format
        product_ids = self.extract_product_ids(results)

        return product_ids

    def extract_product_ids(self, results: str) -> List[str]:
        """Extract product IDs from search results."""
        # Parse results and extract IDs
        # This depends on your result format
        # Example implementation:
        import re

        product_ids = []
        # Assuming format includes product_id fields
        matches = re.findall(r'"product_id":\s*"([^"]+)"', results)
        return matches[:10]  # Top 10

    def calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        relevance_scores: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            retrieved_ids: List of retrieved product IDs (ranked)
            relevant_ids: List of known relevant product IDs
            relevance_scores: Relevance scores for each relevant product (3, 2, 1)
            k: Cutoff rank

        Returns:
            float: NDCG@K score (0 to 1)
        """
        # Create relevance mapping
        relevance_map = dict(zip(relevant_ids, relevance_scores))

        # Assign relevance scores to retrieved items
        retrieved_scores = [
            relevance_map.get(pid, 0)
            for pid in retrieved_ids[:k]
        ]

        # Pad to length k
        retrieved_scores += [0] * (k - len(retrieved_scores))

        # Calculate ideal ranking
        ideal_scores = sorted(relevance_scores, reverse=True)
        ideal_scores += [0] * (k - len(ideal_scores))

        # sklearn expects 2D arrays
        y_true = np.array([ideal_scores])
        y_score = np.array([retrieved_scores])

        return ndcg_score(y_true, y_score)

    def calculate_mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Returns:
            float: MRR score (0 to 1)
        """
        for rank, pid in enumerate(retrieved_ids, start=1):
            if pid in relevant_ids:
                return 1.0 / rank

        return 0.0

    def calculate_precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 10
    ) -> float:
        """Calculate Precision@K."""
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        if not retrieved_k:
            return 0.0

        return len(retrieved_k & relevant_set) / len(retrieved_k)

    def calculate_recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 10
    ) -> float:
        """Calculate Recall@K."""
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        if not relevant_set:
            return 0.0

        return len(retrieved_k & relevant_set) / len(relevant_set)

    def evaluate_query(self, eval_item: Dict) -> Dict:
        """Evaluate a single query."""
        query = eval_item["query"]
        relevant_ids = eval_item["relevant_products"]
        relevance_scores = eval_item.get("relevance_scores", [1] * len(relevant_ids))

        logger.info(f"Evaluating query: {query}")

        # Run search
        retrieved_ids = self.run_search(query, top_k=10)

        # Calculate metrics
        metrics = {
            "query": query,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_ids),
            "ndcg_at_10": self.calculate_ndcg(
                retrieved_ids, relevant_ids, relevance_scores, k=10
            ),
            "mrr": self.calculate_mrr(retrieved_ids, relevant_ids),
            "precision_at_5": self.calculate_precision_at_k(retrieved_ids, relevant_ids, k=5),
            "precision_at_10": self.calculate_precision_at_k(retrieved_ids, relevant_ids, k=10),
            "recall_at_5": self.calculate_recall_at_k(retrieved_ids, relevant_ids, k=5),
            "recall_at_10": self.calculate_recall_at_k(retrieved_ids, relevant_ids, k=10),
        }

        logger.info(f"NDCG@10: {metrics['ndcg_at_10']:.3f}, MRR: {metrics['mrr']:.3f}")

        return metrics

    def evaluate_all(self, evalset: List[Dict]) -> Dict:
        """Evaluate all queries in evalset."""
        all_metrics = []

        for item in evalset:
            try:
                metrics = self.evaluate_query(item)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating query '{item['query']}': {e}")

        # Calculate aggregate metrics
        aggregate = {
            "total_queries": len(all_metrics),
            "mean_ndcg_at_10": np.mean([m["ndcg_at_10"] for m in all_metrics]),
            "mean_mrr": np.mean([m["mrr"] for m in all_metrics]),
            "mean_precision_at_5": np.mean([m["precision_at_5"] for m in all_metrics]),
            "mean_precision_at_10": np.mean([m["precision_at_10"] for m in all_metrics]),
            "mean_recall_at_5": np.mean([m["recall_at_5"] for m in all_metrics]),
            "mean_recall_at_10": np.mean([m["recall_at_10"] for m in all_metrics]),
            "queries": all_metrics,
        }

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Queries: {aggregate['total_queries']}")
        logger.info(f"Mean NDCG@10: {aggregate['mean_ndcg_at_10']:.3f}")
        logger.info(f"Mean MRR: {aggregate['mean_mrr']:.3f}")
        logger.info(f"Mean Precision@10: {aggregate['mean_precision_at_10']:.3f}")
        logger.info(f"Mean Recall@10: {aggregate['mean_recall_at_10']:.3f}")
        logger.info("=" * 60)

        return aggregate

    def write_to_bigquery(self, results: Dict):
        """Write evaluation results to BigQuery."""
        table_id = f"{self.project_id}.eval_results.search_relevance"

        row = {
            "eval_timestamp": np.datetime64('now').astype(str),
            "total_queries": results["total_queries"],
            "mean_ndcg_at_10": results["mean_ndcg_at_10"],
            "mean_mrr": results["mean_mrr"],
            "mean_precision_at_10": results["mean_precision_at_10"],
            "mean_recall_at_10": results["mean_recall_at_10"],
            "query_results": json.dumps(results["queries"]),
        }

        errors = self.bq_client.insert_rows_json(table_id, [row])

        if errors:
            logger.error(f"Failed to write to BigQuery: {errors}")
        else:
            logger.info(f"Results written to {table_id}")

    def check_gate(self, results: Dict, threshold: float = 0.6) -> bool:
        """Check if NDCG@10 meets deployment gate threshold."""
        ndcg = results["mean_ndcg_at_10"]

        if ndcg >= threshold:
            logger.info(f" PASSED: NDCG@10 ({ndcg:.3f}) >= {threshold}")
            return True
        else:
            logger.error(f" FAILED: NDCG@10 ({ndcg:.3f}) < {threshold}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate search relevance")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--collection-path", required=True, help="Vector Search collection path")
    parser.add_argument("--evalset", required=True, help="Path to evalset JSONL file")
    parser.add_argument("--gate-threshold", type=float, default=0.6, help="NDCG@10 threshold for gate")
    parser.add_argument("--write-to-bq", action="store_true", help="Write results to BigQuery")

    args = parser.parse_args()

    evaluator = SearchEvaluator(args.project_id, args.collection_path)

    # Load evalset
    evalset = evaluator.load_evalset(args.evalset)

    # Run evaluation
    results = evaluator.evaluate_all(evalset)

    # Write to BigQuery
    if args.write_to_bq:
        evaluator.write_to_bigquery(results)

    # Check gate
    passed = evaluator.check_gate(results, args.gate_threshold)

    # Exit with appropriate code for CI/CD
    exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

---

## Script 2: eval_quality.py

**Purpose**: Evaluate response quality using Vertex AI Rapid Eval

**Location**: `tests/eval/eval_quality.py`

### Template

```python
#!/usr/bin/env python3
"""Evaluate agent response quality using Vertex AI Rapid Eval.

Usage:
    python eval_quality.py \
        --project-id YOUR_PROJECT \
        --evalset tests/eval/evalsets/quality.jsonl
"""

import argparse
import json
import logging
from typing import List, Dict

from google.cloud import aiplatform
from vertexai.preview.evaluation import EvalTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityEvaluator:
    """Evaluate agent response quality."""

    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)

    def load_evalset(self, evalset_path: str) -> List[Dict]:
        """
        Load quality evaluation dataset.

        Format (JSONL):
        {
          "query": "Show me wireless headphones",
          "reference_response": "Here are wireless headphones...",
          "context": "Product catalog contains 50 wireless headphones...",
          "instruction": "Provide natural, helpful product recommendations"
        }
        """
        evalset = []
        with open(evalset_path, 'r') as f:
            for line in f:
                evalset.append(json.loads(line))

        logger.info(f"Loaded {len(evalset)} quality eval items")
        return evalset

    def run_agent(self, query: str) -> str:
        """Run agent and get response."""
        from app.agent import root_agent

        response = root_agent.run(query)
        return str(response)

    def evaluate_coherence(self, evalset: List[Dict]) -> Dict:
        """
        Evaluate response coherence.

        Measures: Logical flow, clarity, relevance
        """
        # Prepare data for Vertex AI Rapid Eval
        predictions = []
        references = []

        for item in evalset:
            query = item["query"]
            logger.info(f"Generating response for: {query}")

            response = self.run_agent(query)

            predictions.append(response)
            references.append(item.get("reference_response", ""))

        # Run Vertex AI Rapid Eval
        eval_task = EvalTask(
            dataset=evalset,
            metrics=["coherence", "fluency"],
            experiment="product-search-quality-eval"
        )

        results = eval_task.evaluate(
            model=None,  # Using custom responses
            predictions=predictions
        )

        logger.info(f"Coherence Score: {results.summary_metrics.get('coherence', 'N/A')}")

        return results.summary_metrics

    def evaluate_groundedness(self, evalset: List[Dict]) -> Dict:
        """
        Evaluate response groundedness.

        Measures: Whether response is based on retrieved context
        """
        eval_task = EvalTask(
            dataset=evalset,
            metrics=["groundedness"],
            experiment="product-search-groundedness"
        )

        predictions = []
        contexts = []

        for item in evalset:
            query = item["query"]
            response = self.run_agent(query)

            predictions.append(response)
            contexts.append(item.get("context", ""))

        results = eval_task.evaluate(
            model=None,
            predictions=predictions,
            contexts=contexts
        )

        logger.info(f"Groundedness Score: {results.summary_metrics.get('groundedness', 'N/A')}")

        return results.summary_metrics

    def evaluate_safety(self, evalset: List[Dict]) -> Dict:
        """
        Evaluate response safety.

        Measures: No harmful, biased, or inappropriate content
        """
        eval_task = EvalTask(
            dataset=evalset,
            metrics=["safety"],
            experiment="product-search-safety"
        )

        predictions = []

        for item in evalset:
            query = item["query"]
            response = self.run_agent(query)
            predictions.append(response)

        results = eval_task.evaluate(
            model=None,
            predictions=predictions
        )

        logger.info(f"Safety Score: {results.summary_metrics.get('safety', 'N/A')}")

        return results.summary_metrics

    def evaluate_all(self, evalset: List[Dict]) -> Dict:
        """Run all quality evaluations."""
        logger.info("Starting quality evaluation...")

        results = {
            "coherence": self.evaluate_coherence(evalset),
            "groundedness": self.evaluate_groundedness(evalset),
            "safety": self.evaluate_safety(evalset),
        }

        logger.info("\n" + "=" * 60)
        logger.info("QUALITY EVALUATION RESULTS")
        logger.info("=" * 60)
        for metric, scores in results.items():
            logger.info(f"{metric.title()}: {scores}")
        logger.info("=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate response quality")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--evalset", required=True)

    args = parser.parse_args()

    evaluator = QualityEvaluator(args.project_id, args.location)
    evalset = evaluator.load_evalset(args.evalset)
    results = evaluator.evaluate_all(evalset)


if __name__ == "__main__":
    main()
```

---

## Script 3: eval_multimodal.py

**Purpose**: Evaluate image search and multimodal capabilities

**Location**: `tests/eval/eval_multimodal.py`

### Template

```python
#!/usr/bin/env python3
"""Evaluate multimodal (image) search accuracy.

Usage:
    python eval_multimodal.py \
        --project-id YOUR_PROJECT \
        --evalset tests/eval/evalsets/image_search.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

from app.retrievers import search_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalEvaluator:
    """Evaluate image search capabilities."""

    def __init__(self, project_id: str, collection_path: str):
        self.project_id = project_id
        self.collection_path = collection_path

    def load_evalset(self, evalset_path: str) -> List[Dict]:
        """
        Load image search evalset.

        Format (JSONL):
        {
          "image_url": "gs://bucket/test-images/headphones.jpg",
          "expected_category": "audio",
          "expected_products": ["prod-123", "prod-456"],
          "description": "Black wireless headphones on white background"
        }
        """
        evalset = []
        with open(evalset_path, 'r') as f:
            for line in f:
                evalset.append(json.loads(line))

        logger.info(f"Loaded {len(evalset)} image search tests")
        return evalset

    def search_by_image(self, image_url: str, top_k: int = 10) -> List[Dict]:
        """Run image-based search."""
        # Image search using your retriever
        results = search_collection(
            query=image_url,  # Pass image URL for multimodal search
            collection_path=self.collection_path,
            top_k=top_k
        )

        return self.parse_results(results)

    def parse_results(self, results: str) -> List[Dict]:
        """Parse search results into structured format."""
        # Parse based on your result format
        import re

        products = []
        # Extract product info
        matches = re.findall(r'"product_id":\s*"([^"]+)".*?"category":\s*"([^"]+)"', results)

        for product_id, category in matches:
            products.append({"product_id": product_id, "category": category})

        return products

    def calculate_category_precision(
        self,
        retrieved_products: List[Dict],
        expected_category: str,
        k: int = 10
    ) -> float:
        """Calculate what % of top-K results are from expected category."""
        if not retrieved_products:
            return 0.0

        top_k = retrieved_products[:k]
        correct_category = sum(
            1 for p in top_k
            if p.get("category", "").lower() == expected_category.lower()
        )

        return correct_category / len(top_k)

    def calculate_product_recall(
        self,
        retrieved_products: List[Dict],
        expected_products: List[str],
        k: int = 10
    ) -> float:
        """Calculate recall of expected products in top-K."""
        if not expected_products:
            return 1.0  # No expectation

        retrieved_ids = set(p["product_id"] for p in retrieved_products[:k])
        expected_set = set(expected_products)

        return len(retrieved_ids & expected_set) / len(expected_set)

    def evaluate_image_query(self, eval_item: Dict) -> Dict:
        """Evaluate a single image search query."""
        image_url = eval_item["image_url"]
        expected_category = eval_item["expected_category"]
        expected_products = eval_item.get("expected_products", [])

        logger.info(f"Evaluating image: {image_url}")

        # Run image search
        retrieved_products = self.search_by_image(image_url, top_k=10)

        # Calculate metrics
        metrics = {
            "image_url": image_url,
            "expected_category": expected_category,
            "retrieved_count": len(retrieved_products),
            "category_precision_at_10": self.calculate_category_precision(
                retrieved_products, expected_category, k=10
            ),
            "product_recall_at_10": self.calculate_product_recall(
                retrieved_products, expected_products, k=10
            ),
        }

        logger.info(
            f"Category Precision@10: {metrics['category_precision_at_10']:.3f}, "
            f"Product Recall@10: {metrics['product_recall_at_10']:.3f}"
        )

        return metrics

    def evaluate_all(self, evalset: List[Dict]) -> Dict:
        """Evaluate all image queries."""
        all_metrics = []

        for item in evalset:
            try:
                metrics = self.evaluate_image_query(item)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating image '{item['image_url']}': {e}")

        # Aggregate
        aggregate = {
            "total_images": len(all_metrics),
            "mean_category_precision_at_10": sum(m["category_precision_at_10"] for m in all_metrics) / len(all_metrics),
            "mean_product_recall_at_10": sum(m["product_recall_at_10"] for m in all_metrics) / len(all_metrics),
            "images": all_metrics,
        }

        logger.info("\n" + "=" * 60)
        logger.info("MULTIMODAL EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Images: {aggregate['total_images']}")
        logger.info(f"Mean Category Precision@10: {aggregate['mean_category_precision_at_10']:.3f}")
        logger.info(f"Mean Product Recall@10: {aggregate['mean_product_recall_at_10']:.3f}")
        logger.info("=" * 60)

        return aggregate


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal search")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--collection-path", required=True)
    parser.add_argument("--evalset", required=True)

    args = parser.parse_args()

    evaluator = MultimodalEvaluator(args.project_id, args.collection_path)
    evalset = evaluator.load_evalset(args.evalset)
    results = evaluator.evaluate_all(evalset)


if __name__ == "__main__":
    main()
```

---

## Sample Evalsets

### Search Relevance Evalset

**File**: `tests/eval/evalsets/search_relevance.jsonl`

```jsonl
{"query": "wireless headphones under $200", "relevant_products": ["prod-101", "prod-102", "prod-103"], "relevance_scores": [3, 2, 2]}
{"query": "laptop for programming", "relevant_products": ["prod-201", "prod-202"], "relevance_scores": [3, 2]}
{"query": "running shoes size 10", "relevant_products": ["prod-301", "prod-302", "prod-303", "prod-304"], "relevance_scores": [3, 3, 2, 1]}
{"query": "waterproof bluetooth speaker", "relevant_products": ["prod-401", "prod-402"], "relevance_scores": [3, 2]}
{"query": "affordable gaming mouse", "relevant_products": ["prod-501", "prod-502", "prod-503"], "relevance_scores": [3, 2, 2]}
```

### Quality Evalset

**File**: `tests/eval/evalsets/quality.jsonl`

```jsonl
{"query": "Show me laptops for video editing", "reference_response": "Here are high-performance laptops suitable for video editing...", "instruction": "Provide helpful, accurate product recommendations"}
{"query": "What's the best phone under $500?", "reference_response": "Based on reviews and specs, here are the top-rated phones under $500...", "instruction": "Give objective, well-reasoned recommendations"}
{"query": "I need workout clothes", "reference_response": "What type of workout are you doing? Running, gym, yoga?", "instruction": "Ask clarifying questions when query is vague"}
```

### Image Search Evalset

**File**: `tests/eval/evalsets/image_search.jsonl`

```jsonl
{"image_url": "gs://your-bucket/eval-images/headphones-black.jpg", "expected_category": "audio", "expected_products": ["prod-101", "prod-102"]}
{"image_url": "gs://your-bucket/eval-images/laptop-silver.jpg", "expected_category": "electronics", "expected_products": ["prod-201"]}
{"image_url": "gs://your-bucket/eval-images/shoes-running.jpg", "expected_category": "footwear", "expected_products": ["prod-301", "prod-302"]}
```

---

## CI/CD Integration

### Cloud Build Gate

**File**: `deployment/cloudbuild.yaml`

```yaml
steps:
  # ... previous steps (install, build, deploy)

  # Run search evaluation
  - name: 'python:3.10'
    id: 'eval-search'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r requirements.txt
        python tests/eval/eval_search.py \
          --project-id $PROJECT_ID \
          --collection-path $COLLECTION_PATH \
          --evalset tests/eval/evalsets/search_relevance.jsonl \
          --gate-threshold 0.6 \
          --write-to-bq
    env:
      - 'PROJECT_ID=${PROJECT_ID}'
      - 'COLLECTION_PATH=projects/${PROJECT_ID}/locations/us-central1/collections/product-search-collection'

  # If eval-search fails (exit code 1), build stops here
  # Only proceed to deployment if NDCG@10 >= 0.6

  # Deploy to production
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'deploy-prod'
    args:
      - 'run'
      - 'deploy'
      - 'product-search-agent'
      - '--image=gcr.io/${PROJECT_ID}/product-search:${SHORT_SHA}'
      - '--region=us-central1'
```

---

## BigQuery Schema for Results

```sql
CREATE TABLE `your-project.eval_results.search_relevance` (
  eval_timestamp TIMESTAMP,
  total_queries INT64,
  mean_ndcg_at_10 FLOAT64,
  mean_mrr FLOAT64,
  mean_precision_at_10 FLOAT64,
  mean_recall_at_10 FLOAT64,
  query_results JSON
);

CREATE TABLE `your-project.eval_results.quality` (
  eval_timestamp TIMESTAMP,
  coherence_score FLOAT64,
  groundedness_score FLOAT64,
  safety_score FLOAT64,
  fluency_score FLOAT64,
  detailed_results JSON
);
```

---

## Running Evaluations

### Local Testing

```bash
# Search relevance
python tests/eval/eval_search.py \
  --project-id your-project-id \
  --collection-path projects/your-project-id/locations/us-central1/collections/product-search \
  --evalset tests/eval/evalsets/search_relevance.jsonl \
  --gate-threshold 0.6

# Quality
python tests/eval/eval_quality.py \
  --project-id your-project-id \
  --evalset tests/eval/evalsets/quality.jsonl

# Multimodal
python tests/eval/eval_multimodal.py \
  --project-id your-project-id \
  --collection-path projects/your-project-id/locations/us-central1/collections/product-search \
  --evalset tests/eval/evalsets/image_search.jsonl
```

### Makefile Integration

```makefile
# Add to Makefile
eval-search:
	python tests/eval/eval_search.py \
		--project-id $(PROJECT_ID) \
		--collection-path $(COLLECTION_PATH) \
		--evalset tests/eval/evalsets/search_relevance.jsonl \
		--gate-threshold 0.6 \
		--write-to-bq

eval-quality:
	python tests/eval/eval_quality.py \
		--project-id $(PROJECT_ID) \
		--evalset tests/eval/evalsets/quality.jsonl

eval-all: eval-search eval-quality
	@echo "All evaluations complete"
```

---

## Best Practices

1. **Ground Truth Quality**: Manually label at least 50-100 queries for reliable metrics
2. **Continuous Evaluation**: Run evals on every code change via CI/CD
3. **Metric Thresholds**: Set realistic gates (NDCG@10 >= 0.6 is good for product search)
4. **Trend Tracking**: Log results to BigQuery, monitor trends over time
5. **Diverse Queries**: Include vague, specific, multimodal, and edge case queries
6. **Regular Updates**: Refresh evalsets quarterly as catalog changes

---

## Troubleshooting

### Low NDCG Scores

- Check if relevant products are actually in the catalog
- Verify embeddings are properly generated
- Inspect query preprocessing (stopwords, normalization)
- Review top-K results manually

### Evaluation Timeouts

- Reduce evalset size for testing
- Run in parallel (split evalset)
- Increase Cloud Build timeout

### BigQuery Insertion Failures

```python
# Check table exists
bq_client.get_table(table_id)

# Verify schema matches
# Check for JSON serialization issues
```

---

## References

- [NDCG Documentation](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [Vertex AI Rapid Eval](https://cloud.google.com/vertex-ai/docs/generative-ai/models/evaluate-models)
- [MRR and Precision@K](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
