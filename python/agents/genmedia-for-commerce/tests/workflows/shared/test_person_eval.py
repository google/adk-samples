# Copyright 2025 Google LLC
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

"""Tests for shared/person_eval.py - Face comparison with mocked InsightFace."""

import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def face_image_bytes():
    """Create a sample face image as bytes (simple RGB image)."""
    img = Image.new("RGB", (200, 200), color=(255, 200, 180))  # Skin-tone color
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def different_face_bytes():
    """Create a different face image."""
    img = Image.new("RGB", (200, 200), color=(100, 80, 60))  # Different tone
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def full_body_image_bytes():
    """Create a full body image (larger, face would be in upper portion)."""
    img = Image.new("RGB", (400, 600), color=(200, 200, 200))
    # Add a "face" region at top
    for y in range(50, 150):
        for x in range(150, 250):
            img.putpixel((x, y), (255, 200, 180))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestCompareFaces:
    """Tests for compare_faces_insightface function."""

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_compares_identical_faces(self, mock_get_embedding, face_image_bytes):
        """Should return high similarity for identical faces."""
        # Mock identical embeddings (cosine similarity = 1.0)
        embedding = np.ones(512) / np.sqrt(512)  # Normalized
        mock_get_embedding.return_value = embedding

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)

        assert result["distance"] == pytest.approx(0.0, abs=0.01)
        assert result["similarity_percentage"] == pytest.approx(100.0, abs=0.5)
        assert result["model"] == "InsightFace-ArcFace"
        assert result["embeddings_extracted"]

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_compares_different_faces(
        self, mock_get_embedding, face_image_bytes, different_face_bytes
    ):
        """Should return lower similarity for different faces."""
        # Mock different embeddings (cosine similarity = 0.2)
        embedding1 = np.ones(512) / np.sqrt(512)
        embedding2 = np.zeros(512)
        embedding2[:100] = 1.0 / np.sqrt(100)  # Different direction

        mock_get_embedding.side_effect = [embedding1, embedding2]

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, different_face_bytes)

        # Similarity will be low but above 0
        assert result["distance"] >= 0.0
        assert result["distance"] <= 2.0
        assert result["similarity_percentage"] >= 0.0
        assert result["similarity_percentage"] <= 100.0
        assert result["model"] == "InsightFace-ArcFace"

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_uses_correct_model(self, mock_get_embedding, face_image_bytes):
        """Should use InsightFace-ArcFace model."""
        embedding = np.ones(512) / np.sqrt(512)
        mock_get_embedding.return_value = embedding

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)

        assert result["model"] == "InsightFace-ArcFace"

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_uses_arcface_by_default(self, mock_get_embedding, face_image_bytes):
        """Should use InsightFace-ArcFace model by default."""
        embedding = np.ones(512) / np.sqrt(512)
        mock_get_embedding.return_value = embedding

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)

        assert result["model"] == "InsightFace-ArcFace"

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_skips_face_detection(self, mock_get_embedding, face_image_bytes):
        """Should extract embeddings from images (face detection handled by InsightFace)."""
        embedding = np.ones(512) / np.sqrt(512)
        mock_get_embedding.return_value = embedding

        from workflows.shared.person_eval import compare_faces_insightface

        compare_faces_insightface(face_image_bytes, face_image_bytes)

        # Should call embedding extraction twice (once per image)
        assert mock_get_embedding.call_count == 2

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_similarity_bounds(self, mock_get_embedding, face_image_bytes):
        """Should clamp similarity percentage to 0-100 range."""
        # Mock very high similarity (> 1.0)
        embedding = np.ones(512) / np.sqrt(512)
        mock_get_embedding.return_value = embedding

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)

        # Should be clamped to 100
        assert result["similarity_percentage"] <= 100.0
        assert result["similarity_percentage"] >= 0.0

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_similarity_lower_bound(self, mock_get_embedding, face_image_bytes):
        """Should clamp similarity to 0 for very different faces."""
        # Mock orthogonal embeddings (cosine similarity = 0)
        embedding1 = np.zeros(512)
        embedding1[:256] = 1.0 / np.sqrt(256)
        embedding2 = np.zeros(512)
        embedding2[256:] = 1.0 / np.sqrt(256)

        mock_get_embedding.side_effect = [embedding1, embedding2]

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)

        # Similarity should be near 0
        assert result["similarity_percentage"] >= 0.0

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_raises_on_error(self, mock_get_embedding, face_image_bytes):
        """Should handle no face detected (returns None)."""
        mock_get_embedding.return_value = None

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)

        # Should return error result
        assert result["similarity_percentage"] == 0.0
        assert result["distance"] == 2.0
        assert not result["embeddings_extracted"]


class TestEvaluatePersonMatch:
    """Tests for evaluate_person_match function."""

    @patch("workflows.shared.person_eval.crop_face")
    @patch("workflows.shared.person_eval.compare_faces_insightface")
    def test_evaluates_matching_person(
        self, mock_compare, mock_crop, face_image_bytes, full_body_image_bytes
    ):
        """Should evaluate matching person successfully."""
        mock_crop.return_value = face_image_bytes  # Cropped face
        mock_compare.return_value = {
            "distance": 0.2,
            "model": "InsightFace-ArcFace",
            "similarity_percentage": 80.0,
            "embeddings_extracted": True,
        }

        from workflows.shared.person_eval import evaluate_person_match

        result = evaluate_person_match(face_image_bytes, full_body_image_bytes)

        assert result["face_detected"]
        assert result["similarity_percentage"] == 80.0
        assert result["distance"] == 0.2
        mock_crop.assert_called_once()

    @patch("workflows.shared.person_eval.crop_face")
    def test_handles_no_face_detected(
        self, mock_crop, face_image_bytes, full_body_image_bytes
    ):
        """Should handle case when no face is detected in generated image."""
        mock_crop.return_value = None  # No face detected

        from workflows.shared.person_eval import evaluate_person_match

        result = evaluate_person_match(face_image_bytes, full_body_image_bytes)

        assert not result["face_detected"]
        assert result["similarity_percentage"] == 0.0
        assert result["distance"] == 2.0

    @patch("workflows.shared.person_eval.crop_face")
    @patch("workflows.shared.person_eval.compare_faces_insightface")
    def test_uses_correct_model(
        self, mock_compare, mock_crop, face_image_bytes, full_body_image_bytes
    ):
        """Should use InsightFace-ArcFace model."""
        mock_crop.return_value = face_image_bytes
        mock_compare.return_value = {
            "distance": 0.3,
            "model": "InsightFace-ArcFace",
            "similarity_percentage": 70.0,
            "embeddings_extracted": True,
        }

        from workflows.shared.person_eval import evaluate_person_match

        result = evaluate_person_match(face_image_bytes, full_body_image_bytes)

        mock_compare.assert_called_once()
        assert result["model"] == "InsightFace-ArcFace"

    @patch("workflows.shared.person_eval.crop_face")
    @patch("workflows.shared.person_eval.compare_faces_insightface")
    def test_handles_comparison_error(
        self, mock_compare, mock_crop, face_image_bytes, full_body_image_bytes
    ):
        """Should handle errors during face comparison."""
        mock_crop.return_value = face_image_bytes
        mock_compare.side_effect = Exception("Comparison failed")

        from workflows.shared.person_eval import evaluate_person_match

        result = evaluate_person_match(face_image_bytes, full_body_image_bytes)

        assert not result["face_detected"]
        assert result["similarity_percentage"] == 0.0
        assert result["distance"] == 2.0
        assert "error" in result

    @patch("workflows.shared.person_eval.crop_face")
    @patch("workflows.shared.person_eval.compare_faces_insightface")
    def test_returns_correct_result_structure(
        self, mock_compare, mock_crop, face_image_bytes, full_body_image_bytes
    ):
        """Should return result with all required fields."""
        mock_crop.return_value = face_image_bytes
        mock_compare.return_value = {
            "distance": 0.15,
            "model": "InsightFace-ArcFace",
            "similarity_percentage": 85.0,
            "embeddings_extracted": True,
        }

        from workflows.shared.person_eval import evaluate_person_match

        result = evaluate_person_match(face_image_bytes, full_body_image_bytes)

        assert "similarity_percentage" in result
        assert "distance" in result
        assert "model" in result
        assert "face_detected" in result


class TestSubmitEvaluation:
    """Tests for submit_evaluation function."""

    @patch("workflows.shared.person_eval.get_eval_pool")
    def test_submits_to_pool(
        self, mock_get_pool, face_image_bytes, full_body_image_bytes
    ):
        """Should submit evaluation to thread pool."""
        mock_pool = Mock()
        mock_future = Mock()
        mock_pool.submit.return_value = mock_future
        mock_get_pool.return_value = mock_pool

        from workflows.shared.person_eval import submit_evaluation

        result = submit_evaluation(face_image_bytes, full_body_image_bytes)

        assert result == mock_future
        mock_pool.submit.assert_called_once()

    @patch("workflows.shared.person_eval.get_eval_pool")
    def test_passes_model_name(
        self, mock_get_pool, face_image_bytes, full_body_image_bytes
    ):
        """Should pass arguments to evaluation function."""
        mock_pool = Mock()
        mock_get_pool.return_value = mock_pool

        from workflows.shared.person_eval import (
            evaluate_person_match,
            submit_evaluation,
        )

        submit_evaluation(face_image_bytes, full_body_image_bytes)

        # Should submit evaluate_person_match with the two image arguments
        call_args = mock_pool.submit.call_args[0]
        assert call_args[0] == evaluate_person_match
        assert call_args[1] == face_image_bytes
        assert call_args[2] == full_body_image_bytes


class TestGetEvalPool:
    """Tests for get_eval_pool function."""

    @patch("workflows.shared.person_eval._eval_pool", None)
    @patch("workflows.shared.person_eval.ThreadPoolExecutor")
    def test_creates_pool_on_first_call(self, mock_executor_class):
        """Should create thread pool on first call."""
        mock_pool = Mock()
        mock_executor_class.return_value = mock_pool

        from workflows.shared.person_eval import get_eval_pool

        result = get_eval_pool()

        mock_executor_class.assert_called_once()
        assert result == mock_pool

    @patch("workflows.shared.person_eval._eval_pool", None)
    @patch("workflows.shared.person_eval.ThreadPoolExecutor")
    def test_pool_max_workers(self, mock_executor_class):
        """Should create pool with correct max workers."""
        mock_executor_class.return_value = Mock()

        from workflows.shared.person_eval import _POOL_MAX_WORKERS, get_eval_pool

        get_eval_pool()

        call_kwargs = mock_executor_class.call_args.kwargs
        assert call_kwargs["max_workers"] == _POOL_MAX_WORKERS


class TestSimilarityCalculation:
    """Tests for similarity percentage calculation logic."""

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_perfect_match(self, mock_get_embedding, face_image_bytes):
        """Distance 0 should give 100% similarity."""
        embedding = np.ones(512) / np.sqrt(512)
        mock_get_embedding.return_value = embedding

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)
        assert result["similarity_percentage"] == pytest.approx(100.0, abs=0.5)

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_half_similar(self, mock_get_embedding, face_image_bytes):
        """Cosine similarity 0.5 should give 50% similarity."""
        # Create two normalized embeddings with cosine similarity of 0.5
        # Cosine similarity = dot(a, b) when both are normalized
        # To get 0.5, we can use embeddings at 60 degrees apart: cos(60°) = 0.5
        embedding1 = np.zeros(512)
        embedding1[0] = 1.0  # Unit vector along first axis

        embedding2 = np.zeros(512)
        embedding2[0] = 0.5  # cos(60°)
        embedding2[1] = np.sqrt(3) / 2  # sin(60°)

        mock_get_embedding.side_effect = [embedding1, embedding2]

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)
        # Cosine similarity of 0.5 should give 50% similarity
        assert result["similarity_percentage"] >= 49.0
        assert result["similarity_percentage"] <= 51.0

    @patch("workflows.shared.person_eval.get_face_embedding_insightface")
    def test_threshold_boundary(self, mock_get_embedding, face_image_bytes):
        """Should handle threshold boundary cases."""
        # Create embeddings with moderate similarity
        embedding1 = np.ones(512) / np.sqrt(512)
        embedding2 = np.ones(512) * 0.3 / np.linalg.norm(np.ones(512) * 0.3)

        mock_get_embedding.side_effect = [embedding1, embedding2]

        from workflows.shared.person_eval import compare_faces_insightface

        result = compare_faces_insightface(face_image_bytes, face_image_bytes)
        # Similarity should be reasonable
        assert result["similarity_percentage"] >= 0.0
        assert result["similarity_percentage"] <= 100.0
