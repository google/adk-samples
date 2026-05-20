"""Unit tests for export_content.py.

Tests query construction with filters, CSV/JSON output, and config loading.
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import export_content as ec


# ===================================================================
# Export Query Construction
# ===================================================================

class TestExportQueryConstruction:
    """Verify query building with various filters."""

    def test_export_no_filters(self, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        output = str(tmp_path / "out.csv")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("test-project", "products_dataset", output, "csv")

        query = mock_client.query.call_args[0][0]
        assert "content_generated" in query
        assert "WHERE" not in query
        assert "ORDER BY product_id" in query

    def test_export_with_content_type_filter(self, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        output = str(tmp_path / "out.csv")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("test-project", "products_dataset", output, "csv", content_type="description")

        query = mock_client.query.call_args[0][0]
        assert "content_type = @content_type" in query

    def test_export_with_language_filter(self, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        output = str(tmp_path / "out.csv")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("test-project", "products_dataset", output, "csv", language="es")

        query = mock_client.query.call_args[0][0]
        assert "language = @language" in query

    def test_export_with_both_filters(self, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        output = str(tmp_path / "out.csv")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export(
                "test-project", "products_dataset", output, "csv",
                content_type="translation", language="fr",
            )

        query = mock_client.query.call_args[0][0]
        assert "content_type = @content_type" in query
        assert "language = @language" in query
        assert "AND" in query

    def test_export_uses_correct_table_reference(self, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        output = str(tmp_path / "out.csv")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("my-project", "my_dataset", output, "csv")

        query = mock_client.query.call_args[0][0]
        assert "my-project.my_dataset.content_generated" in query


# ===================================================================
# Export Output Formats
# ===================================================================

class TestExportCSVOutput:
    """CSV output tests."""

    def test_export_csv_creates_file(self, tmp_path):
        mock_client = MagicMock()
        mock_row = MagicMock()
        mock_row.items.return_value = [
            ("product_id", "elec-001"),
            ("content_type", "description"),
            ("content", "Great headphones"),
            ("language", "en"),
            ("variant", 1),
        ]
        mock_client.query.return_value.result.return_value = [mock_row]

        output = str(tmp_path / "out.csv")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("test-project", "products_dataset", output, "csv")

        assert Path(output).exists()
        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["product_id"] == "elec-001"

    def test_export_csv_has_headers(self, tmp_path):
        mock_client = MagicMock()
        mock_row = MagicMock()
        mock_row.items.return_value = [
            ("product_id", "p1"),
            ("content_type", "seo_title"),
            ("content", "Best Product"),
        ]
        mock_client.query.return_value.result.return_value = [mock_row]

        output = str(tmp_path / "out.csv")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("test-project", "products_dataset", output, "csv")

        with open(output) as f:
            header = f.readline().strip()
        assert "product_id" in header
        assert "content_type" in header


class TestExportJSONOutput:
    """JSON output tests."""

    def test_export_json_creates_file(self, tmp_path):
        mock_client = MagicMock()
        mock_row = MagicMock()
        mock_row.items.return_value = [
            ("product_id", "groc-001"),
            ("content_type", "translation"),
            ("content", "Aceite de oliva"),
            ("language", "es"),
            ("variant", 1),
        ]
        mock_client.query.return_value.result.return_value = [mock_row]

        output = str(tmp_path / "out.json")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("test-project", "products_dataset", output, "json")

        assert Path(output).exists()
        loaded = json.loads(Path(output).read_text())
        assert len(loaded) == 1
        assert loaded[0]["product_id"] == "groc-001"
        assert loaded[0]["language"] == "es"

    def test_export_json_multiple_rows(self, tmp_path):
        mock_client = MagicMock()
        rows = []
        for pid in ["elec-001", "elec-002", "elec-003"]:
            row = MagicMock()
            row.items.return_value = [
                ("product_id", pid),
                ("content_type", "description"),
                ("content", f"Content for {pid}"),
            ]
            rows.append(row)
        mock_client.query.return_value.result.return_value = rows

        output = str(tmp_path / "out.json")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("test-project", "products_dataset", output, "json")

        loaded = json.loads(Path(output).read_text())
        assert len(loaded) == 3
        assert loaded[2]["product_id"] == "elec-003"


class TestExportNoResults:
    """Verify behavior when no rows are returned."""

    def test_export_no_rows_does_not_create_file(self, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        output = str(tmp_path / "out.csv")
        with patch.object(ec.bigquery, "Client", return_value=mock_client):
            ec.export("test-project", "products_dataset", output, "csv")

        assert not Path(output).exists()


# ===================================================================
# Config Loading (same load_config function)
# ===================================================================

class TestExportConfigLoading:
    """Config loading tests for export_content module."""

    def test_load_yaml_config(self, sample_config_yaml):
        config = ec.load_config(sample_config_yaml)
        assert config["gcp_project_id"] == "test-project-123"

    def test_load_design_spec_frontmatter(self, sample_design_spec):
        config = ec.load_config(sample_design_spec)
        assert config["gcp_project_id"] == "test-project-123"
        assert config["brand_name"] == "Brooks & Co"

    def test_load_nonexistent_file_returns_empty(self):
        config = ec.load_config("/nonexistent/path/config.yaml")
        assert config == {}

    def test_load_empty_yaml_returns_empty(self, tmp_path):
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        config = ec.load_config(str(empty_file))
        assert config == {}
