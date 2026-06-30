"""
test_api.py — Integration tests for the Flask REST API.

Tests the /health, /upload, /query, and /documents endpoints using
the Flask test client with temporary directories.
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, flask_client):
        """Health endpoint returns 200 with status info."""
        res = flask_client.get("/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "embedding_model" in data

    def test_health_includes_model_info(self, flask_client):
        """Health response includes the configured model name."""
        res = flask_client.get("/health")
        data = res.get_json()
        assert data["model"] == "openai/gpt-oss-120b"


class TestUploadEndpoint:
    """Tests for POST /upload."""

    def test_upload_no_file_returns_400(self, flask_client):
        """Missing file field returns 400."""
        res = flask_client.post("/upload")
        assert res.status_code == 400
        assert "error" in res.get_json()

    def test_upload_empty_filename_returns_400(self, flask_client):
        """Empty filename returns 400."""
        data = {"file": (io.BytesIO(b""), "")}
        res = flask_client.post("/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 400

    def test_upload_unsupported_type_returns_400(self, flask_client):
        """A .csv file is rejected with 400."""
        data = {"file": (io.BytesIO(b"a,b,c"), "data.csv")}
        res = flask_client.post("/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 400
        assert "Unsupported" in res.get_json()["error"]

    def test_upload_empty_file_returns_400(self, flask_client):
        """An empty .txt file returns 400."""
        data = {"file": (io.BytesIO(b""), "empty.txt")}
        res = flask_client.post("/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 400

    def test_upload_valid_txt_succeeds(self, flask_client):
        """A valid .txt file is ingested successfully."""
        content = b"This is a test document with enough content to chunk properly. " * 20
        data = {"file": (io.BytesIO(content), "test_doc.txt")}
        res = flask_client.post("/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 200
        result = res.get_json()
        assert result["status"] == "ingested"
        assert result["filename"] == "test_doc.txt"
        assert result["chunk_count"] > 0


class TestDocumentsEndpoint:
    """Tests for GET /documents."""

    def test_documents_empty_initially(self, flask_client):
        """No documents listed when index is empty."""
        res = flask_client.get("/documents")
        assert res.status_code == 200
        data = res.get_json()
        assert data["total"] == 0

    def test_documents_lists_after_upload(self, flask_client):
        """After uploading, the document appears in the list."""
        content = b"Test content for document listing verification. " * 20
        flask_client.post(
            "/upload",
            data={"file": (io.BytesIO(content), "listed_doc.txt")},
            content_type="multipart/form-data",
        )
        res = flask_client.get("/documents")
        data = res.get_json()
        assert data["total"] == 1
        assert data["documents"][0]["filename"] == "listed_doc.txt"


class TestQueryEndpoint:
    """Tests for POST /query."""

    def test_query_no_body_returns_400(self, flask_client):
        """Missing JSON body returns 400."""
        res = flask_client.post("/query")
        assert res.status_code == 400

    def test_query_empty_question_returns_400(self, flask_client):
        """Empty question string returns 400."""
        res = flask_client.post(
            "/query",
            data=json.dumps({"question": "   "}),
            content_type="application/json",
        )
        assert res.status_code == 400

    def test_query_no_documents_returns_helpful_message(self, flask_client):
        """Querying with no documents returns a guidance message."""
        res = flask_client.post(
            "/query",
            data=json.dumps({"question": "What is AI?"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "upload" in data["answer"].lower() or "no documents" in data["answer"].lower()

    @patch("app.generate_answer")
    def test_query_end_to_end_with_mock_llm(self, mock_llm, flask_client):
        """Full flow: upload → query → structured citation response."""
        # Mock LLM to avoid real API calls
        mock_llm.return_value = {
            "answer": "AI is intelligence demonstrated by machines.",
            "model": "openai/gpt-oss-120b",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }

        # Upload a document
        content = (
            b"Artificial intelligence is intelligence demonstrated by machines. "
            b"It encompasses many subfields including machine learning and NLP. "
        ) * 15
        flask_client.post(
            "/upload",
            data={"file": (io.BytesIO(content), "ai_intro.txt")},
            content_type="multipart/form-data",
        )

        # Query
        res = flask_client.post(
            "/query",
            data=json.dumps({"question": "What is artificial intelligence?"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()

        # Verify structured response
        assert "answer" in data
        assert "citations" in data
        assert "query" in data
        assert "model" in data
        assert data["query"] == "What is artificial intelligence?"

        # Verify citation structure
        assert len(data["citations"]) > 0
        cit = data["citations"][0]
        assert "source" in cit
        assert "chunk_index" in cit
        assert "excerpt" in cit
        assert "similarity_score" in cit
        assert "retrieval_method" in cit
