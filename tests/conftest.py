"""
conftest.py — Shared test fixtures for the RAG Document QA test suite.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test config BEFORE importing app modules
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("GROQ_MODEL", "openai/gpt-oss-120b")


@pytest.fixture
def sample_text():
    """Return a reproducible multi-paragraph text for chunking tests."""
    return (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to natural intelligence displayed by animals including humans. "
        "AI research has been defined as the field of study of intelligent agents, "
        "which refers to any system that perceives its environment and takes actions "
        "that maximize its chance of achieving its goals.\n\n"
        "The term 'artificial intelligence' had previously been used to describe "
        "machines that mimic and display human cognitive skills. AI applications "
        "include advanced web search engines, recommendation systems, understanding "
        "human speech, self-driving cars, automated decision-making and competing "
        "at the highest level in strategic game systems.\n\n"
        "As machines become increasingly capable, tasks considered to require "
        "intelligence are often removed from the definition of AI, a phenomenon "
        "known as the AI effect. For instance, optical character recognition is "
        "frequently excluded from things considered to be AI, having become a "
        "routine technology."
    )


@pytest.fixture
def sample_txt_file(sample_text, tmp_path):
    """Write sample text to a .txt file and return its path."""
    file_path = tmp_path / "test_document.txt"
    file_path.write_text(sample_text, encoding="utf-8")
    return file_path


@pytest.fixture
def small_chunks():
    """Return pre-built chunk dicts for index tests (avoids embedding in every test)."""
    return [
        {
            "text": "Artificial intelligence is intelligence demonstrated by machines.",
            "metadata": {"source": "test.txt", "chunk_index": 0, "total_chunks": 3},
        },
        {
            "text": "AI applications include web search and recommendation systems.",
            "metadata": {"source": "test.txt", "chunk_index": 1, "total_chunks": 3},
        },
        {
            "text": "Optical character recognition is frequently excluded from AI.",
            "metadata": {"source": "test.txt", "chunk_index": 2, "total_chunks": 3},
        },
    ]


@pytest.fixture
def temp_faiss_dir(tmp_path):
    """Provide a temporary directory for FAISS index persistence tests."""
    index_dir = tmp_path / "faiss_test_index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def flask_client():
    """Create a Flask test client with a temporary upload/index directory."""
    from config import Config

    # Use temp dirs so tests don't pollute the real project
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        Config.UPLOAD_DIR = tmpdir_path / "uploads"
        Config.FAISS_INDEX_DIR = tmpdir_path / "faiss_index"
        Config.ensure_dirs()

        from app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
