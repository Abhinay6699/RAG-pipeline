"""
test_ingest.py — Tests for document ingestion: parsing, chunking, FAISS index.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest import (
    chunk_text,
    parse_txt,
    parse_document,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
)
from config import Config


class TestParseTxt:
    """Tests for plain-text file parsing."""

    def test_parse_txt_returns_content(self, sample_txt_file):
        """A valid .txt file returns its text content."""
        result = parse_txt(sample_txt_file)
        assert "Artificial intelligence" in result
        assert len(result) > 100

    def test_parse_txt_empty_raises(self, tmp_path):
        """An empty .txt file raises ValueError."""
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            parse_txt(empty)


class TestParseDocument:
    """Tests for the file-type dispatch."""

    def test_unsupported_extension_raises(self, tmp_path):
        """A .csv file is rejected with a clear error."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(csv_file)

    def test_txt_dispatches_correctly(self, sample_txt_file):
        """A .txt file goes through parse_txt."""
        result = parse_document(sample_txt_file)
        assert "Artificial intelligence" in result


class TestChunking:
    """Tests for RecursiveCharacterTextSplitter chunking."""

    def test_chunk_produces_multiple_chunks(self, sample_text):
        """A multi-paragraph text produces more than one chunk."""
        chunks = chunk_text(sample_text, source="test.txt", chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_metadata_present(self, sample_text):
        """Each chunk has source, chunk_index, total_chunks metadata."""
        chunks = chunk_text(sample_text, source="doc.pdf", chunk_size=200, chunk_overlap=20)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["source"] == "doc.pdf"
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["total_chunks"] == len(chunks)
            assert "ingested_at" in chunk["metadata"]

    def test_chunk_size_respected(self, sample_text):
        """No chunk exceeds the configured chunk_size."""
        size = 150
        chunks = chunk_text(sample_text, source="test.txt", chunk_size=size, chunk_overlap=10)
        for chunk in chunks:
            assert len(chunk["text"]) <= size + 10  # small tolerance for splitter

    def test_chunk_text_content_preserved(self, sample_text):
        """Concatenated chunks contain the original text (modulo overlap)."""
        chunks = chunk_text(sample_text, source="test.txt", chunk_size=300, chunk_overlap=0)
        reconstructed = "".join(c["text"] for c in chunks)
        # All words from original should appear
        for word in ["intelligence", "machines", "optical", "AI"]:
            assert word.lower() in reconstructed.lower()

    def test_configurable_parameters(self, sample_text):
        """Different chunk_size produces different chunk counts."""
        small = chunk_text(sample_text, source="t.txt", chunk_size=100, chunk_overlap=10)
        large = chunk_text(sample_text, source="t.txt", chunk_size=500, chunk_overlap=10)
        assert len(small) > len(large)


class TestFAISSIndex:
    """Tests for FAISS index build, save, and load."""

    def test_build_faiss_index(self, small_chunks):
        """build_faiss_index creates a searchable index."""
        index = build_faiss_index(small_chunks)
        assert index is not None
        # Search should return results
        results = index.similarity_search("artificial intelligence", k=2)
        assert len(results) > 0
        assert results[0].page_content  # has text

    def test_save_and_load_faiss(self, small_chunks, temp_faiss_dir):
        """Index survives a save/load round-trip."""
        # Temporarily override the FAISS dir
        original_dir = Config.FAISS_INDEX_DIR
        Config.FAISS_INDEX_DIR = temp_faiss_dir

        try:
            index = build_faiss_index(small_chunks)
            save_faiss_index(index)

            # Verify files written
            assert (temp_faiss_dir / "index.faiss").exists()

            # Load and search
            loaded = load_faiss_index()
            assert loaded is not None
            results = loaded.similarity_search("web search", k=1)
            assert len(results) == 1
            assert "search" in results[0].page_content.lower()
        finally:
            Config.FAISS_INDEX_DIR = original_dir

    def test_load_nonexistent_returns_none(self, temp_faiss_dir):
        """Loading from an empty directory returns None."""
        original_dir = Config.FAISS_INDEX_DIR
        Config.FAISS_INDEX_DIR = temp_faiss_dir

        try:
            result = load_faiss_index()
            assert result is None
        finally:
            Config.FAISS_INDEX_DIR = original_dir
