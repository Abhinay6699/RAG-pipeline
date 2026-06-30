"""
test_retriever.py — Tests for BM25 retrieval and hybrid RRF merge.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever import _tokenize, reciprocal_rank_fusion, sparse_search


class TestTokenizer:
    """Tests for the BM25 tokenizer."""

    def test_basic_tokenization(self):
        """Tokenizes words and lowercases them."""
        tokens = _tokenize("Hello World! This is a Test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_empty_string(self):
        """Empty input yields empty token list."""
        assert _tokenize("") == []

    def test_punctuation_stripping(self):
        """Punctuation is stripped from tokens."""
        tokens = _tokenize("AI, ML, and NLP — these are fields.")
        assert "ai" in tokens
        assert "ml" in tokens
        assert "nlp" in tokens
        assert "," not in tokens


class TestReciprocalRankFusion:
    """Tests for the RRF merge algorithm."""

    def _make_result(self, source, chunk_index, score, method="dense"):
        """Helper to create a mock retrieval result."""
        return {
            "text": f"chunk {chunk_index} from {source}",
            "metadata": {"source": source, "chunk_index": chunk_index},
            "score": score,
            "method": method,
        }

    def test_single_list_preserves_order(self):
        """With one list, RRF preserves the original ranking."""
        results = [
            self._make_result("a.txt", 0, 0.9),
            self._make_result("a.txt", 1, 0.7),
            self._make_result("a.txt", 2, 0.5),
        ]
        fused = reciprocal_rank_fusion([results])
        assert len(fused) == 3
        assert fused[0]["metadata"]["chunk_index"] == 0
        assert fused[1]["metadata"]["chunk_index"] == 1
        assert fused[2]["metadata"]["chunk_index"] == 2

    def test_overlapping_results_get_boosted(self):
        """Documents appearing in both lists get higher fused scores."""
        dense = [
            self._make_result("a.txt", 0, 0.9, "dense"),
            self._make_result("a.txt", 1, 0.7, "dense"),
        ]
        sparse = [
            self._make_result("a.txt", 1, 5.0, "sparse"),  # overlap with dense rank 2
            self._make_result("a.txt", 2, 3.0, "sparse"),
        ]
        fused = reciprocal_rank_fusion([dense, sparse])

        # Chunk 1 appears in both → should have highest fused score
        scores_by_chunk = {
            r["metadata"]["chunk_index"]: r["rrf_score"] for r in fused
        }
        assert scores_by_chunk[1] > scores_by_chunk[0]
        assert scores_by_chunk[1] > scores_by_chunk[2]

    def test_empty_lists_produce_empty(self):
        """Empty input lists produce empty output."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_deduplication(self):
        """Same document in both lists appears only once in output."""
        result_a = self._make_result("a.txt", 0, 0.9)
        fused = reciprocal_rank_fusion([[result_a], [result_a]])
        assert len(fused) == 1

    def test_rrf_scores_are_positive(self):
        """All RRF scores should be positive."""
        results = [
            self._make_result("a.txt", i, 0.5)
            for i in range(5)
        ]
        fused = reciprocal_rank_fusion([results])
        for r in fused:
            assert r["rrf_score"] > 0

    def test_method_set_to_hybrid(self):
        """All fused results have method='hybrid'."""
        results = [self._make_result("a.txt", 0, 0.9, "dense")]
        fused = reciprocal_rank_fusion([results])
        assert fused[0]["method"] == "hybrid"
