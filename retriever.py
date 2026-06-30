"""
retriever.py — Hybrid retrieval engine.

Combines dense retrieval (FAISS similarity search) with sparse keyword
retrieval (BM25) using Reciprocal Rank Fusion (RRF) to merge results.
"""

import logging
import re

from rank_bm25 import BM25Okapi

from config import Config
from ingest import load_faiss_index

logger = logging.getLogger(__name__)


# ── BM25 helpers ──────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25.

    Args:
        text: Input string.

    Returns:
        List of lowercased tokens.
    """
    return re.findall(r"\w+", text.lower())


def _build_bm25_from_faiss(faiss_index) -> tuple[BM25Okapi, list[dict]]:
    """Build a BM25 index from all documents stored in the FAISS index.

    Extracts the stored texts and metadata from the FAISS docstore,
    tokenizes, and builds a BM25Okapi instance.

    Args:
        faiss_index: A loaded LangChain FAISS vector store.

    Returns:
        Tuple of (BM25Okapi instance, list of {text, metadata} dicts).
    """
    docs = []
    for doc_id in faiss_index.docstore._dict:
        doc = faiss_index.docstore._dict[doc_id]
        docs.append({"text": doc.page_content, "metadata": doc.metadata})

    if not docs:
        raise ValueError("FAISS index is empty — no documents to search.")

    tokenized = [_tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs


# ── Dense retrieval ───────────────────────────────────────────────────


def dense_search(query: str, top_k: int | None = None) -> list[dict]:
    """Perform dense similarity search using the FAISS index.

    Args:
        query: The search query.
        top_k: Number of results to return (defaults to Config.TOP_K).

    Returns:
        List of result dicts with 'text', 'metadata', 'score', 'method'.
        Score is converted from L2 distance to a 0-1 similarity.
    """
    k = top_k or Config.TOP_K
    faiss_index = load_faiss_index()
    if faiss_index is None:
        return []

    results = faiss_index.similarity_search_with_score(query, k=k)
    output = []
    for doc, distance in results:
        # Convert L2 distance to a 0-1 similarity score
        # Lower distance = more similar; use 1/(1+d) as a bounded transform
        similarity = 1.0 / (1.0 + float(distance))
        output.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": round(similarity, 4),
            "method": "dense",
        })
    return output


# ── Sparse (BM25) retrieval ──────────────────────────────────────────


def sparse_search(query: str, top_k: int | None = None) -> list[dict]:
    """Perform sparse BM25 keyword search over indexed documents.

    Args:
        query: The search query.
        top_k: Number of results to return (defaults to Config.TOP_K).

    Returns:
        List of result dicts with 'text', 'metadata', 'score', 'method'.
    """
    k = top_k or Config.TOP_K
    faiss_index = load_faiss_index()
    if faiss_index is None:
        return []

    bm25, docs = _build_bm25_from_faiss(faiss_index)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Pair scores with docs and sort descending
    scored = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)

    output = []
    for score, doc in scored[:k]:
        output.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": round(float(score), 4),
            "method": "sparse",
        })
    return output


# ── Reciprocal Rank Fusion ────────────────────────────────────────────


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF assigns each result a score of 1/(k + rank) in each list it
    appears in, then sums scores across lists. This avoids the need to
    normalize scores across incompatible scales (L2 vs BM25).

    Args:
        result_lists: List of ranked result lists.
        k: RRF constant (default 60, per the original paper).

    Returns:
        Merged list sorted by combined RRF score, deduplicated by
        source + chunk_index.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            # Unique key: source filename + chunk index
            key = f"{result['metadata'].get('source', '')}::{result['metadata'].get('chunk_index', rank)}"
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed, so +1
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_score

            # Keep the first occurrence's data (with its original score)
            if key not in doc_map:
                doc_map[key] = result

    # Sort by fused RRF score descending
    sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    output = []
    for key in sorted_keys:
        entry = doc_map[key].copy()
        entry["rrf_score"] = round(fused_scores[key], 6)
        entry["method"] = "hybrid"
        output.append(entry)

    return output


# ── Public hybrid retrieval interface ─────────────────────────────────


def hybrid_search(query: str, top_k: int | None = None) -> list[dict]:
    """Run hybrid retrieval: dense (FAISS) + sparse (BM25) merged via RRF.

    This is the primary retrieval function used by the query pipeline.

    Args:
        query: The user's search query.
        top_k: Number of final results to return (defaults to Config.TOP_K).

    Returns:
        Top-k results sorted by RRF score, each containing 'text',
        'metadata', 'score' (original), 'rrf_score', and 'method'.
    """
    k = top_k or Config.TOP_K

    # Retrieve from both engines with extra candidates for better fusion
    candidate_k = k * 3
    dense_results = dense_search(query, top_k=candidate_k)
    sparse_results = sparse_search(query, top_k=candidate_k)

    if not dense_results and not sparse_results:
        return []

    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    return fused[:k]
