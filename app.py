"""
app.py — Flask REST API for the RAG Document QA system.

Endpoints:
    GET  /           — Serve the web frontend
    GET  /health     — Health check
    GET  /documents  — List ingested documents
    POST /upload     — Upload and ingest a document (PDF, TXT, DOCX)
    POST /query      — Query the knowledge base with hybrid retrieval + LLM
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, render_template

from config import Config
from ingest import ingest_document, load_faiss_index
from llm import generate_answer
from retriever import hybrid_search

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Ensure required directories exist
Config.ensure_dirs()

# Log config warnings at startup
for warning in Config.validate():
    logger.warning("CONFIG: %s", warning)


# ── Error handlers ────────────────────────────────────────────────────


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file uploads exceeding MAX_UPLOAD_SIZE_MB."""
    return jsonify({
        "error": f"File too large. Maximum size is {Config.MAX_UPLOAD_SIZE_MB} MB."
    }), 413


@app.errorhandler(400)
def bad_request(error):
    """Handle malformed requests."""
    return jsonify({"error": str(error.description)}), 400


@app.errorhandler(500)
def internal_error(error):
    """Handle unexpected server errors without exposing stack traces."""
    logger.exception("Internal server error")
    return jsonify({"error": "An internal server error occurred."}), 500


# ── Routes ────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the web frontend."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns:
        JSON with status, model config, and index state.
    """
    faiss_index = load_faiss_index()
    doc_count = 0
    if faiss_index is not None:
        doc_count = len(faiss_index.docstore._dict)

    return jsonify({
        "status": "healthy",
        "model": Config.GROQ_MODEL,
        "embedding_model": Config.EMBEDDING_MODEL,
        "indexed_chunks": doc_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/documents", methods=["GET"])
def list_documents():
    """List all ingested documents with metadata.

    Returns:
        JSON array of documents with filename, chunk count, and upload info.
    """
    faiss_index = load_faiss_index()
    if faiss_index is None:
        return jsonify({"documents": [], "total": 0})

    # Aggregate chunk info by source document
    doc_info: dict[str, dict] = {}
    for doc_id in faiss_index.docstore._dict:
        doc = faiss_index.docstore._dict[doc_id]
        source = doc.metadata.get("source", "unknown")
        if source not in doc_info:
            doc_info[source] = {
                "filename": source,
                "chunk_count": 0,
                "ingested_at": doc.metadata.get("ingested_at", ""),
            }
        doc_info[source]["chunk_count"] += 1

    documents = list(doc_info.values())
    return jsonify({"documents": documents, "total": len(documents)})


@app.route("/upload", methods=["POST"])
def upload_document():
    """Upload and ingest a document.

    Accepts multipart/form-data with a 'file' field. Validates file
    type and size, saves to uploads/, runs ingestion pipeline.

    Returns:
        JSON with ingestion results or error details.
    """
    # ── Validate presence ─────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use 'file' field in multipart form data."}), 400

    file = request.files["file"]
    if file.filename is None or file.filename.strip() == "":
        return jsonify({"error": "No file selected."}), 400

    # ── Validate extension ────────────────────────────────────────
    filename = file.filename.strip()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in Config.ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type: .{ext}. Allowed: {', '.join(sorted(Config.ALLOWED_EXTENSIONS))}"
        }), 400

    # ── Save file ─────────────────────────────────────────────────
    save_path = Config.UPLOAD_DIR / filename
    try:
        file.save(str(save_path))
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # ── Check empty file ──────────────────────────────────────────
    if save_path.stat().st_size == 0:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Uploaded file is empty."}), 400

    # ── Ingest ────────────────────────────────────────────────────
    try:
        result = ingest_document(save_path)
        return jsonify(result), 200
    except ValueError as e:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Ingestion failed for %s", filename)
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500


@app.route("/query", methods=["POST"])
def query():
    """Query the knowledge base.

    Expects JSON body with 'question' field. Runs hybrid retrieval,
    generates a grounded answer via Groq, and returns structured
    citations alongside the answer.

    Returns:
        JSON with answer, citations (source, excerpt, score), model info.
    """
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in JSON body."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    # Optional top_k override
    top_k = data.get("top_k", Config.TOP_K)

    # ── Retrieve ──────────────────────────────────────────────────
    try:
        results = hybrid_search(question, top_k=top_k)
    except Exception as e:
        logger.exception("Retrieval failed")
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    if not results:
        return jsonify({
            "answer": "No documents have been uploaded yet. Please upload documents first.",
            "citations": [],
            "query": question,
            "model": Config.GROQ_MODEL,
        })

    # ── Generate answer ───────────────────────────────────────────
    try:
        llm_result = generate_answer(question, results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("LLM inference failed")
        return jsonify({"error": f"LLM inference failed: {str(e)}"}), 500

    # ── Build structured citations ────────────────────────────────
    citations = []
    for r in results:
        citations.append({
            "source": r["metadata"].get("source", "unknown"),
            "chunk_index": r["metadata"].get("chunk_index", -1),
            "excerpt": r["text"][:300],  # Truncate for response size
            "similarity_score": r.get("score", 0.0),
            "rrf_score": r.get("rrf_score", 0.0),
            "retrieval_method": r.get("method", "hybrid"),
        })

    return jsonify({
        "answer": llm_result["answer"],
        "citations": citations,
        "query": question,
        "model": llm_result["model"],
        "usage": llm_result.get("usage", {}),
    })


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
