"""
ingest.py — Document ingestion pipeline.

Handles:
  1. Parsing uploaded files (PDF, TXT, DOCX) into raw text
  2. Recursive character text splitting into chunks with metadata
  3. Embedding generation via HuggingFace sentence-transformers
  4. Building / merging a FAISS vector store index persisted to disk
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config

logger = logging.getLogger(__name__)


# ── File parsers ──────────────────────────────────────────────────────


def parse_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using PyPDF2.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    from PyPDF2 import PdfReader

    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    if not pages:
        raise ValueError(f"No extractable text found in PDF: {file_path.name}")
    return "\n".join(pages)


def parse_txt(file_path: Path) -> str:
    """Read a plain-text file.

    Args:
        file_path: Path to the .txt file.

    Returns:
        File contents as a string.

    Raises:
        ValueError: If the file is empty.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise ValueError(f"Text file is empty: {file_path.name}")
    return text


def parse_docx(file_path: Path) -> str:
    """Extract text from a DOCX file using python-docx.

    Args:
        file_path: Path to the .docx file.

    Returns:
        Concatenated paragraph text.

    Raises:
        ValueError: If the document contains no text.
    """
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError(f"No text found in DOCX: {file_path.name}")
    return "\n".join(paragraphs)


PARSERS = {
    "pdf": parse_pdf,
    "txt": parse_txt,
    "docx": parse_docx,
}


def parse_document(file_path: Path) -> str:
    """Dispatch to the correct parser based on file extension.

    Args:
        file_path: Path to the uploaded document.

    Returns:
        Extracted raw text.

    Raises:
        ValueError: If the file type is unsupported.
    """
    ext = file_path.suffix.lower().lstrip(".")
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(
            f"Unsupported file type: .{ext}. "
            f"Allowed: {', '.join(sorted(Config.ALLOWED_EXTENSIONS))}"
        )
    return parser(file_path)


# ── Chunking ──────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Split text into overlapping chunks with metadata.

    Uses LangChain's RecursiveCharacterTextSplitter with paragraph →
    sentence → word fallback separators.

    Args:
        text: Raw document text.
        source: Source filename for metadata.
        chunk_size: Characters per chunk (defaults to Config.CHUNK_SIZE).
        chunk_overlap: Overlap between chunks (defaults to Config.CHUNK_OVERLAP).

    Returns:
        List of dicts with keys: 'text', 'metadata' (source, chunk_index,
        total_chunks, ingested_at).
    """
    _chunk_size = chunk_size if chunk_size is not None else Config.CHUNK_SIZE
    _chunk_overlap = chunk_overlap if chunk_overlap is not None else Config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    splits = splitter.split_text(text)
    now = datetime.now(timezone.utc).isoformat()

    chunks = []
    for i, chunk in enumerate(splits):
        chunks.append(
            {
                "text": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(splits),
                    "ingested_at": now,
                },
            }
        )
    logger.info("Chunked '%s' into %d chunks (size=%d, overlap=%d)",
                source, len(splits), _chunk_size, _chunk_overlap)
    return chunks


# ── Embeddings ────────────────────────────────────────────────────────

_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a singleton HuggingFaceEmbeddings instance.

    The model is loaded once and reused across requests to avoid
    repeatedly downloading / loading weights.

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info("Loading embedding model: %s", Config.EMBEDDING_MODEL)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


# ── FAISS index management ────────────────────────────────────────────


def build_faiss_index(chunks: list[dict]) -> FAISS:
    """Build a FAISS index from document chunks.

    Args:
        chunks: List of chunk dicts (with 'text' and 'metadata' keys).

    Returns:
        A FAISS vector store instance.
    """
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = get_embeddings()

    index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    logger.info("Built FAISS index with %d vectors", len(texts))
    return index


def save_faiss_index(index: FAISS) -> None:
    """Persist a FAISS index to disk.

    Args:
        index: The FAISS vector store to save.
    """
    Config.ensure_dirs()
    index.save_local(str(Config.FAISS_INDEX_DIR))
    logger.info("FAISS index saved to %s", Config.FAISS_INDEX_DIR)


def load_faiss_index() -> FAISS | None:
    """Load a FAISS index from disk if it exists.

    Returns:
        The loaded FAISS vector store, or None if no index is found.
    """
    index_path = Config.FAISS_INDEX_DIR / "index.faiss"
    if not index_path.exists():
        logger.info("No existing FAISS index at %s", index_path)
        return None

    embeddings = get_embeddings()
    index = FAISS.load_local(
        str(Config.FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("Loaded FAISS index from %s", Config.FAISS_INDEX_DIR)
    return index


def add_to_faiss_index(chunks: list[dict]) -> FAISS:
    """Add chunks to an existing FAISS index, or create a new one.

    If an index exists on disk, it's loaded and merged with the new
    chunks. Otherwise a fresh index is built. The result is saved.

    Args:
        chunks: List of chunk dicts to add.

    Returns:
        The updated FAISS vector store.
    """
    new_index = build_faiss_index(chunks)
    existing = load_faiss_index()

    if existing is not None:
        existing.merge_from(new_index)
        save_faiss_index(existing)
        return existing

    save_faiss_index(new_index)
    return new_index


# ── Full ingestion pipeline ───────────────────────────────────────────


def ingest_document(file_path: Path) -> dict:
    """Run the full ingestion pipeline for one document.

    Steps: parse → chunk → embed → store in FAISS.

    Args:
        file_path: Path to the uploaded document.

    Returns:
        Summary dict with filename, chunk_count, and status.

    Raises:
        ValueError: On unsupported type, empty file, or parse failure.
    """
    logger.info("Ingesting document: %s", file_path.name)

    # 1. Parse
    raw_text = parse_document(file_path)

    # 2. Chunk
    chunks = chunk_text(raw_text, source=file_path.name)
    if not chunks:
        raise ValueError(f"Document produced zero chunks: {file_path.name}")

    # 3. Embed + store
    add_to_faiss_index(chunks)

    return {
        "filename": file_path.name,
        "chunk_count": len(chunks),
        "status": "ingested",
    }
