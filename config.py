"""
config.py — Central configuration for the RAG Document QA system.

All settings are read from environment variables (loaded from .env via
python-dotenv). No secrets or model names are hardcoded.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent / ".env")


class Config:
    """Application configuration sourced from environment variables."""

    # ── Groq LLM ──────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # ── Embedding model ───────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Chunking ──────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ── Retrieval ─────────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", "15"))

    # ── Paths ─────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    FAISS_INDEX_DIR: Path = BASE_DIR / "faiss_index"

    # ── Upload limits ─────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: set = {"pdf", "txt", "docx"}

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of configuration warnings/errors."""
        issues = []
        if not cls.GROQ_API_KEY:
            issues.append("GROQ_API_KEY is not set — LLM queries will fail.")
        if cls.CHUNK_SIZE <= 0:
            issues.append(f"CHUNK_SIZE must be positive, got {cls.CHUNK_SIZE}.")
        if cls.CHUNK_OVERLAP < 0:
            issues.append(f"CHUNK_OVERLAP must be non-negative, got {cls.CHUNK_OVERLAP}.")
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            issues.append("CHUNK_OVERLAP must be less than CHUNK_SIZE.")
        if cls.TOP_K <= 0:
            issues.append(f"TOP_K must be positive, got {cls.TOP_K}.")
        return issues

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they don't exist."""
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
