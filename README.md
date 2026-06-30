# RAG Document QA System

A production-grade Retrieval-Augmented Generation pipeline for semantic document search and grounded question answering.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Flask REST API              │
                    │   /upload   /query   /health   /documents│
                    └────┬──────────┬──────────────────────────┘
                         │          │
              ┌──────────▼──┐  ┌───▼──────────────────────┐
              │   INGEST    │  │        QUERY PIPELINE     │
              │             │  │                           │
              │ Parse file  │  │  ┌─────────┐ ┌────────┐  │
              │ (PDF/TXT/   │  │  │  FAISS  │ │  BM25  │  │
              │  DOCX)      │  │  │ (Dense) │ │(Sparse)│  │
              │      │      │  │  └────┬────┘ └───┬────┘  │
              │      ▼      │  │       └──┬───────┘       │
              │ Chunk text  │  │          ▼               │
              │ (Recursive  │  │   Reciprocal Rank       │
              │  Splitter)  │  │   Fusion (RRF)          │
              │      │      │  │          │               │
              │      ▼      │  │          ▼               │
              │ Embed       │  │   Groq LLM Inference    │
              │ (MiniLM-L6) │  │   (Grounded Prompt)     │
              │      │      │  │          │               │
              │      ▼      │  │          ▼               │
              │ FAISS Index │  │   Answer + Citations     │
              │ (persist)   │  │                           │
              └─────────────┘  └───────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Flask (REST API + web frontend) |
| Embeddings | HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (persisted to disk) |
| Sparse Retrieval | BM25 (via `rank_bm25`) |
| Hybrid Merge | Reciprocal Rank Fusion |
| LLM Inference | Groq API (`openai/gpt-oss-120b` by default) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Document Parsing | PyPDF2, python-docx |

## Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
# Clone and enter the project
git clone <your-repo-url>
cd RAG

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env and add your Groq API key
# GROQ_API_KEY=gsk_your_key_here
```

### Run the Application

```bash
python app.py
```

The app starts at `http://localhost:5000`.

### Run Tests

```bash
pytest tests/ -v
```

## API Reference

### `GET /health`

Health check with system status.

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "openai/gpt-oss-120b",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "indexed_chunks": 42,
  "timestamp": "2026-06-30T04:00:00+00:00"
}
```

### `POST /upload`

Upload and ingest a document (PDF, TXT, or DOCX).

```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@document.pdf"
```

Response:
```json
{
  "filename": "document.pdf",
  "chunk_count": 15,
  "status": "ingested"
}
```

### `GET /documents`

List all ingested documents.

```bash
curl http://localhost:5000/documents
```

Response:
```json
{
  "documents": [
    {
      "filename": "document.pdf",
      "chunk_count": 15,
      "ingested_at": "2026-06-30T04:00:00+00:00"
    }
  ],
  "total": 1
}
```

### `POST /query`

Query the knowledge base with hybrid retrieval + LLM.

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

Response:
```json
{
  "answer": "The document discusses...",
  "citations": [
    {
      "source": "document.pdf",
      "chunk_index": 3,
      "excerpt": "...relevant chunk text...",
      "similarity_score": 0.82,
      "rrf_score": 0.032787,
      "retrieval_method": "hybrid"
    }
  ],
  "query": "What is the main topic of the document?",
  "model": "openai/gpt-oss-120b",
  "usage": {
    "prompt_tokens": 450,
    "completion_tokens": 80,
    "total_tokens": 530
  }
}
```

## Configuration

All settings are configurable via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | LLM model — swap without code changes |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `1500` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `15` | Number of retrieval results |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum upload file size |

### Swapping the LLM Model

Change `GROQ_MODEL` in `.env` to any Groq-supported model. No code changes required:

```bash
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

## Project Structure

```
RAG/
├── app.py              # Flask REST API
├── config.py           # Environment-based configuration
├── ingest.py           # Document parsing, chunking, embedding, FAISS
├── retriever.py        # Hybrid retrieval (FAISS + BM25 + RRF)
├── llm.py              # Groq LLM inference with grounded prompts
├── requirements.txt    # Pinned dependencies
├── .env.example        # Configuration template
├── .gitignore          # Repo hygiene
├── README.md           # This file
├── templates/
│   └── index.html      # Web frontend
├── static/
│   ├── style.css       # UI stylesheet
│   └── app.js          # Frontend logic
└── tests/
    ├── conftest.py     # Shared fixtures
    ├── test_ingest.py  # Chunking + FAISS tests
    ├── test_retriever.py # BM25 + RRF tests
    ├── test_llm.py     # Prompt construction tests
    └── test_api.py     # Integration tests
```

## License

MIT
