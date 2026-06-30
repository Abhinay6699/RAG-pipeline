"""
llm.py — LLM inference layer using Groq API.

Constructs a grounded prompt from retrieved context chunks and sends it
to the configured Groq model. The prompt explicitly instructs the model
to answer only from provided context.
"""

import logging

from groq import Groq

from config import Config

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a document research assistant. Answer the question using ONLY the provided context excerpts below.

Rules:
1. Base your answer ONLY on the information in the context excerpts.
2. If the context does not contain enough information to answer the question, respond with: "The answer was not found in the uploaded documents."
3. Do NOT use any outside knowledge beyond what is provided in the context.
4. When possible, mention which source document(s) your answer comes from.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """Construct the grounded prompt from a query and retrieved chunks.

    Each chunk is formatted with its source metadata for transparency.

    Args:
        query: The user's question.
        context_chunks: List of retrieval result dicts, each containing
            'text' and 'metadata' (with 'source' key).

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Excerpt {i} — Source: {source}]\n{text}")

    context_str = "\n\n".join(context_parts) if context_parts else "(No context available)"

    return PROMPT_TEMPLATE.format(context=context_str, question=query)


# ── Groq client ───────────────────────────────────────────────────────

_client: Groq | None = None


def _get_client() -> Groq:
    """Return a singleton Groq client instance.

    Returns:
        Configured Groq client.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    global _client
    if _client is None:
        if not Config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = Groq(api_key=Config.GROQ_API_KEY)
    return _client


def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """Generate a grounded answer using the Groq LLM.

    Args:
        query: The user's question.
        context_chunks: Retrieved context chunks from hybrid search.

    Returns:
        Dict with 'answer', 'model', 'usage' (token counts).

    Raises:
        ValueError: If the API key is missing.
        groq.APIError: On Groq API failures.
    """
    prompt = build_prompt(query, context_chunks)
    client = _get_client()

    logger.info("Sending query to Groq model: %s", Config.GROQ_MODEL)

    response = client.chat.completions.create(
        model=Config.GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document research assistant. "
                    "Only answer from provided context. Never fabricate information."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    logger.info("LLM response received — %d tokens used", usage["total_tokens"])

    return {
        "answer": answer,
        "model": Config.GROQ_MODEL,
        "usage": usage,
    }
