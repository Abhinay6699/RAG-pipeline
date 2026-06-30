"""
test_llm.py — Tests for prompt construction and LLM module.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import build_prompt, PROMPT_TEMPLATE


class TestBuildPrompt:
    """Tests for the grounded prompt builder."""

    def test_prompt_contains_question(self):
        """The user's question appears in the prompt."""
        prompt = build_prompt("What is AI?", [])
        assert "What is AI?" in prompt

    def test_prompt_contains_context_excerpts(self):
        """Context chunks are embedded in the prompt with source labels."""
        chunks = [
            {
                "text": "AI is intelligence by machines.",
                "metadata": {"source": "doc1.pdf"},
            },
            {
                "text": "ML is a subset of AI.",
                "metadata": {"source": "doc2.txt"},
            },
        ]
        prompt = build_prompt("What is AI?", chunks)
        assert "AI is intelligence by machines." in prompt
        assert "ML is a subset of AI." in prompt
        assert "doc1.pdf" in prompt
        assert "doc2.txt" in prompt
        assert "[Excerpt 1" in prompt
        assert "[Excerpt 2" in prompt

    def test_prompt_has_grounding_instruction(self):
        """Prompt template instructs model to answer only from context."""
        prompt = build_prompt("test", [])
        assert "ONLY" in prompt
        assert "not found in the uploaded documents" in prompt

    def test_empty_context_handled(self):
        """With no context, prompt shows '(No context available)'."""
        prompt = build_prompt("Question?", [])
        assert "(No context available)" in prompt

    def test_prompt_structure(self):
        """Prompt follows the template structure: Context → Question → Answer."""
        chunks = [{"text": "sample", "metadata": {"source": "s.txt"}}]
        prompt = build_prompt("My question", chunks)
        ctx_pos = prompt.index("Context:")
        q_pos = prompt.index("Question:")
        a_pos = prompt.index("Answer:")
        assert ctx_pos < q_pos < a_pos

    def test_multiple_chunks_numbered(self):
        """Multiple chunks are numbered sequentially."""
        chunks = [
            {"text": f"chunk {i}", "metadata": {"source": "doc.pdf"}}
            for i in range(5)
        ]
        prompt = build_prompt("test", chunks)
        for i in range(1, 6):
            assert f"[Excerpt {i}" in prompt

    def test_missing_source_defaults_to_unknown(self):
        """Chunks without source metadata default to 'unknown'."""
        chunks = [{"text": "no source", "metadata": {}}]
        prompt = build_prompt("test", chunks)
        assert "unknown" in prompt
