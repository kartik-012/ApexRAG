"""Unit tests for LLM generation and mock fallback client."""

from src.generation.llm_client import generate_answer, chat_with_ollama


def test_generate_answer():
    context = ["React state memory triggers re-render. useState is a hook."]
    ans = generate_answer("What is useState?", context)
    assert isinstance(ans, str)
    assert len(ans) > 0


def test_chat_mock_fallback():
    messages = [{"role": "user", "content": "generate 3 variants for QA"}]
    res = chat_with_ollama("llama3.1:8b", messages)
    assert "paraphrase" in res
