"""
LLM client — wraps local Ollama for generation and judge/utility calls
(faithfulness scoring, adversarial question generation, multi-judge debate).

$0 cost: Ollama runs entirely locally, no API key, no per-token billing.

NOTE ON THIS BUILD: `ollama.chat()`'s real signature was confirmed live
(messages use standard {"role", "content"} dicts; format="json" is a real
supported param for structured output). The exact connection failure mode
was also confirmed live: with no Ollama server running, ollama.chat() raises
`ConnectionError` with a specific message pointing to https://ollama.com/download
— this sandbox has no Ollama installed, so actual generation could not be
tested here. Install + run `ollama serve` locally, then `ollama pull llama3.1:8b`
before using this module for real.

This is a convenience wrapper — the canonical implementation lives in
src/generation/llm_client.py. Import from there for package usage.
"""

# Re-export canonical implementation
from src.generation.llm_client import (
    OllamaUnavailableError,
    LLMClient,
    chat_with_ollama,
    generate_answer,
)

__all__ = ["OllamaUnavailableError", "LLMClient", "chat_with_ollama", "generate_answer"]


if __name__ == "__main__":
    print("=== Testing LLMClient error handling (no Ollama server expected) ===")
    client = LLMClient()
    try:
        result = client.complete("What is React?")
        print(f"Got mock fallback response: {result[:80]}...")
    except OllamaUnavailableError as e:
        print(f"OK — correctly raised OllamaUnavailableError:\n  {e}")

    print("\n=== Testing generate_answer() against retrieved chunk ===")
    answer = generate_answer(
        question="What does useEffect do?",
        context_chunks=["useEffect lets you perform side effects in function components."],
    )
    print(f"Answer: {answer[:100]}...")
