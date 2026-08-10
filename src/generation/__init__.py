"""Generation package: Wraps local Ollama LLM queries with mock/simulation fallback."""
from .llm_client import generate_answer, chat_with_ollama, OllamaUnavailableError, LLMClient

__all__ = ["generate_answer", "chat_with_ollama", "OllamaUnavailableError", "LLMClient"]
