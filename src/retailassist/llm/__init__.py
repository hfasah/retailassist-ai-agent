"""Model providers. Pick one with RETAILASSIST_LLM_PROVIDER ("gemini", the default, or "claude")."""

import os

from .base import LLM, LLMAuthError, LLMError, LLMRateLimitError, LLMUnavailableError, ModelTurn, ToolCall, ToolResult

PROVIDERS = ("gemini", "claude")


def get_llm(provider: str | None = None) -> LLM:
    provider = (provider or os.environ.get("RETAILASSIST_LLM_PROVIDER", "gemini")).lower()
    if provider == "gemini":
        from .gemini import GeminiLLM
        return GeminiLLM()
    if provider == "claude":
        from .claude import ClaudeLLM
        return ClaudeLLM()
    raise ValueError(f"Unknown RETAILASSIST_LLM_PROVIDER {provider!r}; use one of {', '.join(PROVIDERS)}.")


__all__ = ["LLM", "LLMAuthError", "LLMError", "LLMRateLimitError", "LLMUnavailableError", "ModelTurn", "PROVIDERS",
           "ToolCall", "ToolResult", "get_llm"]
