"""Shared AI provider identifiers."""

from enum import StrEnum


class LLMProviderName(StrEnum):
    """Supported answer-generation providers."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
