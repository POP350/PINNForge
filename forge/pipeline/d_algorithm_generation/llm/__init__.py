"""LLM providers and JSON transport helpers for the mainline search."""

from .provider import OpenAICompatibleProvider, build_provider

__all__ = ["OpenAICompatibleProvider", "build_provider"]
