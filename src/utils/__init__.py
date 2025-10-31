"""Utility functions for the RAG application."""

from .ollama_utils import fetch_ollama_models, get_default_model
from .persistence_utils import find_persisted_knowledge_base

__all__ = [
    "fetch_ollama_models",
    "get_default_model",
    "find_persisted_knowledge_base",
]

