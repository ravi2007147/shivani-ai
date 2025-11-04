"""Utility functions for the RAG application."""

from .ollama_utils import fetch_ollama_models, get_default_model
from .persistence_utils import find_persisted_knowledge_base
from .text_classifier import PDFClassifier, classify_pdf_content, PDF_CATEGORIES
from .web_scraper import extract_article_from_url, is_valid_url

__all__ = [
    "fetch_ollama_models",
    "get_default_model",
    "find_persisted_knowledge_base",
    "PDFClassifier",
    "classify_pdf_content",
    "PDF_CATEGORIES",
    "extract_article_from_url",
    "is_valid_url",
]

