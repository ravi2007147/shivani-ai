"""Configuration constants for the RAG application."""

import os

# Default values
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "mistral"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Directory paths
CHROMA_DB_DIR = ".chroma_db"
KB_PREFIX = "kb_"

# Text splitting configuration
# Optimized for large knowledge bases
CHUNK_SIZE = 2000  # Larger chunks for better context retention
CHUNK_OVERLAP = 400  # Larger overlap to preserve context boundaries

# Retrieval configuration
RETRIEVAL_K = 10  # Number of documents to retrieve per knowledge base
MAX_CONTEXT_LENGTH = 8000  # Maximum context length for LLM

# Batch processing
BATCH_SIZE = 100  # Process embeddings in batches for large texts

# Auto-discovery configuration
# Maximum number of URLs to extract content from during auto-discovery
# Can be overridden by environment variable MAX_URLS_TO_EXTRACT
MAX_URLS_TO_EXTRACT = int(os.getenv("MAX_URLS_TO_EXTRACT", "3"))

# Maximum number of search results to consider during auto-discovery
# Can be overridden by environment variable MAX_SEARCH_RESULTS
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))

# Maximum number of pages to crawl from the primary website (maximum match website)
# When a maximum match website is found, the system will crawl multiple pages from that website
# Can be overridden by environment variable MAX_PAGES_TO_CRAWL
MAX_PAGES_TO_CRAWL = int(os.getenv("MAX_PAGES_TO_CRAWL", "15"))

# RAG prompt template
RAG_PROMPT_TEMPLATE = """Answer the question using the information provided below. Answer naturally and directly without mentioning sources, context, or where the information came from. Do not use phrases like "Based on the context", "According to the context", "Based on the information provided", or similar meta-commentary.
Answer only with information that is directly relevant to the user's question. Do not mention or comment on unrelated topics.
If the question is not related to the information provided, simply answer based on the question itself.
If you don't know, say you don't know.

Information:
{context}

Question: {question}

Answer:"""

