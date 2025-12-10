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

# Maximum crawl depth for website crawling (0 = only start URL, 1 = start URL + direct links, 2 = up to 2 levels deep)
# Prevents deep recursion into websites (e.g., if most relevant domain has links to another website,
# and that website has links to another, it will stop at level 2 by default)
# Can be overridden by environment variable MAX_CRAWL_DEPTH
MAX_CRAWL_DEPTH = int(os.getenv("MAX_CRAWL_DEPTH", "2"))

# Domain blacklisting configuration
# Minimum relevance ratio (0.0 to 1.0) required to keep a domain from being blacklisted
# If a domain has less than this ratio of relevant pages, it will be blacklisted
# Default: 0.10 (10% of pages must be relevant)
DOMAIN_RELEVANCE_THRESHOLD = float(os.getenv("DOMAIN_RELEVANCE_THRESHOLD", "0.10"))

# Minimum number of pages from a domain before blacklisting is considered
# Default: 10 pages
DOMAIN_MIN_PAGES_FOR_BLACKLIST = int(os.getenv("DOMAIN_MIN_PAGES_FOR_BLACKLIST", "10"))

# Link relevance threshold for automatic filtering
# Links with relevance score less than this threshold will be automatically ignored (not stored in queue)
# Default: 0.2 (20% relevance minimum)
LINK_RELEVANCE_THRESHOLD = float(os.getenv("LINK_RELEVANCE_THRESHOLD", "0.2"))

# Maximum number of low-relevance links (< 0.2) from a domain before auto-blacklisting
# If a domain has this many or more low-relevance links, it will be automatically blacklisted
# Default: 5 links
DOMAIN_MAX_LOW_RELEVANCE_LINKS = int(os.getenv("DOMAIN_MAX_LOW_RELEVANCE_LINKS", "5"))

# RAG prompt template
RAG_PROMPT_TEMPLATE = """Answer the question using the information provided below. Answer naturally and directly without mentioning sources, context, or where the information came from. Do not use phrases like "Based on the context", "According to the context", "Based on the information provided", or similar meta-commentary.
Answer only with information that is directly relevant to the user's question. Do not mention or comment on unrelated topics.
If the question is not related to the information provided, simply answer based on the question itself.
If you don't know, say you don't know.

Information:
{context}

Question: {question}

Answer:"""

