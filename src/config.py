"""Configuration constants for the RAG application."""

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

# RAG prompt template
RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""

