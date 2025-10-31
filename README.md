# RAG with Ollama

A Streamlit application for creating and querying knowledge bases using Ollama locally with RAG (Retrieval Augmented Generation).

## Features

- **Auto-load Ollama Models**: Automatically fetches and displays all installed Ollama models
- **Persistent Knowledge Bases**: Saves embeddings to disk for persistent storage across sessions
- **RAG Support**: Create knowledge bases from text and query them with context
- **Direct Query**: Query models directly without creating a knowledge base
- **Clean Architecture**: Well-structured Python codebase with separation of concerns

## Project Structure

```
shivani-ai/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── app.py                    # Main Streamlit application
│   ├── config.py                 # Configuration constants
│   ├── pages/                    # Streamlit pages
│   │   └── 1_📚_Manage_Knowledge_Bases.py  # KB management page
│   ├── rag/                      # RAG-related modules
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py       # RAG pipeline for querying with context
│   │   └── vectorstore_manager.py # VectorStore management (create/load)
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── kb_manager.py         # Knowledge base management
│       ├── ollama_utils.py       # Ollama API utilities
│       └── persistence_utils.py  # Knowledge base persistence utilities
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
└── README.md                     # This file
```

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv shivani-env
source shivani-env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running locally:
```bash
ollama serve
```

## Usage

1. Run the Streamlit app (choose one method):

**Method 1: Using the run script (recommended)**
```bash
./run.sh
```

**Method 2: Direct command (from project root)**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run src/app.py
```

**Method 3: Install in development mode**
```bash
pip install -e .
streamlit run src/app.py
```

2. The app will automatically:
   - Load all available Ollama models
   - Attempt to load any persisted knowledge bases

3. Create a knowledge base:
   - Enter text in the "RAG Text Box"
   - Click "Create Knowledge Base"
   - The knowledge base will be persisted to `.chroma_db/` directory
   - Input box will clear after successful creation

4. Query:
   - With knowledge base: Queries use context from the knowledge base
   - Without knowledge base: Direct LLM queries

5. Manage Knowledge Bases:
   - Click "📚 Manage Knowledge Bases" button in sidebar to open management page
   - View all saved knowledge bases with details
   - Load, delete, or unload specific knowledge bases
   - See active knowledge base status

## Architecture

### Components

- **`app.py`**: Main Streamlit application with UI logic
- **`config.py`**: Configuration constants (defaults, paths, prompts)
- **`rag/vectorstore_manager.py`**: Handles Chroma vectorstore creation and persistence
- **`rag/rag_pipeline.py`**: Manages RAG pipeline (retrieval + LLM)
- **`utils/ollama_utils.py`**: Ollama API interactions (fetch models)
- **`utils/persistence_utils.py`**: Knowledge base discovery and management

### Design Principles

1. **Separation of Concerns**: UI logic separate from business logic
2. **Modularity**: Each component has a single responsibility
3. **Reusability**: Utility functions can be used independently
4. **Maintainability**: Clear structure makes it easy to extend

## Configuration

Default settings can be modified in `src/config.py`:
- `DEFAULT_OLLAMA_BASE_URL`: Ollama API endpoint
- `DEFAULT_LLM_MODEL`: Default LLM model name
- `DEFAULT_EMBEDDING_MODEL`: Default embedding model name
- `CHUNK_SIZE`: Text chunking size for embeddings
- `CHUNK_OVERLAP`: Overlap between chunks

## Dependencies

- `streamlit`: Web application framework
- `langchain-ollama`: Ollama integration for LangChain
- `langchain-chroma`: Chroma vector database integration
- `langchain-text-splitters`: Text splitting utilities
- `langchain-core`: Core LangChain components
- `chromadb`: Vector database
- `requests`: HTTP library for Ollama API calls

## Notes

- Knowledge bases are stored in `.chroma_db/` directory (gitignored)
- Each knowledge base is stored in a unique directory based on text hash
- The most recently modified knowledge base is automatically loaded on startup

