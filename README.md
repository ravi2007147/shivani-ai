# 🤖 Shivani AI - Personalized RAG-Powered Knowledge Base System

**Build intelligent, context-aware AI assistants with local LLM models and Retrieval Augmented Generation (RAG)**

A production-ready Streamlit web application that enables users to create personalized AI knowledge bases using local Large Language Models (LLMs) through Ollama. This enterprise-grade solution combines the power of Retrieval Augmented Generation (RAG) with vector embeddings to deliver accurate, context-aware responses from your private data—all running completely offline on your local machine.

---

## 🌟 What This Application Does

**Shivani AI** is an intelligent knowledge management platform that allows you to:

- **Create Personalized AI Assistants**: Build multiple AI profiles (e.g., "Ravi Kumar Pundir", "Seema Kumari") with separate knowledge bases
- **Transform Text into Knowledge**: Convert any document, article, or text content into a searchable, queryable knowledge base
- **Get Context-Aware Answers**: Ask questions and receive accurate responses based on your stored knowledge
- **Maintain Complete Privacy**: All processing happens locally—your data never leaves your machine
- **Query Multiple Knowledge Bases**: Search across multiple profiles simultaneously for comprehensive answers
- **Scale to Large Documents**: Optimized to handle very large knowledge bases efficiently

### Business Value

- **100% Local & Private**: No data sent to external servers—perfect for sensitive information
- **Cost-Effective**: No API costs—runs entirely on your local hardware
- **Customizable**: Create unlimited profiles and knowledge bases for different use cases
- **Production-Ready**: Robust error handling, persistent storage, and professional code architecture

---

## 🚀 Key Features

### Core Capabilities

- ✅ **Multi-Profile Support**: Create and manage multiple user profiles, each with their own knowledge bases
- ✅ **Intelligent RAG Pipeline**: Advanced Retrieval Augmented Generation for context-aware responses
- ✅ **Vector Embeddings**: Semantic search using ChromaDB vector database
- ✅ **Auto-Load Models**: Automatically detects and loads all available Ollama models
- ✅ **Persistent Storage**: Knowledge bases saved to disk—survive restarts and refreshes
- ✅ **Cross-Profile Queries**: Query across multiple profiles' knowledge bases simultaneously
- ✅ **Large Document Support**: Optimized chunking and batch processing for handling large texts
- ✅ **Real-Time Progress**: Live progress indicators during knowledge base creation
- ✅ **Knowledge Base Management**: Dedicated interface to view, load, and delete knowledge bases
- ✅ **Direct LLM Queries**: Query models directly without knowledge base context

### Technical Highlights

- 🏗️ **Modular Architecture**: Clean separation of concerns with professional code structure
- 🔒 **Robust Error Handling**: Comprehensive error messages and debugging capabilities
- 📊 **Performance Optimized**: Dynamic chunk sizing, batch processing, and memory monitoring
- 🎯 **SEO & Metadata**: Well-structured codebase with clear documentation
- 🔄 **Session Management**: Intelligent state management across Streamlit reruns

---

## 💻 Technology Stack

### Frontend & Framework
- **Streamlit** (`streamlit>=1.28.0`) - Modern web application framework for Python
- **Streamlit Pages** - Multi-page application architecture

### AI & Machine Learning
- **Ollama** - Local LLM server and model management
- **LangChain** (`langchain>=0.1.0`) - LLM application framework
  - `langchain-ollama` - Ollama integration for LLMs and embeddings
  - `langchain-community` - Community integrations
  - `langchain-chroma` - ChromaDB vector store integration
  - `langchain-text-splitters` - Intelligent text chunking
  - `langchain-core` - Core LangChain components

### Vector Database & Embeddings
- **ChromaDB** (`chromadb>=0.4.15`) - Embedding database for vector search
- **Vector Embeddings** - Semantic text representation using Ollama embedding models

### Utilities & Dependencies
- **Requests** (`requests>=2.31.0`) - HTTP library for Ollama API calls
- **psutil** (`psutil>=5.9.0`) - System and process utilities for memory monitoring

### Development & Architecture
- **Python 3.10+** - Modern Python with type hints
- **Modular Design** - Separation of concerns (UI, business logic, utilities)
- **Configuration Management** - Centralized config for easy customization

---

## 📋 Prerequisites

Before installing, ensure you have:

1. **Python 3.10 or higher** installed
2. **Ollama** installed and running locally
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   ```
3. **Ollama Models** pulled (at least one LLM and one embedding model)
   ```bash
   # Pull a language model
   ollama pull mistral
   
   # Pull an embedding model
   ollama pull nomic-embed-text
   ```

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd shivani-ai
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv shivani-env
source shivani-env/bin/activate  # On Windows: shivani-env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

Ensure Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

---

## 🎯 Usage Guide

### Starting the Application

**Option 1: Direct Streamlit Command**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run src/app.py
```

**Option 2: Development Mode**
```bash
pip install -e .
streamlit run src/app.py
```

The application will open in your default browser at `http://localhost:8501`

### Creating Your First Knowledge Base

1. **Select a Profile**: Choose or create a profile (e.g., "Default", "Ravi Kumar Pundir")
2. **Enter Your Content**: Paste or type text into the "RAG Text Box"
3. **Create Knowledge Base**: Click "🔨 Create Knowledge Base"
4. **Wait for Processing**: The system will:
   - Split your text into semantic chunks
   - Generate vector embeddings using Ollama
   - Store embeddings in ChromaDB
   - Register the knowledge base
5. **Query Your Knowledge**: Enter questions in the "Query Text Box" and get context-aware answers

### Managing Profiles

1. Click "👤 Manage Profiles" button on the main page
2. Create new profiles with descriptive names
3. View knowledge base counts per profile
4. Delete profiles and their associated knowledge bases

### Managing Knowledge Bases

1. Click "📚 Manage Knowledge Bases" in the sidebar
2. View all knowledge bases across all profiles
3. Load specific knowledge bases
4. Delete knowledge bases you no longer need
5. See detailed information (chunks, creation date, preview)

### Querying the System

- **With Knowledge Base**: Responses include context from your stored knowledge
- **Without Knowledge Base**: Direct LLM responses (still powered by Ollama)
- **Multi-Profile**: Select multiple profiles to search across all their knowledge bases

---

## 🏗️ Project Structure

```
shivani-ai/
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── app.py                                # Main Streamlit application & UI
│   ├── config.py                             # Configuration constants
│   │
│   ├── pages/                                # Streamlit multi-page application
│   │   ├── 1_📚_Manage_Knowledge_Bases.py    # Knowledge base management page
│   │   └── 2_👤_Manage_Profiles.py            # Profile management page
│   │
│   ├── rag/                                  # RAG (Retrieval Augmented Generation) module
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py                   # RAG pipeline for querying with context
│   │   └── vectorstore_manager.py           # ChromaDB vector store management
│   │
│   └── utils/                                # Utility functions
│       ├── __init__.py
│       ├── kb_manager.py                     # Knowledge base & profile management
│       ├── ollama_utils.py                   # Ollama API utilities
│       ├── persistence_utils.py              # Knowledge base persistence utilities
│       └── performance_utils.py              # Performance monitoring & optimization
│
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Git ignore patterns
└── README.md                                 # This file
```

### Architecture Overview

**Separation of Concerns:**
- **UI Layer** (`app.py`, `pages/`): Streamlit interface and user interactions
- **Business Logic** (`rag/`): RAG pipeline and vector store operations
- **Utilities** (`utils/`): Reusable helper functions
- **Configuration** (`config.py`): Centralized settings

---

## ⚙️ Configuration

Default settings can be customized in `src/config.py`:

```python
# Ollama Configuration
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "mistral"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Vector Database
CHROMA_DB_DIR = ".chroma_db"  # Storage location

# Text Processing
CHUNK_SIZE = 2000             # Characters per chunk
CHUNK_OVERLAP = 400          # Overlap between chunks

# Retrieval Settings
RETRIEVAL_K = 10             # Documents retrieved per query
MAX_CONTEXT_LENGTH = 8000    # Maximum context for LLM
```

---

## 📊 How It Works

### RAG (Retrieval Augmented Generation) Pipeline

1. **Text Input**: User provides text content
2. **Text Splitting**: Content is intelligently split into semantic chunks
3. **Embedding Generation**: Each chunk is converted to a vector embedding using Ollama
4. **Vector Storage**: Embeddings stored in ChromaDB for fast similarity search
5. **Query Processing**: User questions trigger semantic search across stored embeddings
6. **Context Retrieval**: Most relevant chunks retrieved based on semantic similarity
7. **Response Generation**: LLM generates answer using retrieved context + question
8. **Response Display**: Context-aware answer shown to user with source citations

### Multi-Profile Architecture

- Each profile has its own directory under `.chroma_db/{profile_id}/`
- Knowledge bases are stored with unique hash-based identifiers
- Multiple profiles can be queried simultaneously
- Cross-profile retrieval combines results from all selected profiles

---

## 🔍 SEO Keywords & Technical Terms

This application leverages:
- **Retrieval Augmented Generation (RAG)**
- **Large Language Models (LLMs)**
- **Vector Embeddings**
- **Semantic Search**
- **Knowledge Base Management**
- **Local AI/LLM**
- **Offline AI Assistant**
- **Vector Database**
- **Text Chunking**
- **Context-Aware AI**
- **Private AI**
- **Self-Hosted AI**

---

## 🛠️ Troubleshooting

### Common Issues

**"Cannot connect to Ollama"**
- Ensure Ollama is running: `ollama serve`
- Check Ollama is accessible at `http://localhost:11434`

**"Model not found"**
- Pull the required model: `ollama pull <model-name>`
- Verify with: `ollama list`

**"Permission denied" errors**
- Ensure write permissions on `.chroma_db/` directory
- Check disk space availability

**"Knowledge base not loading"**
- Verify the knowledge base directory exists in `.chroma_db/`
- Check file permissions on ChromaDB files

---

## 📝 Notes

- Knowledge bases are stored locally in `.chroma_db/` (not tracked in git)
- Each knowledge base is identified by an MD5 hash of its content
- Profiles allow organization of knowledge bases by user or context
- All data remains on your local machine—complete privacy
- The system automatically optimizes chunk sizes for large documents

---

## 🔒 Privacy & Security

- **100% Local Processing**: All AI operations happen on your machine
- **No External API Calls**: No data sent to cloud services (beyond Ollama local API)
- **Persistent Storage**: Knowledge bases stored in local file system
- **User-Controlled**: You own and control all your data

---

## 📈 Future Enhancements

Potential features for future versions:
- [ ] Document upload (PDF, DOCX, TXT)
- [ ] Real-time collaborative editing
- [ ] Advanced analytics and insights
- [ ] Integration with external data sources
- [ ] Web scraping capabilities
- [ ] Export/import knowledge bases
- [ ] Advanced search filters
- [ ] Custom prompt templates

---

## 📄 License

[Specify your license here]

---

## 👥 Contributing

[Add contribution guidelines if applicable]

---

## 📧 Support

For issues, questions, or contributions, please contact:

- **Email**: priorcoder@gmail.com
- **LinkedIn**: [Ravi Kumar Pundir](https://www.linkedin.com/in/ravikumarpundir/)

---

**Built with ❤️ using Streamlit, Ollama, LangChain, and ChromaDB**
