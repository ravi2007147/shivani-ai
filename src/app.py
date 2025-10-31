"""Streamlit application for RAG with Ollama."""

import streamlit as st
import os
import shutil
import sys
import time
from pathlib import Path

from langchain_ollama import OllamaLLM

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
)
from src.utils import fetch_ollama_models, get_default_model, find_persisted_knowledge_base
from src.utils.kb_manager import KnowledgeBaseManager
from src.rag import VectorStoreManager, RAGPipeline


# Page configuration
st.set_page_config(
    page_title="RAG with Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'vectorstore': None,
        'vectorstores': [],  # List of all active vectorstores
        'active_kb_ids': [],  # List of active KB IDs
        'retriever': None,
        'llm': None,
        'prompt_template': None,
        'qa_chain': None,
        'rag_text': "",
        'knowledge_base_created': False,
        'ollama_models': [],
        'kb_persist_dir': None,
        'current_kb_id': None,
        'rag_pipeline': None,
        'vectorstore_manager': None,
        'kb_manager': None,
        'selected_profiles': [],  # List of selected profile IDs
        'all_profiles': [],  # List of all available profiles
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Initialize profile manager
    from src.utils.kb_manager import ProfileManager
    if 'profile_manager' not in st.session_state:
        st.session_state.profile_manager = ProfileManager()
    
    # Load profiles
    if not st.session_state.all_profiles:
        st.session_state.all_profiles = st.session_state.profile_manager.load_profiles()


def load_knowledge_base_by_id(
    kb_id: str,
    embedding_model: str,
    ollama_model: str,
    ollama_base_url: str,
) -> tuple[bool, str]:
    """Load a knowledge base by its ID.
    
    Args:
        kb_id: Knowledge base ID
        embedding_model: Name of embedding model to use
        ollama_model: Name of LLM model to use
        ollama_base_url: Base URL for Ollama API
        
    Returns:
        Tuple of (success, message)
    """
    kb_manager = st.session_state.kb_manager
    kb_data = kb_manager.get_knowledge_base(kb_id)
    
    if not kb_data:
        return False, f"Knowledge base '{kb_id}' not found"
    
    persist_dir = kb_data.get("persist_dir")
    if not persist_dir or not os.path.exists(persist_dir):
        return False, f"Knowledge base directory not found: {persist_dir}"
    
    success = load_persisted_knowledge_base(
        persist_dir,
        embedding_model,
        ollama_model,
        ollama_base_url,
    )
    
    if success:
        return True, f"✅ Loaded knowledge base '{kb_id}' successfully"
    else:
        return False, f"❌ Failed to load knowledge base '{kb_id}'"


def load_all_knowledge_bases(
    embedding_model: str,
    ollama_model: str,
    ollama_base_url: str,
    profile_ids: list = None,
) -> bool:
    """Load all persisted knowledge bases and make them active.
    
    Args:
        embedding_model: Name of embedding model to use
        ollama_model: Name of LLM model to use
        ollama_base_url: Base URL for Ollama API
        profile_ids: List of profile IDs to load KBs from (None = all)
        
    Returns:
        True if at least one loaded successfully, False otherwise
    """
    # Use selected profiles or all profiles
    if profile_ids is None:
        profile_ids = st.session_state.selected_profiles if st.session_state.selected_profiles else ["default"]
    
    loaded_vectorstores = []
    loaded_kb_ids = []
    manager = VectorStoreManager()
    
    # Load KBs from all selected profiles
    for profile_id in profile_ids:
        # Initialize KB manager for this profile
        kb_manager = KnowledgeBaseManager(profile_id=profile_id)
        kb_list = kb_manager.list_knowledge_bases()
        
        if not kb_list:
            continue
        
        # Load all knowledge bases from this profile
        for kb in kb_list:
            persist_dir = kb.get("persist_dir")
            kb_id = kb.get("id")
            
            if not persist_dir or not os.path.exists(persist_dir):
                continue
            
            if not os.listdir(persist_dir):
                continue
            
            try:
                # Load vectorstore (create manager with profile_id)
                profile_manager = VectorStoreManager(profile_id=profile_id)
                vectorstore = profile_manager.load_vectorstore(
                    persist_dir,
                    embedding_model,
                    ollama_base_url,
                )
                
                loaded_vectorstores.append(vectorstore)
                loaded_kb_ids.append(f"{profile_id}:{kb_id}")
            except Exception:
                # Skip this KB if it fails to load
                continue
    
    if not loaded_vectorstores:
        return False
    
    try:
        # Create RAG pipeline with all vectorstores
        rag_pipeline = RAGPipeline(
            loaded_vectorstores,
            ollama_model,
            ollama_base_url,
        )
        
        # Initialize LLM if not already initialized
        if st.session_state.llm is None:
            from langchain_ollama import OllamaLLM
            st.session_state.llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
            )
        
        # Save in session state
        st.session_state.vectorstores = loaded_vectorstores
        st.session_state.vectorstore = loaded_vectorstores[0]  # First one for compatibility
        st.session_state.retriever = rag_pipeline.retriever
        st.session_state.llm = rag_pipeline.llm
        st.session_state.prompt_template = rag_pipeline.prompt_template
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.active_kb_ids = loaded_kb_ids
        st.session_state.knowledge_base_created = True
        
        return True
    except Exception as e:
        st.warning(f"⚠️ Could not create RAG pipeline: {str(e)}")
        return False


def load_persisted_knowledge_base(
    persisted_kb_dir: str,
    embedding_model: str,
    ollama_model: str,
    ollama_base_url: str,
) -> bool:
    """Load a single persisted knowledge base and add to active set.
    
    Args:
        persisted_kb_dir: Directory path containing persisted knowledge base
        embedding_model: Name of embedding model to use
        ollama_model: Name of LLM model to use
        ollama_base_url: Base URL for Ollama API
        
    Returns:
        True if loaded successfully, False otherwise
    """
    if not persisted_kb_dir or not os.path.exists(persisted_kb_dir):
        return False
    
    if not os.listdir(persisted_kb_dir):
        # Empty directory, remove it
        try:
            shutil.rmtree(persisted_kb_dir)
        except Exception:
            pass
        return False
    
    try:
        # Initialize vectorstore manager
        manager = VectorStoreManager()
        
        # Load vectorstore
        vectorstore = manager.load_vectorstore(
            persisted_kb_dir,
            embedding_model,
            ollama_base_url,
        )
        
        # Get KB ID from directory name
        kb_id = os.path.basename(persisted_kb_dir)
        
        # If RAG pipeline exists, add this vectorstore to it
        if st.session_state.rag_pipeline:
            st.session_state.rag_pipeline.add_vectorstore(vectorstore)
            st.session_state.vectorstores.append(vectorstore)
            if kb_id not in st.session_state.active_kb_ids:
                st.session_state.active_kb_ids.append(kb_id)
        else:
            # Create new RAG pipeline with this vectorstore
            rag_pipeline = RAGPipeline(
                [vectorstore],
                ollama_model,
                ollama_base_url,
            )
            
            # Initialize LLM if not already initialized
            if st.session_state.llm is None:
                from langchain_ollama import OllamaLLM
                st.session_state.llm = OllamaLLM(
                    model=ollama_model,
                    base_url=ollama_base_url,
                )
            
            # Save in session state
            st.session_state.vectorstores = [vectorstore]
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = rag_pipeline.retriever
            st.session_state.llm = rag_pipeline.llm
            st.session_state.prompt_template = rag_pipeline.prompt_template
            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.active_kb_ids = [kb_id]
            st.session_state.knowledge_base_created = True
        
        return True
    except Exception as e:
        st.warning(f"⚠️ Could not load persisted knowledge base: {str(e)}")
        return False


def create_knowledge_base_with_progress(
    rag_input: str,
    embedding_model: str,
    ollama_model: str,
    ollama_base_url: str,
    profile_id: str = None,
    progress_callback=None,
    title: str = None,
    pdf_metadata: dict = None,
) -> tuple[bool, str, list]:
    """Create a new knowledge base from text with progress updates.
    
    Args:
        rag_input: Text to create knowledge base from
        embedding_model: Name of embedding model to use
        ollama_model: Name of LLM model to use
        ollama_base_url: Base URL for Ollama API
        profile_id: Profile ID to create KB under (None = use selected profile)
        progress_callback: Optional callback function(step, message) for progress updates
        title: Optional title for the knowledge base
        pdf_metadata: Optional PDF metadata dictionary (author, subject, etc.)
        
    Returns:
        Tuple of (success, message, progress_updates)
    """
    if progress_callback:
        progress_callback("Step 1/5", "Validating inputs...")
    
    # Use selected profile or default
    if profile_id is None:
        profile_id = st.session_state.selected_profiles[0] if st.session_state.selected_profiles else "default"
    
    try:
        # Validate inputs
        if not rag_input or not rag_input.strip():
            return False, "Error: RAG input text cannot be empty", []
        
        if not profile_id:
            return False, "Error: Profile ID is required", []
        
        if progress_callback:
            progress_callback("Step 2/5", f"Initializing vectorstore manager for profile: {profile_id}")
        
        # Initialize vectorstore manager with profile
        manager = VectorStoreManager(profile_id=profile_id)
        
        if progress_callback:
            progress_callback("Step 3/5", f"Creating embeddings using model: {embedding_model}")
            progress_callback("Step 3/5", "Connecting to Ollama embedding service...")
            time.sleep(0.1)  # Allow UI to update
        
        # Create vectorstore (this is where embeddings are created)
        # This is the critical step that calls Ollama API to generate embeddings
        try:
            vectorstore, persist_dir = manager.create_vectorstore(
                rag_input,
                embedding_model,
                ollama_base_url,
            )
            if progress_callback:
                progress_callback("Step 3/5", "✅ Embeddings created successfully!")
        except Exception as e:
            if progress_callback:
                progress_callback("Step 3/5", f"❌ Failed: {str(e)}")
            raise
        
        if progress_callback:
            progress_callback("Step 4/5", "Verifying persistence...")
        
        # Verify persistence
        is_valid, file_count = manager.verify_persistence(persist_dir)
        
        # Generate KB ID from persist directory name
        kb_id = os.path.basename(persist_dir)
        
        # Count chunks (accurate count from actual splitting)
        from src.utils.performance_utils import get_chunk_count_estimate, optimize_chunk_size
        
        text_length = len(rag_input)
        optimal_chunk_size, optimal_overlap = optimize_chunk_size(text_length)
        
        # Use optimized sizes for large texts
        if text_length > 50000:
            chunk_size, chunk_overlap = optimal_chunk_size, optimal_overlap
        else:
            from src.config import CHUNK_SIZE, CHUNK_OVERLAP
            chunk_size, chunk_overlap = CHUNK_SIZE, CHUNK_OVERLAP
        
        approx_chunks = get_chunk_count_estimate(text_length, chunk_size, chunk_overlap)
        
        if progress_callback:
            progress_callback("Step 5/5", f"Registering knowledge base: {kb_id}")
        
        # Register in KB manager for this profile
        kb_manager = KnowledgeBaseManager(profile_id=profile_id)
        kb_manager.register_knowledge_base(
            kb_id=kb_id,
            persist_dir=persist_dir,
            text_preview=rag_input,
            chunk_count=approx_chunks,
            title=title,
            pdf_metadata=pdf_metadata,
        )
        
        # Add to active knowledge bases (don't replace existing ones)
        # Use profile_id:kb_id format for consistency
        full_kb_id = f"{profile_id}:{kb_id}"
        
        if st.session_state.rag_pipeline:
            # Add to existing pipeline
            st.session_state.rag_pipeline.add_vectorstore(vectorstore)
            st.session_state.vectorstores.append(vectorstore)
            if full_kb_id not in st.session_state.active_kb_ids:
                st.session_state.active_kb_ids.append(full_kb_id)
        else:
            # Create new RAG pipeline with this vectorstore
            rag_pipeline = RAGPipeline(
                [vectorstore],
                ollama_model,
                ollama_base_url,
            )
            
            # Initialize LLM
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
            )
            
            # Save in session state
            st.session_state.vectorstores = [vectorstore]
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = rag_pipeline.retriever
            st.session_state.llm = llm
            st.session_state.prompt_template = rag_pipeline.prompt_template
            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.active_kb_ids = [full_kb_id]
            st.session_state.knowledge_base_created = True
        
        # Store KB info
        st.session_state.rag_text = rag_input
        st.session_state.kb_persist_dir = persist_dir
        
        # Calculate approximate size info
        from src.utils.performance_utils import estimate_tokens
        approx_tokens = estimate_tokens(rag_input)
        
        if is_valid:
            return True, f"✅ Knowledge base created successfully! (ID: {kb_id}, {file_count} files, {approx_chunks} chunks, ~{approx_tokens:,} tokens)", []
        else:
            return False, f"⚠️ Knowledge base created but persistence verification failed. No files found in {persist_dir}", []
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return False, f"Error creating knowledge base: {str(e)}\n\nTraceback:\n{error_trace}", []


def create_knowledge_base(
    rag_input: str,
    embedding_model: str,
    ollama_model: str,
    ollama_base_url: str,
    profile_id: str = None,
) -> tuple[bool, str]:
    """Create a new knowledge base from text.
    
    Args:
        rag_input: Text to create knowledge base from
        embedding_model: Name of embedding model to use
        ollama_model: Name of LLM model to use
        ollama_base_url: Base URL for Ollama API
        profile_id: Profile ID to create KB under (None = use selected profile)
        
    Returns:
        Tuple of (success, message)
    """
    # Use selected profile or default
    if profile_id is None:
        profile_id = st.session_state.selected_profiles[0] if st.session_state.selected_profiles else "default"
    
    try:
        # Initialize vectorstore manager with profile
        manager = VectorStoreManager(profile_id=profile_id)
        
        # Create vectorstore
        vectorstore, persist_dir = manager.create_vectorstore(
            rag_input,
            embedding_model,
            ollama_base_url,
        )
        
        # Verify persistence
        is_valid, file_count = manager.verify_persistence(persist_dir)
        
        # Generate KB ID from persist directory name
        kb_id = os.path.basename(persist_dir)
        
        # Count chunks (accurate count from actual splitting)
        from src.utils.performance_utils import get_chunk_count_estimate, optimize_chunk_size
        
        text_length = len(rag_input)
        optimal_chunk_size, optimal_overlap = optimize_chunk_size(text_length)
        
        # Use optimized sizes for large texts
        if text_length > 50000:
            chunk_size, chunk_overlap = optimal_chunk_size, optimal_overlap
        else:
            from src.config import CHUNK_SIZE, CHUNK_OVERLAP
            chunk_size, chunk_overlap = CHUNK_SIZE, CHUNK_OVERLAP
        
        approx_chunks = get_chunk_count_estimate(text_length, chunk_size, chunk_overlap)
        
        # Register in KB manager for this profile
        kb_manager = KnowledgeBaseManager(profile_id=profile_id)
        kb_manager.register_knowledge_base(
            kb_id=kb_id,
            persist_dir=persist_dir,
            text_preview=rag_input,
            chunk_count=approx_chunks,
        )
        
        # Add to active knowledge bases (don't replace existing ones)
        # Use profile_id:kb_id format for consistency
        full_kb_id = f"{profile_id}:{kb_id}"
        
        if st.session_state.rag_pipeline:
            # Add to existing pipeline
            st.session_state.rag_pipeline.add_vectorstore(vectorstore)
            st.session_state.vectorstores.append(vectorstore)
            if full_kb_id not in st.session_state.active_kb_ids:
                st.session_state.active_kb_ids.append(full_kb_id)
        else:
            # Create new RAG pipeline with this vectorstore
            rag_pipeline = RAGPipeline(
                [vectorstore],
                ollama_model,
                ollama_base_url,
            )
            
            # Initialize LLM
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
            )
            
            # Save in session state
            st.session_state.vectorstores = [vectorstore]
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = rag_pipeline.retriever
            st.session_state.llm = llm
            st.session_state.prompt_template = rag_pipeline.prompt_template
            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.active_kb_ids = [full_kb_id]
            st.session_state.knowledge_base_created = True
        
        # Store KB info
        st.session_state.rag_text = rag_input
        st.session_state.kb_persist_dir = persist_dir
        
        # Calculate approximate size info
        from src.utils.performance_utils import estimate_tokens
        approx_tokens = estimate_tokens(rag_input)
        
        if is_valid:
            return True, f"✅ Knowledge base created successfully! (ID: {kb_id}, {file_count} files, {approx_chunks} chunks, ~{approx_tokens:,} tokens)"
        else:
            return False, f"⚠️ Knowledge base created but persistence verification failed. No files found in {persist_dir}"
            
    except Exception as e:
        return False, f"Error creating knowledge base: {str(e)}"


def initialize_llm_for_direct_query(ollama_model: str, ollama_base_url: str) -> bool:
    """Initialize LLM for direct queries without RAG.
    
    Args:
        ollama_model: Name of LLM model to use
        ollama_base_url: Base URL for Ollama API
        
    Returns:
        True if successful, False otherwise
    """
    try:
        llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
        )
        
        st.session_state.llm = llm
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.prompt_template = None
        st.session_state.rag_pipeline = None
        st.session_state.rag_text = ""
        st.session_state.knowledge_base_created = False
        
        return True
    except Exception as e:
        return False


# Initialize session state
initialize_session_state()

# Try to load ALL persisted knowledge bases on startup (all KBs should be active)
if not st.session_state.knowledge_base_created and not st.session_state.vectorstores:
    # We'll load all KBs after config is available in sidebar
    pass

# Title
st.title("🤖 RAG with Ollama")
st.markdown("Create a knowledge base from text and query it using Ollama locally")

# Profile Selection (before sidebar to initialize profiles)
from src.utils.kb_manager import ProfileManager

if 'profile_manager' not in st.session_state:
    st.session_state.profile_manager = ProfileManager()

profile_manager = st.session_state.profile_manager
all_profiles = profile_manager.load_profiles()

# Default profile if none exists
if not all_profiles:
    # Create default profile
    profile_manager.create_profile("Default")
    all_profiles = profile_manager.load_profiles()

st.session_state.all_profiles = all_profiles

# Initialize profile selection for querying (multiple profiles)
# This will be set per tab
if 'query_profiles' not in st.session_state:
    if all_profiles:
        profile_options = {p.get('name'): p.get('id') for p in all_profiles}
        profile_names = list(profile_options.keys())
        st.session_state.query_profiles = [all_profiles[0].get('id')] if all_profiles else ["default"]
        st.session_state.selected_profiles = st.session_state.query_profiles  # For KB creation (single)
    else:
        st.session_state.query_profiles = ["default"]
        st.session_state.selected_profiles = ["default"]

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama API endpoint"
    )
    
    # Auto-load models on first load
    if not st.session_state.ollama_models:
        with st.spinner("Loading models..."):
            st.session_state.ollama_models = fetch_ollama_models(ollama_base_url)
    
    # Try to reload persisted knowledge base if it exists (after we have model config)
    if not st.session_state.knowledge_base_created and st.session_state.kb_persist_dir:
        persisted_kb_dir = st.session_state.kb_persist_dir
        if persisted_kb_dir and os.path.exists(persisted_kb_dir):
            # Verify write permissions
            test_file = os.path.join(persisted_kb_dir, ".write_test")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception:
                st.warning("⚠️ Cannot write to persisted directory. Skipping reload.")
                st.session_state.kb_persist_dir = None
                persisted_kb_dir = None
            
            if persisted_kb_dir:
                # Get model configurations (will be set below)
                # We need to wait for them to be set, so we'll handle loading after config is set
                pass
    
    # Manual refresh button
    if st.button("🔄 Refresh Models"):
        with st.spinner("Loading models..."):
            fetched_models = fetch_ollama_models(ollama_base_url)
            st.session_state.ollama_models = fetched_models
            if fetched_models:
                st.success(f"✅ Loaded {len(fetched_models)} model(s)")
            else:
                st.warning("⚠️ Could not fetch models. Make sure Ollama is running.")
    
    # LLM Model Selection
    if st.session_state.ollama_models:
        default_llm = get_default_model(st.session_state.ollama_models, DEFAULT_LLM_MODEL)
        ollama_model = st.selectbox(
            "Ollama LLM Model",
            options=st.session_state.ollama_models,
            index=st.session_state.ollama_models.index(default_llm) if default_llm in st.session_state.ollama_models else 0,
            help="The Ollama model to use for queries"
        )
    else:
        ollama_model = st.text_input(
            "Ollama LLM Model",
            value=DEFAULT_LLM_MODEL,
            help="The Ollama model to use for queries (models not loaded)"
        )
    
    # Embedding Model Selection
    if st.session_state.ollama_models:
        # Filter for embedding models
        embedding_models = [m for m in st.session_state.ollama_models if "embed" in m.lower()]
        if not embedding_models:
            embedding_models = st.session_state.ollama_models
        
        default_embedding = get_default_model(embedding_models, DEFAULT_EMBEDDING_MODEL)
        embedding_model = st.selectbox(
            "Embedding Model",
            options=embedding_models,
            index=embedding_models.index(default_embedding) if default_embedding in embedding_models else 0,
            help="The Ollama embedding model"
        )
    else:
        embedding_model = st.text_input(
            "Embedding Model",
            value=DEFAULT_EMBEDDING_MODEL,
            help="The Ollama embedding model (models not loaded)"
        )
    
    # Auto-load ALL knowledge bases from query profiles on startup (for query tab)
    # Use query_profiles if available, otherwise use selected_profiles
    query_profiles_for_loading = st.session_state.get('query_profiles', st.session_state.get('selected_profiles', ["default"]))
    if not st.session_state.knowledge_base_created and not st.session_state.vectorstores:
        # Load KBs from query profiles
        if query_profiles_for_loading:
            if load_all_knowledge_bases(
                embedding_model,
                ollama_model,
                ollama_base_url,
                profile_ids=query_profiles_for_loading,
            ):
                profile_names = [
                    next((p.get('name') for p in st.session_state.all_profiles if p.get('id') == pid), pid)
                    for pid in query_profiles_for_loading
                ]
                st.success(f"✅ Loaded {len(st.session_state.active_kb_ids)} knowledge base(s) from {len(query_profiles_for_loading)} profile(s) - {', '.join(profile_names)}")

    st.markdown("---")
    
    # Quick Knowledge Base Status (from selected profiles)
    if st.session_state.selected_profiles:
        all_kb_list = []
        for profile_id in st.session_state.selected_profiles:
            kb_manager = KnowledgeBaseManager(profile_id=profile_id)
            kb_list = kb_manager.list_knowledge_bases()
            all_kb_list.extend(kb_list)
        
        kb_list = all_kb_list
    else:
        kb_list = []
    
    if kb_list:
        active_count = len(st.session_state.active_kb_ids) if st.session_state.knowledge_base_created else 0
        total_chunks = sum(kb.get('chunk_count', 0) for kb in kb_list)
        
        st.info(f"📚 {len(kb_list)} knowledge base(s) saved ({total_chunks:,} total chunks)")
        
        # Display active KBs
        if st.session_state.knowledge_base_created and st.session_state.active_kb_ids:
            st.success(f"✅ **{active_count} Active** - All knowledge bases are active for queries")
            with st.expander("View Active Knowledge Bases"):
                for kb_id in st.session_state.active_kb_ids:
                    kb_info = next((kb for kb in kb_list if kb['id'] == kb_id), None)
                    if kb_info:
                        st.write(f"• {kb_id} ({kb_info.get('chunk_count', 0)} chunks)")
        else:
            st.info("💡 All saved knowledge bases will be automatically loaded and active")
        
        # Performance info for large KBs
        if total_chunks > 100:
            from src.utils.performance_utils import check_memory_usage
            mem_info = check_memory_usage()
            if mem_info['rss_mb'] > 0:
                st.caption(f"💾 Memory: {mem_info['rss_mb']:.1f} MB ({mem_info['percent']:.1f}%)")
    
    # Link to management page
    if st.button("📚 Manage Knowledge Bases"):
        st.switch_page("pages/1_📚_Manage_Knowledge_Bases.py")
    
    st.markdown("---")
    
    # Current Knowledge Base Clear
    if st.session_state.knowledge_base_created:
        if st.button("Clear Current Knowledge Base"):
            # Clear session state but don't delete from disk
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.rag_pipeline = None
            st.session_state.kb_persist_dir = None
            st.session_state.current_kb_id = None
            st.session_state.knowledge_base_created = False
            st.session_state.rag_text = ""
            st.success("✅ Current knowledge base cleared (still saved on disk)")
            st.rerun()

# Main content area with Tabs
tab1, tab2 = st.tabs(["📝 Create Knowledge Base", "💬 Query"])

# Tab 1: Create Knowledge Base (Single Profile)
with tab1:
    st.header("📝 Create Knowledge Base")
    st.markdown("Select a profile and enter text to create a knowledge base. The knowledge base will be saved under the selected profile.")
    
    # Single profile selection for KB creation
    if all_profiles:
        profile_options = {p.get('name'): p.get('id') for p in all_profiles}
        profile_names = list(profile_options.keys())
        
        selected_profile_name = st.selectbox(
            "👤 Select Profile (for this knowledge base)",
            options=profile_names,
            index=0 if profile_names else None,
            help="Select ONE profile where this knowledge base will be saved."
        )
        
        # Store single profile for KB creation
        st.session_state.selected_profiles = [profile_options[selected_profile_name]] if selected_profile_name else ["default"]
    else:
        st.session_state.selected_profiles = ["default"]
    
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Manual Text Entry", "Upload PDF"],
        horizontal=True
    )
    
    kb_title = None
    kb_pdf_metadata = None
    rag_input = ""
    
    if input_method == "Manual Text Entry":
        rag_input = st.text_area(
            "RAG Text Box",
            height=500,
            value=st.session_state.rag_text,
            placeholder="Enter your text here to create a knowledge base...\n\nExample:\n\nArtificial intelligence (AI) is transforming industries across the globe. Machine learning algorithms can process vast amounts of data to make predictions and decisions. Natural language processing enables computers to understand and generate human language..."
        )
    else:  # PDF Upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to extract text and create a knowledge base"
        )
        
        if uploaded_file is not None:
            # Optional title field for PDF uploads
            kb_title = st.text_input(
                "Knowledge Base Title (optional)",
                placeholder="Enter a descriptive title for this knowledge base",
                help="Optional: Add a custom title to help identify this knowledge base"
            )
            
            # Process PDF button
            if st.button("📄 Extract Text from PDF", type="secondary"):
                try:
                    with st.spinner("Extracting and processing PDF content..."):
                        from src.utils.pdf_parser import process_pdf_content
                        extracted_text, pdf_metadata = process_pdf_content(uploaded_file, return_metadata=True)
                        
                        if extracted_text:
                            rag_input = extracted_text
                            kb_pdf_metadata = pdf_metadata
                            st.success(f"Successfully extracted {len(extracted_text)} characters from PDF")
                            
                            # Display metadata if available
                            if pdf_metadata:
                                st.info("PDF Metadata extracted:")
                                metadata_col1, metadata_col2 = st.columns(2)
                                with metadata_col1:
                                    if pdf_metadata.get('Title'):
                                        st.write(f"**Title:** {pdf_metadata['Title']}")
                                    if pdf_metadata.get('Author'):
                                        st.write(f"**Author:** {pdf_metadata['Author']}")
                                    if pdf_metadata.get('Subject'):
                                        st.write(f"**Subject:** {pdf_metadata['Subject']}")
                                with metadata_col2:
                                    if pdf_metadata.get('Producer'):
                                        st.write(f"**Producer:** {pdf_metadata['Producer']}")
                                    if pdf_metadata.get('Creator'):
                                        st.write(f"**Creator:** {pdf_metadata['Creator']}")
                            
                            st.info("Text preview:")
                            st.text_area(
                                "Extracted Text Preview",
                                value=extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
                                height=200,
                                disabled=True
                            )
                        else:
                            st.warning("No text could be extracted from the PDF file")
                except ImportError as e:
                    st.error(f"PDF processing library not installed: {str(e)}")
                    st.info("Install with: pip install pymupdf or pip install pypdf")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    if st.button("🔨 Create Knowledge Base", type="primary"):
        if not rag_input or rag_input.strip() == "":
            st.warning("⚠️ Please enter text to create a knowledge base.")
        else:
            try:
                # Get selected profile ID (single profile for KB creation)
                if not st.session_state.selected_profiles:
                    st.warning("⚠️ No profile selected. Using default profile.")
                    profile_id = "default"
                else:
                    profile_id = st.session_state.selected_profiles[0]
                
                # Show initial status
                status_text = st.empty()
                # Get profile name for display
                from src.utils.kb_manager import ProfileManager
                pm = ProfileManager()
                all_profiles_list = pm.load_profiles()
                profile_name = next((p.get('name') for p in all_profiles_list if p.get('id') == profile_id), profile_id)
                status_text.info(f"📝 Creating knowledge base for profile: **{profile_name}**")
                
                # Create knowledge base with progress updates
                success, message, progress_updates = create_knowledge_base_with_progress(
                    rag_input,
                    embedding_model,
                    ollama_model,
                    ollama_base_url,
                    profile_id=profile_id,
                    progress_callback=lambda step, msg: status_text.info(f"📝 {step}: {msg}"),
                    title=kb_title if kb_title else None,
                    pdf_metadata=kb_pdf_metadata
                )
                
                # Clear status
                status_text.empty()
                
                if success:
                    st.success(message)
                    # Clear the input box after successful creation
                    st.session_state.rag_text = ""
                    time.sleep(1)  # Give user time to see success message
                    st.rerun()
                else:
                    st.error(message)
                    st.info("Make sure Ollama is running locally and models are available.")
            except Exception as e:
                st.error(f"❌ Error creating knowledge base: {str(e)}")
                st.exception(e)
                import traceback
                st.code(traceback.format_exc())

# Tab 2: Query (Multiple Profiles)
with tab2:
    st.header("💬 Query")
    st.markdown("Select one or more profiles to search across their knowledge bases. Queries will retrieve information from all selected profiles.")
    
    # Multiple profile selection for querying
    if all_profiles:
        profile_options_query = {p.get('name'): p.get('id') for p in all_profiles}
        profile_names_query = list(profile_options_query.keys())
        
        # Initialize query profiles if not set
        if 'query_profiles' not in st.session_state:
            st.session_state.query_profiles = [all_profiles[0].get('id')] if all_profiles else ["default"]
        
        # Get current selection
        current_selected_names = [name for name, pid in profile_options_query.items() if pid in st.session_state.query_profiles]
        
        selected_profile_names_query = st.multiselect(
            "👤 Select Profile(s) to Query",
            options=profile_names_query,
            default=current_selected_names if current_selected_names else [profile_names_query[0]] if profile_names_query else [],
            help="Select one or more profiles. Queries will search across all selected profiles' knowledge bases."
        )
        
        # Convert names to IDs for querying
        st.session_state.query_profiles = [
            profile_options_query[name] for name in selected_profile_names_query
        ] if selected_profile_names_query else [all_profiles[0].get('id')]
        
        # Update selected_profiles for querying (this affects which KBs are loaded)
        st.session_state.selected_profiles = st.session_state.query_profiles
    else:
        st.session_state.query_profiles = ["default"]
        st.session_state.selected_profiles = ["default"]
    
    st.markdown("---")
    
    query_input = st.text_area(
        "Query Text Box",
        height=300,
        placeholder="Enter your question here... For example:\n\nWhat is artificial intelligence?\n\nWhat are the main concepts?\n\nExplain in detail...",
    )

    if st.button("🚀 Run Query", type="primary"):
        if not query_input or query_input.strip() == "":
            st.error("Please enter a query")
        else:
            with st.spinner("Processing query..."):
                try:
                    # Check if we have a knowledge base (RAG) or just LLM
                    if st.session_state.knowledge_base_created and st.session_state.rag_pipeline:
                        # Use RAG: Retrieve relevant documents
                        result = st.session_state.rag_pipeline.query_with_context(query_input)
                        
                        # Display answer
                        st.subheader("📋 Answer (with RAG context)")
                        st.markdown("---")
                        st.markdown(result["answer"])
                        st.markdown("---")

                        # Display source documents if available
                        if result.get("context_documents"):
                            with st.expander("📚 View Source Documents", expanded=False):
                                for i, doc in enumerate(result["context_documents"][:5], 1):
                                    st.markdown(f"**Source {i}:**")
                                    content = doc.page_content
                                    # Show more content in query tab
                                    preview_content = content[:800] + "..." if len(content) > 800 else content
                                    st.text_area(
                                        f"Content {i}",
                                        value=preview_content,
                                        height=150,
                                        disabled=True,
                                        key=f"source_{i}"
                                    )
                                    st.markdown("---")
                    else:
                        # Direct LLM query without RAG
                        if st.session_state.llm is None:
                            # Initialize LLM on the fly if not already initialized
                            st.session_state.llm = OllamaLLM(
                                model=ollama_model,
                                base_url=ollama_base_url,
                            )
                        
                        # Simple prompt for direct query
                        prompt = f"Question: {query_input}\n\nAnswer:"
                        answer = st.session_state.llm.invoke(prompt)
                        
                        # Display answer
                        st.subheader("📋 Answer (direct LLM response)")
                        st.markdown("---")
                        st.markdown(answer)
                        st.markdown("---")
                        st.info("💡 Tip: Create a knowledge base in the 'Create Knowledge Base' tab for context-aware responses.")

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.info("Make sure Ollama is running locally and the model is available.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Make sure Ollama is running locally. You can start it with: `ollama serve`")
