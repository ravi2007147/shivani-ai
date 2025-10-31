"""Knowledge Base Management Page."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
)
from src.utils import fetch_ollama_models, get_default_model
from src.utils.kb_manager import KnowledgeBaseManager, ProfileManager
from src.rag import VectorStoreManager, RAGPipeline


# Page configuration
st.set_page_config(
    page_title="Manage Knowledge Bases",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Knowledge Base Management")
st.markdown("View, load, update, and delete your saved knowledge bases.")

# Initialize profile manager
if 'profile_manager' not in st.session_state:
    st.session_state.profile_manager = ProfileManager()

profile_manager = st.session_state.profile_manager
all_profiles = profile_manager.load_profiles()

# If no profiles, create default
if not all_profiles:
    profile_manager.create_profile("Default")
    all_profiles = profile_manager.load_profiles()

# Profile selection for KB management (multi-select)
profile_options = {p.get('name'): p.get('id') for p in all_profiles}
profile_names = list(profile_options.keys())

if profile_names:
    selected_profile_names = st.multiselect(
        "👤 Select Profile(s) to View KBs",
        options=profile_names,
        default=profile_names if len(profile_names) <= 3 else profile_names[:1],  # Default: all if <=3, else first one
        help="Select one or more profiles to view their knowledge bases. Leave empty to see all."
    )
    
    # Determine which profile(s) to show KBs from
    if selected_profile_names:
        profile_ids_to_show = [profile_options[name] for name in selected_profile_names]
    else:
        # Show all if none selected
        profile_ids_to_show = [p.get('id') for p in all_profiles]
else:
    profile_ids_to_show = ["default"]

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama API endpoint"
    )
    
    # Auto-load models on first load
    if 'ollama_models' not in st.session_state:
        st.session_state.ollama_models = []
    
    if not st.session_state.ollama_models:
        with st.spinner("Loading models..."):
            st.session_state.ollama_models = fetch_ollama_models(ollama_base_url)
    
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


def load_knowledge_base(kb_id: str, profile_id: str) -> tuple[bool, str]:
    """Load a knowledge base by its ID from a specific profile.
    
    Args:
        kb_id: Knowledge base ID
        profile_id: Profile ID that owns this KB
        
    Returns:
        Tuple of (success, message)
    """
    # Get KB manager for this profile
    kb_manager = KnowledgeBaseManager(profile_id=profile_id)
    kb_data = kb_manager.get_knowledge_base(kb_id)
    
    if not kb_data:
        return False, f"Knowledge base '{kb_id}' not found"
    
    persist_dir = kb_data.get("persist_dir")
    if not persist_dir:
        return False, f"Knowledge base directory not found"
    
    try:
        # Initialize vectorstore manager with profile
        manager = VectorStoreManager(profile_id=profile_id)
        
        # Load vectorstore
        vectorstore = manager.load_vectorstore(
            persist_dir,
            embedding_model,
            ollama_base_url,
        )
        
        # Create RAG pipeline
        rag_pipeline = RAGPipeline(
            [vectorstore],
            ollama_model,
            ollama_base_url,
        )
        
        # Save in session state (accessible across pages)
        st.session_state.vectorstore = vectorstore
        st.session_state.vectorstores = [vectorstore]
        st.session_state.retriever = rag_pipeline.retriever
        st.session_state.llm = rag_pipeline.llm
        st.session_state.prompt_template = rag_pipeline.prompt_template
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.kb_persist_dir = persist_dir
        st.session_state.current_kb_id = f"{profile_id}:{kb_id}"
        st.session_state.rag_text = kb_data.get("text_preview", "")
        st.session_state.knowledge_base_created = True
        
        return True, f"✅ Loaded knowledge base '{kb_id}' successfully"
    except Exception as e:
        return False, f"❌ Failed to load knowledge base: {str(e)}"


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📖 All Knowledge Bases")
    
    # Refresh button
    if st.button("🔄 Refresh List"):
        # Cleanup orphaned metadata for all profiles
        for profile_id in profile_ids_to_show:
            kb_manager = KnowledgeBaseManager(profile_id=profile_id)
            kb_manager.cleanup_orphaned_metadata()
        st.rerun()
    
    # Get KBs from selected profile(s) grouped by profile
    kb_by_profile = {}
    for profile_id in profile_ids_to_show:
        kb_manager = KnowledgeBaseManager(profile_id=profile_id)
        profile_kbs = kb_manager.list_knowledge_bases()
        if profile_kbs:
            profile_name = next((p.get('name') for p in all_profiles if p.get('id') == profile_id), profile_id)
            # Add profile info to each KB
            for kb in profile_kbs:
                kb['profile_id'] = profile_id
                kb['profile_name'] = profile_name
            kb_by_profile[profile_name] = profile_kbs
    
    # Flatten list for total count
    kb_list = []
    for profile_kbs in kb_by_profile.values():
        kb_list.extend(profile_kbs)
    
    if kb_list:
        total_kbs = len(kb_list)
        total_profiles = len(kb_by_profile)
        st.info(f"Found {total_kbs} knowledge base(s) across {total_profiles} profile(s)")
        
        # Display current KB if any
        if st.session_state.get('knowledge_base_created') and st.session_state.get('current_kb_id'):
            current_kb_id = st.session_state.current_kb_id
            # Handle both old format (just kb_id) and new format (profile_id:kb_id)
            if ':' in current_kb_id:
                profile_part, kb_part = current_kb_id.split(':', 1)
                current_kb = next((kb for kb in kb_list if kb.get("profile_id") == profile_part and kb["id"] == kb_part), None)
            else:
                current_kb = next((kb for kb in kb_list if kb["id"] == current_kb_id), None)
            
            if current_kb:
                st.success(f"**Current Active:** {current_kb.get('profile_name', 'Unknown')} - {current_kb['id']}")
        
        # List knowledge bases grouped by profile
        for profile_name, profile_kbs in kb_by_profile.items():
            with st.expander(f"👤 **{profile_name}** ({len(profile_kbs)} KB(s))", expanded=len(kb_by_profile) <= 3):
                for kb in profile_kbs:
                    with st.container():
                        st.markdown("---")
                        
                        # KB header with status
                        current_kb_id = st.session_state.get('current_kb_id', '')
                        # Handle both old format (just kb_id) and new format (profile_id:kb_id)
                        if ':' in current_kb_id:
                            profile_part, kb_part = current_kb_id.split(':', 1)
                            is_current = (kb.get("profile_id") == profile_part and kb["id"] == kb_part)
                        else:
                            is_current = (kb["id"] == current_kb_id)
                        
                        status_badge = "🟢 **ACTIVE**" if is_current else "⚪ Inactive"
                        
                        col_header1, col_header2 = st.columns([3, 1])
                        with col_header1:
                            st.markdown(f"#### {kb['id']} {status_badge}")
                        with col_header2:
                            st.caption(f"{kb['chunk_count']} chunks")
                        
                        # KB details - displayed directly (nested expanders not allowed in Streamlit)
                        show_details = st.checkbox("📄 Show Details", key=f"details_{kb['id']}_{kb.get('profile_id', '')}")
                        if show_details:
                            st.write(f"**Text Preview:**")
                            st.text_area(
                                "Preview",
                                value=kb['text_preview'],
                                height=150,
                                disabled=True,
                                key=f"preview_{kb['id']}_{kb.get('profile_id', '')}"
                            )
                            
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                if kb.get('created_at'):
                                    st.caption(f"📅 Created: {kb['created_at'][:19]}")
                            with col_info2:
                                if kb.get('updated_at'):
                                    st.caption(f"📅 Updated: {kb['updated_at'][:19]}")
                            
                            st.caption(f"📁 Location: {kb.get('persist_dir', 'N/A')}")
                        
                        # Action buttons
                        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                        
                        with col_btn1:
                            if st.button("📂 Load", key=f"load_{kb['id']}_{kb.get('profile_id', '')}", use_container_width=True):
                                profile_id = kb.get('profile_id', 'default')
                                success, message = load_knowledge_base(kb['id'], profile_id)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        with col_btn2:
                            if st.button("🗑️ Delete", key=f"delete_{kb['id']}_{kb.get('profile_id', '')}", type="secondary", use_container_width=True):
                                profile_id = kb.get('profile_id', 'default')
                                kb_manager = KnowledgeBaseManager(profile_id=profile_id)
                                success, msg = kb_manager.delete_knowledge_base(kb['id'])
                                if success:
                                    st.success(msg)
                                    # If this was the current KB, clear session state
                                    current_kb_id = st.session_state.get('current_kb_id', '')
                                    kb_full_id = f"{kb.get('profile_id', '')}:{kb['id']}"
                                    if current_kb_id == kb['id'] or current_kb_id == kb_full_id:
                                        st.session_state.vectorstore = None
                                        st.session_state.vectorstores = []
                                        st.session_state.retriever = None
                                        st.session_state.rag_pipeline = None
                                        st.session_state.kb_persist_dir = None
                                        st.session_state.current_kb_id = None
                                        st.session_state.knowledge_base_created = False
                                        st.session_state.rag_text = ""
                                    st.rerun()
                                else:
                                    st.error(msg)
                        
                        with col_btn3:
                            if is_current:
                                if st.button("🔄 Unload", key=f"unload_{kb['id']}_{kb.get('profile_id', '')}", use_container_width=True):
                                    st.session_state.vectorstore = None
                                    st.session_state.vectorstores = []
                                    st.session_state.retriever = None
                                    st.session_state.rag_pipeline = None
                                    st.session_state.kb_persist_dir = None
                                    st.session_state.current_kb_id = None
                                    st.session_state.knowledge_base_created = False
                                    st.success("✅ Knowledge base unloaded (still saved on disk)")
                                    st.rerun()
                            else:
                                st.empty()
                        
                        st.markdown("---")  # Separator between KBs in same profile
    else:
        st.info("No knowledge bases found. Create one on the main page.")
        st.markdown("---")
        if st.button("🏠 Go to Main Page"):
            st.switch_page("src/app.py")

with col2:
    st.header("ℹ️ Information")
    
    if kb_list:
        st.metric("Total Knowledge Bases", len(kb_list))
        
        total_chunks = sum(kb['chunk_count'] for kb in kb_list)
        st.metric("Total Chunks", total_chunks)
        
        if st.session_state.get('knowledge_base_created'):
            st.success("✅ Knowledge base is currently active")
            st.info(f"Active KB: **{st.session_state.get('current_kb_id', 'None')}**")
        else:
            st.info("ℹ️ No knowledge base is currently active")
            st.info("💡 Load a knowledge base to use it for queries")
    else:
        st.info("No knowledge bases available")
    
    st.markdown("---")
    st.markdown("### 📝 Actions")
    
    if st.button("🏠 Go to Main Page", use_container_width=True):
        st.switch_page("src/app.py")
    
    if st.button("🔄 Refresh Metadata", use_container_width=True):
        # Cleanup orphaned metadata for all profiles
        for profile_id in profile_ids_to_show:
            kb_manager = KnowledgeBaseManager(profile_id=profile_id)
            kb_manager.cleanup_orphaned_metadata()
        st.success("✅ Metadata refreshed")
        st.rerun()

