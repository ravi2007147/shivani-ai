"""Knowledge Base Management Page."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.kb_manager import KnowledgeBaseManager, ProfileManager


# Page configuration
st.set_page_config(
    page_title="Manage Knowledge Bases",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Knowledge Base Management")
st.markdown("View and delete your saved knowledge bases. Knowledge bases are automatically loaded based on selected profiles in the Query tab.")

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
        
        # List knowledge bases grouped by profile
        for profile_name, profile_kbs in kb_by_profile.items():
            with st.expander(f"👤 **{profile_name}** ({len(profile_kbs)} KB(s))", expanded=len(kb_by_profile) <= 3):
                for kb in profile_kbs:
                    with st.container():
                        st.markdown("---")
                        
                        # KB header
                        col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
                        with col_header1:
                            st.markdown(f"#### {kb['id']}")
                        with col_header2:
                            st.caption(f"{kb['chunk_count']} chunks")
                        with col_header3:
                            if kb.get('title'):
                                st.caption(f"📄 {kb['title']}")
                        
                        # KB details - displayed directly
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
                            
                            # Show PDF metadata if available
                            if kb.get('pdf_metadata'):
                                st.markdown("**PDF Metadata:**")
                                pdf_meta = kb['pdf_metadata']
                                if pdf_meta.get('Title'):
                                    st.caption(f"📖 Title: {pdf_meta['Title']}")
                                if pdf_meta.get('Author'):
                                    st.caption(f"✍️ Author: {pdf_meta['Author']}")
                                if pdf_meta.get('Subject'):
                                    st.caption(f"📋 Subject: {pdf_meta['Subject']}")
                            
                            st.caption(f"📁 Location: {kb.get('persist_dir', 'N/A')}")
                        
                        # Action buttons
                        if st.button("🗑️ Delete", key=f"delete_{kb['id']}_{kb.get('profile_id', '')}", type="secondary", use_container_width=True):
                            profile_id = kb.get('profile_id', 'default')
                            kb_manager = KnowledgeBaseManager(profile_id=profile_id)
                            success, msg = kb_manager.delete_knowledge_base(kb['id'])
                            if success:
                                st.success(msg)
                                # If this KB is currently loaded, remove it from session
                                active_kb_ids = st.session_state.get('active_kb_ids', [])
                                kb_full_id = f"{profile_id}:{kb['id']}"
                                if kb_full_id in active_kb_ids:
                                    active_kb_ids.remove(kb_full_id)
                                    st.session_state.active_kb_ids = active_kb_ids
                                    
                                    # If no KBs left, clear everything
                                    if not active_kb_ids:
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
            active_count = len(st.session_state.get('active_kb_ids', []))
            st.success(f"✅ {active_count} knowledge base(s) active for queries")
        else:
            st.info("💡 Knowledge bases are automatically loaded in the Query tab based on selected profiles")
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

