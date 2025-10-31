"""Profile Management Page."""

import streamlit as st
import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import CHROMA_DB_DIR
from src.utils.kb_manager import KnowledgeBaseManager


# Page configuration
st.set_page_config(
    page_title="Manage Profiles",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Profile Management")
st.markdown("Create and manage profiles. Each profile has its own knowledge bases.")


class ProfileManager:
    """Manages user profiles and their knowledge bases."""
    
    def __init__(self):
        """Initialize the profile manager."""
        self.profiles_file = os.path.join(CHROMA_DB_DIR, "profiles.json")
        self._ensure_profiles_file()
    
    def _ensure_profiles_file(self):
        """Ensure the profiles file exists."""
        os.makedirs(os.path.dirname(self.profiles_file), exist_ok=True)
        if not os.path.exists(self.profiles_file):
            with open(self.profiles_file, 'w') as f:
                json.dump({"profiles": []}, f, indent=2)
    
    def load_profiles(self) -> list:
        """Load all profiles from file.
        
        Returns:
            List of profile dictionaries
        """
        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                return data.get("profiles", [])
        except Exception:
            return []
    
    def save_profiles(self, profiles: list):
        """Save profiles to file.
        
        Args:
            profiles: List of profile dictionaries
        """
        try:
            with open(self.profiles_file, 'w') as f:
                json.dump({"profiles": profiles}, f, indent=2)
        except Exception as e:
            st.error(f"Error saving profiles: {str(e)}")
    
    def create_profile(self, name: str) -> tuple[bool, str]:
        """Create a new profile.
        
        Args:
            name: Profile name
            
        Returns:
            Tuple of (success, message)
        """
        if not name or not name.strip():
            return False, "Profile name cannot be empty"
        
        name = name.strip()
        
        profiles = self.load_profiles()
        
        # Check if profile already exists
        if any(p.get("name") == name for p in profiles):
            return False, f"Profile '{name}' already exists"
        
        # Create new profile ID (sanitize for directory name)
        import re
        profile_id = re.sub(r'[^a-z0-9_]', '_', name.lower().replace(" ", "_"))
        # Ensure it doesn't start with a number
        if profile_id and profile_id[0].isdigit():
            profile_id = "profile_" + profile_id
        
        from datetime import datetime
        new_profile = {
            "id": profile_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
        }
        
        profiles.append(new_profile)
        self.save_profiles(profiles)
        
        return True, f"✅ Profile '{name}' created successfully"
    
    def delete_profile(self, profile_id: str) -> tuple[bool, str]:
        """Delete a profile and its knowledge bases.
        
        Args:
            profile_id: Profile ID to delete
            
        Returns:
            Tuple of (success, message)
        """
        profiles = self.load_profiles()
        
        # Find and remove profile
        profile = next((p for p in profiles if p.get("id") == profile_id), None)
        if not profile:
            return False, f"Profile '{profile_id}' not found"
        
        profile_name = profile.get("name")
        profiles = [p for p in profiles if p.get("id") != profile_id]
        self.save_profiles(profiles)
        
        # Delete profile's knowledge bases directory
        profile_dir = os.path.join(CHROMA_DB_DIR, profile_id)
        if os.path.exists(profile_dir):
            try:
                import shutil
                shutil.rmtree(profile_dir)
            except Exception as e:
                return True, f"✅ Profile '{profile_name}' deleted, but some files could not be removed: {str(e)}"
        
        return True, f"✅ Profile '{profile_name}' and all its knowledge bases deleted"
    
    def get_profile(self, profile_id: str) -> dict:
        """Get a profile by ID.
        
        Args:
            profile_id: Profile ID
            
        Returns:
            Profile dictionary or None
        """
        profiles = self.load_profiles()
        return next((p for p in profiles if p.get("id") == profile_id), None)


# Initialize profile manager
if 'profile_manager' not in st.session_state:
    st.session_state.profile_manager = ProfileManager()

profile_manager = st.session_state.profile_manager

# Sidebar
with st.sidebar:
    st.header("Actions")
    
    if st.button("🏠 Go to Main Page", use_container_width=True):
        st.switch_page("src/app.py")
    
    if st.button("📚 Go to Knowledge Bases", use_container_width=True):
        st.switch_page("pages/1_📚_Manage_Knowledge_Bases.py")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 All Profiles")
    
    profiles = profile_manager.load_profiles()
    
    if profiles:
        st.info(f"Found {len(profiles)} profile(s)")
        
        for profile in profiles:
            with st.container():
                st.markdown("---")
                
                col_name, col_actions = st.columns([3, 1])
                
                with col_name:
                    st.markdown(f"### 👤 {profile.get('name', 'Unknown')}")
                    st.caption(f"ID: {profile.get('id', 'N/A')}")
                    if profile.get('created_at'):
                        st.caption(f"Created: {profile.get('created_at')}")
                
                with col_actions:
                    if st.button("🗑️ Delete", key=f"delete_{profile.get('id')}", type="secondary"):
                        success, message = profile_manager.delete_profile(profile.get('id'))
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
    else:
        st.info("No profiles found. Create your first profile below.")
    
    st.markdown("---")
    
    # Create new profile
    st.header("➕ Create New Profile")
    
    with st.form("create_profile_form", clear_on_submit=True):
        profile_name = st.text_input(
            "Profile Name",
            placeholder="e.g., Ravi Kumar Pundir",
            help="Enter a unique name for this profile"
        )
        
        submit = st.form_submit_button("Create Profile", type="primary", use_container_width=True)
        
        if submit:
            if profile_name:
                success, message = profile_manager.create_profile(profile_name)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter a profile name")

with col2:
    st.header("ℹ️ Information")
    
    if profiles:
        st.metric("Total Profiles", len(profiles))
        
        # Count knowledge bases per profile
        kb_counts = {}
        for profile in profiles:
            profile_id = profile.get('id')
            profile_dir = os.path.join(CHROMA_DB_DIR, profile_id)
            if os.path.exists(profile_dir):
                kb_manager = KnowledgeBaseManager(profile_id=profile_id)
                kb_list = kb_manager.list_knowledge_bases()
                kb_counts[profile_id] = len(kb_list)
            else:
                kb_counts[profile_id] = 0
        
        total_kbs = sum(kb_counts.values())
        st.metric("Total Knowledge Bases", total_kbs)
        
        with st.expander("Knowledge Bases per Profile"):
            for profile in profiles:
                profile_id = profile.get('id')
                kb_count = kb_counts.get(profile_id, 0)
                st.write(f"• {profile.get('name')}: {kb_count} KB(s)")
    else:
        st.info("No profiles available")
    
    st.markdown("---")
    st.markdown("### 💡 How it works")
    st.markdown("""
    1. Create profiles for different users or contexts
    2. Each profile has its own knowledge bases
    3. Select profile(s) on the main page to query their knowledge bases
    4. Select multiple profiles to search across all of them
    """)

