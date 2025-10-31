"""Knowledge base management utilities."""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.config import CHROMA_DB_DIR, KB_PREFIX


class ProfileManager:
    """Manages user profiles."""
    
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
        """Load all profiles from file."""
        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                return data.get("profiles", [])
        except Exception:
            return []
    
    def save_profiles(self, profiles: list):
        """Save profiles to file."""
        try:
            with open(self.profiles_file, 'w') as f:
                json.dump({"profiles": profiles}, f, indent=2)
        except Exception as e:
            raise Exception(f"Error saving profiles: {str(e)}")
    
    def get_profile(self, profile_id: str) -> dict:
        """Get a profile by ID."""
        profiles = self.load_profiles()
        return next((p for p in profiles if p.get("id") == profile_id), None)
    
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


class KnowledgeBaseManager:
    """Manages knowledge bases - listing, selecting, deleting."""
    
    def __init__(self, base_dir: str = CHROMA_DB_DIR, profile_id: str = "default"):
        """Initialize the knowledge base manager.
        
        Args:
            base_dir: Base directory containing knowledge bases
            profile_id: Profile ID to manage KBs for (default: "default")
        """
        self.profile_id = profile_id
        # Store KBs in profile-specific directory
        self.base_dir = os.path.join(base_dir, profile_id)
        self.metadata_file = os.path.join(self.base_dir, "kb_metadata.json")
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load metadata for all knowledge bases.
        
        Returns:
            Dictionary mapping KB IDs to their metadata
        """
        if not os.path.exists(self.metadata_file):
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Dict]) -> None:
        """Save metadata for all knowledge bases.
        
        Args:
            metadata: Dictionary mapping KB IDs to their metadata
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass
    
    def register_knowledge_base(
        self,
        kb_id: str,
        persist_dir: str,
        text_preview: str = "",
        chunk_count: int = 0,
        title: str = None,
        pdf_metadata: dict = None,
    ) -> None:
        """Register a new knowledge base in metadata.
        
        Args:
            kb_id: Unique ID for the knowledge base
            persist_dir: Directory path where the KB is stored
            text_preview: Preview of the text (first 1000 chars)
            chunk_count: Number of chunks in the knowledge base
            title: Optional title for the knowledge base
            pdf_metadata: Optional PDF metadata dictionary (author, subject, etc.)
        """
        metadata = self._load_metadata()
        
        kb_data = {
            "persist_dir": persist_dir,
            "text_preview": text_preview[:1000] if text_preview else "",  # Increased from 200 to 1000 for better preview
            "chunk_count": chunk_count,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        if title:
            kb_data["title"] = title
        
        # Add PDF metadata if provided
        if pdf_metadata:
            kb_data["pdf_metadata"] = pdf_metadata
        
        metadata[kb_id] = kb_data
        self._save_metadata(metadata)
    
    def update_knowledge_base(
        self,
        kb_id: str,
        text_preview: str = "",
        chunk_count: int = 0,
    ) -> bool:
        """Update metadata for an existing knowledge base.
        
        Args:
            kb_id: Unique ID for the knowledge base
            text_preview: Preview of the text (first 200 chars)
            chunk_count: Number of chunks in the knowledge base
            
        Returns:
            True if updated, False if KB not found
        """
        metadata = self._load_metadata()
        
        if kb_id not in metadata:
            return False
        
        metadata[kb_id].update({
            "text_preview": text_preview[:200] if text_preview else "",
            "chunk_count": chunk_count,
            "updated_at": datetime.now().isoformat(),
        })
        
        self._save_metadata(metadata)
        return True
    
    def list_knowledge_bases(self) -> List[Dict]:
        """List all registered knowledge bases.
        
        Returns:
            List of knowledge base dictionaries with metadata
        """
        metadata = self._load_metadata()
        kb_list = []
        
        for kb_id, kb_data in metadata.items():
            persist_dir = kb_data.get("persist_dir", "")
            
            # Verify the directory exists and has content
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                kb_list.append({
                    "id": kb_id,
                    "persist_dir": persist_dir,
                    "text_preview": kb_data.get("text_preview", ""),
                    "chunk_count": kb_data.get("chunk_count", 0),
                    "created_at": kb_data.get("created_at", ""),
                    "updated_at": kb_data.get("updated_at", ""),
                    "title": kb_data.get("title", ""),
                    "pdf_metadata": kb_data.get("pdf_metadata", {}),
                })
        
        # Sort by most recently updated
        kb_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return kb_list
    
    def delete_knowledge_base(self, kb_id: str) -> tuple[bool, str]:
        """Delete a knowledge base.
        
        Args:
            kb_id: Unique ID for the knowledge base to delete
            
        Returns:
            Tuple of (success, message)
        """
        metadata = self._load_metadata()
        
        if kb_id not in metadata:
            return False, "Knowledge base not found"
        
        kb_data = metadata[kb_id]
        persist_dir = kb_data.get("persist_dir", "")
        
        # Delete the directory
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
            except Exception as e:
                return False, f"Could not delete directory: {str(e)}"
        
        # Remove from metadata
        del metadata[kb_id]
        self._save_metadata(metadata)
        
        return True, "Knowledge base deleted successfully"
    
    def get_knowledge_base(self, kb_id: str) -> Optional[Dict]:
        """Get metadata for a specific knowledge base.
        
        Args:
            kb_id: Unique ID for the knowledge base
            
        Returns:
            Knowledge base metadata or None if not found
        """
        metadata = self._load_metadata()
        return metadata.get(kb_id)
    
    def cleanup_orphaned_metadata(self) -> None:
        """Remove metadata entries for knowledge bases that no longer exist on disk."""
        metadata = self._load_metadata()
        to_remove = []
        
        for kb_id, kb_data in metadata.items():
            persist_dir = kb_data.get("persist_dir", "")
            if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
                to_remove.append(kb_id)
        
        for kb_id in to_remove:
            del metadata[kb_id]
        
        if to_remove:
            self._save_metadata(metadata)

