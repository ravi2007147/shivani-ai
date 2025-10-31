"""Utilities for managing knowledge base persistence."""

import os
from typing import Optional


def find_persisted_knowledge_base(base_dir: str = ".chroma_db") -> Optional[str]:
    """Find the most recently modified persisted knowledge base directory.
    
    Args:
        base_dir: Base directory containing knowledge bases
        
    Returns:
        Path to the most recently modified knowledge base directory, or None
    """
    if not os.path.exists(base_dir):
        return None
    
    kb_dirs = [
        d for d in os.listdir(base_dir)
        if d.startswith("kb_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not kb_dirs:
        return None
    
    # Return the most recently modified directory
    kb_dirs_with_time = [
        (d, os.path.getmtime(os.path.join(base_dir, d))) for d in kb_dirs
    ]
    kb_dirs_with_time.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(base_dir, kb_dirs_with_time[0][0])

