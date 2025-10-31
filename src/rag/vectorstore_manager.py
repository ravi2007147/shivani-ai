"""Manages Chroma vectorstore creation and persistence."""

import os
import sys
import hashlib
import time
import shutil
from pathlib import Path
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    CHROMA_DB_DIR,
    KB_PREFIX,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
)
from src.utils.performance_utils import optimize_chunk_size


class VectorStoreManager:
    """Manages creation, persistence, and loading of Chroma vectorstores."""
    
    def __init__(self, base_dir: str = None, profile_id: str = None):
        """Initialize the vectorstore manager.
        
        Args:
            base_dir: Base directory for storing vectorstores (optional, will use profile_id if provided)
            profile_id: Profile ID to store vectorstores under (optional)
        """
        # If profile_id is provided, create profile-specific directory
        if profile_id:
            self.base_dir = os.path.join(CHROMA_DB_DIR, profile_id)
            self.profile_id = profile_id
        elif base_dir:
            self.base_dir = base_dir
            self.profile_id = None
        else:
            # Default to base CHROMA_DB_DIR
            self.base_dir = CHROMA_DB_DIR
            self.profile_id = None
        
        os.makedirs(self.base_dir, mode=0o755, exist_ok=True)
        os.chmod(self.base_dir, 0o755)
    
    def _get_persist_dir(self, text: str) -> str:
        """Generate a unique directory path for a text.
        
        Args:
            text: Input text to create hash from
            
        Returns:
            Absolute path to persistence directory
        """
        rag_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        persist_dir = os.path.join(self.base_dir, f"{KB_PREFIX}{rag_hash}")
        return os.path.abspath(persist_dir)
    
    def _clean_directory(self, persist_dir: str) -> None:
        """Remove existing directory and ensure clean state.
        
        Note: This is only used when recreating the same KB. Different texts
        create different directories based on hash, so they don't conflict.
        
        Args:
            persist_dir: Directory path to clean
        """
        if os.path.exists(persist_dir):
            time.sleep(0.2)  # Wait for any locks to release
            try:
                # Remove all files individually
                for root, dirs, files in os.walk(persist_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.chmod(file_path, 0o777)
                            os.remove(file_path)
                        except Exception:
                            pass
                shutil.rmtree(persist_dir)
                time.sleep(0.2)
            except Exception:
                pass
    
    def _verify_write_permissions(self, persist_dir: str) -> None:
        """Verify that we can write to the directory.
        
        Args:
            persist_dir: Directory path to verify
            
        Raises:
            Exception: If write permissions are not available
        """
        test_file = os.path.join(persist_dir, ".write_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            # Try to fix permissions and retry
            try:
                os.chmod(persist_dir, 0o777)
                os.chmod(self.base_dir, 0o777)
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception:
                raise Exception(
                    f"Cannot write to directory {persist_dir}. "
                    f"Please check permissions: {str(e)}"
                )
    
    def _fix_permissions(self, persist_dir: str) -> None:
        """Fix permissions on all files and directories.
        Uses more permissive settings to avoid readonly database errors.
        
        Args:
            persist_dir: Directory path to fix permissions for
        """
        if not os.path.exists(persist_dir):
            return
        
        # Fix permissions recursively with more permissive settings
        for root, dirs, files in os.walk(persist_dir):
            # Fix directory permissions - use 777 for directories
            try:
                os.chmod(root, 0o777)
            except Exception:
                pass
            
            # Fix file permissions - use 666 for files (read/write for all)
            # This is important for SQLite database files
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Database files need write permissions
                    if file.endswith('.sqlite3') or file.endswith('.db') or file.endswith('.sqlite'):
                        os.chmod(file_path, 0o666)
                    else:
                        os.chmod(file_path, 0o666)
                except Exception:
                    pass
        
        # Also fix the persist_dir itself
        try:
            os.chmod(persist_dir, 0o777)
        except Exception:
            pass
    
    def create_vectorstore(
        self,
        text: str,
        embedding_model: str,
        base_url: str,
    ) -> tuple[Chroma, str]:
        """Create a new vectorstore from text and persist it.
        Optimized for large knowledge bases with batch processing.
        
        Args:
            text: Text to create embeddings from
            embedding_model: Name of the embedding model to use
            base_url: Base URL for Ollama API
            
        Returns:
            Tuple of (vectorstore, persist_directory_path)
        """
        # Create documents
        documents = [Document(page_content=text)]
        
        # Dynamically optimize chunk size for large texts
        text_length = len(text)
        optimal_chunk_size, optimal_overlap = optimize_chunk_size(text_length)
        
        # Use optimized chunk size for very large texts, otherwise use default
        if text_length > 50000:
            chunk_size = optimal_chunk_size
            chunk_overlap = optimal_overlap
        else:
            chunk_size = CHUNK_SIZE
            chunk_overlap = CHUNK_OVERLAP
        
        # Optimized text splitter for large documents
        # Use separators to preserve semantic boundaries
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Preserve paragraph boundaries
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
        )
        
        # Get persistence directory
        persist_dir = self._get_persist_dir(text)
        
        # Clean existing directory if it exists
        self._clean_directory(persist_dir)
        
        # Ensure all parent directories exist with proper permissions
        # This is critical for Chroma to create the SQLite database
        current_path = persist_dir
        path_parts = []
        while current_path and current_path != os.path.dirname(current_path):
            path_parts.insert(0, current_path)
            current_path = os.path.dirname(current_path)
        
        # Create and set permissions for all parent directories
        for path in path_parts:
            os.makedirs(path, mode=0o777, exist_ok=True)
            try:
                os.chmod(path, 0o777)  # Full permissions for owner, group, others
            except Exception:
                pass
        
        # Specifically set permissions on the persist directory
        os.makedirs(persist_dir, mode=0o777, exist_ok=True)
        os.chmod(persist_dir, 0o777)
        
        # Also ensure base_dir has proper permissions
        try:
            os.chmod(self.base_dir, 0o777)
        except Exception:
            pass
        
        # Verify write permissions
        self._verify_write_permissions(persist_dir)
        
        # Create vectorstore - this is where embeddings are actually created
        # Chroma.from_documents will:
        # 1. Generate embeddings for each document chunk using Ollama
        # 2. Store them in the Chroma vectorstore
        # 3. Persist to disk
        
        try:
            # Show where Chroma DB will be created
            print(f"[DEBUG] =========================================")
            print(f"[DEBUG] CHROMA DATABASE LOCATION:")
            print(f"[DEBUG]   Base Directory: {self.base_dir}")
            print(f"[DEBUG]   Full Persist Path: {persist_dir}")
            print(f"[DEBUG]   Absolute Path: {os.path.abspath(persist_dir)}")
            print(f"[DEBUG] =========================================")
            
            # Test embedding connection first to catch errors early
            try:
                print(f"[DEBUG] Testing embedding connection with model '{embedding_model}' at {base_url}...")
                test_embedding = embeddings.embed_query("test")
                print(f"[DEBUG] ✅ Embedding test successful. Vector dimension: {len(test_embedding)}")
            except Exception as e:
                error_msg = f"Failed to connect to embedding model '{embedding_model}' at {base_url}: {str(e)}"
                print(f"[ERROR] {error_msg}")
                raise Exception(f"{error_msg}\n\nMake sure:\n1. Ollama is running (check: ollama serve)\n2. The embedding model '{embedding_model}' is installed (check: ollama list)\n3. Ollama API is accessible at {base_url}")
            
            print(f"[DEBUG] Creating vectorstore with {len(texts)} document chunks")
            print(f"[DEBUG] Will save to: {persist_dir}")
            
            # Create vectorstore - this calls Ollama embedding API for each chunk
            print(f"[DEBUG] Starting Chroma.from_documents (this will create embeddings via Ollama)...")
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
            print(f"[DEBUG] ✅ Chroma.from_documents completed successfully")
            
            # Verify the vectorstore has content
            try:
                count = vectorstore._collection.count()
                print(f"[DEBUG] ✅ Vectorstore verified: contains {count} document(s)")
                if count == 0:
                    raise Exception("Vectorstore was created but contains 0 documents - embedding creation may have failed silently")
            except (AttributeError, Exception) as e:
                # Try alternative method to verify
                try:
                    # Try a simple search to verify it works
                    results = vectorstore.similarity_search("test", k=1)
                    print(f"[DEBUG] ✅ Vectorstore verified: similarity search successful, found {len(results)} result(s)")
                except Exception as verify_e:
                    print(f"[WARNING] Could not verify vectorstore: {str(verify_e)}")
                    # Don't fail if verification fails, might still work
            
        except Exception as e:
            # If embedding creation fails, provide more context
            print(f"[ERROR] Failed to create vectorstore: {str(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"[ERROR] Full traceback:\n{traceback_str}")
            
            # Provide helpful error message based on error type
            error_str = str(e)
            if "ConnectionError" in error_str or "timeout" in error_str.lower() or "refused" in error_str.lower():
                raise Exception(f"Cannot connect to Ollama at {base_url}.\n\nMake sure Ollama is running:\n  ollama serve\n\nError: {error_str}")
            elif "model" in error_str.lower() and ("not found" in error_str.lower() or "does not exist" in error_str.lower()):
                raise Exception(f"Embedding model '{embedding_model}' not found.\n\nInstall it with:\n  ollama pull {embedding_model}\n\nError: {error_str}")
            else:
                raise Exception(f"Failed to create embeddings: {error_str}\n\nMake sure:\n1. Ollama is running (ollama serve)\n2. The embedding model '{embedding_model}' is installed (ollama pull {embedding_model})\n3. Ollama API is accessible at {base_url}")
        
        # Wait for files to be written
        print(f"[DEBUG] Waiting for Chroma to persist files...")
        time.sleep(1.0)  # Give Chroma more time to write
        
        # Fix permissions on created files IMMEDIATELY after creation
        # This is critical - Chroma creates SQLite files that need write access
        self._fix_permissions(persist_dir)
        
        # Also fix permissions on parent directories
        try:
            parent_dir = os.path.dirname(persist_dir)
            if os.path.exists(parent_dir):
                os.chmod(parent_dir, 0o777)
                # Fix any files in parent dir too (like Chroma metadata)
                for item in os.listdir(parent_dir):
                    item_path = os.path.join(parent_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.chmod(item_path, 0o666)
                        elif os.path.isdir(item_path):
                            os.chmod(item_path, 0o777)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[DEBUG] Warning: Could not fix parent directory permissions: {e}")
        
        # Verify files were created
        print(f"[DEBUG] =========================================")
        print(f"[DEBUG] VERIFYING CHROMA DATABASE CREATION:")
        print(f"[DEBUG]   Persist directory exists: {os.path.exists(persist_dir)}")
        if os.path.exists(persist_dir):
            all_items = os.listdir(persist_dir)
            files_created = [f for f in all_items if os.path.isfile(os.path.join(persist_dir, f))]
            dirs_created = [f for f in all_items if os.path.isdir(os.path.join(persist_dir, f))]
            print(f"[DEBUG]   Files created: {len(files_created)}")
            print(f"[DEBUG]   Directories created: {len(dirs_created)}")
            if files_created:
                print(f"[DEBUG]   File names: {files_created[:5]}...")  # Show first 5
            if dirs_created:
                print(f"[DEBUG]   Directory names: {dirs_created[:5]}...")  # Show first 5
            print(f"[DEBUG]   Full path: {os.path.abspath(persist_dir)}")
        else:
            print(f"[DEBUG]   ⚠️ WARNING: Persist directory does not exist!")
        print(f"[DEBUG] =========================================")
        
        return vectorstore, persist_dir
    
    def load_vectorstore(
        self,
        persist_dir: str,
        embedding_model: str,
        base_url: str,
    ) -> Chroma:
        """Load an existing vectorstore from disk.
        
        Args:
            persist_dir: Directory path containing the persisted vectorstore
            embedding_model: Name of the embedding model to use
            base_url: Base URL for Ollama API
            
        Returns:
            Loaded Chroma vectorstore
        """
        abs_persist_dir = os.path.abspath(persist_dir)
        
        # Fix permissions before loading
        self._fix_permissions(abs_persist_dir)
        
        # Create embeddings
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
        )
        
        # Load vectorstore
        vectorstore = Chroma(
            persist_directory=abs_persist_dir,
            embedding_function=embeddings,
        )
        
        return vectorstore
    
    def verify_persistence(self, persist_dir: str) -> tuple[bool, int]:
        """Verify that a vectorstore was persisted correctly.
        
        Args:
            persist_dir: Directory path to verify
            
        Returns:
            Tuple of (is_valid, file_count)
        """
        time.sleep(0.3)  # Wait for any final writes
        try:
            all_files = []
            for root, dirs, files in os.walk(persist_dir):
                all_files.extend([os.path.join(root, f) for f in files])
            file_count = len(all_files)
            return (file_count > 0, file_count)
        except Exception:
            return (False, 0)

