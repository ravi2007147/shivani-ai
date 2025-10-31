"""Performance utilities for handling large knowledge bases."""

import os
import hashlib
from typing import List, Dict
from langchain_core.documents import Document


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text (rough approximation).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 characters per token
    return len(text) // 4


def get_chunk_count_estimate(text_length: int, chunk_size: int, chunk_overlap: int) -> int:
    """Estimate number of chunks for a text.
    
    Args:
        text_length: Length of text in characters
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Estimated chunk count
    """
    if chunk_size <= chunk_overlap:
        return 1 if text_length > 0 else 0
    
    effective_chunk_size = chunk_size - chunk_overlap
    return max(1, (text_length + effective_chunk_size - 1) // effective_chunk_size)


def optimize_chunk_size(text_length: int) -> tuple[int, int]:
    """Dynamically optimize chunk size based on text length.
    
    Args:
        text_length: Length of text in characters
        
    Returns:
        Tuple of (optimal_chunk_size, optimal_overlap)
    """
    if text_length < 5000:
        return 1000, 200
    elif text_length < 50000:
        return 2000, 400
    elif text_length < 200000:
        return 3000, 600
    else:
        return 4000, 800


def check_memory_usage() -> Dict[str, float]:
    """Check memory usage statistics.
    
    Returns:
        Dictionary with memory usage info
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "percent": process.memory_percent(),
        }
    except ImportError:
        return {
            "rss_mb": 0,
            "vms_mb": 0,
            "percent": 0,
        }


def should_use_batch_processing(text_length: int, chunk_count: int) -> bool:
    """Determine if batch processing should be used.
    
    Args:
        text_length: Length of text
        chunk_count: Number of chunks
        
    Returns:
        True if batch processing recommended
    """
    # Use batch processing for large texts or many chunks
    return text_length > 100000 or chunk_count > 50

