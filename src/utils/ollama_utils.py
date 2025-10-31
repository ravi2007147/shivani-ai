"""Utilities for interacting with Ollama API."""

from typing import List
import requests


def fetch_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Fetch all available Ollama models from the API.
    
    Args:
        base_url: The base URL for the Ollama API
        
    Returns:
        List of available model names (base names without tags)
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Extract model names and strip tags (e.g., "mistral:latest" -> "mistral")
            models = []
            seen_models = set()
            for model in data.get("models", []):
                model_name = model["name"]
                # Split by ':' to get base model name (remove tags like :latest, :7b, etc.)
                base_name = model_name.split(":")[0]
                # Only add unique base names
                if base_name not in seen_models:
                    models.append(base_name)
                    seen_models.add(base_name)
            return models
        return []
    except Exception:
        # Silently return empty list - warning will be shown in UI
        return []


def get_default_model(models: List[str], preferred: str = "mistral") -> str:
    """Get the default model, preferring the specified model if available.
    
    Args:
        models: List of available models
        preferred: Preferred model name
        
    Returns:
        Default model name
    """
    if not models:
        return preferred
    if preferred in models:
        return preferred
    return models[0] if models else preferred

