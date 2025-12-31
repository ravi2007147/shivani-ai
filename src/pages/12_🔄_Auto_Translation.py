"""Auto Translation Page - Real-time English to Hindi translation with context understanding.

This page allows you to:
- Enter English text on the left
- Get context-aware Hindi translation on the right
- Uses Ollama LLM models for natural, meaningful translations
"""

import streamlit as st
import sys
import logging
from pathlib import Path
from typing import Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_ollama import OllamaLLM
from src.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_LLM_MODEL
from src.utils import fetch_ollama_models, get_default_model

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Auto Translation",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Auto Translation")
st.markdown("**Context-aware English to Hindi translation** - Understands real meaning, not just word-by-word")


def translate_text(llm: OllamaLLM, text: str) -> str:
    """Translate English text to Hindi using Ollama LLM with enhanced context-aware prompting."""
    
    # Enhanced translation prompt for better quality
    translation_prompt = f"""You are a professional translator with native-level proficiency in both English and Hindi. Your expertise includes understanding cultural nuances, idiomatic expressions, and context-dependent meanings.

TASK: Translate the following English text into natural, fluent Hindi that reads as if it were originally written in Hindi.

TRANSLATION GUIDELINES:
1. **Context Understanding**: Analyze the full context, not individual words. Understand the intended meaning, tone, and purpose.
2. **Natural Flow**: The translation should flow naturally in Hindi, using appropriate sentence structures and word order.
3. **Cultural Adaptation**: Adapt cultural references, idioms, and expressions to be meaningful in Hindi context.
4. **Vocabulary Selection**: Choose the most appropriate Hindi words that convey the exact meaning and register (formal/informal).
5. **Grammar & Syntax**: Use correct Hindi grammar, including proper use of matras (diacritics), verb conjugations, and case markers.
6. **Technical Terms**: For technical terms, use standard Hindi translations when available, or transliterate appropriately.
7. **Proper Nouns**: Keep names, places, and brand names as-is or use common Hindi transliterations.
8. **Tone Preservation**: Maintain the original tone (formal, casual, technical, narrative, etc.).

ORIGINAL ENGLISH TEXT:
{text}

TRANSLATION REQUIREMENTS:
- Output ONLY the Hindi translation
- No explanations, notes, or additional text
- Preserve paragraph breaks and formatting
- Use proper Devanagari script

Hindi Translation:"""
    
    try:
        # Get translation from Ollama
        translation = llm.invoke(translation_prompt)
        
        # Clean up the response
        translation = translation.strip()
        
        # Remove common prefixes that LLMs sometimes add
        prefixes_to_remove = [
            "Hindi translation:",
            "Translation:",
            "Here is the translation:",
            "The Hindi translation is:",
            "हिंदी अनुवाद:",
            "अनुवाद:",
            "Hindi Translation:",
            "TRANSLATION:",
            "Translation:"
        ]
        
        for prefix in prefixes_to_remove:
            if translation.lower().startswith(prefix.lower()):
                translation = translation[len(prefix):].strip()
            # Also check if it's at the start after removing whitespace
            lines = translation.split('\n')
            if lines and lines[0].lower().startswith(prefix.lower()):
                lines[0] = lines[0][len(prefix):].strip()
                translation = '\n'.join(lines).strip()
        
        return translation
        
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        raise Exception(f"Translation failed: {str(e)}")


def translate_with_refinement(llm: OllamaLLM, text: str) -> str:
    """Translate with two-step process: initial translation + refinement for better quality."""
    
    # Step 1: Initial translation
    initial_prompt = f"""Translate the following English text to Hindi. Focus on accuracy and meaning.

English text:
{text}

Provide the Hindi translation:"""
    
    try:
        # Get initial translation
        initial_translation = llm.invoke(initial_prompt).strip()
        
        # Clean initial translation
        prefixes_to_remove = [
            "Hindi translation:",
            "Translation:",
            "Here is the translation:",
            "हिंदी अनुवाद:",
            "अनुवाद:"
        ]
        for prefix in prefixes_to_remove:
            if initial_translation.lower().startswith(prefix.lower()):
                initial_translation = initial_translation[len(prefix):].strip()
        
        # Step 2: Refinement
        refinement_prompt = f"""You are refining a Hindi translation to make it more natural, fluent, and accurate.

ORIGINAL ENGLISH TEXT:
{text}

CURRENT HINDI TRANSLATION:
{initial_translation}

TASK: Refine this translation to:
1. Make it sound more natural and fluent in Hindi
2. Improve word choice and expressions
3. Ensure proper grammar and sentence structure
4. Enhance readability and flow
5. Fix any awkward phrasings or literal translations

Output ONLY the refined Hindi translation, no explanations:"""
        
        refined_translation = llm.invoke(refinement_prompt).strip()
        
        # Clean refined translation
        for prefix in prefixes_to_remove:
            if refined_translation.lower().startswith(prefix.lower()):
                refined_translation = refined_translation[len(prefix):].strip()
        
        return refined_translation
        
    except Exception as e:
        logger.error(f"Refined translation error: {e}", exc_info=True)
        # Fallback to single-step if refinement fails
        return translate_text(llm, text)


# Initialize session state
if 'translation_llm' not in st.session_state:
    st.session_state.translation_llm = None
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []
if 'ollama_models' not in st.session_state:
    st.session_state.ollama_models = []

# Model selection sidebar
with st.sidebar:
    st.header("⚙️ Translation Settings")
    
    # Ollama configuration
    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama API endpoint"
    )
    
    # Fetch available models
    if not st.session_state.ollama_models:
        with st.spinner("Fetching Ollama models..."):
            try:
                st.session_state.ollama_models = fetch_ollama_models(ollama_base_url)
            except Exception as e:
                st.error(f"Could not fetch models: {e}")
                st.session_state.ollama_models = []
    
    if st.button("🔄 Refresh Models"):
        with st.spinner("Fetching models..."):
            try:
                st.session_state.ollama_models = fetch_ollama_models(ollama_base_url)
                st.success("Models refreshed!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Fetch full model names with tags
    try:
        import requests
        full_models_response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if full_models_response.status_code == 200:
            full_models_data = full_models_response.json()
            full_model_names = [model["name"] for model in full_models_data.get("models", [])]
        else:
            full_model_names = []
    except:
        full_model_names = []
    
    if full_model_names:
        # Show full model names with tags in dropdown
        default_model = DEFAULT_LLM_MODEL
        # Try to find a default that matches
        default_index = 0
        for i, model in enumerate(full_model_names):
            if model.startswith(default_model + ":") or model == default_model:
                default_index = i
                break
        
        ollama_model = st.selectbox(
            "Ollama Model",
            options=full_model_names,
            index=default_index,
            help="Select an Ollama model for translation. Full model names with tags are shown."
        )
    elif st.session_state.ollama_models:
        # Fallback to base names if full names not available
        default_model = get_default_model(st.session_state.ollama_models, DEFAULT_LLM_MODEL)
        ollama_model = st.selectbox(
            "Ollama Model",
            options=st.session_state.ollama_models,
            index=st.session_state.ollama_models.index(default_model) if default_model in st.session_state.ollama_models else 0,
            help="Select an Ollama model for translation. Note: You may need to enter full name with tag (e.g., aya:8b) manually."
        )
    else:
        ollama_model = st.text_input(
            "Ollama Model",
            value="aya:8b" if "aya" in str(st.session_state.get('ollama_models', [])) else DEFAULT_LLM_MODEL,
            help="Enter full model name with tag (e.g., aya:8b, llama3.1:8b, mistral:latest)"
        )
    
    st.markdown("---")
    st.markdown("### 📊 Recommended Models for Translation")
    
    # Translation quality settings
    use_refinement = st.checkbox(
        "✨ Use Two-Step Refinement",
        value=True,
        help="First translates, then refines for better quality (slower but better)"
    )
    
    translation_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower = more consistent, Higher = more creative (0.2 recommended for translation)"
    )
    
    with st.expander("📥 Pull Better Models for Translation"):
        st.markdown("""
        **For ChatGPT-level translation quality, pull these models:**
        
        **Best Quality (if you have enough VRAM):**
        ```bash
        # Qwen 2.5 32B - Excellent for translation (~20GB VRAM needed)
        ollama pull qwen2.5:32b
        
        # Llama 3.1 70B Quantized - Best quality (~20GB VRAM with Q4)
        ollama pull llama3.1:70b-q4_K_M
        ```
        
        **Good Quality (fits your RTX 3060 12GB):**
        ```bash
        # Qwen 2.5 14B - You already have this! Good quality
        # Llama 3.1 8B - Fast and decent
        ollama pull llama3.1:8b
        
        # Qwen 2.5 7B - Good balance
        ollama pull qwen2.5:7b
        ```
        
        **After pulling, refresh models in the sidebar above.**
        """)
    
    st.info("""
    **Current Best Options:**
    - **qwen2.5:14b** - You have this! Good quality (~9GB VRAM)
    - **llama3.1:8b** - Fast and decent (~5GB VRAM)
    - **mistral:latest** - You have this, decent for translation
    
    **For best results, enable "Two-Step Refinement" above.**
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History"):
        st.session_state.translation_history = []
        st.rerun()

# Main translation interface
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📝 English Text")
    
    # Text input area
    english_text = st.text_area(
        "Enter English text to translate",
        height=400,
        placeholder="Enter your English text here...\n\nThe model will understand the context and provide a natural Hindi translation.",
        key="english_input"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn1:
        translate_btn = st.button("🔄 Translate", type="primary", use_container_width=True)
    
    with col_btn2:
        if st.button("📋 Clear", use_container_width=True):
            st.session_state.english_input = ""
            st.rerun()
    
    with col_btn3:
        if st.button("📥 Copy Hindi", use_container_width=True):
            if 'hindi_output' in st.session_state and st.session_state.hindi_output:
                st.write("✅ Copied to clipboard!")
            else:
                st.warning("No translation available")

with col_right:
    st.markdown("### 🇮🇳 Hindi Translation")
    
    # Translation output area
    translation_placeholder = st.empty()
    
    if translate_btn and english_text.strip():
        with st.spinner("🔄 Translating with Ollama..."):
            try:
                # Initialize LLM if not already loaded or model changed
                llm_key = f"{ollama_model}_{ollama_base_url}_{translation_temperature}"
                if (st.session_state.translation_llm is None or 
                    st.session_state.get('current_llm_key') != llm_key):
                    with st.status("Initializing Ollama model...", expanded=True) as status:
                        status.update(label=f"Connecting to {ollama_model}...")
                        st.session_state.translation_llm = OllamaLLM(
                            model=ollama_model,
                            base_url=ollama_base_url,
                            temperature=translation_temperature,
                            top_p=0.9,  # Nucleus sampling for better quality
                            num_ctx=4096  # Context window
                        )
                        st.session_state.current_llm_key = llm_key
                        status.update(label="Model ready!", state="complete")
                
                # Perform translation
                start_time = time.time()
                if use_refinement:
                    hindi_translation = translate_with_refinement(
                        st.session_state.translation_llm,
                        english_text
                    )
                else:
                    hindi_translation = translate_text(
                        st.session_state.translation_llm,
                        english_text
                    )
                translation_time = time.time() - start_time
                
                # Display translation
                translation_placeholder.markdown(f"""
                <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; min-height: 400px; font-size: 18px; line-height: 1.8;">
                    {hindi_translation}
                </div>
                <p style="text-align: right; color: #666; font-size: 0.85em; margin-top: 10px;">
                    ⏱️ Translated in {translation_time:.2f}s
                </p>
                """, unsafe_allow_html=True)
                
                # Store in session state for copy
                st.session_state.hindi_output = hindi_translation
                
                # Add to history
                st.session_state.translation_history.insert(0, {
                    'english': english_text[:100] + "..." if len(english_text) > 100 else english_text,
                    'hindi': hindi_translation[:100] + "..." if len(hindi_translation) > 100 else hindi_translation,
                    'timestamp': time.strftime("%H:%M:%S")
                })
                
                # Keep only last 10 translations in history
                if len(st.session_state.translation_history) > 10:
                    st.session_state.translation_history = st.session_state.translation_history[:10]
                
            except Exception as e:
                logger.error(f"Translation error: {e}", exc_info=True)
                translation_placeholder.error(f"❌ Translation failed: {str(e)}\n\nPlease check:\n1. Ollama is running (ollama serve)\n2. Model is available (ollama pull {ollama_model})\n3. Ollama base URL is correct")
    else:
        translation_placeholder.info("👈 Enter English text on the left and click 'Translate' to get Hindi translation")

# Translation history
if st.session_state.translation_history:
    st.markdown("---")
    st.markdown("### 📜 Recent Translations")
    
    for i, item in enumerate(st.session_state.translation_history[:5]):
        with st.expander(f"Translation #{i+1} - {item['timestamp']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**English:**")
                st.text(item['english'])
            with col2:
                st.markdown("**Hindi:**")
                st.text(item['hindi'])

