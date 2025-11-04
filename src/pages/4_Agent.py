"""Agent page for video script generation."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_ollama import OllamaLLM
from src.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_LLM_MODEL
from src.utils import fetch_ollama_models, get_default_model, extract_article_from_url, is_valid_url


# Page configuration
st.set_page_config(
    page_title="Agent - Video Script Writer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agent")
st.markdown("Generate structured video scripts from articles using AI")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama API endpoint"
    )
    
    # Load models
    if 'ollama_models_agent' not in st.session_state:
        st.session_state.ollama_models_agent = fetch_ollama_models(DEFAULT_OLLAMA_BASE_URL)
    
    # Model selection
    if st.session_state.ollama_models_agent:
        default_model = get_default_model(st.session_state.ollama_models_agent, DEFAULT_LLM_MODEL)
        ollama_model = st.selectbox(
            "LLM Model",
            options=st.session_state.ollama_models_agent,
            index=st.session_state.ollama_models_agent.index(default_model) if default_model in st.session_state.ollama_models_agent else 0,
            help="LLM model used for script generation"
        )
    else:
        ollama_model = st.text_input(
            "LLM Model",
            value=DEFAULT_LLM_MODEL,
            help="LLM model used for script generation"
        )
    
    # Refresh models button
    if st.button("🔄 Refresh Models"):
        with st.spinner("Loading models..."):
            fetched_models = fetch_ollama_models(ollama_base_url)
            st.session_state.ollama_models_agent = fetched_models
            if fetched_models:
                st.success(f"✅ Loaded {len(fetched_models)} model(s)")
            else:
                st.warning("⚠️ Could not fetch models. Make sure Ollama is running.")

# Main content area
st.markdown("---")

# Input mode selection
input_mode = st.radio(
    "Select Input Mode:",
    ["Manual Text Entry", "From URL"],
    horizontal=True,
    help="Choose to either enter text manually or fetch content from a URL"
)

description_input = ""

if input_mode == "Manual Text Entry":
    # Manual text entry
    description_input = st.text_area(
        "Description",
        height=300,
        placeholder="Enter your article text here...",
        help="Paste the full article text you want to convert into a video script"
    )
else:
    # URL input
    url_input = st.text_input(
        "Article URL",
        placeholder="https://example.com/article",
        help="Enter the URL of the article you want to convert into a video script"
    )
    
    col1, col2 = st.columns([1, 9])
    with col1:
        fetch_button = st.button("🔍 Fetch", type="secondary", use_container_width=True)
    
    # Fetch content from URL
    if fetch_button:
        if not url_input or not url_input.strip():
            st.error("⚠️ Please enter a URL")
        elif not is_valid_url(url_input):
            st.error("⚠️ Please enter a valid URL (must start with http:// or https://)")
        else:
            with st.spinner("Fetching and extracting article content..."):
                try:
                    extracted_text = extract_article_from_url(url_input)
                    if extracted_text:
                        st.success("✅ Article extracted successfully!")
                        # Store in session state and display in text area
                        st.session_state.extracted_text = extracted_text
                        description_input = extracted_text
                        st.rerun()
                    else:
                        st.error("⚠️ Could not extract article content from the URL. Please try another URL or use manual text entry.")
                except Exception as e:
                    st.error(f"❌ Error fetching content: {str(e)}")
    
    # Display extracted text in text area for user to review/edit
    if 'extracted_text' in st.session_state and st.session_state.extracted_text:
        description_input = st.text_area(
            "Extracted Article Content",
            value=st.session_state.extracted_text,
            height=300,
            help="Review and edit the extracted content before generating the script"
        )
        if st.button("🗑️ Clear Extracted Content"):
            st.session_state.extracted_text = ""
            st.rerun()

# Generate button
if st.button("Generate", type="primary", use_container_width=True):
    if not description_input or description_input.strip() == "":
        st.error("⚠️ Please enter article text to generate a script")
    else:
        with st.spinner("Generating video script..."):
            try:
                # Initialize LLM
                llm = OllamaLLM(
                    model=ollama_model,
                    base_url=ollama_base_url,
                )
                
                # Create the prompt
                prompt = f"""SYSTEM:

You are an expert AI video scriptwriter. 

Your job is to convert an article into a structured, short video script that can be used for automatic video generation.

Follow these strict rules:

- The final output must be valid JSON (no explanations, no markdown formatting).

- Each scene must include a narration, a visual_prompt for image generation, and an estimated duration_seconds.

- The visual_prompt must describe the visual style clearly, as if for a text-to-image model (Stable Diffusion or SDXL).

- Keep total duration around 60–120 seconds.

- Each narration should be 1–2 sentences long, simple and conversational.

- Avoid long introductions — start engagingly.

- Use natural tone and smooth flow between scenes.

USER:

Convert the following article into a structured video script.

ARTICLE:

{description_input}"""
                
                # Generate response
                result = llm.invoke(prompt)
                
                # Display result
                st.markdown("---")
                st.subheader("📹 Generated Video Script")
                st.markdown(result)
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error generating script: {str(e)}")
                st.info("Make sure Ollama is running locally and the model is available.")

# Footer
st.markdown("---")
st.info("💡 Tip: Make sure Ollama is running locally. You can start it with: `ollama serve`")

