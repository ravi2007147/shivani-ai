"""Tools page for PDF categorization and other utilities."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_LLM_MODEL
from src.utils.text_classifier import PDFClassifier, classify_pdf_content, PDF_CATEGORIES
from src.utils.pdf_parser import process_pdf_content
from src.utils import fetch_ollama_models, get_default_model


# Page configuration
st.set_page_config(
    page_title="Tools",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Tools")
st.markdown("Utility tools for PDF analysis and categorization")

# Tabbed interface
tab1 = st.tabs(["📊 Find PDF Category"])

# Tab 1: Find PDF Category
with tab1[0]:
    st.header("📊 Find PDF Category")
    st.markdown("Upload a PDF file to automatically categorize its content type.")
    
    # Load models
    if 'ollama_models_tools' not in st.session_state:
        st.session_state.ollama_models_tools = fetch_ollama_models(DEFAULT_OLLAMA_BASE_URL)
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=DEFAULT_OLLAMA_BASE_URL,
            help="Ollama API endpoint"
        )
        
        # Model selection
        if st.session_state.ollama_models_tools:
            default_model = get_default_model(st.session_state.ollama_models_tools, DEFAULT_LLM_MODEL)
            ollama_model = st.selectbox(
                "Classification Model",
                options=st.session_state.ollama_models_tools,
                index=st.session_state.ollama_models_tools.index(default_model) if default_model in st.session_state.ollama_models_tools else 0,
                help="LLM model used for classification"
            )
        else:
            ollama_model = st.text_input(
                "Classification Model",
                value=DEFAULT_LLM_MODEL,
                help="LLM model used for classification"
            )
        
        # Refresh models button
        if st.button("🔄 Refresh Models"):
            with st.spinner("Loading models..."):
                fetched_models = fetch_ollama_models(ollama_base_url)
                st.session_state.ollama_models_tools = fetched_models
                if fetched_models:
                    st.success(f"✅ Loaded {len(fetched_models)} model(s)")
                else:
                    st.warning("⚠️ Could not fetch models. Make sure Ollama is running.")
    
    # Main content area
    st.markdown("---")
    
    # PDF upload section
    uploaded_file = st.file_uploader(
        "Upload a PDF file for categorization",
        type=["pdf"],
        help="Upload a PDF document to analyze and categorize"
    )
    
    if uploaded_file is not None:
        # Show file details
        file_name = uploaded_file.name
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", file_name)
        with col2:
            st.metric("File Size", f"{file_size / 1024:.2f} KB")
        
        st.markdown("---")
        
        # Extract and classify button
        if st.button("🔍 Analyze and Categorize PDF", type="primary"):
            try:
                with st.spinner("Extracting text from PDF..."):
                    # Extract text from PDF
                    extracted_text, pdf_metadata = process_pdf_content(uploaded_file, return_metadata=True)
                    
                    if not extracted_text or len(extracted_text.strip()) < 100:
                        st.error("Could not extract sufficient text from PDF for analysis.")
                        st.info("Please make sure the PDF contains readable text content.")
                    else:
                        st.success(f"✅ Extracted {len(extracted_text)} characters from PDF")
                        
                        # Display PDF metadata if available
                        if pdf_metadata:
                            with st.expander("📄 PDF Metadata", expanded=False):
                                if pdf_metadata.get('Title'):
                                    st.write(f"**Title:** {pdf_metadata['Title']}")
                                if pdf_metadata.get('Author'):
                                    st.write(f"**Author:** {pdf_metadata['Author']}")
                                if pdf_metadata.get('Subject'):
                                    st.write(f"**Subject:** {pdf_metadata['Subject']}")
                                if pdf_metadata.get('Producer'):
                                    st.write(f"**Producer:** {pdf_metadata['Producer']}")
                                if pdf_metadata.get('Creator'):
                                    st.write(f"**Creator:** {pdf_metadata['Creator']}")
                        
                        st.markdown("---")
                        
                        # Classify the content
                        with st.spinner("Analyzing content and determining category..."):
                            classification_result = classify_pdf_content(
                                text=extracted_text,
                                model=ollama_model,
                                base_url=ollama_base_url,
                                max_length=2000
                            )
                        
                        # Display classification results
                        st.subheader("📊 Classification Results")
                        
                        # Main category result
                        category_name = classification_result.get("category_name", "Unknown")
                        confidence = classification_result.get("confidence", 0.0)
                        reasoning = classification_result.get("reasoning", "No reasoning provided")
                        
                        # Visual confidence indicator
                        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"
                        
                        result_col1, result_col2 = st.columns([2, 1])
                        with result_col1:
                            st.markdown(f"### Category: **{category_name}**")
                        with result_col2:
                            st.markdown(f"**Confidence:** <span style='color: {confidence_color}'>{confidence*100:.1f}%</span>", unsafe_allow_html=True)
                        
                        # Progress bar for confidence
                        st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                        
                        # Reasoning
                        st.info(f"**Analysis:** {reasoning}")
                        
                        # Show all available categories
                        with st.expander("📋 All Available Categories", expanded=False):
                            st.write("The PDF was categorized into one of the following categories:")
                            for cat_key, cat_name in PDF_CATEGORIES.items():
                                if cat_name == category_name:
                                    st.markdown(f"✅ **{cat_name}** (selected)")
                                else:
                                    st.markdown(f"- {cat_name}")
                        
                        # Error handling
                        if "error" in classification_result:
                            st.warning(f"⚠️ Classification completed with warning: {classification_result['error']}")
                        
                        st.markdown("---")
                        
                        # Text preview
                        with st.expander("📄 Text Preview (First 500 chars)", expanded=False):
                            st.text(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
                
            except ImportError as e:
                print(e)
                st.error(f"PDF processing library not installed: {str(e)}")
                st.info("Install with: pip install pymupdf or pip install pypdf")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.exception(e)
    
    else:
        # Show info when no file is uploaded
        st.info("👆 Upload a PDF file above to get started with content categorization.")
        
        # Display available categories
        st.markdown("### Available Categories")
        st.write("The classifier can categorize PDFs into the following categories:")
        
        category_cols = st.columns(2)
        for idx, (cat_key, cat_name) in enumerate(PDF_CATEGORIES.items()):
            with category_cols[idx % 2]:
                st.markdown(f"- {cat_name}")

