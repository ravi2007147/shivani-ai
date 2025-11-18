"""Test Website Score Page - Test and debug relevance detection algorithm."""

import streamlit as st
import sys
from pathlib import Path
import time
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_BASE_URL
from src.utils.url_data_extractor import URLDataExtractor
from src.relevance_detector import RelevanceDetector


# Page configuration
st.set_page_config(
    page_title="Test Website Score",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Test Website Score")
st.markdown("Test and debug the relevance detection algorithm. Enter a URL and keyword to see detailed relevance scores.")

# Initialize session state for results
if 'test_results' not in st.session_state:
    st.session_state.test_results = None

# Configuration section
with st.expander("⚙️ Configuration", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        embedding_model = st.text_input(
            "Embedding Model",
            value=DEFAULT_EMBEDDING_MODEL,
            help="Ollama embedding model to use for semantic similarity"
        )
    with col2:
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=DEFAULT_OLLAMA_BASE_URL,
            help="Ollama API base URL"
        )

# Input section
st.markdown("### 📝 Input")
col1, col2 = st.columns([2, 1])

with col1:
    test_url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        help="Enter the URL of the website to test"
    )

with col2:
    keyword = st.text_input(
        "Keyword/Term",
        placeholder="example",
        help="Enter the keyword or search term to test relevance"
    )

# Options
col1, col2 = st.columns(2)
with col1:
    headless_mode = st.checkbox(
        "Headless Mode",
        value=True,
        help="Run browser in headless mode (uncheck to see browser for debugging)"
    )
with col2:
    extract_title = st.checkbox(
        "Extract Page Title",
        value=True,
        help="Extract and use page title for relevance scoring"
    )

# Test button
test_button = st.button(
    "🧪 Test Relevance Score",
    type="primary",
    use_container_width=True
)

# Results section
if test_button:
    if not test_url or not keyword:
        st.error("❌ Please provide both URL and keyword to test.")
    else:
        # Validate URL
        if not test_url.startswith(('http://', 'https://')):
            st.error("❌ Invalid URL. Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("🔍 Testing relevance score... This may take a minute."):
                try:
                    # Step 1: Extract content from URL
                    st.info("📄 Step 1/3: Extracting content from URL...")
                    progress_bar = st.progress(0)
                    
                    url_extractor = URLDataExtractor(
                        headless=headless_mode,
                        ollama_model="mistral",  # Not used for extraction, just for initialization
                        ollama_base_url=ollama_base_url
                    )
                    
                    progress_bar.progress(20)
                    success, content, error = url_extractor.extract_content(test_url)
                    
                    if not success:
                        st.error(f"❌ Failed to extract content: {error}")
                        url_extractor._close_browser()
                    else:
                        progress_bar.progress(40)
                        
                        # Extract page title if enabled
                        page_title = ""
                        if extract_title:
                            try:
                                page_title = url_extractor.page.title() if url_extractor.page else ""
                            except Exception:
                                page_title = ""
                        
                        url_extractor._close_browser()
                        progress_bar.progress(60)
                        
                        # Step 2: Calculate relevance score
                        st.info("📊 Step 2/3: Calculating relevance scores...")
                        
                        relevance_detector = RelevanceDetector(
                            embedding_model=embedding_model,
                            ollama_base_url=ollama_base_url
                        )
                        
                        progress_bar.progress(80)
                        
                        relevance_results = relevance_detector.calculate_relevance_score(
                            search_term=keyword,
                            url=test_url,
                            content=content,
                            title=page_title
                        )
                        
                        progress_bar.progress(100)
                        time.sleep(0.5)  # Small delay for visual feedback
                        progress_bar.empty()
                        
                        # Step 3: Display results
                        st.success("✅ Relevance score calculated successfully!")
                        
                        # Store results in session state
                        st.session_state.test_results = {
                            'url': test_url,
                            'keyword': keyword,
                            'title': page_title,
                            'content_length': len(content),
                            'content_preview': content[:500] + "..." if len(content) > 500 else content,
                            'relevance': relevance_results
                        }
                        
                        # Display summary
                        st.markdown("### 📊 Results Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        final_score = relevance_results.get('final_score', 0.0)
                        confidence = relevance_results.get('confidence', 'unknown')
                        is_relevant = relevance_results.get('is_relevant', False)
                        threshold = relevance_results.get('threshold', 0.5)
                        
                        with col1:
                            st.metric(
                                "Final Score",
                                f"{final_score:.3f}",
                                f"{final_score - threshold:.3f}" if final_score > threshold else f"{final_score - threshold:.3f}",
                                delta_color="normal" if is_relevant else "inverse"
                            )
                        
                        with col2:
                            st.metric(
                                "Confidence",
                                confidence.upper(),
                                "✓" if confidence == 'high' else ("~" if confidence == 'medium' else "✗")
                            )
                        
                        with col3:
                            st.metric(
                                "Relevance Status",
                                "✅ RELEVANT" if is_relevant else "❌ NOT RELEVANT",
                                f"Threshold: {threshold}"
                            )
                        
                        with col4:
                            st.metric(
                                "Content Length",
                                f"{len(content):,}",
                                "characters"
                            )
                        
                        # Detailed breakdown
                        st.markdown("### 📈 Detailed Score Breakdown")
                        
                        # Score components
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("#### Score Components")
                            
                            url_title_score = relevance_results.get('url_title_score', 0.0)
                            semantic_score = relevance_results.get('semantic_similarity_score', 0.0)
                            keyword_density = relevance_results.get('keyword_density_score', 0.0)
                            domain_authority = relevance_results.get('domain_authority_score', 0.0)
                            negative_penalty = relevance_results.get('negative_penalty', 0.0)
                            
                            # URL/Title Score (45% weight)
                            st.markdown(f"**1. URL & Title Exactness** (45% weight)")
                            st.progress(url_title_score, text=f"Score: {url_title_score:.3f}")
                            st.caption(f"Weighted contribution: {url_title_score * 0.45:.3f}")
                            
                            # Semantic Similarity (70% weight)
                            st.markdown(f"**2. Semantic Embedding Similarity** (70% weight)")
                            st.progress(semantic_score, text=f"Score: {semantic_score:.3f}")
                            st.caption(f"Weighted contribution: {semantic_score * 0.70:.3f}")
                            
                            # Keyword Density (10% weight)
                            st.markdown(f"**3. Keyword Density** (10% weight)")
                            st.progress(keyword_density, text=f"Score: {keyword_density:.3f}")
                            st.caption(f"Weighted contribution: {keyword_density * 0.10:.3f}")
                            
                            # Domain Authority (15% weight)
                            st.markdown(f"**4. Domain Trust/Authority** (15% weight)")
                            domain_score_display = domain_authority if domain_authority >= 0 else 0.0
                            st.progress(domain_score_display, text=f"Score: {domain_authority:.3f}")
                            st.caption(f"Weighted contribution: {domain_authority * 0.15:.3f}")
                            
                            # Negative Penalty
                            if negative_penalty < 0:
                                st.markdown(f"**5. Negative Keyword Penalty**")
                                st.warning(f"Penalty: {negative_penalty:.3f}")
                            
                        with col2:
                            st.markdown("#### Score Calculation")
                            
                            # Formula breakdown
                            st.markdown("**Formula:**")
                            st.code(f"""
final_score = (
    0.45 × url_title_score      ({url_title_score:.3f})
  + 0.70 × semantic_similarity  ({semantic_score:.3f})
  + 0.10 × keyword_density      ({keyword_density:.3f})
  + 0.15 × domain_authority     ({domain_authority:.3f})
  + negative_penalty            ({negative_penalty:.3f})
)
                            """.strip())
                            
                            # Calculate intermediate values
                            weighted_url_title = url_title_score * 0.45
                            weighted_semantic = semantic_score * 0.70
                            weighted_keyword = keyword_density * 0.10
                            weighted_domain = domain_authority * 0.15
                            
                            st.markdown("**Weighted Contributions:**")
                            st.markdown(f"- URL/Title: `{weighted_url_title:.3f}`")
                            st.markdown(f"- Semantic: `{weighted_semantic:.3f}`")
                            st.markdown(f"- Keyword: `{weighted_keyword:.3f}`")
                            st.markdown(f"- Domain: `{weighted_domain:.3f}`")
                            if negative_penalty < 0:
                                st.markdown(f"- Penalty: `{negative_penalty:.3f}`")
                            
                            st.markdown(f"**Final Score: `{final_score:.3f}`**")
                            st.markdown(f"**Threshold: `{threshold}`**")
                            
                            if final_score >= threshold:
                                st.success(f"✅ Score ≥ Threshold → **RELEVANT**")
                            else:
                                st.error(f"❌ Score < Threshold → **NOT RELEVANT**")
                        
                        # Additional information
                        st.markdown("### ℹ️ Additional Information")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Page Information")
                            st.info(f"**URL:** {test_url}")
                            if page_title:
                                st.info(f"**Title:** {page_title}")
                            st.info(f"**Content Length:** {len(content):,} characters")
                            
                            # Show content preview
                            with st.expander("📄 Content Preview", expanded=False):
                                st.text_area(
                                    "First 1000 characters:",
                                    content[:1000] + "..." if len(content) > 1000 else content,
                                    height=200,
                                    disabled=True
                                )
                        
                        with col2:
                            st.markdown("#### Test Parameters")
                            st.info(f"**Keyword:** {keyword}")
                            st.info(f"**Embedding Model:** {embedding_model}")
                            st.info(f"**Ollama Base URL:** {ollama_base_url}")
                            st.info(f"**Headless Mode:** {'Yes' if headless_mode else 'No'}")
                            st.info(f"**Title Extraction:** {'Enabled' if extract_title else 'Disabled'}")
                        
                        # Raw results (for debugging)
                        with st.expander("🔧 Raw Results (JSON)", expanded=False):
                            st.json(relevance_results)
                        
                except Exception as e:
                    st.error(f"❌ Error during testing: {str(e)}")
                    import traceback
                    with st.expander("📋 Error Details", expanded=False):
                        st.code(traceback.format_exc())
                    
                    # Make sure browser is closed
                    try:
                        if 'url_extractor' in locals():
                            url_extractor._close_browser()
                    except Exception:
                        pass

# Display previous results if available
if st.session_state.test_results and not test_button:
    st.markdown("---")
    st.markdown("### 📋 Previous Test Results")
    results = st.session_state.test_results
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**URL:** {results['url']}")
        st.info(f"**Keyword:** {results['keyword']}")
    with col2:
        final_score = results['relevance'].get('final_score', 0.0)
        confidence = results['relevance'].get('confidence', 'unknown')
        is_relevant = results['relevance'].get('is_relevant', False)
        st.metric("Previous Score", f"{final_score:.3f}", confidence.upper())
        st.caption("✅ Relevant" if is_relevant else "❌ Not Relevant")
    
    st.info("Click '🧪 Test Relevance Score' again to run a new test.")




