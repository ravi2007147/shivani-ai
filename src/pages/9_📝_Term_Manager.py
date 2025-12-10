"""Term Manager Page - Self-Learning System.

This page allows you to:
- View all terms stored in the database
- Gather more information about terms using unvisited links or auto-discovery
- Manage terms and their associated content/links
"""

import streamlit as st
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.term_db import TermDBManager

logger = logging.getLogger(__name__)

# Initialize session state variables if not already initialized
if 'selected_profiles' not in st.session_state:
    # Try to load profiles from ProfileManager
    try:
        from src.utils.kb_manager import ProfileManager
        profile_manager = ProfileManager()
        all_profiles = profile_manager.load_profiles()
        st.session_state.all_profiles = all_profiles
        if all_profiles:
            st.session_state.selected_profiles = [all_profiles[0].get('id')]
        else:
            st.session_state.selected_profiles = ["default"]
    except Exception as e:
        logger.warning(f"Could not load profiles: {str(e)}")
        st.session_state.all_profiles = []
        st.session_state.selected_profiles = ["default"]

if 'all_profiles' not in st.session_state:
    try:
        from src.utils.kb_manager import ProfileManager
        profile_manager = ProfileManager()
        st.session_state.all_profiles = profile_manager.load_profiles()
    except Exception as e:
        logger.warning(f"Could not load profiles: {str(e)}")
        st.session_state.all_profiles = []


def _assess_new_information(
    term_name: str,
    existing_content: str,
    new_content: str,
    ollama_model: str = "mistral",
    ollama_base_url: str = "http://localhost:11434"
) -> Dict:
    """Assess if newly discovered content contains meaningful new information.
    
    Args:
        term_name: Term/topic name
        existing_content: Existing content from database (before this gathering)
        new_content: Newly discovered content
        ollama_model: LLM model name
        ollama_base_url: Ollama base URL
        
    Returns:
        Dictionary with assessment results:
            - has_meaningful_new_info: bool
            - meaningful_score: float (0.0 to 1.0)
            - summary: str
            - new_information_points: List[str]
    """
    try:
        from langchain_ollama import OllamaLLM
        import json
        import re
        
        llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.2
        )
        
        # Truncate content if too long (keep first 15000 chars each)
        existing_preview = existing_content[:15000] if len(existing_content) > 15000 else existing_content
        new_preview = new_content[:15000] if len(new_content) > 15000 else new_content
        
        existing_length = len(existing_content)
        new_length = len(new_content)
        
        prompt = f"""You are an information assessment system. Your task is to determine if newly discovered content contains meaningful NEW information compared to existing content.

Topic/Term: {term_name}

EXISTING CONTENT ({existing_length} characters total, showing first {len(existing_preview)}):
---
{existing_preview}
{'... [Content truncated for context]' if existing_length > 15000 else ''}
---

NEW CONTENT ({new_length} characters total, showing first {len(new_preview)}):
---
{new_preview}
{'... [Content truncated for context]' if new_length > 15000 else ''}
---

Instructions:
1. Compare the NEW content with the EXISTING content
2. Determine if the new content contains MEANINGFUL NEW INFORMATION that:
   - Adds new facts, details, or insights not in existing content
   - Provides additional context, examples, or explanations
   - Contains updated or more recent information
   - Introduces new concepts, features, or aspects of the topic
3. Ignore minor differences, reformulations, or redundant information
4. Rate how meaningful the new information is (0.0 to 1.0):
   - 0.8-1.0: Highly meaningful - adds substantial new information, facts, or insights
   - 0.5-0.79: Moderately meaningful - adds some new information or context
   - 0.3-0.49: Somewhat meaningful - adds minor new details or clarifications
   - 0.0-0.29: Not meaningful - mostly redundant or overlaps significantly with existing content

Return ONLY a valid JSON object with this structure:
{{
  "has_meaningful_new_info": true or false,
  "meaningful_score": 0.0 to 1.0,
  "summary": "Brief summary of assessment (2-3 sentences)",
  "new_information_points": [
    "First key new information point",
    "Second new information point",
    "Third new information point"
  ]
}}

Answer:"""
        
        response = llm.invoke(prompt).strip()
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            # Validate result structure
            if 'has_meaningful_new_info' in result and 'meaningful_score' in result:
                result['meaningful_score'] = float(result.get('meaningful_score', 0.0))
                result['summary'] = result.get('summary', 'Assessment completed.')
                result['new_information_points'] = result.get('new_information_points', [])
                return result
        
        # Fallback if JSON parsing fails
        logger.warning("Failed to parse assessment JSON, using fallback")
        has_meaningful = "yes" in response.lower() or "meaningful" in response.lower() or "new information" in response.lower()
        return {
            'has_meaningful_new_info': has_meaningful,
            'meaningful_score': 0.5 if has_meaningful else 0.2,
            'summary': "Assessment completed, but detailed analysis unavailable.",
            'new_information_points': []
        }
        
    except Exception as e:
        logger.error(f"Error assessing new information: {str(e)}", exc_info=True)
        return {
            'has_meaningful_new_info': False,
            'meaningful_score': 0.0,
            'summary': f"Assessment failed: {str(e)}",
            'new_information_points': []
        }

# Page configuration
st.set_page_config(
    page_title="Term Manager",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Term Manager")
st.markdown("**Self-Learning System** - Manage terms and gather additional information automatically.")
st.markdown("Terms are automatically saved when you use the 'Content Analysis' button in 'Create Knowledge Base' → 'Structured URL Input'.")

# Initialize term database manager
term_db = TermDBManager()

# Sidebar filters
with st.sidebar:
    st.markdown("### 🔍 Filters")
    
    status_filter = st.selectbox(
        "Status",
        ["All", "active", "completed", "archived"],
        index=0
    )
    
    sort_by = st.selectbox(
        "Sort By",
        ["Updated (Newest)", "Updated (Oldest)", "Created (Newest)", "Created (Oldest)", "Term Name"],
        index=0
    )

# Get all terms
all_terms = term_db.list_terms(status=None if status_filter == "All" else status_filter)

# Sort terms
if sort_by == "Updated (Newest)":
    all_terms.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
elif sort_by == "Updated (Oldest)":
    all_terms.sort(key=lambda x: x.get('updated_at', ''), reverse=False)
elif sort_by == "Created (Newest)":
    all_terms.sort(key=lambda x: x.get('created_at', ''), reverse=True)
elif sort_by == "Created (Oldest)":
    all_terms.sort(key=lambda x: x.get('created_at', ''), reverse=False)
elif sort_by == "Term Name":
    all_terms.sort(key=lambda x: x.get('term', '').lower())

# Display terms
if not all_terms:
    st.info("📭 **No terms found.** Terms will be automatically added when you use 'Content Analysis' in 'Create Knowledge Base' → 'Structured URL Input'.")
    st.markdown("""
    ### How to add terms:
    1. Go to **Create Knowledge Base** tab
    2. Select **Structured URL Input**
    3. Enter a URL (e.g., `https://example.com`)
    4. Click **Content Analysis** button
    5. The term will be automatically saved to the database!
    """)
else:
    st.markdown(f"**Found {len(all_terms)} term(s)**")
    st.markdown("---")
    
    # Display each term
    for idx, term_data in enumerate(all_terms):
        term_id = term_data['id']
        term_name = term_data['term']
        status = term_data['status']
        created_at = term_data['created_at']
        updated_at = term_data['updated_at']
        last_gathered_at = term_data.get('last_gathered_at')
        original_url = term_data.get('original_url')
        domain = term_data.get('domain')
        
        # Get statistics
        stats = term_db.get_term_stats(term_id)
        content_count = stats.get('content_count', 0)
        total_links = stats.get('total_links', 0)
        visited_links = stats.get('visited_links', 0)
        unvisited_links = stats.get('unvisited_links', 0)
        avg_relevance_score = stats.get('avg_relevance_score', 0.0)
        
        # Create expandable section for each term
        with st.expander(
            f"**{term_name}** | Status: {status} | Content: {content_count} | Links: {total_links} ({unvisited_links} unvisited)",
            expanded=False
        ):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Term:** {term_name}")
                if original_url:
                    st.markdown(f"**Original URL:** [{original_url}]({original_url})")
                if domain:
                    st.markdown(f"**Domain:** {domain}")
                st.markdown(f"**Status:** {status}")
            
            with col2:
                st.markdown("**📊 Statistics**")
                st.metric("Content Items", content_count)
                st.metric("Total Links", total_links)
                st.metric("Unvisited Links", unvisited_links)
                if avg_relevance_score > 0:
                    st.metric("Avg Relevance", f"{avg_relevance_score:.2f}")
            
            with col3:
                st.markdown("**📅 Timestamps**")
                if created_at:
                    try:
                        created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        st.caption(f"Created: {created_dt.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.caption(f"Created: {created_at}")
                
                if updated_at:
                    try:
                        updated_dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        st.caption(f"Updated: {updated_dt.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.caption(f"Updated: {updated_at}")
                
                if last_gathered_at:
                    try:
                        gathered_dt = datetime.fromisoformat(last_gathered_at.replace('Z', '+00:00'))
                        st.caption(f"Last Gathered: {gathered_dt.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.caption(f"Last Gathered: {last_gathered_at}")
            
            st.markdown("---")
            
            # Action buttons
            col_action1, col_action2, col_action3, col_action4, col_action5 = st.columns(5)
            
            with col_action1:
                gather_button = st.button(
                    "🔍 Gather More Information",
                    key=f"gather_{term_id}",
                    use_container_width=True,
                    type="primary"
                )
            
            with col_action2:
                view_content_button = st.button(
                    "📄 View Content",
                    key=f"view_content_{term_id}",
                    use_container_width=True
                )
            
            with col_action3:
                view_links_button = st.button(
                    "🔗 View Links",
                    key=f"view_links_{term_id}",
                    use_container_width=True
                )
            
            with col_action4:
                status_options = ["active", "completed", "archived"]
                current_status_idx = status_options.index(status) if status in status_options else 0
                new_status = st.selectbox(
                    "Change Status",
                    status_options,
                    index=current_status_idx,
                    key=f"status_{term_id}"
                )
                if new_status != status:
                    term_db.update_term_status(term_id, new_status)
                    st.rerun()
            
            with col_action5:
                delete_term_button = st.button(
                    "🗑️ Delete Term",
                    key=f"delete_term_{term_id}",
                    use_container_width=True,
                    type="secondary"
                )
            
            # Handle gather more information button
            if gather_button:
                with st.spinner(f"🔍 Gathering more information about '{term_name}'..."):
                    try:
                        # Get ALL unvisited links (no limit - process all of them)
                        unvisited_links_list = term_db.get_unvisited_links(term_id, limit=None)  # No limit - get all
                        unvisited_urls = [link['url'] for link in unvisited_links_list]
                        
                        if not unvisited_urls:
                            st.info("ℹ️ No unvisited links found. Starting web search to discover new links...")
                        
                        # Get configuration from session state or defaults
                        ollama_model = st.session_state.get('ollama_model', 'mistral')
                        ollama_base_url = st.session_state.get('ollama_base_url', 'http://localhost:11434')
                        embedding_model = st.session_state.get('embedding_model', 'nomic-embed-text')
                        
                        # Import necessary components
                        from src.learning_system import ContentAnalyzer, AutoDiscoveryAgent
                        from src.rag import VectorStoreManager
                        from src.config import MAX_URLS_TO_EXTRACT, MAX_SEARCH_RESULTS, MAX_PAGES_TO_CRAWL, DOMAIN_RELEVANCE_THRESHOLD, DOMAIN_MIN_PAGES_FOR_BLACKLIST, LINK_RELEVANCE_THRESHOLD, DOMAIN_MAX_LOW_RELEVANCE_LINKS
                        
                        # Get profile ID (with fallback to default)
                        profile_id = (
                            st.session_state.selected_profiles[0] 
                            if st.session_state.get('selected_profiles') and len(st.session_state.selected_profiles) > 0 
                            else "default"
                        )
                        
                        # Initialize content analyzer
                        rag_pipeline = st.session_state.get('rag_pipeline')
                        vectorstores = st.session_state.get('vectorstores', [])
                        
                        analyzer = ContentAnalyzer(
                            rag_pipeline=rag_pipeline,
                            vectorstores=vectorstores,
                            retriever=st.session_state.get('retriever'),
                            ollama_model=ollama_model,
                            ollama_base_url=ollama_base_url
                        )
                        
                        # Initialize vectorstore manager
                        vectorstore_manager = VectorStoreManager(profile_id=profile_id)
                        
                        discovery_result = None
                        extracted_content = []
                        discovered_links_list = []
                        source_urls = []
                        
                        # STEP 1: Process ALL unvisited links (if available)
                        if unvisited_urls:
                            st.info(f"🔗 Found {len(unvisited_urls)} unvisited links. Processing ALL links...")
                            st.info(f"   ℹ️ Newly discovered links will be saved for the next round.")
                            
                            # Initialize auto-discovery agent to use its extraction methods
                            auto_agent = AutoDiscoveryAgent(
                                ollama_model=ollama_model,
                                ollama_base_url=ollama_base_url,
                                headless=False,
                                search_engine="duckduckgo"
                            )
                            
                            # Extract content from ALL unvisited links directly (no web search)
                            try:
                                extracted_content = auto_agent.extract_content_from_urls(
                                    urls=unvisited_urls,
                                    max_content_length=10000,
                                    topic=term_name
                                )
                                
                                # Collect discovered links from extracted pages (these will be saved for next round)
                                for item in extracted_content:
                                    discovered_links = item.get('discovered_links', [])
                                    if discovered_links:
                                        discovered_links_list.extend(discovered_links)
                                
                                visited_count = sum(1 for item in extracted_content if item.get('success'))
                                st.success(f"✅ Processed {visited_count}/{len(unvisited_urls)} unvisited links")
                                
                                # Mark only the URLs we actually visited as visited
                                # (Don't mark unvisited URLs that failed)
                                visited_urls_this_round = set()
                                for item in extracted_content:
                                    if item.get('url'):
                                        visited_urls_this_round.add(item.get('url'))
                                
                                # Mark visited URLs
                                for url in visited_urls_this_round:
                                    term_db.mark_link_visited(term_id, url)
                                
                                # Log discovered links count
                                if discovered_links_list:
                                    unique_new_links = len(set(link.get('url', '') for link in discovered_links_list if link.get('url')))
                                    st.info(f"   🔗 Discovered {unique_new_links} new links (saved for next round)")
                                
                                # Skip web search since we processed all unvisited links
                                st.info("✅ All unvisited links processed. Newly discovered links saved for next round.")
                            finally:
                                # Always close browser after extraction
                                try:
                                    auto_agent._close_browser()
                                    logger.info("Browser closed after unvisited links extraction")
                                except Exception as e:
                                    logger.warning(f"Error closing browser: {str(e)}")
                        
                        # STEP 2: Only fall back to web search if NO unvisited links were found
                        if not unvisited_urls:
                            st.info("🔍 No unvisited links found. Starting web search...")
                            
                            # Trigger auto-discovery (web search + extraction)
                            discovery_result = analyzer.trigger_auto_discovery(
                                topic=term_name,
                                knowledge_template=None,
                                vectorstore_manager=vectorstore_manager,
                                embedding_model=embedding_model,
                                profile_id=profile_id,
                                max_search_results=MAX_SEARCH_RESULTS,
                                max_urls_to_extract=MAX_URLS_TO_EXTRACT,
                                max_pages_to_crawl=MAX_PAGES_TO_CRAWL,
                                is_from_url=bool(original_url)
                            )
                            
                            if discovery_result.get('success'):
                                # Merge extracted content from web search
                                web_extracted_content = discovery_result.get('extracted_content', [])
                                extracted_content.extend(web_extracted_content)
                                
                                # Merge discovered links from web search (these will be saved for next round)
                                web_discovered_links = discovery_result.get('discovered_links', [])
                                if web_discovered_links:
                                    discovered_links_list.extend(web_discovered_links)
                                
                                # Get source URLs from web search
                                source_urls = discovery_result.get('source_urls', [])
                                
                                # Mark visited URLs from web search
                                for item in web_extracted_content:
                                    if item.get('url'):
                                        term_db.mark_link_visited(term_id, item.get('url'))
                        
                        # Process results (either from unvisited links, web search, or both)
                        if extracted_content or (discovery_result and discovery_result.get('success')):
                            # Calculate successful content count BEFORE storing
                            successful_content = sum(1 for item in extracted_content if item.get('success'))
                            
                            # Get existing content BEFORE storing new content (for comparison)
                            existing_content_list_before = term_db.get_term_content(term_id)
                            existing_count_before = len(existing_content_list_before)
                            
                            # Combine existing content for comparison
                            existing_content_text = ""
                            for content_item in existing_content_list_before:
                                existing_content_text += content_item.get('content', '') + "\n\n"
                            
                            # Combine new content for comparison
                            new_content_text = ""
                            for item in extracted_content:
                                if item.get('success') and item.get('content'):
                                    new_content_text += item.get('content', '') + "\n\n"
                            
                            # If we have content from unvisited links (not from web search), we need to:
                            # 1. Verify content relevance
                            # 2. Summarize knowledge
                            # 3. Create vectorstore
                            # 4. Register in KB manager
                            # 5. Add to RAG pipeline
                            knowledge_vectorstore = None
                            knowledge_persist_dir = None
                            knowledge_kb_id = None
                            knowledge_to_display = None
                            
                            # Process content through knowledge base pipeline if:
                            # 1. We have content from unvisited links only (no web search), OR
                            # 2. We have content from both unvisited links AND web search (but web search already created KB)
                            # For case 2, web search already created KB, so we don't need to create another one
                            # For case 1, we need to create KB ourselves
                            
                            # Create knowledge base if:
                            # - We have content from unvisited links AND
                            # - Web search didn't happen (no discovery_result) OR web search failed
                            # Web search already creates KB when successful, so we don't duplicate
                            needs_kb_creation = (
                                extracted_content and 
                                (not discovery_result or not discovery_result.get('success'))
                            )
                            
                            if needs_kb_creation:
                                # Content from unvisited links only (or web search failed) - need to process through pipeline
                                st.info("📝 Processing content through knowledge base pipeline...")
                                
                                try:
                                    # Initialize auto-discovery agent for verification and summarization
                                    pipeline_agent = AutoDiscoveryAgent(
                                        ollama_model=ollama_model,
                                        ollama_base_url=ollama_base_url,
                                        headless=False,
                                        search_engine="duckduckgo"
                                    )
                                    
                                    try:
                                        # Step 1: Verify content relevance
                                        st.info("   🔍 Verifying content relevance...")
                                        verified_content = pipeline_agent.verify_content_relevance(term_name, extracted_content)
                                        verified_count = sum(1 for item in verified_content if item.get('verified') and item.get('success') and item.get('content'))
                                        
                                        if verified_count == 0:
                                            st.warning("   ⚠️ No verified relevant content found. Content may not be relevant to the term.")
                                        else:
                                            st.success(f"   ✅ Verified {verified_count}/{successful_content} content items as relevant")
                                            
                                            # Step 2: Summarize knowledge
                                            st.info("   🤖 Summarizing knowledge...")
                                            # Create mock search_results for summarize_knowledge
                                            mock_search_results = {
                                                'query': term_name,
                                                'organic_results': [],
                                                'total_results': 0,
                                                'success': True
                                            }
                                            
                                            summarize_success, summarized_knowledge, summarize_error, summary_source_urls = pipeline_agent.summarize_knowledge(
                                                topic=term_name,
                                                search_results=mock_search_results,
                                                extracted_content=verified_content,
                                                knowledge_template=None
                                            )
                                            
                                            if summarize_success and summarized_knowledge:
                                                st.success(f"   ✅ Knowledge summarized ({len(summarized_knowledge)} characters)")
                                                
                                                # Store summarized knowledge for display
                                                knowledge_to_display = summarized_knowledge
                                                
                                                # Step 3: Create vectorstore
                                                st.info("   💾 Creating knowledge base...")
                                                knowledge_vectorstore, knowledge_persist_dir = vectorstore_manager.create_vectorstore(
                                                    summarized_knowledge,
                                                    embedding_model,
                                                    ollama_base_url,
                                                    source_urls=summary_source_urls,
                                                    topic=term_name
                                                )
                                                
                                                knowledge_kb_id = os.path.basename(knowledge_persist_dir)
                                                
                                                # Step 4: Register in KB manager
                                                from src.utils.kb_manager import KnowledgeBaseManager
                                                kb_manager = KnowledgeBaseManager(profile_id=profile_id)
                                                kb_manager.register_knowledge_base(
                                                    kb_id=knowledge_kb_id,
                                                    persist_dir=knowledge_persist_dir,
                                                    text_preview=summarized_knowledge[:1000],
                                                    chunk_count=0,
                                                    title=f"Term Manager: {term_name}"
                                                )
                                                
                                                # Step 5: Add to RAG pipeline
                                                if st.session_state.rag_pipeline:
                                                    st.session_state.rag_pipeline.add_vectorstore(knowledge_vectorstore)
                                                if st.session_state.vectorstores:
                                                    st.session_state.vectorstores.append(knowledge_vectorstore)
                                                
                                                st.success("   ✅ Knowledge base created and added to RAG pipeline!")
                                                
                                                # Update source_urls from summary
                                                if summary_source_urls:
                                                    source_urls.extend(summary_source_urls)
                                            else:
                                                st.warning(f"   ⚠️ Could not summarize knowledge: {summarize_error}")
                                    finally:
                                        # Always close browser
                                        pipeline_agent._close_browser()
                                        logger.info("Browser closed after pipeline processing")
                                        
                                except Exception as e:
                                    logger.error(f"Error processing content through pipeline: {str(e)}", exc_info=True)
                                    st.warning(f"⚠️ Error processing through pipeline: {str(e)}")
                            
                            # Store new content in database and update link relevance
                            for item in extracted_content:
                                if item.get('success') and item.get('content'):
                                    url = item.get('url', '')
                                    is_relevant = item.get('verified', False) or item.get('relevance_score', 0.0) > 0.5
                                    relevance_score = item.get('relevance_score')
                                    
                                    # Store content
                                    term_db.add_content(
                                        term_id=term_id,
                                        content=item.get('content', ''),
                                        source_url=url,
                                        relevance_score=relevance_score,
                                        confidence_score=item.get('confidence_score', 1.0),
                                        chunk_count=0
                                    )
                                    
                                    # Update link relevance information
                                    # Also check if link should be removed if relevance is too low
                                    if url:
                                        term_db.update_link_relevance(
                                            term_id=term_id,
                                            url=url,
                                            is_relevant=is_relevant,
                                            relevance_score=relevance_score
                                        )
                                        
                                        # Note: Links with low relevance are already marked as visited
                                        # Domain blacklisting will happen in the analysis step below
                                        # No need to remove from unvisited queue here since we already visited it
                            
                            # Collect all URLs we visited in this round (to avoid marking duplicates)
                            visited_urls_set = set()
                            for item in extracted_content:
                                if item.get('url'):
                                    visited_urls_set.add(item.get('url'))
                            
                            # Remove duplicates from discovered links and filter out already-visited URLs
                            # Note: We don't filter by relevance_score here because relevance is calculated AFTER visiting
                            # Newly discovered links should NOT be processed in this round - save for next round
                            unique_discovered_links = {}
                            
                            for link in discovered_links_list:
                                url = link.get('url', '')
                                if not url or url in visited_urls_set:
                                    continue
                                
                                # Store all discovered links (relevance will be calculated when visited)
                                # Links from blacklisted domains are already filtered by get_unvisited_links
                                if url not in unique_discovered_links:
                                    unique_discovered_links[url] = link
                            
                            discovered_urls = [link['url'] for link in unique_discovered_links.values()]
                            
                            # Store ALL discovered links as unvisited (for next round)
                            # Prioritize related links but store all related ones + many others
                            if discovered_urls:
                                # Prioritize related links
                                related_urls = [link['url'] for link in unique_discovered_links.values() if link.get('is_related', False)]
                                other_urls = [url for url in discovered_urls if url not in related_urls]
                                
                                # Store ALL related links (no limit), and up to 50 other links for next round
                                urls_to_store = related_urls + other_urls[:50]
                                
                                if urls_to_store:
                                    term_db.add_links(
                                        term_id=term_id,
                                        urls=urls_to_store,
                                        visited=False,  # Save as unvisited for next round
                                        link_type='external'
                                    )
                                    st.info(f"   💾 Saved {len(urls_to_store)} newly discovered links for next round ({len(related_urls)} related, {len(other_urls[:50])} other)")
                            
                            # Store source URLs from web search if available (also for next round)
                            # Note: Source URLs from web search typically don't have relevance scores yet,
                            # so we'll store them all (they'll be filtered when processed)
                            if source_urls:
                                # Filter out URLs we already visited
                                new_source_urls = [url for url in source_urls if url not in visited_urls_set]
                                if new_source_urls:
                                    term_db.add_links(
                                        term_id=term_id,
                                        urls=new_source_urls,
                                        visited=False,  # Save as unvisited for next round
                                        link_type='external'
                                    )
                                    st.info(f"   💾 Saved {len(new_source_urls)} source URLs from web search for next round")
                            
                            # Note: URLs we visited are already marked as visited during processing above
                            # This section only stores NEW links for next round
                            
                            # Analyze visited links for low relevance and auto-blacklist domains
                            # Relevance scores are calculated AFTER visiting, so we check visited links
                            st.info("🔍 Analyzing visited links for domain relevance...")
                            
                            # Get domain statistics (includes low_relevance_count)
                            # Pass LINK_RELEVANCE_THRESHOLD to count low-relevance links correctly
                            domain_stats = term_db.get_domain_statistics(term_id, relevance_threshold=LINK_RELEVANCE_THRESHOLD)
                            
                            # Auto-blacklist domains with too many low-relevance visited links
                            auto_blacklisted_domains = []
                            total_low_relevance = 0
                            
                            for domain, stats in domain_stats.items():
                                low_relevance_count = stats.get('low_relevance_count', 0)
                                pages_visited = stats.get('pages_visited', 0)
                                pages_relevant = stats.get('pages_relevant', 0)
                                
                                total_low_relevance += low_relevance_count
                                
                                # Auto-blacklist domain if it has >= DOMAIN_MAX_LOW_RELEVANCE_LINKS low-relevance visited links
                                if low_relevance_count >= DOMAIN_MAX_LOW_RELEVANCE_LINKS:
                                    # Check if domain is already blacklisted
                                    if not term_db.is_domain_blacklisted(term_id, domain):
                                        reason = f"Auto-blacklisted: {low_relevance_count} visited links with low relevance (<{LINK_RELEVANCE_THRESHOLD})"
                                        term_db.blacklist_domain(
                                            term_id=term_id,
                                            domain=domain,
                                            pages_visited=pages_visited,
                                            pages_relevant=pages_relevant,
                                            reason=reason
                                        )
                                        auto_blacklisted_domains.append(domain)
                                        logger.info(f"Auto-blacklisted domain {domain}: {low_relevance_count} low-relevance visited links")
                            
                            if total_low_relevance > 0:
                                st.info(f"   📊 Found {total_low_relevance} visited link(s) with low relevance (<{LINK_RELEVANCE_THRESHOLD})")
                                
                            if auto_blacklisted_domains:
                                st.warning(f"   ⚠️ Auto-blacklisted {len(auto_blacklisted_domains)} domain(s): {', '.join(auto_blacklisted_domains)}")
                                st.caption(f"   ℹ️ Domains with >= {DOMAIN_MAX_LOW_RELEVANCE_LINKS} visited links having relevance < {LINK_RELEVANCE_THRESHOLD} are excluded from future processing.")
                            
                            # Also analyze domain relevance ratio (for domains with many visited pages)
                            # This is a secondary check for domains that may not have enough low-relevance links
                            # but still have a low overall relevance ratio
                            st.info("🔍 Analyzing domain relevance ratio...")
                            blacklisted_count = 0
                            
                            # Re-fetch domain stats (since we might have blacklisted some domains)
                            domain_stats_ratio = term_db.get_domain_statistics(term_id, relevance_threshold=LINK_RELEVANCE_THRESHOLD)
                            
                            for domain, stats in domain_stats_ratio.items():
                                pages_visited = stats['pages_visited']
                                pages_relevant = stats['pages_relevant']
                                relevance_ratio = stats['relevance_ratio']
                                
                                # Skip if already blacklisted (from previous check)
                                if term_db.is_domain_blacklisted(term_id, domain):
                                    continue
                                
                                # Blacklist domain if:
                                # 1. We've visited at least MIN_PAGES_FOR_BLACKLIST pages
                                # 2. Relevance ratio is below threshold
                                if pages_visited >= DOMAIN_MIN_PAGES_FOR_BLACKLIST and relevance_ratio < DOMAIN_RELEVANCE_THRESHOLD:
                                    reason = f"Low relevance ratio: {pages_relevant}/{pages_visited} pages relevant ({relevance_ratio:.1%} < {DOMAIN_RELEVANCE_THRESHOLD:.1%} threshold)"
                                    term_db.blacklist_domain(
                                        term_id=term_id,
                                        domain=domain,
                                        pages_visited=pages_visited,
                                        pages_relevant=pages_relevant,
                                        reason=reason
                                    )
                                    blacklisted_count += 1
                                    logger.info(f"Blacklisted domain {domain}: {relevance_ratio:.1%} relevance ratio ({pages_relevant}/{pages_visited} relevant)")
                            
                            if blacklisted_count > 0:
                                st.warning(f"   ⚠️ Blacklisted {blacklisted_count} domain(s) with low relevance ratio (<{DOMAIN_RELEVANCE_THRESHOLD:.1%})")
                                st.caption(f"   ℹ️ Domains with <{DOMAIN_RELEVANCE_THRESHOLD:.1%} relevant pages (min {DOMAIN_MIN_PAGES_FOR_BLACKLIST} visited) are excluded from future searches.")
                            
                            if not auto_blacklisted_domains and blacklisted_count == 0:
                                if total_low_relevance == 0:
                                    st.success(f"   ✅ All domains and links meet relevance thresholds")
                                else:
                                    st.info(f"   ℹ️ Found {total_low_relevance} low-relevance links, but no domains need blacklisting")
                            
                            # Assess if meaningful new information was found
                            assessment_result = None
                            try:
                                # Assess if new meaningful information was found
                                if new_content_text.strip() and existing_content_text.strip():
                                    assessment_result = _assess_new_information(
                                        term_name=term_name,
                                        existing_content=existing_content_text,
                                        new_content=new_content_text,
                                        ollama_model=ollama_model,
                                        ollama_base_url=ollama_base_url
                                    )
                                elif new_content_text.strip() and not existing_content_text.strip():
                                    # First time gathering - new content is always meaningful
                                    assessment_result = {
                                        'has_meaningful_new_info': True,
                                        'meaningful_score': 1.0,
                                        'summary': "This is the first content gathered for this term, so all information is new.",
                                        'new_information_points': []
                                    }
                            except Exception as e:
                                logger.warning(f"Could not assess new information: {str(e)}")
                            
                            # Update last gathered timestamp
                            term_db.update_term_last_gathered(term_id)
                            
                            # Count saved links for next round
                            saved_for_next_round = len(discovered_urls) if discovered_urls else 0
                            related_links_count = sum(1 for link in unique_discovered_links.values() if link.get('is_related', False)) if unique_discovered_links else 0
                            
                            st.success(f"✅ Successfully gathered more information about '{term_name}'!")
                            st.info(f"   - Added {successful_content} new content items")
                            st.info(f"   - Processed {len(visited_urls_set)} links in this round")
                            if saved_for_next_round > 0:
                                st.info(f"   - Saved {saved_for_next_round} newly discovered links for next round ({related_links_count} related to topic)")
                                st.caption("💡 **Tip:** Click 'Gather More Information' again to process the newly discovered links.")
                            
                            # Display assessment of new information
                            if assessment_result:
                                has_meaningful_info = assessment_result.get('has_meaningful_new_info', False)
                                meaningful_score = assessment_result.get('meaningful_score', 0.0)
                                assessment_summary = assessment_result.get('summary', '')
                                new_information_points = assessment_result.get('new_information_points', [])
                                
                                st.markdown("---")
                                
                                if has_meaningful_info:
                                    st.success(f"🎯 **Meaningful New Information Found!** (Score: {meaningful_score:.2f}/1.0)")
                                    st.info(assessment_summary)
                                    
                                    if new_information_points:
                                        with st.expander("📋 Key New Information Points", expanded=False):
                                            for i, point in enumerate(new_information_points, 1):
                                                st.markdown(f"{i}. {point}")
                                else:
                                    st.warning(f"⚠️ **Limited New Information** (Score: {meaningful_score:.2f}/1.0)")
                                    st.info(assessment_summary)
                                    st.caption("💡 Tip: The new content may overlap significantly with existing information, or may not add substantial value.")
                            
                            elif successful_content > 0:
                                # If assessment failed but we have content, show basic info
                                st.info("ℹ️ **Content gathered**, but assessment could not be performed.")
                            
                            # Show discovered knowledge preview (from web search or pipeline)
                            # Note: knowledge_to_display is set in the pipeline processing section above
                            if discovery_result and discovery_result.get('success'):
                                # Web search was successful - use its knowledge
                                knowledge_to_display = discovery_result.get('knowledge', '')
                                # Update RAG pipeline if vectorstore was created
                                if discovery_result.get('vectorstore'):
                                    if st.session_state.rag_pipeline:
                                        st.session_state.rag_pipeline.add_vectorstore(discovery_result['vectorstore'])
                                    if st.session_state.vectorstores:
                                        st.session_state.vectorstores.append(discovery_result['vectorstore'])
                                    st.success("✅ New knowledge added to active knowledge bases!")
                            
                            # Display knowledge if available (from either web search or pipeline)
                            if knowledge_to_display:
                                with st.expander("📚 New Knowledge Discovered", expanded=False):
                                    preview = knowledge_to_display[:2000] + "..." if len(knowledge_to_display) > 2000 else knowledge_to_display
                                    st.markdown(preview)
                            
                            # Rerun to refresh the page
                            st.rerun()
                        elif discovery_result and not discovery_result.get('success'):
                            error_msg = discovery_result.get('error', 'Unknown error')
                            st.error(f"❌ Failed to gather information: {error_msg}")
                        else:
                            st.warning("⚠️ No content was extracted. Try again or check if links are accessible.")
                    
                    except Exception as e:
                        import traceback
                        error_trace = traceback.format_exc()
                        st.error(f"❌ Error gathering information: {str(e)}")
                        with st.expander("Technical Details"):
                            st.code(error_trace)
                        logger.error(f"Error gathering information for term {term_id}: {error_trace}")
            
            # Handle view content button
            if view_content_button:
                content_list = term_db.get_term_content(term_id)
                if content_list:
                    st.markdown(f"**📄 Content Items ({len(content_list)})**")
                    st.caption("💡 Click the delete button (🗑️) next to any content item to remove it.")
                    
                    # Handle delete confirmation first (before showing content)
                    delete_content_key = f'delete_content_confirm_{term_id}'
                    if delete_content_key in st.session_state:
                        delete_content_info = st.session_state[delete_content_key]
                        content_id_to_delete = delete_content_info['content_id']
                        url_to_delete = delete_content_info.get('source_url', 'N/A')
                        
                        st.markdown("---")
                        st.warning("🗑️ **Delete Content Confirmation**")
                        st.markdown(f"**Content ID:** {content_id_to_delete}")
                        st.markdown(f"**Source URL:** {url_to_delete}")
                        
                        col_confirm1, col_confirm2 = st.columns([1, 3])
                        
                        with col_confirm1:
                            if st.button("✅ Delete Content", key=f"delete_content_confirm_{term_id}", type="primary"):
                                try:
                                    deleted = term_db.delete_content(term_id, content_id_to_delete)
                                    if deleted:
                                        del st.session_state[delete_content_key]
                                        st.success(f"✅ Deleted content item ID: {content_id_to_delete}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Content not found or could not be deleted")
                                except Exception as e:
                                    st.error(f"❌ Error deleting content: {str(e)}")
                                    logger.error(f"Error deleting content: {str(e)}", exc_info=True)
                        
                        with col_confirm2:
                            if st.button("❌ Cancel", key=f"cancel_delete_content_{term_id}"):
                                del st.session_state[delete_content_key]
                                st.rerun()
                        
                        st.markdown("---")
                    
                    # Display content items with delete buttons
                    for i, content_item in enumerate(content_list):
                        content_id = content_item.get('id')
                        source_url = content_item.get('source_url', 'N/A')
                        relevance_score = content_item.get('relevance_score')
                        confidence_score = content_item.get('confidence_score', 1.0)
                        
                        # Build relevance display
                        relevance_display = f"Relevance: {relevance_score:.3f}" if relevance_score is not None else "Relevance: N/A"
                        
                        col_expander, col_delete = st.columns([9, 1])
                        
                        with col_expander:
                            with st.expander(f"Content #{i+1} | URL: {source_url[:50]}... | {relevance_display}"):
                                st.markdown(f"**Content ID:** {content_id}")
                                st.markdown(f"**Source URL:** {source_url}")
                                if relevance_score is not None:
                                    st.markdown(f"**Relevance Score:** {relevance_score:.3f}")
                                if confidence_score:
                                    st.markdown(f"**Confidence Score:** {confidence_score:.3f}")
                                
                                content_text = content_item.get('content', '')
                                preview = content_text[:1000] + "..." if len(content_text) > 1000 else content_text
                                st.text_area(
                                    "Content Preview",
                                    value=preview,
                                    height=200,
                                    disabled=True,
                                    key=f"content_preview_{term_id}_{content_id}"
                                )
                                if len(content_text) > 1000:
                                    st.caption(f"Showing first 1000 characters of {len(content_text)} total characters")
                        
                        with col_delete:
                            delete_content_button_key = f"delete_content_{term_id}_{content_id}"
                            delete_content_button = st.button("🗑️", key=delete_content_button_key, help=f"Delete content ID {content_id}")
                            
                            if delete_content_button:
                                # Store delete action in session state
                                st.session_state[delete_content_key] = {
                                    'content_id': content_id,
                                    'source_url': source_url
                                }
                                st.rerun()
                else:
                    st.info("No content items found for this term.")
            
            # Handle view links button
            if view_links_button:
                links_list = term_db.get_term_links(term_id)
                if links_list:
                    visited_links = [link for link in links_list if link.get('visited')]
                    unvisited_links = [link for link in links_list if not link.get('visited')]
                    
                    st.markdown(f"**🔗 Links ({len(links_list)} total)**")
                    
                    if unvisited_links:
                        st.markdown(f"**Unvisited Links ({len(unvisited_links)})**")
                        st.caption("💡 Click the delete button (🗑️) next to any unvisited link to remove it. You can choose to delete only the link or also blacklist its domain.")
                        
                        # Handle delete confirmation first (before showing links)
                        delete_confirmation_key = f'delete_link_confirm_{term_id}'
                        if delete_confirmation_key in st.session_state:
                            delete_info = st.session_state[delete_confirmation_key]
                            url_to_delete = delete_info['url']
                            domain_to_delete = delete_info['domain']
                            
                            st.markdown("---")
                            st.warning("🗑️ **Delete Link Confirmation**")
                            st.markdown(f"**URL:** {url_to_delete}")
                            st.markdown(f"**Domain:** {domain_to_delete}")
                            
                            col_confirm1, col_confirm2, col_confirm3 = st.columns([1, 1, 2])
                            
                            with col_confirm1:
                                if st.button("✅ Delete Link Only", key=f"delete_only_{term_id}", type="primary"):
                                    try:
                                        deleted = term_db.delete_link(term_id, url_to_delete)
                                        if deleted:
                                            del st.session_state[delete_confirmation_key]
                                            st.success(f"✅ Deleted link: {url_to_delete}")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Link not found or could not be deleted")
                                    except Exception as e:
                                        st.error(f"❌ Error deleting link: {str(e)}")
                                        logger.error(f"Error deleting link: {str(e)}", exc_info=True)
                            
                            with col_confirm2:
                                if st.button("🚫 Delete & Blacklist Domain", key=f"delete_blacklist_{term_id}", type="secondary"):
                                    try:
                                        # Delete the link first
                                        deleted = term_db.delete_link(term_id, url_to_delete)
                                        
                                        if not deleted:
                                            st.error(f"❌ Link not found or could not be deleted")
                                        else:
                                            # Blacklist the domain if it's not already blacklisted
                                            if domain_to_delete and domain_to_delete != 'Unknown':
                                                if not term_db.is_domain_blacklisted(term_id, domain_to_delete):
                                                    # Get domain stats if available
                                                    domain_stats = term_db.get_domain_statistics(term_id)
                                                    stats = domain_stats.get(domain_to_delete, {})
                                                    
                                                    pages_visited = stats.get('pages_visited', 0)
                                                    pages_relevant = stats.get('pages_relevant', 0)
                                                    
                                                    reason = f"Manually blacklisted when deleting link: {url_to_delete}"
                                                    term_db.blacklist_domain(
                                                        term_id=term_id,
                                                        domain=domain_to_delete,
                                                        pages_visited=pages_visited,
                                                        pages_relevant=pages_relevant,
                                                        reason=reason
                                                    )
                                                    st.success(f"✅ Deleted link and blacklisted domain: {domain_to_delete}")
                                                else:
                                                    st.success(f"✅ Deleted link. Domain {domain_to_delete} was already blacklisted.")
                                            
                                            del st.session_state[delete_confirmation_key]
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error deleting link and blacklisting domain: {str(e)}")
                                        logger.error(f"Error deleting link and blacklisting domain: {str(e)}", exc_info=True)
                            
                            with col_confirm3:
                                if st.button("❌ Cancel", key=f"cancel_delete_{term_id}"):
                                    del st.session_state[delete_confirmation_key]
                                    st.rerun()
                            
                            st.markdown("---")
                        
                        # Display unvisited links with relevance scores
                        for i, link in enumerate(unvisited_links):
                            url = link['url']
                            domain = link.get('domain', 'Unknown')
                            link_type = link.get('link_type', 'external')
                            relevance_score = link.get('relevance_score')
                            is_relevant = link.get('is_relevant', False)
                            
                            col_link, col_delete = st.columns([4, 1])
                            
                            with col_link:
                                # Build display string with relevance information
                                relevance_display = ""
                                if relevance_score is not None:
                                    relevance_display = f" | Relevance: {relevance_score:.3f}"
                                elif is_relevant:
                                    relevance_display = " | Relevance: ✅"
                                
                                st.markdown(f"- [{url}]({url}) | Domain: {domain} | Type: {link_type}{relevance_display}")
                            
                            with col_delete:
                                # Use a URL-safe key (replace special chars)
                                url_key = url.replace('/', '_').replace(':', '_').replace('.', '_').replace('?', '_').replace('&', '_').replace('=', '_')[:40]
                                delete_key = f"delete_link_{term_id}_{i}_{url_key}"
                                delete_button = st.button("🗑️", key=delete_key, help=f"Delete {url}")
                                
                                if delete_button:
                                    # Store delete action in session state
                                    st.session_state[delete_confirmation_key] = {
                                        'url': url,
                                        'domain': domain,
                                        'link_type': link_type
                                    }
                                    st.rerun()
                    
                    if visited_links:
                        with st.expander(f"Visited Links ({len(visited_links)})", expanded=False):
                            for link in visited_links[:20]:  # Show first 20
                                visited_at = link.get('visited_at', 'N/A')
                                domain = link.get('domain', 'Unknown')
                                is_relevant = link.get('is_relevant', False)
                                relevance_score = link.get('relevance_score')
                                
                                # Build relevance display
                                if relevance_score is not None:
                                    relevance_display = f"Relevance: {relevance_score:.3f}"
                                elif is_relevant:
                                    relevance_display = "Relevance: ✅"
                                else:
                                    relevance_display = "Relevance: ❌"
                                
                                st.markdown(f"- [{link['url']}]({link['url']}) | Domain: {domain} | Visited: {visited_at} | {relevance_display}")
                            if len(visited_links) > 20:
                                st.caption(f"... and {len(visited_links) - 20} more visited links")
                else:
                    st.info("No links found for this term.")
            
            # Handle delete term button
            delete_term_confirm_key = f'delete_term_confirm_{term_id}'
            if delete_term_button:
                # Store confirmation in session state
                st.session_state[delete_term_confirm_key] = True
                st.rerun()
            
            if delete_term_confirm_key in st.session_state:
                st.markdown("---")
                st.error("⚠️ **Delete Term Confirmation**")
                st.warning(f"⚠️ **WARNING: This will permanently delete ALL data for '{term_name}'!**")
                st.markdown("**This action will delete:**")
                st.markdown(f"- ❌ Term: {term_name}")
                st.markdown(f"- ❌ All content items ({content_count})")
                st.markdown(f"- ❌ All links ({total_links})")
                st.markdown(f"- ❌ All domain blacklist entries")
                st.markdown("**This action cannot be undone!**")
                
                col_confirm1, col_confirm2 = st.columns([1, 3])
                
                with col_confirm1:
                    if st.button("✅ Confirm Delete", key=f"confirm_delete_term_{term_id}", type="primary"):
                        try:
                            term_db.delete_term(term_id)
                            del st.session_state[delete_term_confirm_key]
                            st.success(f"✅ Successfully deleted term '{term_name}' and all associated data!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting term: {str(e)}")
                            logger.error(f"Error deleting term {term_id}: {str(e)}", exc_info=True)
                
                with col_confirm2:
                    if st.button("❌ Cancel", key=f"cancel_delete_term_{term_id}"):
                        del st.session_state[delete_term_confirm_key]
                        st.rerun()
            
            st.markdown("---")

