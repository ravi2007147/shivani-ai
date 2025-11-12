"""Article Search page for managing RSS feeds and articles."""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import RSSDB
from src.utils.rss_fetcher import fetch_and_save_feed, fetch_rss_feed
from src.utils.link_extractor import extract_domain
from src.utils.article_analyzer import analyze_article_content
from src.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_LLM_MODEL
from src.utils import fetch_ollama_models, get_default_model
from langchain_ollama import OllamaLLM

# Page configuration
st.set_page_config(
    page_title="Article Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Article Search")
st.markdown("Manage RSS feeds and articles")

# Initialize database
db = RSSDB()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # AI Configuration for article analysis
    st.subheader("🤖 AI Configuration")
    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama API endpoint",
        key="article_analyzer_ollama_url"
    )
    
    # Load models
    if 'ollama_models_article' not in st.session_state:
        st.session_state.ollama_models_article = fetch_ollama_models(ollama_base_url)
    
    # Model selection
    if st.session_state.ollama_models_article:
        default_model = get_default_model(st.session_state.ollama_models_article, DEFAULT_LLM_MODEL)
        ollama_model = st.selectbox(
            "AI Model for Article Analysis",
            options=st.session_state.ollama_models_article,
            index=st.session_state.ollama_models_article.index(default_model) if default_model in st.session_state.ollama_models_article else 0,
            help="LLM model used for analyzing article content",
            key="article_analyzer_model"
        )
    else:
        ollama_model = st.text_input(
            "AI Model for Article Analysis",
            value=DEFAULT_LLM_MODEL,
            help="LLM model used for analyzing article content",
            key="article_analyzer_model_input"
        )
    
    # Refresh models button
    if st.button("🔄 Refresh Models", key="refresh_models_article"):
        with st.spinner("Loading models..."):
            fetched_models = fetch_ollama_models(ollama_base_url)
            st.session_state.ollama_models_article = fetched_models
            if fetched_models:
                st.success(f"✅ Loaded {len(fetched_models)} model(s)")
            else:
                st.warning("⚠️ Could not fetch models. Make sure Ollama is running.")
    
    st.markdown("---")
    
    # Auto-check feeds
    auto_check = st.checkbox("Auto-check feeds on load", value=False)
    if auto_check:
        feeds_due = db.get_feeds_due_for_check(limit=10)
        if feeds_due:
            with st.spinner(f"Checking {len(feeds_due)} feed(s)..."):
                for feed in feeds_due:
                    # Check if domain is paused before fetching
                    feed_domain = extract_domain(feed['url'])
                    if db.is_domain_paused(feed_domain):
                        st.warning(f"⏸️ {feed['name']}: Domain {feed_domain} is paused")
                        continue
                    
                    success, message, saved_count, domain = fetch_and_save_feed(db, feed['id'], max_items=10)
                    if success:
                        st.success(f"✅ {feed['name']}: {message}")
                    else:
                        if "paused" in message.lower():
                            st.error(f"⏸️ {feed['name']}: {message}")
                        else:
                            st.warning(f"⚠️ {feed['name']}: {message}")
    
    st.markdown("---")
    
    # Paused domains management
    st.subheader("⏸️ Paused Domains")
    paused_domains = db.get_paused_domains()
    
    if paused_domains:
        st.warning(f"⚠️ {len(paused_domains)} domain(s) are currently paused")
        for paused_domain in paused_domains:
            with st.expander(f"⏸️ {paused_domain['domain']}", expanded=False):
                col_pause1, col_pause2 = st.columns([3, 1])
                with col_pause1:
                    if paused_domain.get('pause_reason'):
                        st.markdown(f"**Reason:** {paused_domain['pause_reason']}")
                    if paused_domain.get('last_error'):
                        st.markdown(f"**Last Error:** {paused_domain['last_error']}")
                    if paused_domain.get('error_count'):
                        st.markdown(f"**Error Count:** {paused_domain['error_count']}")
                    if paused_domain.get('paused_at'):
                        paused_at = datetime.fromisoformat(paused_domain['paused_at']) if isinstance(paused_domain['paused_at'], str) else paused_domain['paused_at']
                        st.markdown(f"**Paused At:** {paused_at.strftime('%Y-%m-%d %H:%M:%S')}")
                with col_pause2:
                    if st.button("▶️ Resume", key=f"resume_{paused_domain['domain']}", type="primary"):
                        success, msg = db.resume_domain(paused_domain['domain'])
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
    else:
        st.success("✅ No domains are currently paused")
    
    st.markdown("---")
    
    # Auto-delete old article links (not feed links)
    st.subheader("🗑️ Cleanup")
    auto_delete_enabled = st.checkbox("Auto-delete old article links on load", value=True, 
                                      help="Automatically delete old article links (not RSS feeds) when the page loads")
    days_old = st.number_input("Delete article links older than (days)", min_value=1, max_value=365, value=10, step=1,
                               help="Article links older than this many days (based on published date) will be deleted. RSS feeds are not affected.")
    
    # Check how many old articles exist
    old_count = db.get_old_articles_count(days_old=days_old)
    
    # Initialize or update cleanup tracking
    cleanup_key = "last_cleanup_time"
    cleanup_days_key = "last_cleanup_days"
    
    # Reset cleanup timestamp if days_old changed
    if cleanup_days_key in st.session_state:
        if st.session_state[cleanup_days_key] != days_old:
            # Days threshold changed, reset cleanup timestamp
            if cleanup_key in st.session_state:
                del st.session_state[cleanup_key]
    
    st.session_state[cleanup_days_key] = days_old
    
    # Auto-delete on load if enabled
    if auto_delete_enabled and old_count > 0:
        # Use timestamp to prevent running cleanup too frequently (max once per 30 seconds)
        cleanup_interval = 30  # seconds
        should_run_cleanup = True
        
        if cleanup_key in st.session_state:
            last_cleanup = st.session_state[cleanup_key]
            if isinstance(last_cleanup, str):
                last_cleanup = datetime.fromisoformat(last_cleanup)
            time_since_cleanup = (datetime.now() - last_cleanup).total_seconds()
            if time_since_cleanup < cleanup_interval:
                should_run_cleanup = False
        
        if should_run_cleanup:
            with st.spinner(f"Cleaning up article links older than {days_old} days..."):
                success, message, deleted_count = db.delete_old_articles(days_old=days_old)
                st.session_state[cleanup_key] = datetime.now().isoformat()
                if success and deleted_count > 0:
                    st.success(f"✅ {message}")
                    # Refresh the count after deletion
                    old_count = db.get_old_articles_count(days_old=days_old)
                elif success:
                    st.info(message)
                else:
                    st.error(message)
        else:
            # Show status if cleanup was recently run
            if old_count > 0:
                st.info(f"📊 {old_count} article link(s) older than {days_old} days (cleanup ran recently)")
            else:
                st.success(f"✅ All article links older than {days_old} days have been deleted")
    elif old_count > 0:
        # Show count and manual delete button if auto-delete is disabled
        st.info(f"📊 {old_count} article link(s) older than {days_old} days")
        if st.button("🗑️ Delete Old Article Links Now", type="primary", use_container_width=True):
            with st.spinner(f"Deleting article links older than {days_old} days..."):
                success, message, deleted_count = db.delete_old_articles(days_old=days_old)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    else:
        st.success(f"✅ No article links older than {days_old} days")

# Main tabs
tab1, tab2 = st.tabs(["Manage Links", "Manage Keywords"])

with tab1:
    # Sub-tabs for Manage Links
    subtab1, subtab2 = st.tabs(["Manage Feed", "View Articles"])
    
    with subtab1:
        st.subheader("Manage RSS Feeds")
        st.markdown("Add, edit, or delete RSS feeds")
        
        # Get all feeds
        feeds = db.get_feeds()
        
        # Add new feed form
        with st.expander("➕ Add New Feed", expanded=False):
            with st.form("add_feed_form"):
                feed_name = st.text_input("Feed Name *", placeholder="e.g., Tech News")
                feed_url = st.text_input("RSS Feed URL *", placeholder="https://example.com/feed.xml")
                check_frequency = st.number_input(
                    "Check Frequency (times per day) *",
                    min_value=1,
                    max_value=24,
                    value=1,
                    help="How many times per day to check for new articles"
                )
                
                submit_feed = st.form_submit_button("Add Feed", type="primary")
                
                if submit_feed:
                    if not feed_name or not feed_url:
                        st.error("Please fill in all required fields")
                    else:
                        # Validate URL by trying to fetch
                        with st.spinner("Validating RSS feed..."):
                            domain = extract_domain(feed_url)
                            # Check if domain is paused
                            if db.is_domain_paused(domain):
                                st.warning(f"⚠️ Domain {domain} is paused. Resume it first to add feeds from this domain.")
                            else:
                                success, message, articles, extracted_domain, status_code = fetch_rss_feed(
                                    feed_url, max_items=1, domain_pause_check=db.is_domain_paused
                                )
                                if success:
                                    # Add feed to database
                                    success_db, msg_db = db.add_feed(feed_name, feed_url, check_frequency)
                                    if success_db:
                                        st.success(msg_db)
                                        st.rerun()
                                    else:
                                        st.error(msg_db)
                                else:
                                    st.error(f"Invalid RSS feed: {message}")
                                    # Check if domain should be paused
                                    if status_code in [403, 429, 503, 504] or "Connection" in message or "Timeout" in message:
                                        db.pause_domain(domain, reason=f"HTTP {status_code}" if status_code else "Connection error", error_message=message)
                                        st.warning(f"⚠️ Domain {domain} has been paused due to connection issues. You can resume it manually.")
        
        # Display existing feeds
        if feeds:
            st.markdown("### Existing Feeds")
            
            for feed in feeds:
                feed_domain = extract_domain(feed['url'])
                is_domain_paused = db.is_domain_paused(feed_domain)
                # Show pause indicator in title
                pause_indicator = " ⏸️" if is_domain_paused else ""
                with st.expander(f"📰 {feed['name']} {'✅' if feed['is_active'] else '❌'}{pause_indicator}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        # Show pause indicator
                        if is_domain_paused:
                            pause_info = db.get_domain_pause_info(feed_domain)
                            st.error(f"⏸️ **Domain Paused:** {feed_domain}")
                            if pause_info:
                                if pause_info.get('pause_reason'):
                                    st.markdown(f"**Reason:** {pause_info['pause_reason']}")
                                if pause_info.get('last_error'):
                                    st.markdown(f"**Last Error:** {pause_info['last_error'][:100]}...")
                                if pause_info.get('paused_at'):
                                    paused_at = datetime.fromisoformat(pause_info['paused_at']) if isinstance(pause_info['paused_at'], str) else pause_info['paused_at']
                                    st.markdown(f"**Paused At:** {paused_at.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            st.markdown(f"**URL:** {feed['url']}")
                        
                        st.markdown(f"**Check Frequency:** {feed['check_frequency_per_day']} times per day")
                        if feed['last_checked']:
                            last_checked = datetime.fromisoformat(feed['last_checked']) if isinstance(feed['last_checked'], str) else feed['last_checked']
                            st.markdown(f"**Last Checked:** {last_checked.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            st.markdown("**Last Checked:** Never")
                        if feed['next_check']:
                            next_check = datetime.fromisoformat(feed['next_check']) if isinstance(feed['next_check'], str) else feed['next_check']
                            st.markdown(f"**Next Check:** {next_check.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Article count
                        article_count = db.get_article_count(feed_id=feed['id'])
                        st.markdown(f"**Articles:** {article_count}")
                    
                    with col2:
                        # Edit feed
                        if st.button("✏️ Edit", key=f"edit_{feed['id']}"):
                            st.session_state[f"editing_feed_{feed['id']}"] = True
                            st.rerun()
                        
                        # Run manually (bypasses all restrictions)
                        if st.button("🔄 Run Manually", key=f"run_manual_{feed['id']}", type="primary"):
                            with st.spinner(f"Manually fetching feed: {feed['name']}..."):
                                # Check if domain is paused - show warning but allow manual run
                                if is_domain_paused:
                                    st.warning(f"⚠️ Domain {feed_domain} is paused, but proceeding with manual fetch...")
                                
                                success, message, saved_count, domain = fetch_and_save_feed(
                                    db, feed['id'], max_items=50, bypass_pause=True
                                )
                                if success:
                                    st.success(f"✅ {message}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Error: {message}")
                                    if "paused" in message.lower():
                                        st.info("💡 Use the 'Paused Domains' section to resume this domain.")
                                    st.rerun()
                    
                    with col3:
                        # Toggle active status
                        if feed['is_active']:
                            if st.button("⏸️ Deactivate", key=f"deactivate_{feed['id']}"):
                                db.update_feed(feed['id'], is_active=0)
                                st.success("Feed deactivated")
                                st.rerun()
                        else:
                            if st.button("▶️ Activate", key=f"activate_{feed['id']}"):
                                db.update_feed(feed['id'], is_active=1)
                                st.success("Feed activated")
                                st.rerun()
                        
                        # Delete feed
                        if st.button("🗑️ Delete", key=f"delete_{feed['id']}"):
                            st.session_state[f"confirm_delete_{feed['id']}"] = True
                    
                    # Edit form (shown when editing)
                    if st.session_state.get(f"editing_feed_{feed['id']}", False):
                        st.markdown("---")
                        st.markdown("### Edit Feed")
                        with st.form(f"edit_feed_form_{feed['id']}"):
                            edit_name = st.text_input("Feed Name", value=feed['name'])
                            edit_url = st.text_input("RSS Feed URL", value=feed['url'])
                            edit_frequency = st.number_input(
                                "Check Frequency (times per day)",
                                min_value=1,
                                max_value=24,
                                value=feed['check_frequency_per_day']
                            )
                            
                            col_edit1, col_edit2 = st.columns(2)
                            with col_edit1:
                                update_btn = st.form_submit_button("Update", type="primary")
                            with col_edit2:
                                cancel_btn = st.form_submit_button("Cancel")
                            
                            if update_btn:
                                success, msg = db.update_feed(
                                    feed['id'],
                                    name=edit_name,
                                    url=edit_url,
                                    check_frequency_per_day=edit_frequency
                                )
                                if success:
                                    st.success(msg)
                                    st.session_state[f"editing_feed_{feed['id']}"] = False
                                    st.rerun()
                                else:
                                    st.error(msg)
                            
                            if cancel_btn:
                                st.session_state[f"editing_feed_{feed['id']}"] = False
                                st.rerun()
                    
                    # Delete confirmation
                    if st.session_state.get(f"confirm_delete_{feed['id']}", False):
                        st.warning(f"⚠️ Are you sure you want to delete '{feed['name']}'? This will also delete all articles from this feed.")
                        col_del1, col_del2 = st.columns(2)
                        with col_del1:
                            if st.button("Yes, Delete", key=f"confirm_del_{feed['id']}", type="primary"):
                                success, msg = db.delete_feed(feed['id'])
                                if success:
                                    st.success(msg)
                                    st.session_state[f"confirm_delete_{feed['id']}"] = False
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with col_del2:
                            if st.button("Cancel", key=f"cancel_del_{feed['id']}"):
                                st.session_state[f"confirm_delete_{feed['id']}"] = False
                                st.rerun()
        else:
            st.info("No feeds added yet. Add a feed using the form above.")
    
    with subtab2:
        st.subheader("View Articles")
        st.markdown("Browse articles from all feeds")
        
        # Filter options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            # Filter by feed
            feeds = db.get_feeds(active_only=False)
            feed_options = ["All Feeds"] + [f"{f['name']} (ID: {f['id']})" for f in feeds]
            selected_feed = st.selectbox("Filter by Feed", feed_options)
            
            if selected_feed != "All Feeds":
                selected_feed_id = int(selected_feed.split("ID: ")[1].split(")")[0])
            else:
                selected_feed_id = None
        
        with col_filter2:
            # Limit results
            limit = st.number_input("Max Articles", min_value=10, max_value=100, value=20, step=10)
        
        with col_filter3:
            # Order by
            order_by = st.selectbox("Order By", ["published_at", "created_at", "title"])
        
        # Get articles
        articles = db.get_articles(feed_id=selected_feed_id, limit=limit, order_by=order_by)
        
        if articles:
            st.markdown(f"### Found {len(articles)} articles")
            
            for article in articles:
                article_id = article['id']
                article_key = f"article_{article_id}"
                
                # Check if analyzing this article
                is_analyzing = st.session_state.get(f"analyzing_{article_id}", False)
                
                with st.expander(f"📄 {article['title'] or 'Untitled'}", expanded=is_analyzing):
                    col_art1, col_art2 = st.columns([3, 1])
                    
                    with col_art1:
                        st.markdown(f"**Feed:** {article.get('feed_name', 'Unknown')}")
                        if article.get('link'):
                            st.markdown(f"**Link:** [{article['link']}]({article['link']})")
                        if article.get('description'):
                            st.markdown(f"**Description:** {article['description'][:500]}...")
                        if article.get('published_at'):
                            published_at = article['published_at']
                            if isinstance(published_at, str):
                                try:
                                    published_at = datetime.fromisoformat(published_at)
                                except:
                                    pass
                            if isinstance(published_at, datetime):
                                st.markdown(f"**Published:** {published_at.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown(f"**Added:** {datetime.fromisoformat(article['created_at']).strftime('%Y-%m-%d %H:%M:%S') if isinstance(article['created_at'], str) else article['created_at']}")
                    
                    with col_art2:
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            # Analyze button
                            if st.button("🔍 Analyze", key=f"analyze_{article_id}", type="primary"):
                                st.session_state[f"analyzing_{article_id}"] = True
                                st.rerun()
                        
                        with col_btn2:
                            # Delete button
                            if st.button("🗑️ Delete", key=f"del_article_{article_id}"):
                                success, msg = db.delete_article(article_id)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                    
                    # Show analysis results if analyzing
                    if st.session_state.get(f"analyzing_{article_id}", False):
                        st.markdown("---")
                        st.markdown("### 🔍 Analysis Results")
                        
                        # Check if domain is paused
                        article_domain = extract_domain(article.get('link', ''))
                        is_domain_paused = db.is_domain_paused(article_domain) if article_domain else False
                        
                        if is_domain_paused:
                            st.warning(f"⚠️ Domain {article_domain} is paused. Proceeding with analysis anyway...")
                        
                        # Create domain pause check function (bypass for manual analysis)
                        def check_pause(d: str) -> bool:
                            return False  # Don't check pause during manual analysis
                        
                        # Step 1: Extract content from article link
                        st.info("📥 Step 1/3: Extracting content from article link...")
                        
                        # Import web scraper
                        from src.utils.web_scraper import extract_article_from_url
                        
                        # Extract content
                        with st.spinner("Fetching and extracting content from the article URL..."):
                            try:
                                extracted_content = extract_article_from_url(article.get('link', ''), timeout=30)
                            except Exception as e:
                                extracted_content = None
                                st.error(f"❌ Error extracting content: {str(e)}")
                        
                        if not extracted_content or len(extracted_content.strip()) < 100:
                            st.error(f"❌ Failed to extract content: Content too short or empty (got {len(extracted_content) if extracted_content else 0} characters)")
                            st.session_state[f"analyzing_{article_id}"] = False
                            if st.button("🔄 Retry Analysis", key=f"retry_extract_{article_id}"):
                                st.session_state[f"analyzing_{article_id}"] = True
                                st.rerun()
                        else:
                            st.success(f"✅ Step 1 completed: Extracted {len(extracted_content)} characters from article")
                            
                            # Show extracted content preview
                            with st.expander("📄 Extracted Raw Content Preview", expanded=False):
                                st.text_area(
                                    "Raw Content",
                                    value=extracted_content[:3000] + ("..." if len(extracted_content) > 3000 else ""),
                                    height=150,
                                    disabled=True,
                                    key=f"extracted_preview_{article_id}"
                                )
                                st.caption(f"Total length: {len(extracted_content)} characters")
                            
                            # Step 2: Analyze content with LLM
                            st.info("🤖 Step 2/3: Analyzing content with AI model...")
                            
                            # Get active keywords for matching
                            active_keywords = db.get_active_keywords()
                            
                            with st.spinner(f"Analyzing content with {ollama_model} (this may take a moment)..."):
                                success, message, extracted_content_raw, structured_data, raw_response, keyword_match = analyze_article_content(
                                    article_title=article.get('title', 'Untitled'),
                                    article_description=article.get('description', ''),
                                    article_link=article.get('link', ''),
                                    ollama_model=ollama_model,
                                    ollama_base_url=ollama_base_url,
                                    domain_pause_check=check_pause,
                                    keywords=active_keywords if active_keywords else None
                                )
                                
                                # Use the raw content returned from analyzer (which includes filtered content)
                                if extracted_content_raw:
                                    extracted_content = extracted_content_raw
                            
                            # Check if analysis succeeded
                            if not success:
                                st.error(f"❌ Step 2 failed: {message}")
                                # Show extracted content even if analysis failed
                                if extracted_content:
                                    with st.expander("📄 Extracted Raw Content (Analysis Failed)", expanded=False):
                                        st.text_area(
                                            "Raw Content",
                                            value=extracted_content,
                                            height=300,
                                            disabled=True,
                                            key=f"extracted_error_{article_id}"
                                        )
                                        st.caption(f"Total length: {len(extracted_content)} characters")
                                
                                # Reset analyzing state on error
                                st.session_state[f"analyzing_{article_id}"] = False
                                
                                # Option to retry
                                if st.button("🔄 Retry Analysis", key=f"retry_{article_id}"):
                                    st.session_state[f"analyzing_{article_id}"] = True
                                    st.rerun()
                            else:
                                # Show results if analysis succeeded
                                st.success("✅ Step 2 completed: Content analyzed successfully")
                                
                                # Step 3: Keyword matching (if keywords available)
                                if active_keywords and keyword_match:
                                    st.info("🔑 Step 3/3: Keyword matching completed")
                                elif active_keywords:
                                    st.info("🔑 Step 3/3: Keyword matching skipped (no matches found)")
                                else:
                                    st.info("ℹ️ No active keywords - keyword matching skipped")
                                
                                st.markdown("---")
                                st.markdown("### 📊 Structured Analysis Results")
                                
                                # Show feed item reference
                                with st.expander("📋 Feed Item Reference", expanded=False):
                                    st.markdown(f"**Title:** {article.get('title', 'Untitled')}")
                                    if article.get('description'):
                                        st.markdown(f"**Description:** {article.get('description')}")
                                    st.markdown(f"**Link:** [{article.get('link', '')}]({article.get('link', '')})")
                                
                                # Show structured data if available
                                if structured_data:
                                    # Display structured information
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.markdown("#### 🎯 Main Topic")
                                        st.info(structured_data.get('topic', 'Unknown topic'))
                                        
                                        st.markdown("#### 📝 Summary")
                                        st.markdown(structured_data.get('summary', 'No summary available'))
                                        
                                        st.markdown("#### 🔑 Key Points")
                                        key_points = structured_data.get('key_points', [])
                                        if key_points:
                                            for i, point in enumerate(key_points, 1):
                                                st.markdown(f"{i}. {point}")
                                        else:
                                            st.info("No key points extracted")
                                    
                                    with col2:
                                        # Relevance score
                                        relevance = structured_data.get('relevance', 0.5)
                                        st.metric("Relevance Score", f"{relevance:.2f}", delta=f"{relevance*100:.0f}%")
                                        
                                        # Category
                                        category = structured_data.get('category', 'other')
                                        category_emoji = {
                                            'release': '🚀',
                                            'update': '🔄',
                                            'tutorial': '📚',
                                            'news': '📰',
                                            'announcement': '📢',
                                            'opinion': '💭',
                                            'other': '📄'
                                        }.get(category, '📄')
                                        st.markdown(f"**Category:** {category_emoji} {category.title()}")
                                        
                                        # Action
                                        action = structured_data.get('action', 'monitor')
                                        action_emoji = {
                                            'write': '✍️',
                                            'monitor': '👀',
                                            'ignore': '❌'
                                        }.get(action, '👀')
                                        st.markdown(f"**Action:** {action_emoji} {action.title()}")
                                        
                                        # Version info
                                        version_info = structured_data.get('version_info')
                                        if version_info:
                                            st.markdown(f"**Version Info:** {version_info}")
                                        
                                        # Freshness hint
                                        freshness_hint = structured_data.get('freshness_hint')
                                        if freshness_hint:
                                            st.success(f"✨ {freshness_hint}")
                                    
                                    # Tech entities
                                    tech_entities = structured_data.get('tech_entities', [])
                                    if tech_entities:
                                        st.markdown("#### 🔧 Technical Entities")
                                        st.markdown(", ".join([f"`{entity}`" for entity in tech_entities]))
                                    
                                    # Show structured JSON
                                    with st.expander("📋 Structured JSON Data", expanded=False):
                                        import json
                                        st.code(json.dumps(structured_data, indent=2), language="json")
                                    
                                    # Show keyword match and article recommendation
                                    if keyword_match:
                                        st.markdown("---")
                                        st.markdown("### 🔑 Keyword Match & Article Recommendation")
                                        
                                        matched_keywords = keyword_match.get('matched_keywords', [])
                                        should_write = keyword_match.get('should_write_article', False)
                                        seo_potential = keyword_match.get('seo_potential', 'low')
                                        reasoning = keyword_match.get('reasoning', '')
                                        article_recommendation = keyword_match.get('article_recommendation', '')
                                        article_titles = keyword_match.get('article_titles', [])
                                        
                                        if matched_keywords:
                                            st.success(f"✅ Matched Keywords: {', '.join([f'**{kw}**' for kw in matched_keywords])}")
                                        else:
                                            st.info("ℹ️ No keywords matched")
                                        
                                        # SEO Potential
                                        seo_colors = {
                                            'high': '🟢',
                                            'medium': '🟡',
                                            'low': '🔴'
                                        }
                                        seo_emoji = seo_colors.get(seo_potential.lower(), '⚪')
                                        st.metric("SEO Potential", f"{seo_emoji} {seo_potential.upper()}")
                                        
                                        # Article Recommendation
                                        if should_write:
                                            st.success("✍️ **Recommendation: Write Article**")
                                            st.markdown("💡 **This is a good opportunity to write an article early and get good SEO/traffic!**")
                                        else:
                                            st.warning("👀 **Recommendation: Monitor Only**")
                                        
                                        # Reasoning
                                        if reasoning:
                                            with st.expander("📝 Reasoning", expanded=True):
                                                st.markdown(reasoning)
                                        
                                        # Article Recommendation Details
                                        if article_recommendation:
                                            with st.expander("💡 Article Recommendation Details", expanded=True):
                                                st.markdown(article_recommendation)
                                        
                                        # Article Title Suggestions
                                        if article_titles and len(article_titles) > 0:
                                            st.markdown("---")
                                            st.markdown("### 📝 Suggested Article Titles")
                                            if len(article_titles) < 5 and should_write:
                                                st.warning(f"⚠️ Only {len(article_titles)} article title(s) generated. Expected at least 5.")
                                            else:
                                                st.info(f"Here are {len(article_titles)} article title suggestions that would be good targets for SEO and content creation:")
                                            for i, title in enumerate(article_titles, 1):
                                                st.markdown(f"{i}. **{title}**")
                                            st.caption("💡 These titles are optimized for SEO and target specific search queries related to the content.")
                                        elif should_write:
                                            st.warning("⚠️ No article titles were generated. The analysis recommended writing an article but no titles were provided.")
                                        
                                        # Active keywords info
                                        if active_keywords:
                                            st.caption(f"ℹ️ Checked against {len(active_keywords)} active keyword(s)")
                                else:
                                    # Fallback: show raw response if JSON parsing failed
                                    st.warning("⚠️ Could not parse structured JSON. Showing raw response:")
                                    if raw_response:
                                        st.text_area(
                                            "Raw LLM Response",
                                            value=raw_response,
                                            height=400,
                                            key=f"raw_response_{article_id}",
                                            help="Raw response from LLM (JSON parsing failed)"
                                        )
                                
                                # Show extracted raw content (full)
                                with st.expander("📄 Extracted Raw Content (Full)", expanded=False):
                                    if extracted_content:
                                        st.text_area(
                                            "Raw Content (Filtered)",
                                            value=extracted_content,
                                            height=300,
                                            disabled=True,
                                            key=f"extracted_full_{article_id}",
                                            help="Content extracted from the article link (navigation and unwanted elements filtered out)"
                                        )
                                        st.caption(f"Total length: {len(extracted_content)} characters")
                                        st.info("ℹ️ Navigation elements, social buttons, and unwanted UI elements have been filtered out")
                                    else:
                                        st.info("No content extracted")
                                
                                # Reset analyzing state
                                st.session_state[f"analyzing_{article_id}"] = False
                                
                                # Option to analyze again
                                if st.button("🔄 Analyze Again", key=f"reanalyze_{article_id}"):
                                    st.session_state[f"analyzing_{article_id}"] = True
                                    st.rerun()
        else:
            st.info("No articles found. Add feeds and fetch articles to get started.")

with tab2:
    st.subheader("🔑 Manage Keywords")
    st.markdown("Manage keywords for article topic matching and content discovery")
    
    # Add keyword form
    with st.expander("➕ Add New Keyword", expanded=False):
        col_key1, col_key2 = st.columns([2, 1])
        
        with col_key1:
            new_keyword = st.text_input("Keyword", key="new_keyword", 
                                       help="Keyword to track (e.g., 'React', 'Python', 'AI', 'Machine Learning')")
            new_keyword_desc = st.text_area("Description (Optional)", key="new_keyword_desc",
                                           help="Description of what this keyword represents")
        
        with col_key2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("Add Keyword", key="add_keyword_btn", type="primary"):
                if new_keyword and new_keyword.strip():
                    success, msg = db.add_keyword(new_keyword.strip(), new_keyword_desc.strip() if new_keyword_desc else None)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("⚠️ Please enter a keyword")
    
    # Display existing keywords
    keywords = db.get_keywords(active_only=False)
    
    if keywords:
        st.markdown(f"### Found {len(keywords)} keyword(s)")
        
        for keyword in keywords:
            keyword_id = keyword['id']
            is_editing = st.session_state.get(f"editing_keyword_{keyword_id}", False)
            
            if is_editing:
                # Edit mode
                col_edit1, col_edit2, col_edit3 = st.columns([3, 1, 1])
                
                with col_edit1:
                    edit_keyword = st.text_input("Keyword", value=keyword['keyword'], 
                                                key=f"edit_keyword_{keyword_id}")
                    edit_desc = st.text_area("Description", value=keyword.get('description', '') or '', 
                                           key=f"edit_desc_{keyword_id}")
                    edit_active = st.checkbox("Active", value=bool(keyword['is_active']), 
                                            key=f"edit_active_{keyword_id}")
                
                with col_edit2:
                    update_btn = st.button("Update", key=f"update_keyword_{keyword_id}", type="primary")
                    cancel_btn = st.button("Cancel", key=f"cancel_edit_keyword_{keyword_id}")
                
                with col_edit3:
                    st.write("")  # Spacing
                
                if update_btn:
                    success, msg = db.update_keyword(
                        keyword_id,
                        keyword=edit_keyword.strip() if edit_keyword else None,
                        description=edit_desc.strip() if edit_desc else None,
                        is_active=edit_active
                    )
                    if success:
                        st.success(msg)
                        st.session_state[f"editing_keyword_{keyword_id}"] = False
                        st.rerun()
                    else:
                        st.error(msg)
                
                if cancel_btn:
                    st.session_state[f"editing_keyword_{keyword_id}"] = False
                    st.rerun()
            else:
                # Display mode
                col_kw1, col_kw2, col_kw3, col_kw4 = st.columns([3, 1, 1, 1])
                
                with col_kw1:
                    status_emoji = "✅" if keyword['is_active'] else "❌"
                    st.markdown(f"**{status_emoji} {keyword['keyword']}**")
                    if keyword.get('description'):
                        st.caption(keyword['description'])
                    created_at = datetime.fromisoformat(keyword['created_at']) if isinstance(keyword['created_at'], str) else keyword['created_at']
                    st.caption(f"Added: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col_kw2:
                    if st.button("✏️ Edit", key=f"edit_btn_keyword_{keyword_id}"):
                        st.session_state[f"editing_keyword_{keyword_id}"] = True
                        st.rerun()
                
                with col_kw3:
                    if keyword['is_active']:
                        if st.button("❌ Deactivate", key=f"deactivate_keyword_{keyword_id}"):
                            success, msg = db.update_keyword(keyword_id, is_active=False)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        if st.button("✅ Activate", key=f"activate_keyword_{keyword_id}"):
                            success, msg = db.update_keyword(keyword_id, is_active=True)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                
                with col_kw4:
                    # Delete confirmation
                    if st.session_state.get(f"confirm_delete_keyword_{keyword_id}", False):
                        st.warning(f"⚠️ Are you sure you want to delete '{keyword['keyword']}'?")
                        col_del1, col_del2 = st.columns(2)
                        with col_del1:
                            if st.button("Yes, Delete", key=f"confirm_del_keyword_{keyword_id}", type="primary"):
                                success, msg = db.delete_keyword(keyword_id)
                                if success:
                                    st.success(msg)
                                    st.session_state[f"confirm_delete_keyword_{keyword_id}"] = False
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with col_del2:
                            if st.button("Cancel", key=f"cancel_del_keyword_{keyword_id}"):
                                st.session_state[f"confirm_delete_keyword_{keyword_id}"] = False
                                st.rerun()
                    else:
                        if st.button("🗑️ Delete", key=f"del_keyword_{keyword_id}"):
                            st.session_state[f"confirm_delete_keyword_{keyword_id}"] = True
                            st.rerun()
    else:
        st.info("No keywords added yet. Add a keyword using the form above.")
