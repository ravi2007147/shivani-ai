"""RSS feed fetching utility with domain pause support."""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from src.utils.link_extractor import extract_rss_articles, extract_domain

logger = logging.getLogger(__name__)

def fetch_rss_feed(url: str, max_items: int = 10, domain_pause_check=None) -> tuple[bool, str, List[Dict], Optional[str], Optional[int]]:
    """Fetch and parse an RSS feed using common link extractor.
    
    Args:
        url: RSS feed URL
        max_items: Maximum number of items to return
        domain_pause_check: Optional function(domain) -> bool to check if domain is paused
        
    Returns:
        Tuple of (success, message, articles, domain, status_code)
    """
    return extract_rss_articles(url, max_items, domain_pause_check)

def fetch_and_save_feed(rss_db, feed_id: int, max_items: int = 10, bypass_pause: bool = False) -> tuple[bool, str, int, Optional[str]]:
    """Fetch a feed and save articles to database.
    
    Args:
        rss_db: RSSDB instance
        feed_id: Feed ID to fetch
        max_items: Maximum number of items to fetch and save
        bypass_pause: If True, bypass domain pause check (for manual runs)
        
    Returns:
        Tuple of (success, message, count of new articles saved, domain)
    """
    # Get feed info
    feed = rss_db.get_feed(feed_id)
    if not feed:
        return False, "Feed not found", 0, None
    
    url = feed['url']
    domain = extract_domain(url)
    
    # Check if domain is paused (unless bypassing)
    if not bypass_pause and rss_db.is_domain_paused(domain):
        return False, f"Domain {domain} is paused - manual resume required", 0, domain
    
    # Create domain pause check function (only if not bypassing)
    def check_pause(d: str) -> bool:
        if bypass_pause:
            return False  # Don't check pause when bypassing
        return rss_db.is_domain_paused(d)
    
    # Fetch feed using common extractor
    success, message, articles, extracted_domain, status_code = extract_rss_articles(
        url, max_items, check_pause if not bypass_pause else None
    )
    
    if not success:
        # Check if we should pause the domain (only if not bypassing)
        should_pause = False
        pause_reason = None
        
        if status_code in [403, 429, 503, 504]:
            should_pause = True
            pause_reason = f"HTTP {status_code}"
        elif "Connection" in message or "Timeout" in message or "SSL" in message:
            should_pause = True
            pause_reason = "Connection error"
        
        if should_pause and not bypass_pause:
            # Pause the domain
            pause_success, pause_msg = rss_db.pause_domain(
                domain=domain,
                reason=pause_reason,
                error_message=message
            )
            if pause_success:
                return False, f"{message} - Domain {domain} paused. Resume manually.", 0, domain
            else:
                return False, f"{message} - Failed to pause domain: {pause_msg}", 0, domain
        
        # If bypassing, just return the error without pausing
        if bypass_pause:
            return False, f"{message} (Manual run - domain not paused)", 0, domain
        
        return False, message, 0, domain
    
    # Save articles (only new ones - database UNIQUE constraint handles duplicates)
    saved_count = 0
    skipped_count = 0
    for article in articles:
        success, msg = rss_db.add_article(
            feed_id=feed_id,
            title=article['title'],
            link=article['link'],
            description=article['description'],
            published_at=article['published_at']
        )
        if success:
            saved_count += 1
        else:
            # Article already exists (duplicate)
            skipped_count += 1
    
    # Update last_checked timestamp
    rss_db.update_last_checked(feed_id)
    
    # Prepare success message
    if saved_count > 0:
        message = f"Saved {saved_count} new article(s)"
        if skipped_count > 0:
            message += f", {skipped_count} already existed"
    else:
        if skipped_count > 0:
            message = f"All {skipped_count} article(s) already exist in database"
        else:
            message = "No articles found"
    
    return True, message, saved_count, domain
