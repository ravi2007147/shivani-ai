"""Common link extraction utility with domain pause/resume functionality."""

import requests
import feedparser
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse
import logging
import time

logger = logging.getLogger(__name__)

# HTTP status codes that should trigger domain pause
PAUSE_STATUS_CODES = {
    403,  # Forbidden
    429,  # Too Many Requests
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# Connection error types that should trigger pause
PAUSE_ERROR_TYPES = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.SSLError,
)


def extract_domain(url: str) -> str:
    """Extract domain from URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        Domain name (e.g., 'example.com')
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        return domain
    except Exception:
        return ""


def fetch_url_content(
    url: str,
    timeout: int = 30,
    headers: Optional[Dict] = None,
    max_retries: int = 1
) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
    """Fetch content from URL with error handling.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        headers: Optional HTTP headers
        max_retries: Maximum number of retries
        
    Returns:
        Tuple of (success, content, status_code, error_message)
        - success: True if request succeeded
        - content: Response content (bytes) if successful, None otherwise
        - status_code: HTTP status code
        - error_message: Error message if failed
    """
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            status_code = response.status_code
            
            # Check if status code should trigger pause
            if status_code in PAUSE_STATUS_CODES:
                error_msg = f"HTTP {status_code}: {response.reason}"
                logger.warning(f"Domain should be paused: {url} - {error_msg}")
                return False, None, status_code, error_msg
            
            # Check for successful response
            if response.status_code == 200:
                return True, response.content, status_code, None
            else:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                return False, None, status_code, error_msg
                
        except PAUSE_ERROR_TYPES as e:
            error_msg = f"Connection error: {str(e)}"
            logger.warning(f"Domain should be paused: {url} - {error_msg}")
            return False, None, None, error_msg
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            if attempt < max_retries:
                time.sleep(1)  # Wait before retry
                continue
            return False, None, None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error fetching {url}: {error_msg}", exc_info=True)
            return False, None, None, error_msg
    
    return False, None, None, "Max retries exceeded"


def extract_rss_articles(
    url: str,
    max_items: int = 10,
    domain_pause_check: Optional[callable] = None
) -> Tuple[bool, str, List[Dict], Optional[str], Optional[int]]:
    """Extract articles from RSS feed URL.
    
    Args:
        url: RSS feed URL
        max_items: Maximum number of items to return
        domain_pause_check: Optional function(domain) -> bool to check if domain is paused.
                          If None, pause check is bypassed.
        
    Returns:
        Tuple of (success, message, articles, domain, status_code)
        - success: True if extraction succeeded
        - message: Status message
        - articles: List of article dictionaries
        - domain: Domain name (for pause tracking)
        - status_code: HTTP status code if available
    """
    domain = extract_domain(url)
    
    # Check if domain is paused (only if pause_check function is provided)
    if domain_pause_check is not None and domain_pause_check(domain):
        return False, f"Domain {domain} is paused", [], domain, None
    
    # First, try to fetch the URL content
    success, content, status_code, error_msg = fetch_url_content(url, timeout=30)
    
    if not success:
        # Check if we should pause the domain
        should_pause = False
        pause_reason = None
        
        if status_code in PAUSE_STATUS_CODES:
            should_pause = True
            pause_reason = f"HTTP {status_code}"
        elif error_msg and any(err_type in error_msg for err_type in ["Connection", "Timeout", "SSL"]):
            should_pause = True
            pause_reason = "Connection error"
        
        if should_pause:
            return False, f"Failed to fetch feed: {error_msg}", [], domain, status_code
        
        return False, f"Failed to fetch feed: {error_msg}", [], domain, status_code
    
    # Parse the feed content
    try:
        # Parse feed from content
        feed = feedparser.parse(content)
        
        # Check for parsing errors
        if feed.bozo and feed.bozo_exception:
            error_msg = f"Feed parsing error: {str(feed.bozo_exception)}"
            logger.warning(error_msg)
            return False, error_msg, [], domain, None
        
        # Check if feed has entries
        if not feed.entries:
            return False, "No entries found in feed", [], domain, None
        
        articles = []
        for entry in feed.entries[:max_items]:
            # Extract published date
            published_at = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    published_at = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                try:
                    published_at = datetime(*entry.updated_parsed[:6])
                except Exception:
                    pass
            
            # Extract description
            description = None
            if hasattr(entry, 'summary'):
                description = entry.summary
            elif hasattr(entry, 'description'):
                description = entry.description
            
            # Extract link
            link = None
            if hasattr(entry, 'link'):
                link = entry.link
            elif hasattr(entry, 'id'):
                link = entry.id
            
            if not link:
                continue  # Skip entries without links
            
            article = {
                'title': entry.get('title', 'Untitled'),
                'link': link,
                'description': description,
                'published_at': published_at
            }
            articles.append(article)
        
        return True, f"Successfully fetched {len(articles)} articles", articles, domain, status_code
        
    except Exception as e:
        error_msg = f"Error parsing feed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg, [], domain, None


def extract_link_content(
    url: str,
    domain_pause_check: Optional[callable] = None
) -> Tuple[bool, Optional[str], Optional[str], Optional[int], Optional[str]]:
    """Extract content from a single link/article URL.
    
    Args:
        url: Article URL
        domain_pause_check: Optional function(domain) -> bool to check if domain is paused
        
    Returns:
        Tuple of (success, content, domain, status_code, error_message)
    """
    domain = extract_domain(url)
    
    # Check if domain is paused
    if domain_pause_check and domain_pause_check(domain):
        return False, None, domain, None, f"Domain {domain} is paused"
    
    # Fetch URL content
    success, content, status_code, error_msg = fetch_url_content(url, timeout=30)
    
    if not success:
        # Check if we should pause the domain
        should_pause = False
        if status_code in PAUSE_STATUS_CODES:
            should_pause = True
        elif error_msg and any(err_type in error_msg for err_type in ["Connection", "Timeout", "SSL"]):
            should_pause = True
        
        if should_pause:
            return False, None, domain, status_code, error_msg
        
        return False, None, domain, status_code, error_msg
    
    # Convert content to string if it's bytes
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = content.decode('latin-1')
            except Exception:
                return False, None, domain, status_code, "Failed to decode content"
    
    return True, content, domain, status_code, None

