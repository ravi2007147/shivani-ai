"""Web scraping utilities for extracting article content from URLs."""

import re
from typing import Optional
import requests
from urllib.parse import urlparse


def extract_article_from_url(url: str, timeout: int = 10) -> Optional[str]:
    """Extract article text content from a URL.
    
    Uses advanced HTML parsing to extract only the main article content,
    removing navigation, ads, and other non-content elements.
    
    Args:
        url: The URL to scrape
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        Extracted article text, or None if extraction fails
    """
    try:
        # Try readability-lxml first (best for article extraction)
        from readability import Document
        return _extract_with_readability(url, timeout, Document)
    except ImportError:
        try:
            # Fallback to beautifulsoup4
            from bs4 import BeautifulSoup
            return _extract_with_beautifulsoup(url, timeout, BeautifulSoup)
        except ImportError:
            # Last resort: basic text extraction
            return _extract_basic(url, timeout)


def _extract_with_readability(url: str, timeout: int, Document) -> Optional[str]:
    """Extract article using readability-lxml library.
    
    This is the best method for extracting article content as it
    uses Mozilla's readability algorithm to identify the main content.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse with readability
        doc = Document(response.text)
        article_html = doc.summary()
        
        # Extract text from HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(article_html, 'lxml')
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up the text
        text = _clean_extracted_text(text)
        
        return text if text else None
        
    except Exception as e:
        print(f"Readability extraction failed: {e}")
        # Fallback to beautifulsoup
        try:
            from bs4 import BeautifulSoup
            return _extract_with_beautifulsoup(url, timeout, BeautifulSoup)
        except ImportError:
            return _extract_basic(url, timeout)


def _extract_with_beautifulsoup(url: str, timeout: int, BeautifulSoup) -> Optional[str]:
    """Extract article using beautifulsoup4 library.
    
    Uses heuristics to identify the main article content:
    - Looks for article, main, or content tags
    - Finds divs with high text density
    - Removes common non-content elements (nav, aside, script, style, etc.)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove script, style, and other non-content tags
        for element in soup(['script', 'style', 'nav', 'aside', 'header', 'footer', 
                             'iframe', 'embed', 'object', 'noscript']):
            element.decompose()
        
        # Try to find main article content
        article = None
        
        # Strategy 1: Look for semantic HTML5 tags
        for tag_name in ['article', 'main', '[role="main"]']:
            article = soup.select_one(tag_name)
            if article:
                break
        
        # Strategy 2: Look for common content class names
        if not article:
            for class_name in ['content', 'post-content', 'article-content', 'entry-content',
                               'article-body', 'post-body', 'story-body', 'article-text']:
                article = soup.select_one(f'.{class_name}, #{class_name}')
                if article:
                    break
        
        # Strategy 3: Find body tag if nothing else works
        if not article:
            article = soup.find('body')
        
        if not article:
            return None
        
        # Extract text
        text = article.get_text(separator='\n', strip=True)
        
        # Clean up the text
        text = _clean_extracted_text(text)
        
        return text if text else None
        
    except Exception as e:
        print(f"BeautifulSoup extraction failed: {e}")
        return _extract_basic(url, timeout)


def _extract_basic(url: str, timeout: int) -> Optional[str]:
    """Basic text extraction without special parsing.
    
    Last resort method that just gets all text from the page.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Simple regex to extract text (very basic)
        text = response.text
        
        # Remove script and style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        # Clean up
        text = _clean_extracted_text(text)
        
        return text if text else None
        
    except Exception as e:
        print(f"Basic extraction failed: {e}")
        return None


def _clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            # Skip very short lines that are likely navigation/UI elements
            if len(line) > 20 or any(c.isalpha() for c in line):
                lines.append(line)
    
    # Join lines and normalize whitespace
    text = '\n'.join(lines)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Replace multiple newlines
    
    # Remove common non-content patterns
    text = re.sub(r'Share.*?Facebook.*?Twitter.*?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Cookie.*?accept.*?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Skip to.*?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Click here.*?', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL.
    
    Args:
        url: String to check
        
    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


