"""HTML text extraction utilities."""

import re
import logging
from typing import Optional, Tuple
from html import unescape

logger = logging.getLogger(__name__)


def extract_text_from_html(html_content: str, use_llm: bool = False, llm=None) -> Tuple[bool, str, Optional[str]]:
    """Extract text content from HTML.
    
    Args:
        html_content: HTML content as string
        use_llm: Whether to use LLM for intelligent extraction (default: False)
        llm: Optional LLM instance for extraction (required if use_llm=True)
        
    Returns:
        Tuple of (success, extracted_text, error_message)
    """
    if not html_content or not html_content.strip():
        return False, "", "HTML content is empty"
    
    try:
        if use_llm and llm:
            return _extract_with_llm(html_content, llm)
        else:
            return _extract_with_python(html_content)
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}", exc_info=True)
        return False, "", f"Error extracting text: {str(e)}"


def _extract_with_python(html_content: str) -> Tuple[bool, str, Optional[str]]:
    """Extract text from HTML using Python libraries.
    
    Tries multiple strategies:
    1. BeautifulSoup (if available) - best results
    2. html2text (if available) - good for readable text
    3. Basic regex extraction - fallback
    """
    try:
        # Strategy 1: Try BeautifulSoup (most reliable)
        try:
            from bs4 import BeautifulSoup
            return _extract_with_beautifulsoup(html_content, BeautifulSoup)
        except ImportError:
            logger.warning("BeautifulSoup not available, trying html2text")
        
        # Strategy 2: Try html2text (good for readable text)
        try:
            import html2text
            return _extract_with_html2text(html_content, html2text)
        except ImportError:
            logger.warning("html2text not available, using basic extraction")
        
        # Strategy 3: Basic regex extraction
        return _extract_basic(html_content)
        
    except Exception as e:
        logger.error(f"Error in Python extraction: {str(e)}", exc_info=True)
        return False, "", f"Error in Python extraction: {str(e)}"


def _extract_with_beautifulsoup(html_content: str, BeautifulSoup) -> Tuple[bool, str, Optional[str]]:
    """Extract text using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and other non-content tags
        for element in soup(['script', 'style', 'nav', 'aside', 'header', 'footer', 
                             'iframe', 'embed', 'object', 'noscript', 'meta', 'link']):
            element.decompose()
        
        # Try to find main article content
        content = None
        
        # Strategy 1: Look for semantic HTML5 tags
        for selector in ['article', 'main', '[role="main"]']:
            try:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(separator='\n', strip=True)
                    if content and len(content.strip()) > 100:
                        logger.info(f"Found content using selector: {selector}")
                        break
            except Exception:
                continue
        
        # Strategy 2: Look for common content class names
        if not content or len(content.strip()) < 100:
            for class_name in ['.content', '.post-content', '.article-content', '.entry-content',
                               '.article-body', '.post-body', '.story-body', '.article-text', 
                               '.main-content', '.post', '.entry']:
                try:
                    element = soup.select_one(class_name)
                    if element:
                        content = element.get_text(separator='\n', strip=True)
                        if content and len(content.strip()) > 100:
                            logger.info(f"Found content using class: {class_name}")
                            break
                except Exception:
                    continue
        
        # Strategy 3: Get body text if nothing else works
        if not content or len(content.strip()) < 100:
            body = soup.find('body')
            if body:
                content = body.get_text(separator='\n', strip=True)
                logger.info("Using body text as content")
        
        if not content or len(content.strip()) < 50:
            return False, "", "No meaningful content extracted from HTML"
        
        # Clean the content
        content = _clean_extracted_text(content)
        
        return True, content, None
        
    except Exception as e:
        logger.error(f"Error in BeautifulSoup extraction: {str(e)}", exc_info=True)
        return False, "", f"Error in BeautifulSoup extraction: {str(e)}"


def _extract_with_html2text(html_content: str, html2text) -> Tuple[bool, str, Optional[str]]:
    """Extract text using html2text library."""
    try:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines
        
        text = h.handle(html_content)
        
        if not text or len(text.strip()) < 50:
            return False, "", "No meaningful content extracted from HTML"
        
        # Clean the content
        text = _clean_extracted_text(text)
        
        return True, text, None
        
    except Exception as e:
        logger.error(f"Error in html2text extraction: {str(e)}", exc_info=True)
        return False, "", f"Error in html2text extraction: {str(e)}"


def _extract_basic(html_content: str) -> Tuple[bool, str, Optional[str]]:
    """Basic text extraction using regex (fallback method)."""
    try:
        # Remove script and style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove other non-content tags
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<aside[^>]*>.*?</aside>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<noscript[^>]*>.*?</noscript>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags but preserve structure
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</h[1-6]>', '\n\n', text, flags=re.IGNORECASE)
        
        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Decode HTML entities
        text = unescape(text)
        
        # Clean the content
        text = _clean_extracted_text(text)
        
        if not text or len(text.strip()) < 50:
            return False, "", "No meaningful content extracted from HTML"
        
        return True, text, None
        
    except Exception as e:
        logger.error(f"Error in basic extraction: {str(e)}", exc_info=True)
        return False, "", f"Error in basic extraction: {str(e)}"


def _extract_with_llm(html_content: str, llm) -> Tuple[bool, str, Optional[str]]:
    """Extract text from HTML using LLM for intelligent extraction.
    
    This method uses the LLM to understand the HTML structure and extract
    the main content, ignoring navigation, ads, and other non-content elements.
    """
    try:
        # Truncate HTML if too long (keep first 50000 chars for context)
        html_preview_length = min(50000, len(html_content))
        html_preview = html_content[:html_preview_length]
        html_length = len(html_content)
        is_truncated = html_length > html_preview_length
        
        prompt = f"""You are an AI HTML content extractor. Extract the main textual content from the provided HTML, ignoring navigation menus, headers, footers, advertisements, and other non-content elements.

HTML Content ({html_preview_length} characters shown of {html_length} total):
---
{html_preview}
{'... [HTML continues, but truncated for context]' if is_truncated else ''}
---

Instructions:
1. Extract only the main content text from the HTML
2. Ignore navigation menus, headers, footers, advertisements, and sidebar content
3. Preserve the structure and readability of the content
4. Remove HTML tags but keep the text content
5. Clean up excessive whitespace
6. Return ONLY the extracted text content, no explanations or meta-commentary

Extracted Text Content:
"""
        
        logger.info("Using LLM for HTML text extraction")
        extracted_text = llm.invoke(prompt)
        
        if not extracted_text or len(extracted_text.strip()) < 50:
            return False, "", "LLM returned empty or insufficient content"
        
        # Clean the extracted text
        extracted_text = _clean_extracted_text(extracted_text)
        
        return True, extracted_text, None
        
    except Exception as e:
        logger.error(f"Error in LLM extraction: {str(e)}", exc_info=True)
        return False, "", f"Error in LLM extraction: {str(e)}"


def _clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line and len(line) > 10:  # Filter out very short lines
            lines.append(line)
    
    # Join lines and normalize whitespace
    text = '\n'.join(lines)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Replace multiple newlines
    
    # Remove common non-content patterns
    patterns = [
        r'Share.*?Facebook.*?Twitter.*?',
        r'Cookie.*?accept.*?',
        r'Skip to.*?',
        r'Click here.*?',
        r'Subscribe.*?newsletter.*?',
        r'Sign up.*?',
        r'Privacy.*?Policy.*?',
        r'Terms.*?Service.*?',
        r'Loading.*?',
        r'Please enable.*?JavaScript.*?',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()

