"""LAYER 3: Auto-Discovery Agent - Learning brain that searches the web and stores knowledge.

This agent is only activated when memory is missing. It:
1. Searches the web (using Playwright)
2. Extracts required knowledge from search results
3. Summarizes using a template
4. Stores in vector DB as long-term memory
"""

import os
import logging
import time
import re
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import quote_plus, urlparse
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from src.rag import VectorStoreManager # For storing in vector DB
from src.config import MAX_URLS_TO_EXTRACT, MAX_SEARCH_RESULTS, MAX_PAGES_TO_CRAWL, MAX_CRAWL_DEPTH
from src.relevance_detector import RelevanceDetector

logger = logging.getLogger(__name__)


class AutoDiscoveryAgent:
    """Auto-Discovery Agent that searches the web and stores knowledge in vector DB.
    
    This is Layer 3 of the learning system. It performs:
    1. Web search using Playwright (supports multiple search engines)
    2. Content extraction from search results
    3. Knowledge summarization using LLM
    4. Storage in vector DB as long-term memory
    """
    
    def __init__(
        self,
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434",
        headless: bool = False,  # Changed to False for testing/debugging
        search_engine: str = "duckduckgo"  # Options: "duckduckgo", "google", "bing"
    ):
        """Initialize the Auto-Discovery Agent.
        
        Args:
            ollama_model: Ollama model name for LLM operations
            ollama_base_url: Ollama base URL
            headless: Whether to run browser in headless mode (default: False for testing)
            search_engine: Search engine to use ("duckduckgo", "google", "bing")
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.headless = headless
        self.search_engine = search_engine.lower()
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._browser_initialized = False
        self.llm = None
    
    def _get_llm(self) -> OllamaLLM:
        """Get or create LLM instance.
        
        Returns:
            OllamaLLM instance
        """
        if self.llm is None:
            self.llm = OllamaLLM(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=0.2
            )
        return self.llm
    
    def _initialize_browser(self):
        """Initialize Playwright browser instance (only once)."""
        if self._browser_initialized:
            return
        
        try:
            logger.info("Initializing Playwright browser for Auto-Discovery...")
            
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                raise ImportError("Playwright is not installed. Please install it with: pip install playwright")
            
            # Check if browsers are installed
            try:
                self.playwright = sync_playwright().start()
            except Exception as e:
                error_msg = str(e).lower()
                if 'browser' in error_msg or 'chromium' in error_msg or 'executable' in error_msg:
                    raise RuntimeError(
                        "Playwright browsers are not installed. Please run: playwright install chromium\n"
                        f"Original error: {str(e)}"
                    )
                raise
            
            # Launch Chrome browser
            try:
                self.browser = self.playwright.chromium.launch(
                    headless=self.headless,
                    args=['--disable-blink-features=AutomationControlled', '--no-sandbox', '--disable-setuid-sandbox']
                )
            except Exception as e:
                error_msg = str(e).lower()
                if 'browser' in error_msg or 'chromium' in error_msg or 'executable' in error_msg:
                    raise RuntimeError(
                        "Failed to launch Chromium browser. Please ensure Playwright browsers are installed:\n"
                        "  playwright install chromium\n"
                        f"Original error: {str(e)}"
                    )
                raise
            
            # Create a new page
            self.page = self.browser.new_page()
            
            # Set user agent to avoid detection
            self.page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            self._browser_initialized = True
            logger.info("Browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing browser: {str(e)}", exc_info=True)
            try:
                self._close_browser()
            except Exception:
                pass
            raise
    
    def _close_browser(self):
        """Close the browser instance."""
        try:
            if self.page:
                self.page.close()
                self.page = None
            if self.browser:
                self.browser.close()
                self.browser = None
            if self.playwright:
                self.playwright.stop()
                self.playwright = None
            self._browser_initialized = False
            logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}", exc_info=True)
    
    def search_web(self, query: str, max_results: int = 10) -> Dict[str, any]:
        """Search the web using the configured search engine.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 10)
            
        Returns:
            Dictionary with search results containing:
                - query: Original query
                - total_results: Total number of results found
                - organic_results: List of search results with title, url, snippet
                - success: True if successful
                - error: Error message if failed
        """
        logger.info(f"      - Initializing browser for web search...")
        self._initialize_browser()
        logger.info(f"      ✅ Browser initialized")
        
        try:
            logger.info(f"      - Routing to {self.search_engine} search engine...")
            if self.search_engine == "duckduckgo":
                logger.info(f"      - Using DuckDuckGo search")
                result = self._search_duckduckgo(query, max_results)
            elif self.search_engine == "google":
                logger.info(f"      - Using Google search")
                result = self._search_google(query, max_results)
            elif self.search_engine == "bing":
                logger.info(f"      - Using Bing search")
                result = self._search_bing(query, max_results)
            else:
                # Default to DuckDuckGo
                logger.warning(f"      ⚠️ Unknown search engine: {self.search_engine}, using DuckDuckGo")
                result = self._search_duckduckgo(query, max_results)
            
            if result.get('success'):
                logger.info(f"      ✅ Web search completed: {result.get('total_results', 0)} results found")
            else:
                logger.error(f"      ❌ Web search failed: {result.get('error', 'Unknown error')}")
            
            return result
        except Exception as e:
            logger.error(f"      ❌ Error searching web: {str(e)}", exc_info=True)
            return {
                'query': query,
                'total_results': 0,
                'organic_results': [],
                'success': False,
                'error': str(e)
            }
    
    def _search_duckduckgo(self, query: str, max_results: int = 10) -> Dict[str, any]:
        """Search using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            encoded_query = quote_plus(query)
            # Use regular DuckDuckGo search (not HTML version) for better results
            search_url = f"https://duckduckgo.com/?q={encoded_query}"
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            logger.info(f"   - Search URL: {search_url}")
            
            # Navigate to search page
            self.page.goto(search_url, wait_until="networkidle", timeout=30000)
            logger.info(f"   - Page loaded, waiting for content...")
            
            # Wait for search results to load (DuckDuckGo uses JavaScript)
            time.sleep(3)  # Give it time to render
            
            # Try to wait for results container
            try:
                self.page.wait_for_selector('article[data-testid="result"]', timeout=10000)
                logger.info(f"   - Results container found")
            except PlaywrightTimeoutError:
                logger.warning(f"   - Results container not found, trying alternative selectors...")
                # Try alternative wait
                time.sleep(2)
            
            # Take a screenshot for debugging (if not headless)
            if not self.headless:
                try:
                    screenshot_path = "/tmp/duckduckgo_search.png"
                    self.page.screenshot(path=screenshot_path, full_page=True)
                    logger.info(f"   - Screenshot saved to: {screenshot_path}")
                except Exception as e:
                    logger.warning(f"   - Could not save screenshot: {str(e)}")
            
            results = {
                'query': query,
                'total_results': 0,
                'organic_results': [],
                'success': True,
                'error': None
            }
            
            # Try multiple selector strategies for DuckDuckGo
            # Modern DuckDuckGo uses article[data-testid="result"]
            selectors_to_try = [
                'article[data-testid="result"]',
                '.result',
                '.web-result',
                '[data-testid="result"]',
                'div.result',
                'div[data-testid="result"]'
            ]
            
            result_elements = []
            for selector in selectors_to_try:
                try:
                    result_elements = self.page.query_selector_all(selector)
                    if result_elements:
                        logger.info(f"   - Found {len(result_elements)} results using selector: {selector}")
                        break
                except Exception as e:
                    logger.debug(f"   - Selector '{selector}' failed: {str(e)}")
                    continue
            
            # If no results found, log page content for debugging
            if not result_elements:
                logger.warning(f"   - No results found with any selector")
                logger.warning(f"   - Page title: {self.page.title()}")
                logger.warning(f"   - Page URL: {self.page.url}")
                # Try to get page text to see what's there
                try:
                    body_text = self.page.query_selector('body').inner_text()[:500]
                    logger.warning(f"   - Page content preview: {body_text}")
                except Exception:
                    pass
            
            for i, element in enumerate(result_elements[:max_results]):
                try:
                    # Try multiple selector strategies for title
                    title = ""
                    title_selectors = [
                        'h2 a',
                        'a[data-testid="result-title-a"]',
                        'a.result__a',
                        'a.web-result__link',
                        'h3 a'
                    ]
                    for title_selector in title_selectors:
                        title_elem = element.query_selector(title_selector)
                        if title_elem:
                            title = title_elem.inner_text().strip()
                            if title:
                                break
                    
                    # Try multiple selector strategies for URL
                    url = ""
                    url_selectors = [
                        'h2 a',
                        'a[data-testid="result-title-a"]',
                        'a.result__a',
                        'a.web-result__link',
                        'h3 a'
                    ]
                    for url_selector in url_selectors:
                        url_elem = element.query_selector(url_selector)
                        if url_elem:
                            url = url_elem.get_attribute('href') or url_elem.get_attribute('data-testid')
                            if url:
                                # Clean up DuckDuckGo redirect URLs
                                if url.startswith('//'):
                                    url = 'https:' + url
                                elif url.startswith('/l/?kh='):
                                    # DuckDuckGo redirect URL, extract actual URL
                                    try:
                                        from urllib.parse import parse_qs, urlparse
                                        parsed = urlparse(url)
                                        if 'uddg' in parsed.query:
                                            url = parse_qs(parsed.query).get('uddg', [url])[0]
                                    except Exception:
                                        pass
                                break
                    
                    # Try multiple selector strategies for snippet
                    snippet = ""
                    snippet_selectors = [
                        '[data-result="snippet"]',
                        '.result__snippet',
                        '.web-result__snippet',
                        'span[data-testid="result-snippet"]'
                    ]
                    for snippet_selector in snippet_selectors:
                        snippet_elem = element.query_selector(snippet_selector)
                        if snippet_elem:
                            snippet = snippet_elem.inner_text().strip()
                            if snippet:
                                break
                    
                    if title and url:
                        results['organic_results'].append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'position': i + 1
                        })
                        logger.info(f"      [{i+1}] {title[:50]}... - {url}")
                except Exception as e:
                    logger.warning(f"      Error extracting result {i+1}: {str(e)}")
                    continue
            
            results['total_results'] = len(results['organic_results'])
            logger.info(f"Found {results['total_results']} results from DuckDuckGo")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {str(e)}", exc_info=True)
            return {
                'query': query,
                'total_results': 0,
                'organic_results': [],
                'success': False,
                'error': str(e)
            }
    
    def _search_google(self, query: str, max_results: int = 10) -> Dict[str, any]:
        """Search using Google.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            encoded_query = quote_plus(query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            logger.info(f"Searching Google for: {query}")
            
            # Navigate to search page
            self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)  # Wait for results to load
            
            # Wait for search results container
            try:
                self.page.wait_for_selector('div#search, div[data-ved], div.g', timeout=10000)
            except PlaywrightTimeoutError:
                logger.warning("Search results container not found")
            
            results = {
                'query': query,
                'total_results': 0,
                'organic_results': [],
                'success': True,
                'error': None
            }
            
            # Extract search results
            # Google uses class "g" for search results
            result_elements = self.page.query_selector_all('div.g')
            
            for i, element in enumerate(result_elements[:max_results]):
                try:
                    # Extract title
                    title_elem = element.query_selector('h3')
                    title = title_elem.inner_text().strip() if title_elem else ""
                    
                    # Extract URL
                    link_elem = element.query_selector('a')
                    url = link_elem.get_attribute('href') if link_elem else ""
                    
                    # Extract snippet
                    snippet_elem = element.query_selector('span[style*="-webkit-line-clamp"], .VwiC3b')
                    snippet = snippet_elem.inner_text().strip() if snippet_elem else ""
                    
                    if title and url and url.startswith('http'):
                        results['organic_results'].append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'position': i + 1
                        })
                except Exception as e:
                    logger.warning(f"Error extracting result {i+1}: {str(e)}")
                    continue
            
            results['total_results'] = len(results['organic_results'])
            logger.info(f"Found {results['total_results']} results from Google")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Google: {str(e)}", exc_info=True)
            return {
                'query': query,
                'total_results': 0,
                'organic_results': [],
                'success': False,
                'error': str(e)
            }
    
    def _search_bing(self, query: str, max_results: int = 10) -> Dict[str, any]:
        """Search using Bing.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            encoded_query = quote_plus(query)
            search_url = f"https://www.bing.com/search?q={encoded_query}"
            
            logger.info(f"Searching Bing for: {query}")
            
            # Navigate to search page
            self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)  # Wait for results to load
            
            results = {
                'query': query,
                'total_results': 0,
                'organic_results': [],
                'success': True,
                'error': None
            }
            
            # Extract search results
            # Bing uses class "b_algo" for search results
            result_elements = self.page.query_selector_all('li.b_algo')
            
            for i, element in enumerate(result_elements[:max_results]):
                try:
                    # Extract title
                    title_elem = element.query_selector('h2 a')
                    title = title_elem.inner_text().strip() if title_elem else ""
                    
                    # Extract URL
                    url_elem = element.query_selector('h2 a')
                    url = url_elem.get_attribute('href') if url_elem else ""
                    
                    # Extract snippet
                    snippet_elem = element.query_selector('.b_caption p')
                    snippet = snippet_elem.inner_text().strip() if snippet_elem else ""
                    
                    if title and url:
                        results['organic_results'].append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'position': i + 1
                        })
                except Exception as e:
                    logger.warning(f"Error extracting result {i+1}: {str(e)}")
                    continue
            
            results['total_results'] = len(results['organic_results'])
            logger.info(f"Found {results['total_results']} results from Bing")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Bing: {str(e)}", exc_info=True)
            return {
                'query': query,
                'total_results': 0,
                'organic_results': [],
                'success': False,
                'error': str(e)
            }
    
    def extract_content_from_urls(self, urls: List[str], max_content_length: int = 5000, topic: str = None) -> List[Dict[str, str]]:
        """Extract content from URLs using Playwright.
        
        This method reuses the existing browser instance from AutoDiscoveryAgent.
        
        Args:
            urls: List of URLs to extract content from
            max_content_length: Maximum content length per URL (default: 5000)
            
        Returns:
            List of dictionaries with url and content
        """
        extracted_content = []
        
        # Ensure browser is initialized
        logger.info("   - Initializing browser...")
        self._initialize_browser()
        logger.info("   ✅ Browser initialized")
        
        for i, url in enumerate(urls, 1):
            try:
                logger.info(f"   📄 [{i}/{len(urls)}] Extracting content from: {url}")
                
                # Navigate to URL
                logger.info(f"      - Navigating to URL...")
                try:
                    self.page.goto(url, wait_until='networkidle', timeout=30000)
                    logger.info(f"      ✅ Page loaded")
                except PlaywrightTimeoutError:
                    logger.warning(f"      ⚠️ Network idle timeout, proceeding with current page state")
                except Exception as e:
                    logger.warning(f"      ❌ Failed to navigate: {str(e)}")
                    extracted_content.append({
                        'url': url,
                        'content': '',
                        'success': False,
                        'error': f"Failed to navigate: {str(e)}"
                    })
                    continue
                
                # Wait for content to load
                logger.info(f"      - Waiting for content to load...")
                self.page.wait_for_timeout(2000)
                
                try:
                    self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                except PlaywrightTimeoutError:
                    pass
                
                # Extract content using similar strategy as URLDataExtractor
                logger.info(f"      - Extracting content...")
                content = None
                selector_used = None
                
                # Strategy 1: Look for semantic HTML5 tags
                for selector in ['article', 'main', '[role="main"]', 'main article']:
                    try:
                        element = self.page.query_selector(selector)
                        if element:
                            content = element.inner_text()
                            if content and len(content.strip()) > 100:
                                selector_used = selector
                                logger.info(f"      ✅ Found content using selector: {selector}")
                                break
                    except Exception:
                        continue
                
                # Strategy 2: Look for common content class names
                if not content or len(content.strip()) < 100:
                    for selector in ['.content', '.post-content', '.article-content', '.entry-content',
                                     '.article-body', '.post-body', '.story-body', '.article-text', '.main-content']:
                        try:
                            element = self.page.query_selector(selector)
                            if element:
                                content = element.inner_text()
                                if content and len(content.strip()) > 100:
                                    selector_used = selector
                                    logger.info(f"      ✅ Found content using selector: {selector}")
                                    break
                        except Exception:
                            continue
                
                # Strategy 3: Get body text if nothing else works
                if not content or len(content.strip()) < 100:
                    body = self.page.query_selector('body')
                    if body:
                        content = body.inner_text()
                        selector_used = 'body'
                        logger.info(f"      ✅ Using body text as content")
                
                if content and len(content.strip()) > 50:
                    # Clean content (remove excessive whitespace)
                    lines = []
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and len(line) > 10:
                            lines.append(line)
                    content = '\n'.join(lines)
                    
                    # Truncate if too long
                    original_length = len(content)
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    
                    # Verify content with LLM immediately if topic is provided
                    is_relevant = True
                    verification_reason = "Not verified (no topic)"
                    
                    if topic:
                        logger.info(f"      🔍 Verifying content relevance with LLM...")
                        is_relevant, verification_reason = self._verify_single_page_with_llm(
                            topic=topic,
                            url=url,
                            content=content,
                            relevance_score=None  # Will be calculated later
                        )
                        
                        if is_relevant:
                            logger.info(f"      ✅ Content verified as RELEVANT")
                        else:
                            logger.warning(f"      ⚠️ Content verified as NOT RELEVANT - discarding: {verification_reason}")
                    
                    # Extract and analyze links from this page
                    discovered_links = []
                    if topic:
                        discovered_links = self._extract_and_analyze_links_from_page(url, topic)
                    
                    # Only add to extracted_content if verified or if no topic provided
                    if is_relevant or not topic:
                        item = {
                            'url': url,
                            'content': content if is_relevant or not topic else '',  # Clear content if not relevant
                            'success': True,
                            'verified': is_relevant if topic else None,
                            'verification_reason': verification_reason if topic else None,
                            'discovered_links': discovered_links  # Include discovered links
                        }
                        extracted_content.append(item)
                        
                        if is_relevant or not topic:
                            logger.info(f"      ✅ Content extracted and {'verified' if topic else 'stored'}: {original_length if original_length > max_content_length else len(content)} chars")
                            if discovered_links:
                                related_count = sum(1 for l in discovered_links if l.get('is_related'))
                                logger.info(f"      🔗 Discovered {len(discovered_links)} links ({related_count} related to topic)")
                    else:
                        # Page is not relevant - skip it, but still save discovered links if any
                        if discovered_links:
                            extracted_content.append({
                                'url': url,
                                'content': '',
                                'success': False,
                                'verified': False,
                                'error': 'Page not relevant to topic',
                                'discovered_links': discovered_links  # Still save links even if page not relevant
                            })
                        logger.info(f"      ⏭️ Skipping non-relevant page: {url}")
                else:
                    logger.warning(f"      ⚠️ No meaningful content extracted from {url}")
                    extracted_content.append({
                        'url': url,
                        'content': '',
                        'success': False,
                        'error': 'No meaningful content extracted',
                        'verified': False
                    })
                    
            except Exception as e:
                logger.error(f"      ❌ Error extracting content: {str(e)}")
                extracted_content.append({
                    'url': url,
                    'content': '',
                    'success': False,
                    'error': str(e)
                })
        
        logger.info(f"   ✅ Content extraction completed")
        return extracted_content
    
    def _get_domain_from_url(self, url: str) -> Optional[str]:
        """Extract domain from URL.
        
        Args:
            url: URL string
            
        Returns:
            Domain name or None
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove 'www.' if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return None
    
    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain.
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            True if same domain, False otherwise
        """
        domain1 = self._get_domain_from_url(url1)
        domain2 = self._get_domain_from_url(url2)
        return domain1 is not None and domain2 is not None and domain1 == domain2
    
    def _is_internal_link(self, base_url: str, link_url: str) -> bool:
        """Check if a link is an internal link (same domain).
        
        Args:
            base_url: Base URL (website URL)
            link_url: Link URL to check
            
        Returns:
            True if internal link, False otherwise
        """
        # Handle relative URLs
        if link_url.startswith('/') or link_url.startswith('./') or link_url.startswith('../'):
            return True
        
        # Handle absolute URLs
        if link_url.startswith('http://') or link_url.startswith('https://'):
            return self._is_same_domain(base_url, link_url)
        
        # Handle anchor links
        if link_url.startswith('#'):
            return True
        
        # Assume relative URLs are internal
        return True
    
    def _is_link_related_to_topic(self, url: str, topic: str, link_text: str = None) -> Tuple[bool, str]:
        """Check if a link is SEO-friendly or semantically related to a topic.
        
        Args:
            url: Link URL
            topic: Topic/term to check against
            link_text: Optional link text/anchor text
            
        Returns:
            Tuple of (is_related, reason)
        """
        try:
            from urllib.parse import urlparse, unquote
            
            # Normalize topic for comparison
            topic_lower = topic.lower().strip()
            topic_words = topic_lower.split()
            
            # Parse URL
            parsed = urlparse(url)
            path_lower = parsed.path.lower()
            domain_lower = parsed.netloc.lower()
            
            # Extract keywords from URL path (SEO-friendly URLs)
            path_segments = [seg for seg in path_lower.split('/') if seg]
            
            # Check 1: Domain contains topic
            if any(word in domain_lower for word in topic_words if len(word) > 3):
                return True, f"Domain contains topic keyword: {topic}"
            
            # Check 2: URL path contains topic keywords (SEO-friendly)
            url_text = ' '.join(path_segments)
            if any(word in url_text for word in topic_words if len(word) > 3):
                return True, f"URL path contains topic keyword: {topic}"
            
            # Check 3: Link text contains topic
            if link_text:
                link_text_lower = link_text.lower()
                if any(word in link_text_lower for word in topic_words if len(word) > 3):
                    return True, f"Link text contains topic keyword: {topic}"
            
            # Check 4: URL contains common content-related keywords along with topic
            content_keywords = ['about', 'services', 'products', 'company', 'blog', 'article', 
                               'guide', 'tutorial', 'help', 'support', 'faq', 'contact',
                               'team', 'careers', 'news', 'press', 'resources']
            
            # Decode URL-encoded characters
            decoded_path = unquote(path_lower)
            decoded_url = unquote(url.lower())
            
            # If URL has content keywords AND topic-related segments
            has_content_keyword = any(kw in decoded_url for kw in content_keywords)
            url_segments_text = ' '.join([unquote(seg) for seg in path_segments])
            
            # Check if any topic word appears in URL segments
            topic_in_url = any(word in url_segments_text for word in topic_words if len(word) > 3)
            
            if has_content_keyword and topic_in_url:
                return True, f"URL contains content keyword and topic: {topic}"
            
            # Check 5: Use LLM for semantic analysis if available (lightweight check)
            # For now, return False for strict filtering
            # Can be enhanced with LLM-based semantic similarity
            
            return False, "Link does not appear to be related to topic"
            
        except Exception as e:
            logger.warning(f"Error checking link relation: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def _extract_and_analyze_links_from_page(self, current_url: str, topic: str = None) -> List[Dict[str, str]]:
        """Extract and analyze links from a page to find related links.
        
        Args:
            current_url: Current page URL
            topic: Optional topic to check link relevance against
            
        Returns:
            List of dictionaries with link information (url, link_text, is_related, reason)
        """
        discovered_links = []
        
        try:
            # Find all links on the page
            links = self.page.query_selector_all('a[href]')
            
            for link in links:
                try:
                    href = link.get_attribute('href')
                    if not href:
                        continue
                    
                    # Normalize URL
                    normalized_url = self._normalize_url(current_url, href)
                    if not normalized_url:
                        continue
                    
                    # Skip if same as current URL
                    if normalized_url == current_url:
                        continue
                    
                    # Get link text
                    link_text = None
                    try:
                        link_text = link.inner_text().strip()
                    except:
                        pass
                    
                    # Check if link is related to topic (if topic provided)
                    is_related = False
                    reason = ""
                    
                    if topic:
                        is_related, reason = self._is_link_related_to_topic(
                            normalized_url,
                            topic,
                            link_text
                        )
                    
                    # Add to discovered links
                    discovered_links.append({
                        'url': normalized_url,
                        'link_text': link_text or '',
                        'is_related': is_related,
                        'reason': reason,
                        'domain': self._get_domain_from_url(normalized_url)
                    })
                    
                except Exception as e:
                    logger.debug(f"Error extracting link: {str(e)}")
                    continue
            
            logger.info(f"      🔗 Extracted {len(discovered_links)} links from page ({sum(1 for l in discovered_links if l['is_related'])} related to topic)")
            
        except Exception as e:
            logger.warning(f"Error extracting links from page: {str(e)}")
        
        return discovered_links
    
    def _normalize_url(self, base_url: str, link_url: str) -> Optional[str]:
        """Normalize a URL (convert relative to absolute).
        
        Args:
            base_url: Base URL
            link_url: Link URL (can be relative or absolute)
            
        Returns:
            Normalized absolute URL or None
        """
        try:
            from urllib.parse import urljoin, urlparse
            
            # Skip anchor links, javascript links, mailto, etc.
            if link_url.startswith('#') or link_url.startswith('javascript:') or link_url.startswith('mailto:'):
                return None
            
            # Skip data URIs
            if link_url.startswith('data:'):
                return None
            
            # Normalize relative URLs
            if link_url.startswith('/') or link_url.startswith('./') or link_url.startswith('../'):
                normalized = urljoin(base_url, link_url)
            elif link_url.startswith('http://') or link_url.startswith('https://'):
                normalized = link_url
            else:
                # Assume relative URL
                normalized = urljoin(base_url, link_url)
            
            # Parse to clean up the URL
            parsed = urlparse(normalized)
            
            # Filter out common non-content pages (but keep /about, /contact, /faq, /help, /support as they may contain useful info)
            path_lower = parsed.path.lower().rstrip('/')
            
            # Skip if path is just root or empty
            if not path_lower or path_lower == '/' or path_lower == '/index' or path_lower == '/index.html':
                # Allow root page (homepage) - don't filter it
                pass
            else:
                # Filter out common non-content patterns (as standalone path segments)
                non_content_patterns = [
                    '/login', '/signin', '/sign-up', '/signup', '/register', '/registration',
                    '/logout', '/signout', '/privacy-policy', '/terms-of-service', '/terms-of-use',
                    '/cookie-policy', '/legal', '/disclaimer', '/sitemap', '/robots.txt',
                    '/feed', '/rss', '/xml', '/json',
                    '/cart', '/checkout', '/payment', '/account', '/profile', '/settings',
                    '/admin', '/dashboard', '/wp-admin', '/wp-login',
                    '/_next', '/static', '/assets', '/css', '/js', '/img', '/images', '/media',
                    '/auth', '/oauth', '/callback', '/redirect', '/api/', '/ajax/'
                ]
                
                # Check if path matches any non-content pattern
                # Match if path starts with the pattern followed by / or end of string
                for pattern in non_content_patterns:
                    if path_lower == pattern or path_lower.startswith(pattern + '/') or path_lower.startswith(pattern + '?'):
                        return None
                
                # Also skip file extensions that are not HTML pages
                file_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.css', '.js', '.zip', '.tar', '.gz', '.mp4', '.mp3', '.avi']
                for ext in file_extensions:
                    if path_lower.endswith(ext):
                        return None
            
            # Remove fragments
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"
            
            return clean_url
        except Exception:
            return None
    
    def crawl_website_pages(
        self,
        start_url: str,
        max_pages: int = 15,
        max_content_length: int = 10000,  # Increased to 10000 to capture more content per page
        topic: str = None,  # Topic for real-time LLM verification
        max_depth: int = None  # Maximum crawl depth (default: MAX_CRAWL_DEPTH from config)
    ) -> List[Dict[str, str]]:
        """Crawl multiple pages from a website starting from a given URL.
        
        This method:
        1. Starts from the given URL (homepage or entry page) - depth 0
        2. Extracts all internal links from the page
        3. Visits those links (depth 1) to extract content
        4. Optionally continues to depth 2 if max_depth allows
        5. Continues until max_pages is reached or max_depth is exceeded or no more internal links are found
        
        Args:
            start_url: Starting URL (homepage or entry page)
            max_pages: Maximum number of pages to crawl (default: 15)
            max_content_length: Maximum content length per page (default: 10000)
            topic: Topic for real-time LLM verification
            max_depth: Maximum crawl depth (0 = only start URL, 1 = start + direct links, 2 = up to 2 levels deep)
                      Default: MAX_CRAWL_DEPTH from config
            
        Returns:
            List of dictionaries with url and content
        """
        if max_depth is None:
            max_depth = MAX_CRAWL_DEPTH
        
        extracted_content = []
        visited_urls = set()
        # Use tuples of (url, depth) to track depth level
        urls_to_visit = [(start_url, 0)]  # Start URL is at depth 0
        base_domain = self._get_domain_from_url(start_url)
        
        if not base_domain:
            logger.error(f"   ❌ Could not extract domain from URL: {start_url}")
            return extracted_content
        
        logger.info(f"   🌐 Starting website crawl from: {start_url}")
        logger.info(f"   - Domain: {base_domain}")
        logger.info(f"   - Max pages to crawl: {max_pages}")
        logger.info(f"   - Max crawl depth: {max_depth} (0 = only start URL, 1 = start + direct links, 2 = up to 2 levels)")
        
        # Ensure browser is initialized
        self._initialize_browser()
        
        page_count = 0
        while urls_to_visit and page_count < max_pages:
            current_url, current_depth = urls_to_visit.pop(0)
            
            # Skip if depth exceeds max_depth
            if current_depth > max_depth:
                logger.debug(f"      ⏭️ Skipping {current_url} - depth {current_depth} exceeds max_depth {max_depth}")
                continue
            
            # Skip if already visited
            if current_url in visited_urls:
                continue
            
            # Skip if not same domain
            if not self._is_same_domain(start_url, current_url):
                continue
            
            visited_urls.add(current_url)
            page_count += 1
            
            try:
                logger.info(f"   📄 [{page_count}/{max_pages}] [Depth {current_depth}/{max_depth}] Crawling: {current_url}")
                
                # Navigate to URL
                try:
                    self.page.goto(current_url, wait_until='networkidle', timeout=30000)
                except PlaywrightTimeoutError:
                    logger.warning(f"      ⚠️ Network idle timeout for {current_url}")
                except Exception as e:
                    logger.warning(f"      ❌ Failed to navigate to {current_url}: {str(e)}")
                    continue
                
                # Wait for content to load
                self.page.wait_for_timeout(2000)
                try:
                    self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                except PlaywrightTimeoutError:
                    pass
                
                # Extract content
                content = None
                
                # Strategy 1: Look for semantic HTML5 tags
                for selector in ['article', 'main', '[role="main"]', 'main article']:
                    try:
                        element = self.page.query_selector(selector)
                        if element:
                            content = element.inner_text()
                            if content and len(content.strip()) > 100:
                                break
                    except Exception:
                        continue
                
                # Strategy 2: Look for common content class names
                if not content or len(content.strip()) < 100:
                    for selector in ['.content', '.post-content', '.article-content', '.entry-content',
                                     '.article-body', '.post-body', '.story-body', '.article-text', '.main-content']:
                        try:
                            element = self.page.query_selector(selector)
                            if element:
                                content = element.inner_text()
                                if content and len(content.strip()) > 100:
                                    break
                        except Exception:
                            continue
                
                # Strategy 3: Get body text if nothing else works
                if not content or len(content.strip()) < 100:
                    body = self.page.query_selector('body')
                    if body:
                        content = body.inner_text()
                
                # Store content if found
                if content and len(content.strip()) > 50:
                    # Clean content
                    lines = []
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and len(line) > 10:
                            lines.append(line)
                    content = '\n'.join(lines)
                    
                    # Truncate if too long
                    original_length = len(content)
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    
                    # Verify content with LLM immediately if topic is provided
                    is_relevant = True
                    verification_reason = "Not verified (no topic)"
                    
                    if topic:
                        logger.info(f"      🔍 Verifying content relevance with LLM...")
                        is_relevant, verification_reason = self._verify_single_page_with_llm(
                            topic=topic,
                            url=current_url,
                            content=content,
                            relevance_score=None  # Will be calculated later
                        )
                        
                        if is_relevant:
                            logger.info(f"      ✅ Content verified as RELEVANT: {len(content)} chars")
                        else:
                            logger.warning(f"      ⚠️ Content verified as NOT RELEVANT - discarding: {verification_reason}")
                    
                    # Extract and analyze links from this page
                    discovered_links = []
                    if topic:
                        discovered_links = self._extract_and_analyze_links_from_page(current_url, topic)
                    
                    # Only add to extracted_content if verified or if no topic provided (for backward compatibility)
                    if is_relevant or not topic:
                        item = {
                            'url': current_url,
                            'content': content if is_relevant or not topic else '',  # Clear content if not relevant
                            'success': True,
                            'verified': is_relevant if topic else None,
                            'verification_reason': verification_reason if topic else None,
                            'discovered_links': discovered_links  # Include discovered links
                        }
                        extracted_content.append(item)
                        
                        if is_relevant or not topic:
                            logger.info(f"      ✅ Content extracted and {'verified' if topic else 'stored'}: {len(content)} chars")
                            if discovered_links:
                                related_count = sum(1 for l in discovered_links if l.get('is_related'))
                                logger.info(f"      🔗 Discovered {len(discovered_links)} links ({related_count} related to topic)")
                    else:
                        # Still save discovered links even if page not relevant
                        if discovered_links:
                            extracted_content.append({
                                'url': current_url,
                                'content': '',
                                'success': False,
                                'verified': False,
                                'error': 'Page not relevant to topic',
                                'discovered_links': discovered_links
                            })
                        logger.info(f"      ⏭️ Skipping non-relevant page: {current_url}")
                else:
                    logger.warning(f"      ⚠️ No meaningful content from {current_url}")
                    extracted_content.append({
                        'url': current_url,
                        'content': '',
                        'success': False,
                        'error': 'No meaningful content extracted',
                        'verified': False
                    })
                
                # Extract internal links from the page (only if we haven't reached max_pages and haven't exceeded depth)
                # Only add links if current_depth < max_depth (so we don't process links beyond max_depth)
                if page_count < max_pages and current_depth < max_depth:
                    try:
                        # Find all links on the page
                        links = self.page.query_selector_all('a[href]')
                        
                        new_links_count = 0
                        for link in links:
                            try:
                                href = link.get_attribute('href')
                                if not href:
                                    continue
                                
                                # Normalize URL
                                normalized_url = self._normalize_url(current_url, href)
                                if not normalized_url:
                                    continue
                                
                                # Check if it's an internal link
                                if not self._is_internal_link(start_url, normalized_url):
                                    continue
                                
                                # Check if already visited or queued
                                # Check against visited_urls and urls_to_visit (which now contains tuples)
                                is_visited = normalized_url in visited_urls
                                is_queued = any(url == normalized_url for url, _ in urls_to_visit)
                                
                                if not is_visited and not is_queued:
                                    # Add to queue if same domain and depth allows
                                    if self._is_same_domain(start_url, normalized_url):
                                        next_depth = current_depth + 1
                                        urls_to_visit.append((normalized_url, next_depth))
                                        new_links_count += 1
                                        logger.debug(f"      ➕ Found internal link (depth {next_depth}): {normalized_url}")
                            except Exception:
                                continue
                        
                        logger.info(f"      🔗 Found {new_links_count} new internal links to visit (will be at depth {current_depth + 1})")
                    except Exception as e:
                        logger.warning(f"      ⚠️ Error extracting links: {str(e)}")
                elif current_depth >= max_depth:
                    logger.debug(f"      ⏭️ Skipping link extraction - depth {current_depth} >= max_depth {max_depth}")
                
            except Exception as e:
                logger.error(f"      ❌ Error crawling {current_url}: {str(e)}")
                extracted_content.append({
                    'url': current_url,
                    'content': '',
                    'success': False,
                    'error': str(e)
                })
        
        logger.info(f"   ✅ Website crawl completed: {page_count} pages crawled")
        return extracted_content
    
    def _verify_single_page_with_llm(self, topic: str, url: str, content: str, relevance_score: float = None) -> Tuple[bool, str]:
        """Verify a single page's relevance to the topic using LLM.
        
        This method verifies EACH page immediately after extraction to ensure
        only relevant content is saved to the knowledge base.
        
        Args:
            topic: Topic name (e.g., "Priorcoder")
            url: URL of the page
            content: Extracted content from the page
            relevance_score: Optional relevance score from detector (for context)
            
        Returns:
            Tuple of (is_relevant, reason)
        """
        if not content or len(content.strip()) < 100:
            return False, "Content too short"
        
        llm = self._get_llm()
        content_preview = content[:2000] if len(content) > 2000 else content
        
        # Build verification prompt
        score_context = ""
        if relevance_score is not None:
            score_context = f"\nAutomated Relevance Score: {relevance_score:.3f} (0.0 to 1.0, where 1.0 is most relevant)\n"
        
        verification_prompt = f"""You are a content verification system. Your task is to determine if the provided web content is relevant to the topic "{topic}".

Topic: {topic}
Content URL: {url}{score_context}
Content Preview:
---
{content_preview}
---

Instructions:
1. Read the content carefully
2. Determine if the content is relevant to "{topic}"
3. Content is RELEVANT if it:
   - Mentions "{topic}" or related terms
   - Provides information about "{topic}"
   - Discusses features, structure, or details about "{topic}"
   - Is about the same subject as "{topic}"
   - Contains semantically related information about "{topic}"
4. Content is NOT RELEVANT if it:
   - Does not mention "{topic}" at all
   - Is about a completely different topic
   - Is just navigation, ads, or boilerplate text
   - Has no meaningful information about "{topic}"
   - Is off-topic or unrelated content

Answer ONLY with "RELEVANT" or "NOT_RELEVANT". Do not provide any explanation or additional text.

Answer:"""
        
        try:
            response = llm.invoke(verification_prompt).strip().upper()
            
            is_relevant = (
                "RELEVANT" in response or 
                "YES" in response or
                response.startswith("RELEVANT")
            )
            
            reason = "LLM verified as relevant" if is_relevant else "LLM verified as not relevant"
            
            return is_relevant, reason
            
        except Exception as e:
            logger.error(f"Error verifying page {url} with LLM: {str(e)}")
            # On error, be conservative - reject the page
            return False, f"Verification error: {str(e)}"
    
    def verify_content_relevance(self, topic: str, extracted_content: List[Dict]) -> List[Dict]:
        """Verify content relevance to topic using Relevance Detector and LLM.
        
        This method uses the multi-factor Relevance Detector to score content,
        then uses LLM verification as a final check for high-scoring content.
        Only content that is verified as relevant to the topic will be used.
        
        Args:
            topic: Topic name (e.g., "Priorcoder")
            extracted_content: List of extracted content dictionaries
            
        Returns:
            List of content dictionaries with 'verified' field and relevance scores added
        """
        from src.relevance_detector import RelevanceDetector
        
        verified_content = []
        
        logger.info(f"   - Verifying {len(extracted_content)} content items using Relevance Detector...")
        
        # Initialize Relevance Detector
        relevance_detector = RelevanceDetector(
            embedding_model=self.ollama_model,
            ollama_base_url=self.ollama_base_url
        )
        
        # Prepare pages for relevance scoring
        pages_to_score = []
        for item in extracted_content:
            if item.get('success') and item.get('content'):
                # Skip if already has relevance score (from earlier scoring)
                if 'relevance_score' not in item:
                    pages_to_score.append({
                        'url': item.get('url', ''),
                        'content': item.get('content', ''),
                        'title': item.get('title', '')
                    })
                else:
                    # Already scored, just mark for verification
                    verified_content.append(item)
        
        # Score pages using Relevance Detector
        if pages_to_score:
            logger.info(f"   📊 Scoring {len(pages_to_score)} pages using Relevance Detector...")
            ranked_pages = relevance_detector.rank_pages_by_relevance(topic, pages_to_score)
            
            # Add relevance scores to extracted_content
            for ranked_page in ranked_pages:
                url = ranked_page.get('url', '')
                for item in extracted_content:
                    if item.get('url') == url:
                        item['relevance_score'] = ranked_page.get('relevance_score', 0.0)
                        item['relevance_details'] = ranked_page.get('relevance_details', {})
                        item['confidence'] = ranked_page.get('confidence', 'unknown')
                        break
            
            # Add scored items to verified_content
            for ranked_page in ranked_pages:
                url = ranked_page.get('url', '')
                for item in extracted_content:
                    if item.get('url') == url and item not in verified_content:
                        verified_content.append(item)
                        break
        
        # Now filter based on relevance score (use 0.4 as threshold for multi-factor scoring)
        relevance_threshold = 0.4
        logger.info(f"   🔍 Filtering content based on relevance score (threshold: {relevance_threshold})...")
        
        for i, item in enumerate(verified_content, 1):
            if not item.get('success') or not item.get('content'):
                item['verified'] = False
                continue
            
            url = item.get('url', 'N/A')
            relevance_score = item.get('relevance_score', 0.0)
            
            # Skip if content is too short
            if len(item.get('content', '').strip()) < 100:
                logger.warning(f"      [{i}] {url}: Content too short - skipping")
                item['verified'] = False
                continue
            
            # ALWAYS verify with LLM - no auto-approval based on score
            # This ensures every page is verified before being saved
            content = item.get('content', '')
            is_relevant, reason = self._verify_single_page_with_llm(
                topic=topic,
                url=url,
                content=content,
                relevance_score=relevance_score
            )
            
            item['verified'] = is_relevant
            item['verification_reason'] = reason
            
            if is_relevant:
                logger.info(f"      ✅ [{i}/{len(verified_content)}] LLM verified as RELEVANT ({relevance_score:.3f}): {url[:50]}...")
            else:
                logger.warning(f"      ⚠️ [{i}/{len(verified_content)}] LLM verified as NOT_RELEVANT ({relevance_score:.3f}): {url[:50]}... - {reason}")
                # Clear content for non-relevant items
                item['original_content_length'] = len(item.get('content', ''))
                item['content'] = ''
        
        verified_count = sum(1 for item in verified_content if item.get('verified') and item.get('success') and item.get('content'))
        logger.info(f"   ✅ Verification completed: {verified_count}/{len(extracted_content)} items verified as relevant")
        
        # Log relevance score distribution
        relevant_scores = [item.get('relevance_score', 0.0) for item in verified_content if item.get('verified') and item.get('success')]
        if relevant_scores:
            avg_score = sum(relevant_scores) / len(relevant_scores)
            max_score = max(relevant_scores)
            min_score = min(relevant_scores)
            logger.info(f"   📊 Relevance score stats: avg={avg_score:.3f}, max={max_score:.3f}, min={min_score:.3f}")
        
        return verified_content
    
    def summarize_knowledge(self, topic: str, search_results: Dict, extracted_content: List[Dict], knowledge_template: Optional[str] = None) -> Tuple[bool, str, Optional[str], List[str]]:
        """Summarize knowledge using LLM.
        
        Args:
            topic: Topic name (e.g., "Upwork")
            search_results: Search results dictionary
            extracted_content: List of extracted content from URLs
            knowledge_template: Optional template for knowledge structure
            
        Returns:
            Tuple of (success, summarized_knowledge, error_message, source_urls)
        """
        try:
            llm = self._get_llm()
            
            # Build content from search results and extracted content
            content_parts = []
            
            # Add search result snippets
            for result in search_results.get('organic_results', [])[:5]:
                content_parts.append(f"Title: {result.get('title', '')}")
                content_parts.append(f"Snippet: {result.get('snippet', '')}")
                content_parts.append(f"URL: {result.get('url', '')}")
                content_parts.append("---")
            
            # Add extracted content (only verified and successful items)
            verified_items = [item for item in extracted_content if item.get('verified') and item.get('success') and item.get('content')]
            
            if not verified_items:
                return False, "", "No verified relevant content found to summarize", []
            
            # Collect source URLs for storage
            # Sort verified items by relevance score (highest first)
            verified_items_sorted = sorted(
                verified_items,
                key=lambda x: x.get('relevance_score', 0.0),
                reverse=True
            )
            
            # Collect all unique URLs from verified content
            source_urls = []
            seen_urls = set()
            url_relevance_map = {}  # Map URL to relevance score
            
            for item in verified_items_sorted:
                url = item.get('url', '')
                relevance_score = item.get('relevance_score', 0.0)
                if url and url not in seen_urls:
                    source_urls.append(url)
                    seen_urls.add(url)
                    url_relevance_map[url] = relevance_score
            
            # Limit to top pages to avoid overwhelming the summary
            # But collect all unique URLs for metadata
            # Use top pages by relevance score
            max_urls_for_summary = min(10, len(verified_items_sorted))  # Use up to 10 URLs for summary
            
            # Add content from top URLs (by relevance) to summary
            for item in verified_items_sorted[:max_urls_for_summary]:
                url = item.get('url', '')
                relevance_score = item.get('relevance_score', 0.0)
                confidence = item.get('confidence', 'unknown')
                content_parts.append(f"Content from {url} (Relevance: {relevance_score:.3f}, Confidence: {confidence}):")
                content_parts.append(item.get('content', ''))
                content_parts.append("---")
            
            # If there are more URLs, mention them
            if len(verified_items_sorted) > max_urls_for_summary:
                content_parts.append(f"\nNote: Additional {len(verified_items_sorted) - max_urls_for_summary} pages were crawled and verified (with relevance scores ranging from {verified_items_sorted[-1].get('relevance_score', 0.0):.3f} to {verified_items_sorted[max_urls_for_summary].get('relevance_score', 0.0):.3f}), but only the top {max_urls_for_summary} most relevant pages are included in this summary for readability.")
                content_parts.append("---")
            
            combined_content = "\n".join(content_parts)
            
            # Truncate if too long (keep first 50000 chars to accommodate more pages)
            # Increased from 30000 to 50000 to handle more crawled pages
            if len(combined_content) > 50000:
                combined_content = combined_content[:50000] + "... [Content truncated]"
            
            # Build prompt - emphasize extracting only relevant information
            if knowledge_template:
                prompt = f"""You are a knowledge extraction system. Extract and summarize ONLY information that is directly relevant to "{topic}" from the provided web search results and verified content.

IMPORTANT: 
- Only extract information that is directly related to "{topic}"
- Discard any information that is not about "{topic}"
- If content is not relevant to "{topic}", do not include it
- Focus only on factual information about "{topic}"

Knowledge Template (structure to follow):
{knowledge_template}

Verified Search Results and Content:
---
{combined_content}
---

Instructions:
1. Read the verified content carefully
2. Extract ONLY information that is directly relevant to "{topic}"
3. Structure the information according to the provided template
4. Focus on:
   - What is {topic}? (definition, purpose)
   - How does {topic} work? (functionality, process)
   - Key features and structure of {topic}
   - Important details about {topic}
5. DO NOT include:
   - Information about other topics
   - Navigation elements
   - Advertisements
   - Unrelated content
6. Provide accurate, well-organized information
7. Use clear headings and bullet points
8. Include only verified, relevant facts about "{topic}"

Summarized Knowledge about {topic}:
"""
            else:
                # Default template structure
                prompt = f"""You are a knowledge extraction system. Extract and summarize ONLY information that is directly relevant to "{topic}" from the provided web search results and verified content.

IMPORTANT: 
- Only extract information that is directly related to "{topic}"
- Discard any information that is not about "{topic}"
- If content is not relevant to "{topic}", do not include it
- Focus only on factual information about "{topic}"

Verified Search Results and Content:
---
{combined_content}
---

Instructions:
1. Read the verified content carefully
2. Extract ONLY information that is directly relevant to "{topic}"
3. Structure the information to answer:
   - What is {topic}? (definition, purpose, overview)
   - How does {topic} work? (functionality, how it operates)
   - What are the key features and structure of {topic}?
   - What are the important details about {topic}? (key facts, characteristics)
4. DO NOT include:
   - Information about other topics
   - Navigation elements
   - Advertisements
   - Unrelated content
5. Provide accurate, well-organized information
6. Use clear headings and bullet points
7. Include only verified, relevant facts about "{topic}"
8. Be concise but comprehensive

Summarized Knowledge about {topic}:
"""
            
            logger.info(f"   - Invoking LLM to summarize knowledge about {topic}...")
            logger.info(f"   - Using {len(verified_items)} verified content items")
            logger.info(f"   - Combined content length: {len(combined_content)} characters")
            
            summarized_knowledge = llm.invoke(prompt)
            
            if not summarized_knowledge or len(summarized_knowledge.strip()) < 100:
                logger.error(f"   ❌ LLM returned insufficient content ({len(summarized_knowledge) if summarized_knowledge else 0} characters)")
                return False, "", f"LLM returned insufficient content (minimum 100 characters required, got {len(summarized_knowledge) if summarized_knowledge else 0})", source_urls
            
            # Verify the summarized knowledge is actually about the topic
            topic_lower = topic.lower()
            summarized_lower = summarized_knowledge.lower()
            
            # Check if topic is mentioned in the summary
            if topic_lower not in summarized_lower and len(topic_lower) > 3:
                # Topic not mentioned - might be irrelevant summary
                logger.warning(f"   ⚠️ Topic '{topic}' not mentioned in summary - may not be relevant")
                # Still return it, but log the warning
            
            # Append source URLs to the knowledge for LLM reference
            # This allows the LLM to see URLs when querying and decide to collect more information
            if source_urls:
                source_urls_text = "\n\n---\n\n## Source URLs\n\n"
                source_urls_text += f"The following {len(source_urls)} URLs were used to collect this information:\n\n"
                
                # Group URLs by domain for better readability
                domains = {}
                for url in source_urls:
                    domain = self._get_domain_from_url(url)
                    if domain:
                        if domain not in domains:
                            domains[domain] = []
                        domains[domain].append(url)
                    else:
                        # If domain can't be extracted, put in "Other" category
                        if "Other" not in domains:
                            domains["Other"] = []
                        domains["Other"].append(url)
                
                # Display URLs grouped by domain
                for domain, urls in domains.items():
                    source_urls_text += f"### {domain} ({len(urls)} pages):\n\n"
                    # Show first 20 URLs per domain, then summarize
                    urls_to_show = urls[:20]
                    for i, url in enumerate(urls_to_show, 1):
                        source_urls_text += f"{i}. {url}\n"
                    if len(urls) > 20:
                        source_urls_text += f"... and {len(urls) - 20} more pages from this domain\n"
                    source_urls_text += "\n"
                
                source_urls_text += "**Note:** These URLs can be used to collect additional information if needed. "
                source_urls_text += "For example, if this is a company, you might want to explore their services, "
                source_urls_text += "products, or other pages on their website to gather more comprehensive knowledge.\n"
                
                summarized_knowledge = summarized_knowledge.strip() + source_urls_text
            
            logger.info(f"   ✅ Successfully summarized knowledge about {topic} ({len(summarized_knowledge)} characters)")
            logger.info(f"   - Summary preview: {summarized_knowledge[:200]}...")
            logger.info(f"   - Source URLs: {len(source_urls)} URLs included")
            
            # Group URLs by domain for logging
            domains = {}
            for url in source_urls:
                domain = self._get_domain_from_url(url)
                if domain:
                    if domain not in domains:
                        domains[domain] = []
                    domains[domain].append(url)
            
            # Log URLs grouped by domain
            for domain, urls in domains.items():
                logger.info(f"      - {domain}: {len(urls)} pages")
                # Log first 5 URLs per domain
                for i, url in enumerate(urls[:5], 1):
                    logger.info(f"         {i}. {url}")
                if len(urls) > 5:
                    logger.info(f"         ... and {len(urls) - 5} more pages")
            
            return True, summarized_knowledge.strip(), None, source_urls
            
        except Exception as e:
            logger.error(f"Error summarizing knowledge: {str(e)}", exc_info=True)
            return False, "", f"Error summarizing knowledge: {str(e)}", []
    
    def discover_and_store(
        self,
        topic: str,
        knowledge_template: Optional[str] = None,
        max_search_results: int = None,
        max_urls_to_extract: int = None,
        max_pages_to_crawl: int = None,
        vectorstore_manager=None,
        embedding_model: str = "nomic-embed-text",
        profile_id: str = "default",
        is_from_url: bool = False
    ) -> Dict[str, any]:
        """Discover knowledge about a topic and store it in vector DB.
        
        This is the main method that performs the complete auto-discovery process:
        1. Search the web
        2. Extract content from URLs
        3. Summarize knowledge
        4. Store in vector DB
        
        Args:
            topic: Topic to discover (e.g., "Upwork")
            knowledge_template: Optional template for knowledge structure
            max_search_results: Maximum number of search results (default: from config)
            max_urls_to_extract: Maximum number of URLs to extract content from (default: from config)
            max_pages_to_crawl: Maximum number of pages to crawl from primary website (default: from config)
            vectorstore_manager: VectorStoreManager instance for storing
            embedding_model: Embedding model name
            profile_id: Profile ID to store knowledge under
            is_from_url: Whether the topic is from a URL (affects search query format)
            
        Returns:
            Dictionary with:
                - success: True if successful
                - knowledge: Summarized knowledge
                - vectorstore: Created vectorstore
                - persist_dir: Persistence directory
                - kb_id: Knowledge base ID
                - error: Error message if failed
        """
        # Use config defaults if not provided
        if max_search_results is None:
            max_search_results = MAX_SEARCH_RESULTS
        if max_urls_to_extract is None:
            max_urls_to_extract = MAX_URLS_TO_EXTRACT
        if max_pages_to_crawl is None:
            max_pages_to_crawl = MAX_PAGES_TO_CRAWL
        
        try:
            logger.info("=" * 80)
            logger.info(f"🔍 Starting auto-discovery for topic: {topic}")
            logger.info("=" * 80)
            logger.info(f"   - Max search results: {max_search_results}")
            logger.info(f"   - Max URLs to extract: {max_urls_to_extract}")
            logger.info(f"   - Max pages to crawl from primary website: {max_pages_to_crawl}")
            
            # Step 1: Build search query (simplified for URL case)
            logger.info("📝 Step 1/6: Building search query...")
            
            if is_from_url:
                # For URL case, use direct term search (e.g., "Priorcoder")
                search_query = topic.strip().capitalize()
                logger.info(f"   ✅ Search query (URL mode - direct term): {search_query}")
            else:
                # For question case, use topic name (simpler than question format)
                search_query = topic.strip().capitalize()
                logger.info(f"   ✅ Search query (Question mode): {search_query}")
            
            # Step 2: Search the web
            logger.info(f"🌐 Step 2/6: Searching the web using {self.search_engine}...")
            logger.info(f"   - Search engine: {self.search_engine}")
            logger.info(f"   - Max results: {max_search_results}")
            logger.info(f"   - Query type: {'Direct term (URL)' if is_from_url else 'Topic-based'}")
            logger.info(f"   - Search query: {search_query}")
            
            # Search with the query
            search_results = self.search_web(search_query, max_results=max_search_results)
            
            if not search_results.get('success') or not search_results.get('organic_results'):
                error_msg = search_results.get('error', 'Unknown error')
                logger.error(f"   ❌ Failed to get search results: {error_msg}")
                logger.info("=" * 80)
                return {
                    'success': False,
                    'knowledge': '',
                    'vectorstore': None,
                    'persist_dir': None,
                    'kb_id': None,
                    'error': f"Failed to get search results: {error_msg}"
                }
            
            logger.info(f"   ✅ Found {len(search_results['organic_results'])} search results")
            for i, result in enumerate(search_results['organic_results'][:5], 1):
                logger.info(f"      {i}. {result.get('title', 'N/A')} - {result.get('url', 'N/A')}")
            
            # Step 3: Use Relevance Detector to identify maximum match website
            logger.info(f"📄 Step 3/6: Extracting content from search results...")
            logger.info(f"   🔍 Using Relevance Detector to identify maximum match website...")
            
            if not search_results.get('organic_results'):
                logger.error("   ❌ No search results found")
                return {
                    'success': False,
                    'knowledge': '',
                    'vectorstore': None,
                    'persist_dir': None,
                    'kb_id': None,
                    'error': "No search results found"
                }
            
            # Initialize Relevance Detector
            relevance_detector = RelevanceDetector(
                embedding_model=embedding_model,
                ollama_base_url=self.ollama_base_url
            )
            
            # Prepare pages for relevance scoring (extract titles from search results first)
            pages_to_score = []
            for result in search_results['organic_results']:
                pages_to_score.append({
                    'url': result.get('url', ''),
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'content': result.get('snippet', '')  # Use snippet for initial scoring
                })
            
            # Rank pages by relevance using search result snippets
            logger.info(f"   📊 Ranking {len(pages_to_score)} search results by relevance...")
            ranked_pages = relevance_detector.rank_pages_by_relevance(topic, pages_to_score)
            
            # Identify the maximum match website (highest relevance score)
            primary_page = ranked_pages[0]
            primary_url = primary_page.get('url', '')
            primary_domain = self._get_domain_from_url(primary_url)
            primary_relevance_score = primary_page.get('relevance_score', 0.0)
            primary_confidence = primary_page.get('confidence', 'unknown')
            
            logger.info(f"   🎯 Maximum match website identified: {primary_domain}")
            logger.info(f"   - Primary URL: {primary_url}")
            logger.info(f"   - Title: {primary_page.get('title', 'N/A')}")
            logger.info(f"   - Relevance Score: {primary_relevance_score:.3f}")
            logger.info(f"   - Confidence: {primary_confidence}")
            
            # Adjust max_pages_to_crawl based on relevance score
            # Higher relevance = crawl more pages
            if primary_relevance_score >= 0.8:
                adjusted_max_pages = int(max_pages_to_crawl * 1.2)  # 20% more pages for very relevant sites
                logger.info(f"   📈 High relevance score ({primary_relevance_score:.3f}), increasing crawl to {adjusted_max_pages} pages")
            elif primary_relevance_score >= 0.6:
                adjusted_max_pages = max_pages_to_crawl  # Use configured value
                logger.info(f"   📊 Medium-high relevance score ({primary_relevance_score:.3f}), using {adjusted_max_pages} pages")
            else:
                adjusted_max_pages = int(max_pages_to_crawl * 0.8)  # 20% fewer pages for lower relevance
                logger.info(f"   📉 Lower relevance score ({primary_relevance_score:.3f}), reducing crawl to {adjusted_max_pages} pages")
            
            # Step 3a: Crawl multiple pages from the primary website
            logger.info(f"   🌐 Crawling {adjusted_max_pages} pages from primary website...")
            logger.info(f"   - Starting from: {primary_url}")
            logger.info(f"   - Relevance-based crawling: {adjusted_max_pages} pages")
            logger.info(f"   - Real-time LLM verification enabled for each page")
            primary_content = self.crawl_website_pages(
                start_url=primary_url,
                max_pages=adjusted_max_pages,
                max_content_length=10000,  # Increased to capture more content per page
                topic=topic  # Pass topic for real-time LLM verification
            )
            
            primary_successful = sum(1 for item in primary_content if item.get('success'))
            logger.info(f"   ✅ Crawled {primary_successful}/{len(primary_content)} pages from primary website")
            
            # Step 3b: Score and rank crawled pages by relevance
            logger.info(f"   🔍 Scoring crawled pages by relevance...")
            if primary_content:
                # Prepare pages for relevance scoring (with full content now)
                crawled_pages_to_score = []
                for item in primary_content:
                    if item.get('success') and item.get('content'):
                        crawled_pages_to_score.append({
                            'url': item.get('url', ''),
                            'content': item.get('content', ''),
                            'title': ''  # We don't have titles for crawled pages
                        })
                
                if crawled_pages_to_score:
                    # Rank crawled pages by relevance
                    ranked_crawled_pages = relevance_detector.rank_pages_by_relevance(topic, crawled_pages_to_score)
                    
                    # Filter out low-relevance pages (optional - can keep all if desired)
                    # For now, we'll keep all pages but mark relevance scores
                    for i, page in enumerate(ranked_crawled_pages):
                        # Find corresponding item in primary_content and add relevance info
                        url = page.get('url', '')
                        for item in primary_content:
                            if item.get('url') == url:
                                item['relevance_score'] = page.get('relevance_score', 0.0)
                                item['relevance_details'] = page.get('relevance_details', {})
                                item['confidence'] = page.get('confidence', 'unknown')
                                break
                    
                    # Log top relevant pages
                    logger.info(f"   📊 Top relevant crawled pages:")
                    for i, page in enumerate(ranked_crawled_pages[:5], 1):
                        logger.info(f"      {i}. {page.get('url', 'N/A')}: {page.get('relevance_score', 0.0):.3f} ({page.get('confidence', 'unknown')})")
            
            # Step 3c: Extract content from additional URLs (if max_urls_to_extract > 1)
            additional_content = []
            if max_urls_to_extract > 1:
                # Get additional URLs from ranked results (skip the primary one)
                additional_urls = []
                for ranked_page in ranked_pages[1:max_urls_to_extract]:
                    url = ranked_page.get('url', '')
                    relevance_score = ranked_page.get('relevance_score', 0.0)
                    # Skip if same domain as primary (already crawled)
                    # Only extract from pages with reasonable relevance (>= 0.4)
                    if url and not self._is_same_domain(primary_url, url) and relevance_score >= 0.4:
                        additional_urls.append(url)
                        logger.info(f"      ✅ Selected additional URL: {url} (relevance: {relevance_score:.3f})")
                
                if additional_urls:
                    logger.info(f"   📄 Extracting content from {len(additional_urls)} additional URLs...")
                    logger.info(f"   - Real-time LLM verification enabled for each URL")
                    additional_content = self.extract_content_from_urls(additional_urls, topic=topic)
                    
                    # Score additional URLs by relevance
                    if additional_content:
                        additional_pages_to_score = []
                        for item in additional_content:
                            if item.get('success') and item.get('content'):
                                # Get title from search results
                                title = ''
                                for result in search_results['organic_results']:
                                    if result.get('url') == item.get('url'):
                                        title = result.get('title', '')
                                        break
                                
                                additional_pages_to_score.append({
                                    'url': item.get('url', ''),
                                    'content': item.get('content', ''),
                                    'title': title
                                })
                        
                        if additional_pages_to_score:
                            ranked_additional_pages = relevance_detector.rank_pages_by_relevance(topic, additional_pages_to_score)
                            
                            # Add relevance info to additional_content
                            for page in ranked_additional_pages:
                                url = page.get('url', '')
                                for item in additional_content:
                                    if item.get('url') == url:
                                        item['relevance_score'] = page.get('relevance_score', 0.0)
                                        item['relevance_details'] = page.get('relevance_details', {})
                                        item['confidence'] = page.get('confidence', 'unknown')
                                        break
                    
                    additional_successful = sum(1 for item in additional_content if item.get('success'))
                    logger.info(f"   ✅ Successfully extracted content from {additional_successful}/{len(additional_content)} additional URLs")
            
            # Combine all extracted content
            extracted_content = primary_content + additional_content
            
            # Collect all discovered links from extracted content
            all_discovered_links = []
            for item in extracted_content:
                discovered_links = item.get('discovered_links', [])
                if discovered_links:
                    all_discovered_links.extend(discovered_links)
            
            # Remove duplicates (by URL)
            unique_discovered_links = {}
            for link in all_discovered_links:
                url = link.get('url', '')
                if url and url not in unique_discovered_links:
                    unique_discovered_links[url] = link
            
            discovered_links_list = list(unique_discovered_links.values())
            related_links = [l for l in discovered_links_list if l.get('is_related', False)]
            
            logger.info(f"   🔗 Discovered {len(discovered_links_list)} unique links ({len(related_links)} related to topic)")
            
            successful_extractions = sum(1 for item in extracted_content if item.get('success'))
            logger.info(f"   ✅ Total: Successfully extracted content from {successful_extractions}/{len(extracted_content)} URLs")
            
            # Log summary
            logger.info(f"   📊 Content extraction summary:")
            logger.info(f"      - Primary website pages: {primary_successful}/{len(primary_content)}")
            if additional_content:
                logger.info(f"      - Additional URLs: {sum(1 for item in additional_content if item.get('success'))}/{len(additional_content)}")
            
            for item in extracted_content[:10]:  # Log first 10 items
                if item.get('success'):
                    content_length = len(item.get('content', ''))
                    logger.info(f"      ✅ {item.get('url', 'N/A')}: {content_length} characters")
                else:
                    logger.warning(f"      ⚠️ {item.get('url', 'N/A')}: {item.get('error', 'Failed')}")
            
            # Step 4: Verify content relevance using LLM
            logger.info("🔍 Step 4/6: Verifying content relevance using LLM...")
            logger.info(f"   - Topic: {topic}")
            logger.info(f"   - Verifying {successful_extractions} extracted content items...")
            
            verified_content = self.verify_content_relevance(topic, extracted_content)
            
            # Count verified items that have content (non-verified items have content cleared)
            verified_count = sum(1 for item in verified_content if item.get('verified') and item.get('success') and item.get('content'))
            logger.info(f"   ✅ Verified {verified_count}/{successful_extractions} content items as relevant to '{topic}'")
            
            if verified_count == 0:
                logger.error(f"   ❌ No verified relevant content found for topic: {topic}")
                logger.error(f"   - All extracted content was not relevant to '{topic}'")
                logger.error(f"   - This could mean:")
                logger.error(f"     1. The search query didn't find relevant results")
                logger.error(f"     2. The extracted content was not about '{topic}'")
                logger.error(f"     3. The content verification is too strict")
                logger.info("=" * 80)
                return {
                    'success': False,
                    'knowledge': '',
                    'vectorstore': None,
                    'persist_dir': None,
                    'kb_id': None,
                    'error': f"No verified relevant content found for topic '{topic}'. All extracted content was not relevant.",
                    'verified_content_count': 0,
                    'total_content_count': successful_extractions
                }
            
            for item in verified_content:
                if item.get('verified') and item.get('content'):
                    logger.info(f"      ✅ {item.get('url', 'N/A')}: Relevant content ({len(item.get('content', ''))} chars)")
                elif item.get('success'):
                    logger.warning(f"      ⚠️ {item.get('url', 'N/A')}: Not relevant - discarded")
            
            # Step 5: Summarize knowledge from verified content only
            logger.info("🤖 Step 5/6: Summarizing knowledge using LLM...")
            logger.info(f"   - Topic: {topic}")
            logger.info(f"   - LLM Model: {self.ollama_model}")
            logger.info(f"   - Knowledge template: {'Provided' if knowledge_template else 'None'}")
            logger.info(f"   - Using {verified_count} verified content items")
            logger.info(f"   - Only verified, relevant content will be used for summarization")
            
            summarize_success, summarized_knowledge, summarize_error, source_urls = self.summarize_knowledge(
                topic,
                search_results,
                verified_content,  # Use only verified content
                knowledge_template
            )
            
            if not summarize_success:
                logger.error(f"   ❌ Failed to summarize knowledge: {summarize_error}")
                
                # Check if the issue is no verified content
                if "No verified relevant content" in summarize_error:
                    logger.error(f"   ❌ No verified relevant content found for topic: {topic}")
                    logger.error(f"   - This means the extracted content was not relevant to '{topic}'")
                    logger.error(f"   - Consider trying a different search query or topic")
                
                logger.info("=" * 80)
                return {
                    'success': False,
                    'knowledge': '',
                    'vectorstore': None,
                    'persist_dir': None,
                    'kb_id': None,
                    'error': f"Failed to summarize knowledge: {summarize_error}",
                    'verified_content_count': verified_count,
                    'total_content_count': successful_extractions,
                    'source_urls': []
                }
            
            knowledge_length = len(summarized_knowledge)
            logger.info(f"   ✅ Knowledge summarized: {knowledge_length} characters")
            logger.info(f"   - Preview: {summarized_knowledge[:200]}...")
            logger.info(f"   - Source URLs: {len(source_urls)} URLs collected")
            
            # Step 6: Store in vector DB with source URLs
            logger.info("💾 Step 6/6: Storing knowledge in vector DB...")
            if vectorstore_manager:
                try:
                    logger.info(f"   - Profile ID: {profile_id}")
                    logger.info(f"   - Embedding model: {embedding_model}")
                    logger.info(f"   - Knowledge length: {knowledge_length} characters")
                    logger.info(f"   - Source URLs: {len(source_urls)} URLs to store")
                    
                    # Create vectorstore with source URLs as metadata
                    logger.info("   - Creating vectorstore with source URLs...")
                    vectorstore, persist_dir = vectorstore_manager.create_vectorstore(
                        summarized_knowledge,
                        embedding_model,
                        self.ollama_base_url,
                        source_urls=source_urls,  # Pass source URLs
                        topic=topic  # Pass topic for metadata
                    )
                    
                    # Get KB ID from persist directory
                    kb_id = os.path.basename(persist_dir)
                    
                    logger.info(f"   ✅ Successfully stored knowledge in vector DB")
                    logger.info(f"      - KB ID: {kb_id}")
                    logger.info(f"      - Persist directory: {persist_dir}")
                    logger.info("=" * 80)
                    logger.info(f"✅ AUTO-DISCOVERY COMPLETED: {topic}")
                    logger.info("=" * 80)
                    
                    return {
                        'success': True,
                        'knowledge': summarized_knowledge,
                        'vectorstore': vectorstore,
                        'persist_dir': persist_dir,
                        'kb_id': kb_id,
                        'error': None,
                        'search_results': search_results,
                        'extracted_content': extracted_content,
                        'source_urls': source_urls,  # Include source URLs in result
                        'discovered_links': discovered_links_list  # Include discovered links from pages
                    }
                except Exception as e:
                    logger.error(f"   ❌ Error storing knowledge in vector DB: {str(e)}", exc_info=True)
                    logger.info("=" * 80)
                    return {
                        'success': False,
                        'knowledge': summarized_knowledge,  # Return knowledge even if storage fails
                        'vectorstore': None,
                        'persist_dir': None,
                        'kb_id': None,
                        'error': f"Failed to store in vector DB: {str(e)}",
                        'source_urls': source_urls
                    }
            else:
                logger.warning("   ⚠️ VectorStoreManager not provided, skipping storage")
                logger.info("=" * 80)
                # Return knowledge even if vectorstore_manager is not provided
                return {
                    'success': True,
                    'knowledge': summarized_knowledge,
                    'vectorstore': None,
                    'persist_dir': None,
                    'kb_id': None,
                    'error': None,
                    'search_results': search_results,
                    'extracted_content': extracted_content,
                    'source_urls': source_urls,
                    'discovered_links': discovered_links_list  # Include discovered links from pages
                }
                
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ ERROR IN AUTO-DISCOVERY: {str(e)}", exc_info=True)
            logger.error("=" * 80)
            return {
                'success': False,
                'knowledge': '',
                'vectorstore': None,
                'persist_dir': None,
                'kb_id': None,
                'error': f"Error in auto-discovery: {str(e)}",
                'source_urls': []
            }
        finally:
            # Clean up browser
            logger.info("🧹 Cleaning up browser...")
            self._close_browser()
            logger.info("   ✅ Browser closed")
    
    def _build_search_query(self, topic: str, is_from_url: bool = False) -> str:
        """Build search query from topic.
        
        Args:
            topic: Topic name (e.g., "Upwork" or "Priorcoder")
            is_from_url: Whether the topic is from a URL (if True, use direct term search)
            
        Returns:
            Search query string
        """
        if is_from_url:
            # For URL case, search directly for the term (more effective)
            # Clean topic name (remove extra spaces, capitalize properly)
            topic_clean = topic.strip().capitalize()
            return topic_clean
        else:
            # For question case, use question format but simpler
            # Use just the topic name for better search results
            topic_clean = topic.strip().capitalize()
            return topic_clean
    
    def _generate_search_queries(self, topic: str, is_from_url: bool = False) -> List[str]:
        """Generate multiple search queries for comprehensive coverage.
        
        Args:
            topic: Topic name (e.g., "Priorcoder")
            is_from_url: Whether the topic is from a URL
            
        Returns:
            List of search query strings
        """
        topic_clean = topic.strip().capitalize()
        
        if is_from_url:
            # For URL case, use direct term searches with variations
            queries = [
                topic_clean,
                f"{topic_clean} platform",
                f"{topic_clean} website",
                f"about {topic_clean}",
            ]
        else:
            # For question case, use question-based queries
            queries = [
                topic_clean,
                f"what is {topic_clean}",
                f"how does {topic_clean} work",
                f"{topic_clean} features",
            ]
        
        # Remove duplicates and empty strings
        queries = list(dict.fromkeys([q.strip() for q in queries if q.strip()]))
        return queries[:3]  # Limit to 3 queries
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup browser."""
        self._close_browser()

