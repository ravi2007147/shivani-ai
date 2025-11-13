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
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus, urlparse
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

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
    
    def extract_content_from_urls(self, urls: List[str], max_content_length: int = 5000) -> List[Dict[str, str]]:
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
                        logger.info(f"      ✅ Content extracted: {original_length} chars (truncated to {max_content_length})")
                    else:
                        logger.info(f"      ✅ Content extracted: {len(content)} chars")
                    
                    extracted_content.append({
                        'url': url,
                        'content': content,
                        'success': True
                    })
                else:
                    logger.warning(f"      ⚠️ No meaningful content extracted from {url}")
                    extracted_content.append({
                        'url': url,
                        'content': '',
                        'success': False,
                        'error': 'No meaningful content extracted'
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
    
    def verify_content_relevance(self, topic: str, extracted_content: List[Dict]) -> List[Dict]:
        """Verify content relevance to topic using LLM.
        
        This method filters out irrelevant content before summarizing.
        Only content that is verified as relevant to the topic will be used.
        
        Args:
            topic: Topic name (e.g., "Priorcoder")
            extracted_content: List of extracted content dictionaries
            
        Returns:
            List of content dictionaries with 'verified' field added
        """
        llm = self._get_llm()
        verified_content = []
        
        logger.info(f"   - Verifying {len(extracted_content)} content items...")
        
        for i, item in enumerate(extracted_content, 1):
            if not item.get('success') or not item.get('content'):
                # Keep failed extractions as-is (they'll be filtered out later)
                item['verified'] = False
                verified_content.append(item)
                continue
            
            content = item.get('content', '')
            url = item.get('url', 'N/A')
            
            # Skip if content is too short
            if len(content.strip()) < 100:
                logger.warning(f"      [{i}] {url}: Content too short ({len(content)} chars) - skipping verification")
                item['verified'] = False
                verified_content.append(item)
                continue
            
            # Truncate content for verification (first 2000 chars should be enough)
            content_preview = content[:2000] if len(content) > 2000 else content
            
            # Verify relevance using LLM
            verification_prompt = f"""You are a content verification system. Your task is to determine if the provided web content is relevant to the topic "{topic}".

Topic: {topic}
Content URL: {url}
Content Preview:
---
{content_preview}
---

Instructions:
1. Read the content carefully
2. Determine if the content is relevant to "{topic}"
3. Content is relevant if it:
   - Mentions "{topic}" or related terms
   - Provides information about "{topic}"
   - Discusses features, structure, or details about "{topic}"
   - Is about the same subject as "{topic}"
4. Content is NOT relevant if it:
   - Does not mention "{topic}" at all
   - Is about a completely different topic
   - Is just navigation, ads, or boilerplate text
   - Has no meaningful information about "{topic}"

Answer ONLY with "RELEVANT" or "NOT_RELEVANT". Do not provide any explanation or additional text.

Answer:"""
            
            try:
                logger.info(f"      [{i}/{len(extracted_content)}] Verifying: {url[:50]}...")
                response = llm.invoke(verification_prompt).strip().upper()
                
                # Check if response indicates relevance
                is_relevant = (
                    "RELEVANT" in response or 
                    "YES" in response or
                    response.startswith("RELEVANT")
                )
                item['verified'] = is_relevant
                
                if is_relevant:
                    logger.info(f"      ✅ Verified as RELEVANT: {url}")
                else:
                    logger.warning(f"      ⚠️ Verified as NOT_RELEVANT - discarding: {url}")
                    # Clear content for non-relevant items (don't use in summarization)
                    # But keep the item in the list for logging purposes
                    item['original_content_length'] = len(item.get('content', ''))
                    item['content'] = ''  # Clear content to exclude from summarization
                
                verified_content.append(item)
                
            except Exception as e:
                logger.error(f"      ❌ Error verifying content from {url}: {str(e)}")
                # On error, assume relevant (better to include than exclude)
                logger.warning(f"      ⚠️ Assuming relevant due to verification error")
                item['verified'] = True
                verified_content.append(item)
        
        verified_count = sum(1 for item in verified_content if item.get('verified') and item.get('success') and item.get('content'))
        logger.info(f"   ✅ Verification completed: {verified_count}/{len(extracted_content)} items verified as relevant")
        
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
            source_urls = []
            for item in verified_items[:3]:  # Use top 3 verified URLs
                url = item.get('url', '')
                if url:
                    source_urls.append(url)
                content_parts.append(f"Content from {url}:")
                content_parts.append(item.get('content', ''))
                content_parts.append("---")
            
            combined_content = "\n".join(content_parts)
            
            # Truncate if too long (keep first 30000 chars)
            if len(combined_content) > 30000:
                combined_content = combined_content[:30000] + "... [Content truncated]"
            
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
                source_urls_text += "The following URLs were used to collect this information:\n\n"
                for i, url in enumerate(source_urls, 1):
                    source_urls_text += f"{i}. {url}\n"
                source_urls_text += "\n**Note:** These URLs can be used to collect additional information if needed. "
                source_urls_text += "For example, if this is a company, you might want to explore their services, "
                source_urls_text += "products, or other pages on their website to gather more comprehensive knowledge.\n"
                
                summarized_knowledge = summarized_knowledge.strip() + source_urls_text
            
            logger.info(f"   ✅ Successfully summarized knowledge about {topic} ({len(summarized_knowledge)} characters)")
            logger.info(f"   - Summary preview: {summarized_knowledge[:200]}...")
            logger.info(f"   - Source URLs: {len(source_urls)} URLs included")
            for i, url in enumerate(source_urls, 1):
                logger.info(f"      {i}. {url}")
            
            return True, summarized_knowledge.strip(), None, source_urls
            
        except Exception as e:
            logger.error(f"Error summarizing knowledge: {str(e)}", exc_info=True)
            return False, "", f"Error summarizing knowledge: {str(e)}", []
    
    def discover_and_store(
        self,
        topic: str,
        knowledge_template: Optional[str] = None,
        max_search_results: int = 10,
        max_urls_to_extract: int = 3,
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
            max_search_results: Maximum number of search results (default: 10)
            max_urls_to_extract: Maximum number of URLs to extract content from (default: 3)
            vectorstore_manager: VectorStoreManager instance for storing
            embedding_model: Embedding model name
            profile_id: Profile ID to store knowledge under
            
        Returns:
            Dictionary with:
                - success: True if successful
                - knowledge: Summarized knowledge
                - vectorstore: Created vectorstore
                - persist_dir: Persistence directory
                - kb_id: Knowledge base ID
                - error: Error message if failed
        """
        try:
            logger.info("=" * 80)
            logger.info(f"🔍 Starting auto-discovery for topic: {topic}")
            logger.info("=" * 80)
            
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
            
            # Step 3: Extract content from top URLs
            logger.info(f"📄 Step 3/6: Extracting content from top {max_urls_to_extract} URLs...")
            urls_to_extract = [result['url'] for result in search_results['organic_results'][:max_urls_to_extract]]
            logger.info(f"   - URLs to extract: {len(urls_to_extract)}")
            for i, url in enumerate(urls_to_extract, 1):
                logger.info(f"      {i}. {url}")
            
            extracted_content = self.extract_content_from_urls(urls_to_extract)
            
            successful_extractions = sum(1 for item in extracted_content if item.get('success'))
            logger.info(f"   ✅ Successfully extracted content from {successful_extractions}/{len(extracted_content)} URLs")
            for item in extracted_content:
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
                        'source_urls': source_urls  # Include source URLs in result
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
                    'source_urls': source_urls
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

