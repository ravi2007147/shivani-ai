"""Google Search Evaluator for Article Titles using Playwright."""

import logging
import time
import json
from typing import List, Dict, Optional, Tuple
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from langchain_ollama import OllamaLLM
import re

logger = logging.getLogger(__name__)


class SearchEvaluator:
    """Evaluates article titles by checking Google search results for competition."""
    
    def __init__(self, headless: bool = True):
        """Initialize the search evaluator.
        
        Args:
            headless: Whether to run browser in headless mode
        """
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._browser_initialized = False
    
    def _initialize_browser(self):
        """Initialize Playwright browser instance (only once)."""
        if self._browser_initialized:
            return
        
        try:
            logger.info("Initializing Playwright browser...")
            
            # Check if playwright is installed
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
            # Clean up on error
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
    
    def search_google(self, query: str, max_results: int = 10) -> Dict:
        """Search Google for a query and extract results.
        
        Args:
            query: Search query (article title)
            max_results: Maximum number of results to extract
            
        Returns:
            Dictionary with search results:
            {
                "query": query,
                "total_results": estimated_total,
                "organic_results": [
                    {
                        "title": "Result title",
                        "url": "Result URL",
                        "snippet": "Result snippet/description"
                    },
                    ...
                ],
                "success": True/False,
                "error": error_message if failed
            }
        """
        if not self._browser_initialized:
            self._initialize_browser()
        
        try:
            # Construct Google search URL with proper encoding
            from urllib.parse import quote_plus
            encoded_query = quote_plus(query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            logger.info(f"Searching Google for: {query}")
            
            # Navigate to Google search
            self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait for search results to load
            time.sleep(3)  # Give JavaScript time to render
            
            # Wait for search results container to appear
            try:
                self.page.wait_for_selector('div#search, div[data-ved], div.g', timeout=10000)
            except PlaywrightTimeoutError:
                logger.warning("Search results container not found, proceeding anyway")
            
            # Extract search results
            results = {
                "query": query,
                "total_results": 0,
                "organic_results": [],
                "success": True,
                "error": None
            }
            
            # Try to extract total results count
            try:
                # Look for "About X results" text
                results_text = self.page.locator('text=/About [0-9,]+ results/').first
                if results_text.count() > 0:
                    results_str = results_text.inner_text()
                    # Extract number from "About 1,234,567 results"
                    numbers = re.findall(r'[\d,]+', results_str.replace(',', ''))
                    if numbers:
                        results["total_results"] = int(numbers[0])
            except Exception as e:
                logger.warning(f"Could not extract total results count: {str(e)}")
            
            # Extract organic search results
            try:
                # Try multiple selectors for Google search results
                # Modern Google uses various structures: div.g, div[data-ved], div.MjjYud, etc.
                selectors = [
                    'div.g:has(h3)',
                    'div[data-ved]:has(h3)',
                    'div.MjjYud:has(h3)',
                    'div.tF2Cxc:has(h3)',
                    'div.g'
                ]
                
                result_elements = []
                for selector in selectors:
                    try:
                        elements = self.page.locator(selector).all()
                        if elements:
                            result_elements = elements
                            logger.info(f"Found {len(elements)} results using selector: {selector}")
                            break
                    except Exception:
                        continue
                
                if not result_elements:
                    # Fallback: try to get all divs with h3 (likely search results)
                    result_elements = self.page.locator('div:has(h3)').all()
                
                for i, element in enumerate(result_elements[:max_results]):
                    try:
                        # Extract title from h3
                        title = ""
                        title_selectors = ['h3', 'h3 a', 'h3.LC20lb', 'h3.DKV0Md']
                        for title_sel in title_selectors:
                            try:
                                title_elem = element.locator(title_sel).first
                                if title_elem.count() > 0:
                                    title = title_elem.inner_text().strip()
                                    if title:
                                        break
                            except Exception:
                                continue
                        
                        # Extract URL from link
                        url = ""
                        url_selectors = ['a[href^="http"]', 'a[href*="/url"]', 'a', 'h3 a']
                        for url_sel in url_selectors:
                            try:
                                url_elem = element.locator(url_sel).first
                                if url_elem.count() > 0:
                                    href = url_elem.get_attribute('href')
                                    if href:
                                        # Clean up Google redirect URLs
                                        if '/url?q=' in href or '/url?url=' in href:
                                            import urllib.parse
                                            try:
                                                parsed = urllib.parse.urlparse(href)
                                                query_params = urllib.parse.parse_qs(parsed.query)
                                                # Try 'q' first, then 'url'
                                                if 'q' in query_params:
                                                    url = query_params['q'][0]
                                                elif 'url' in query_params:
                                                    url = query_params['url'][0]
                                                else:
                                                    url = href
                                            except Exception:
                                                url = href
                                        elif href.startswith('http') and 'google.com' not in href.lower():
                                            url = href
                                        
                                        # Validate URL
                                        if url and url.startswith('http') and 'google.com' not in url.lower():
                                            break
                            except Exception:
                                continue
                        
                        # Extract snippet
                        snippet = ""
                        snippet_selectors = [
                            'span[data-sncf="1"]',
                            'div[data-sncf="1"]',
                            'span:not(:has(span))',
                            'div[style*="line-height"] span',
                            '.VwiC3b'
                        ]
                        for snippet_sel in snippet_selectors:
                            try:
                                snippet_elem = element.locator(snippet_sel).first
                                if snippet_elem.count() > 0:
                                    snippet = snippet_elem.inner_text().strip()
                                    if snippet and len(snippet) > 20:  # Ensure meaningful snippet
                                        break
                            except Exception:
                                continue
                        
                        # Only add if we have at least title and URL
                        if title and url:
                            # Skip Google's own pages
                            if 'google.com' not in url.lower():
                                results["organic_results"].append({
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet[:500] if snippet else ""  # Limit snippet length
                                })
                    except Exception as e:
                        logger.warning(f"Error extracting result {i}: {str(e)}")
                        continue
                
                logger.info(f"Extracted {len(results['organic_results'])} search results for: {query}")
                
                # If we got no results, try alternative method
                if len(results["organic_results"]) == 0:
                    logger.warning("No results extracted with primary method, trying fallback")
                    # Fallback: extract from page text and links
                    try:
                        # Get all links that look like search results
                        all_links = self.page.locator('a[href*="http"]').all()
                        seen_urls = set()
                        for link in all_links:
                            try:
                                href = link.get_attribute('href')
                                if not href:
                                    continue
                                
                                # Clean Google redirect URLs
                                if '/url?q=' in href or '/url?url=' in href:
                                    import urllib.parse
                                    try:
                                        parsed = urllib.parse.urlparse(href)
                                        query_params = urllib.parse.parse_qs(parsed.query)
                                        # Try 'q' first, then 'url'
                                        if 'q' in query_params:
                                            href = query_params['q'][0]
                                        elif 'url' in query_params:
                                            href = query_params['url'][0]
                                    except Exception:
                                        pass  # Keep original href if parsing fails
                                
                                # Skip Google, YouTube, and already seen URLs
                                if 'google.com' in href.lower() or 'youtube.com' in href.lower():
                                    continue
                                if href in seen_urls:
                                    continue
                                seen_urls.add(href)
                                
                                # Get link text as title
                                title_text = link.inner_text().strip()
                                if title_text and len(title_text) > 10 and len(title_text) < 200:
                                    results["organic_results"].append({
                                        "title": title_text,
                                        "url": href,
                                        "snippet": ""
                                    })
                                    if len(results["organic_results"]) >= max_results:
                                        break
                            except Exception:
                                continue
                        
                        if results["organic_results"]:
                            logger.info(f"Fallback method extracted {len(results['organic_results'])} results")
                    except Exception as e2:
                        logger.error(f"Fallback extraction also failed: {str(e2)}")
                
            except Exception as e:
                logger.error(f"Error extracting organic results: {str(e)}", exc_info=True)
            
            return results
            
        except PlaywrightTimeoutError as e:
            error_msg = f"Timeout searching Google for '{query}': {str(e)}"
            logger.error(error_msg)
            return {
                "query": query,
                "total_results": 0,
                "organic_results": [],
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Error searching Google for '{query}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "query": query,
                "total_results": 0,
                "organic_results": [],
                "success": False,
                "error": error_msg
            }
    
    def evaluate_search_results(
        self,
        article_title: str,
        search_results: Dict,
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434"
    ) -> Dict:
        """Evaluate search results to determine if article title is a good contender.
        
        Args:
            article_title: The article title to evaluate
            search_results: Search results dictionary from search_google
            ollama_model: Ollama model to use for evaluation
            ollama_base_url: Ollama base URL
            
        Returns:
            Dictionary with evaluation results:
            {
                "article_title": article_title,
                "is_good_contender": True/False,
                "competition_level": "low|medium|high",
                "relevant_results_count": number,
                "reasoning": "Explanation",
                "recommendation": "write|discard|monitor"
            }
        """
        try:
            if not search_results.get("success"):
                return {
                    "article_title": article_title,
                    "is_good_contender": False,
                    "competition_level": "unknown",
                    "relevant_results_count": 0,
                    "reasoning": f"Search failed: {search_results.get('error', 'Unknown error')}",
                    "recommendation": "discard"
                }
            
            organic_results = search_results.get("organic_results", [])
            total_results = search_results.get("total_results", 0)
            
            # Prepare search results text for LLM
            results_text = f"Total search results: {total_results}\n\n"
            results_text += f"Top {len(organic_results)} organic results:\n\n"
            
            for i, result in enumerate(organic_results, 1):
                results_text += f"{i}. {result.get('title', '')}\n"
                results_text += f"   URL: {result.get('url', '')}\n"
                if result.get('snippet'):
                    results_text += f"   Snippet: {result.get('snippet', '')}\n"
                results_text += "\n"
            
            # Create prompt for LLM evaluation
            prompt = f"""You are an SEO expert and content strategist. Your task is to evaluate if an article title is a good contender for content creation based on Google search results.

Article Title to Evaluate: "{article_title}"

Google Search Results:
{results_text}

Instructions:
1. Analyze the search results to determine how many relevant, high-quality results exist for this article title
2. Assess the competition level:
   - LOW: Few relevant results (0-3), opportunity to rank easily
   - MEDIUM: Some relevant results (4-7), moderate competition
   - HIGH: Many relevant results (8+), high competition, saturated market
3. Determine if this is a good contender:
   - GOOD CONTENDER: Low to medium competition, opportunity to create unique content
   - NOT GOOD: High competition with many established, authoritative sources
4. Provide a recommendation: "write", "discard", or "monitor"

Return ONLY a valid JSON object with the following structure:

{{
  "is_good_contender": true,
  "competition_level": "low|medium|high",
  "relevant_results_count": 5,
  "reasoning": "Explanation of why this is or isn't a good contender based on search results",
  "recommendation": "write|discard|monitor"
}}

Important:
- Return ONLY valid JSON, no markdown, no code fences, no explanatory text
- Focus on the QUALITY and RELEVANCE of existing results, not just quantity
- If many high-quality, authoritative sources exist, competition is HIGH and should be discarded
- If few or low-quality results exist, it's a GOOD CONTENDER for content creation
- Consider if the existing results directly answer the search query or if there's room for unique content

JSON Response:
"""
            
            # Use Ollama LLM to evaluate
            logger.info(f"Evaluating article title with LLM: {article_title}")
            llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0.3
            )
            
            response = llm.invoke(prompt)
            
            if not response or len(response.strip()) < 50:
                logger.warning("LLM evaluation returned insufficient content")
                return {
                    "article_title": article_title,
                    "is_good_contender": len(organic_results) < 5,  # Fallback: good if fewer than 5 results
                    "competition_level": "high" if len(organic_results) >= 8 else "medium" if len(organic_results) >= 4 else "low",
                    "relevant_results_count": len(organic_results),
                    "reasoning": "LLM evaluation failed, using fallback logic based on result count",
                    "recommendation": "discard" if len(organic_results) >= 8 else "monitor"
                }
            
            # Parse JSON response
            evaluation = self._parse_evaluation_response(response)
            
            if evaluation:
                evaluation["article_title"] = article_title
                evaluation["relevant_results_count"] = len(organic_results)
                return evaluation
            else:
                # Fallback evaluation
                logger.warning("Failed to parse LLM evaluation response, using fallback")
                return {
                    "article_title": article_title,
                    "is_good_contender": len(organic_results) < 5,
                    "competition_level": "high" if len(organic_results) >= 8 else "medium" if len(organic_results) >= 4 else "low",
                    "relevant_results_count": len(organic_results),
                    "reasoning": "Failed to parse LLM response, using fallback evaluation",
                    "recommendation": "discard" if len(organic_results) >= 8 else "write" if len(organic_results) < 4 else "monitor"
                }
                
        except Exception as e:
            error_msg = f"Error evaluating search results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "article_title": article_title,
                "is_good_contender": False,
                "competition_level": "unknown",
                "relevant_results_count": len(search_results.get("organic_results", [])),
                "reasoning": error_msg,
                "recommendation": "discard"
            }
    
    def _parse_evaluation_response(self, response: str) -> Optional[Dict]:
        """Parse evaluation JSON response from LLM.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed evaluation dictionary, or None if parsing fails
        """
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
            
            # Strategy 1: Try to find JSON object using balanced braces
            brace_count = 0
            start_idx = -1
            
            for i, char in enumerate(response):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        json_str = response[start_idx:i+1]
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, dict):
                                return self._normalize_evaluation_data(data)
                        except json.JSONDecodeError:
                            pass
            
            # Strategy 2: Try direct JSON parse
            try:
                data = json.loads(response)
                if isinstance(data, dict):
                    return self._normalize_evaluation_data(data)
            except json.JSONDecodeError:
                pass
            
            # Strategy 3: Try to extract JSON from text using regex
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        return self._normalize_evaluation_data(data)
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            logger.warning(f"Error parsing evaluation response: {str(e)}")
        
        return None
    
    def _normalize_evaluation_data(self, data: Dict) -> Dict:
        """Normalize evaluation data with default values.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Normalized dictionary with all required fields
        """
        return {
            "is_good_contender": bool(data.get("is_good_contender", False)),
            "competition_level": data.get("competition_level", "unknown").lower(),
            "reasoning": data.get("reasoning", "No reasoning provided"),
            "recommendation": data.get("recommendation", "monitor").lower()
        }
    
    def evaluate_article_titles(
        self,
        article_titles: List[str],
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434",
        delay_between_searches: float = 2.0
    ) -> List[Dict]:
        """Evaluate multiple article titles by searching Google and analyzing results.
        
        Args:
            article_titles: List of article titles to evaluate
            ollama_model: Ollama model to use for evaluation
            ollama_base_url: Ollama base URL
            delay_between_searches: Delay in seconds between searches (to avoid rate limiting)
            
        Returns:
            List of evaluation dictionaries, one for each article title
        """
        if not article_titles:
            return []
        
        # Initialize browser once
        self._initialize_browser()
        
        evaluations = []
        
        try:
            for i, title in enumerate(article_titles):
                logger.info(f"Evaluating article title {i+1}/{len(article_titles)}: {title}")
                
                # Search Google
                search_results = self.search_google(title, max_results=10)
                
                # Evaluate search results with LLM
                evaluation = self.evaluate_search_results(
                    title,
                    search_results,
                    ollama_model,
                    ollama_base_url
                )
                
                # Add search results to evaluation
                evaluation["search_results"] = search_results
                
                evaluations.append(evaluation)
                
                # Delay between searches to avoid rate limiting
                if i < len(article_titles) - 1:
                    time.sleep(delay_between_searches)
            
        finally:
            # Close browser when done
            self._close_browser()
        
        return evaluations
    
    def __enter__(self):
        """Context manager entry."""
        self._initialize_browser()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._close_browser()


def evaluate_article_titles(
    article_titles: List[str],
    ollama_model: str = "mistral",
    ollama_base_url: str = "http://localhost:11434",
    headless: bool = True,
    delay_between_searches: float = 2.0
) -> List[Dict]:
    """Convenience function to evaluate article titles.
    
    Args:
        article_titles: List of article titles to evaluate
        ollama_model: Ollama model to use for evaluation
        ollama_base_url: Ollama base URL
        headless: Whether to run browser in headless mode
        delay_between_searches: Delay in seconds between searches
        
    Returns:
        List of evaluation dictionaries
    """
    evaluator = SearchEvaluator(headless=headless)
    return evaluator.evaluate_article_titles(
        article_titles,
        ollama_model,
        ollama_base_url,
        delay_between_searches
    )
