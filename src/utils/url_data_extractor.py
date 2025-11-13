"""URL data extraction and structuring using Playwright and Ollama."""

import json
import logging
import re
from typing import Optional, Dict, Tuple
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


class URLDataExtractor:
    """Extract and structure data from URLs using Playwright and Ollama."""
    
    def __init__(self, headless: bool = True, ollama_model: str = "mistral", ollama_base_url: str = "http://localhost:11434"):
        """Initialize the URL data extractor.
        
        Args:
            headless: Whether to run browser in headless mode
            ollama_model: Ollama model to use for structuring data
            ollama_base_url: Ollama base URL
        """
        self.headless = headless
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
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
    
    def extract_content(self, url: str, timeout: int = 30000, wait_for_selector: Optional[str] = None, wait_time: int = 2000) -> Tuple[bool, str, Optional[str]]:
        """Extract content from a URL using Playwright.
        
        Args:
            url: URL to scrape
            timeout: Page load timeout in milliseconds (default: 30000)
            wait_for_selector: Optional CSS selector to wait for before extracting content
            wait_time: Additional wait time in milliseconds after page load (default: 2000)
            
        Returns:
            Tuple of (success, content, error_message)
        """
        try:
            self._initialize_browser()
            
            logger.info(f"Navigating to URL: {url}")
            
            # Navigate to the URL
            try:
                self.page.goto(url, wait_until='networkidle', timeout=timeout)
            except PlaywrightTimeoutError:
                # Still try to get content even if networkidle times out
                logger.warning(f"Network idle timeout for {url}, proceeding with current page state")
            except Exception as e:
                return False, "", f"Failed to navigate to URL: {str(e)}"
            
            # Wait for specific selector if provided
            if wait_for_selector:
                try:
                    logger.info(f"Waiting for selector: {wait_for_selector}")
                    self.page.wait_for_selector(wait_for_selector, timeout=10000)
                except PlaywrightTimeoutError:
                    logger.warning(f"Selector {wait_for_selector} not found, proceeding anyway")
            
            # Additional wait time for JavaScript to finish loading
            if wait_time > 0:
                self.page.wait_for_timeout(wait_time)
            
            # Wait for page to be fully loaded
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=5000)
            except PlaywrightTimeoutError:
                pass  # Continue anyway
            
            # Extract text content
            try:
                # Try to get main content first
                content = None
                
                # Strategy 1: Look for semantic HTML5 tags
                for selector in ['article', 'main', '[role="main"]', 'main article']:
                    try:
                        element = self.page.query_selector(selector)
                        if element:
                            content = element.inner_text()
                            if content and len(content.strip()) > 100:
                                logger.info(f"Found content using selector: {selector}")
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
                                    logger.info(f"Found content using selector: {selector}")
                                    break
                        except Exception:
                            continue
                
                # Strategy 3: Get body text if nothing else works
                if not content or len(content.strip()) < 100:
                    body = self.page.query_selector('body')
                    if body:
                        content = body.inner_text()
                        logger.info("Using body text as content")
                
                if not content or len(content.strip()) < 50:
                    return False, "", "No meaningful content extracted from the page"
                
                # Clean the content
                content = self._clean_content(content)
                
                return True, content, None
                
            except Exception as e:
                return False, "", f"Failed to extract content: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}", exc_info=True)
            return False, "", f"Error: {str(e)}"
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content.
        
        Args:
            content: Raw content text
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        # Remove excessive whitespace
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                lines.append(line)
        
        # Join lines and normalize whitespace
        content = '\n'.join(lines)
        content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with single
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Replace multiple newlines
        
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
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        return content.strip()
    
    def structure_data(self, content: str, json_template: Optional[str] = None, url: Optional[str] = None) -> Tuple[bool, str, Optional[Dict], Optional[str]]:
        """Structure extracted content using Ollama LLM.
        
        Args:
            content: Extracted content from URL
            json_template: Optional JSON template/schema to structure the data
            url: Optional URL for metadata (default: None)
            
        Returns:
            Tuple of (success, structured_json_string, structured_dict, error_message)
        """
        try:
            logger.info(f"Structuring data with Ollama model: {self.ollama_model}")
            
            # Initialize Ollama LLM
            llm = OllamaLLM(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=0.2  # Lower temperature for more consistent structured output
            )
            
            # Truncate content if too long (keep first 30000 chars for better context)
            content_preview_length = min(30000, len(content))
            content_preview = content[:content_preview_length]
            content_length = len(content)
            is_truncated = content_length > content_preview_length
            
            # Create prompt based on whether template is provided
            if json_template and json_template.strip():
                # Validate JSON template
                try:
                    template_dict = json.loads(json_template)
                    template_description = json.dumps(template_dict, indent=2)
                except json.JSONDecodeError:
                    return False, "", None, "Invalid JSON template. Please provide a valid JSON structure."
                
                prompt = f"""You are an AI data extraction system. Extract structured information from the provided web page content and format it according to the specified JSON template.

JSON Template (target structure):
```json
{template_description}
```

Web Page Content ({content_preview_length} characters shown of {content_length} total):
---
{content_preview}
{'... [Content continues, but truncated for context]' if is_truncated else ''}
---

Instructions:
1. Extract relevant information from the web page content
2. Structure the data according to the provided JSON template
3. Fill in as many fields as possible from the available content
4. If a field cannot be determined from the content, use null
5. Maintain the exact structure and field names from the template
6. Return ONLY valid JSON, no markdown, no code blocks, no explanations

Important:
- Return ONLY the JSON object matching the template structure
- Do not include any text before or after the JSON
- Use null for missing values
- Preserve the exact field names from the template

JSON Response:
"""
            else:
                # No template provided - extract common structured information
                prompt = f"""You are an AI data extraction system. Extract structured information from the provided web page content.

Web Page Content ({content_preview_length} characters shown of {content_length} total):
---
{content_preview}
{'... [Content continues, but truncated for context]' if is_truncated else ''}
---

Instructions:
1. Analyze the web page content
2. Extract key information in a structured format
3. Include: title, description, main_content, key_points (array), metadata (object), and any other relevant structured data
4. Return the data as a valid JSON object

Return ONLY valid JSON with this structure:
{{
  "title": "Page title or main heading",
  "description": "Brief description or summary",
  "main_content": "Main text content",
  "key_points": ["point1", "point2", "point3"],
  "metadata": {{
    "url": "{url or ''}",
    "extracted_at": "timestamp if available",
    "content_length": {content_length}
  }},
  "additional_fields": {{}}
}}

Important:
- Return ONLY the JSON object, no markdown, no code blocks, no explanations
- Use null for missing values
- Be specific and accurate in your extraction

JSON Response:
"""
            
            # Get structured response from Ollama
            logger.info("Sending prompt to Ollama...")
            raw_response = llm.invoke(prompt)
            
            if not raw_response or len(raw_response.strip()) < 10:
                return False, "", None, "LLM returned empty or insufficient response"
            
            # Parse JSON from response
            structured_dict = self._parse_json_response(raw_response)
            
            if structured_dict:
                structured_json = json.dumps(structured_dict, indent=2)
                return True, structured_json, structured_dict, None
            else:
                # Return raw response if JSON parsing fails
                return True, raw_response, None, "Could not parse JSON from LLM response, returning raw text"
                
        except Exception as e:
            logger.error(f"Error structuring data: {str(e)}", exc_info=True)
            return False, "", None, f"Error structuring data: {str(e)}"
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Remove markdown code blocks if present
            cleaned_response = response.strip()
            
            # Remove ```json and ``` markers
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Try to parse JSON
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Try to find JSON object in the response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
                
                # If that fails, return None
                return None
                
        except Exception as e:
            logger.error(f"Error parsing JSON response: {str(e)}", exc_info=True)
            return None
    
    def extract_and_structure(
        self,
        url: str,
        json_template: Optional[str] = None,
        timeout: int = 30000,
        wait_for_selector: Optional[str] = None,
        wait_time: int = 2000
    ) -> Tuple[bool, str, Optional[str], Optional[Dict], Optional[str]]:
        """Extract content from URL and structure it using Ollama.
        
        Args:
            url: URL to scrape
            json_template: Optional JSON template/schema to structure the data
            timeout: Page load timeout in milliseconds
            wait_for_selector: Optional CSS selector to wait for before extracting content
            wait_time: Additional wait time in milliseconds after page load
            
        Returns:
            Tuple of (success, content, structured_json_string, structured_dict, error_message)
        """
        try:
            # Extract content
            success, content, error_msg = self.extract_content(url, timeout, wait_for_selector, wait_time)
            
            if not success:
                return False, "", None, None, error_msg
            
            # Structure data
            struct_success, structured_json, structured_dict, struct_error = self.structure_data(content, json_template, url)
            
            if not struct_success:
                # Return content even if structuring fails
                return True, content, None, None, struct_error
            
            return True, content, structured_json, structured_dict, None
            
        except Exception as e:
            logger.error(f"Error in extract_and_structure: {str(e)}", exc_info=True)
            return False, "", None, None, f"Error: {str(e)}"
        finally:
            # Clean up browser
            self._close_browser()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup browser."""
        self._close_browser()


def extract_structured_data_from_url(
    url: str,
    json_template: Optional[str] = None,
    ollama_model: str = "mistral",
    ollama_base_url: str = "http://localhost:11434",
    timeout: int = 30000,
    wait_for_selector: Optional[str] = None,
    wait_time: int = 2000,
    headless: bool = True
) -> Tuple[bool, str, Optional[str], Optional[Dict], Optional[str]]:
    """Convenience function to extract and structure data from a URL.
    
    Args:
        url: URL to scrape
        json_template: Optional JSON template/schema to structure the data
        ollama_model: Ollama model to use for structuring
        ollama_base_url: Ollama base URL
        timeout: Page load timeout in milliseconds
        wait_for_selector: Optional CSS selector to wait for before extracting content
        wait_time: Additional wait time in milliseconds after page load
        headless: Whether to run browser in headless mode
        
    Returns:
        Tuple of (success, content, structured_json_string, structured_dict, error_message)
    """
    extractor = URLDataExtractor(
        headless=headless,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url
    )
    
    try:
        return extractor.extract_and_structure(
            url=url,
            json_template=json_template,
            timeout=timeout,
            wait_for_selector=wait_for_selector,
            wait_time=wait_time
        )
    finally:
        extractor._close_browser()

