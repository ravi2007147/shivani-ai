"""Article content analyzer using Ollama LLM with structured JSON output."""

import json
import re
import logging
from typing import Dict, Optional, Tuple, List
from langchain_ollama import OllamaLLM

from src.utils.link_extractor import extract_domain
from src.utils.web_scraper import extract_article_from_url
from src.utils.content_filter import extract_main_content, filter_unwanted_content

logger = logging.getLogger(__name__)

def analyze_article_content(
    article_title: str,
    article_description: str,
    article_link: str,
    ollama_model: str = "mistral",
    ollama_base_url: str = "http://localhost:11434",
    domain_pause_check: Optional[callable] = None,
    keywords: Optional[List[str]] = None
) -> Tuple[bool, str, Optional[str], Optional[Dict], Optional[str], Optional[Dict]]:
    """Analyze article content by extracting structured information from the article link.
    
    Args:
        article_title: Article title from feed
        article_description: Article description from feed
        article_link: Article URL to scrape
        ollama_model: Ollama model to use for analysis
        ollama_base_url: Ollama base URL
        domain_pause_check: Optional function(domain) -> bool to check if domain is paused
        
    Returns:
        Tuple of (success, message, extracted_content, structured_data, raw_response, keyword_match)
        - success: True if analysis succeeded
        - message: Status message
        - extracted_content: Raw content extracted from the link
        - structured_data: Parsed JSON with structured fields (topic, summary, tech_entities, etc.)
        - raw_response: Raw LLM response (fallback if JSON parsing fails)
        - keyword_match: Dictionary with keyword match information (matched_keywords, should_write_article, article_recommendation)
    """
    try:
        # Step 1: Check if domain is paused (unless bypassing)
        domain = extract_domain(article_link)
        if domain_pause_check and domain_pause_check(domain):
            # Don't block, just warn - allow manual analysis
            logger.warning(f"Domain {domain} is paused, but proceeding with analysis")
        
        # Step 2: Extract content from article link using web scraper
        logger.info(f"Extracting content from article link: {article_link}")
        raw_content = extract_article_from_url(article_link, timeout=30)
        
        if not raw_content or len(raw_content.strip()) < 100:
            return False, "Extracted content is too short or empty", raw_content, None, None
        
        # Step 2.5: Filter out unwanted content (navigation, social buttons, etc.)
        logger.info("Filtering unwanted content (navigation, social buttons, etc.)")
        content = extract_main_content(raw_content, min_length=100)
        
        # If filtering removed too much, use less aggressive filtering
        if len(content) < 100 and len(raw_content) > 100:
            logger.warning("Aggressive filtering removed too much content, using less aggressive filtering")
            content = filter_unwanted_content(raw_content, aggressive=False)
        
        # Final check
        if not content or len(content.strip()) < 100:
            logger.warning("Content too short after filtering, using raw content")
            content = raw_content
        
        logger.info(f"Content filtered: {len(raw_content)} -> {len(content)} characters")
        
        # Step 3: Prepare combined feed description + scraped content
        # Combine feed description and scraped content for better context
        combined_text = f"""Feed Summary:
Title: {article_title}
Description: {article_description or 'No description available'}

Article Content:
{content}
"""
        
        # Step 4: Create structured prompt for LLM analysis
        # Truncate content if too long (keep first 25000 chars for better context)
        content_preview_length = min(25000, len(content))
        content_preview = content[:content_preview_length]
        content_length = len(content)
        is_truncated = content_length > content_preview_length
        
        prompt = f"""You are an AI news extractor and content analyzer. Your task is to analyze the feed item description and the scraped article content to extract structured information.

Feed Item Reference:
- Title: {article_title}
- Description: {article_description or 'No description available'}
- URL: {article_link}

Scraped Article Content ({content_preview_length} characters shown of {content_length} total):
---
{content_preview}
{'... [Content continues, but truncated for context]' if is_truncated else ''}
---

Instructions:
1. Analyze both the feed description (what people say about the article) and the scraped content (what the article actually says)
2. Focus ONLY on the actual article content - ignore navigation elements, social media buttons, "Read more", "Share", "Subscribe", Reddit subreddit links (r/), user references (u/), and similar UI elements
3. Identify the main topic, key technical points, and why this might be important
4. Extract product names, tools, frameworks, version numbers, dates, and technical entities
5. Determine relevance (0-1 scale) based on how important/interesting this is
6. Categorize the content (e.g., "release", "update", "tutorial", "news", "announcement", "opinion")
7. Suggest an action: "write" (worth writing about), "monitor" (keep an eye on), or "ignore" (not significant)

IMPORTANT - EXCLUDE FROM ANALYSIS:
- Navigation elements: "Menu", "Home", "About", "Contact", "Privacy", "Terms", "Cookie", "Skip to content", etc.
- Social media buttons: "Share", "Follow", "Subscribe", "Join", "Upvote", "Downvote", "Comments", etc.
- Reddit-specific elements: "r/subreddit", "u/username", "Go to", "Members", "Posted by", "[D]", "[R]", "[P]", etc.
- UI elements: "Read more", "Share", "Save", "Click here", "Learn more", "See more", "Show more", etc.
- Footer/Header noise: "Copyright", "All rights reserved", "Powered by", "Built with", etc.
- Advertisement text: "Advertisement", "Sponsored content", "Promoted content", etc.
- Newsletter prompts: "Subscribe to our newsletter", "Sign up for updates", "Get notifications", etc.
- Reddit post metadata: "points", "comments", "hours ago", "days ago", etc.

Focus ONLY on the actual article content, main topic, technical details, and meaningful information. Ignore all UI elements, navigation, and social media buttons.

Return ONLY a valid JSON object with the following structure. Do not include any text before or after the JSON:

{{
  "topic": "Main topic or announcement in one sentence",
  "summary": "2-3 sentence summary of the key points",
  "tech_entities": ["list", "of", "technologies", "tools", "frameworks", "products", "mentioned"],
  "relevance": 0.85,
  "category": "release|update|tutorial|news|announcement|opinion|other",
  "action": "write|monitor|ignore",
  "key_points": [
    "First key technical point or update",
    "Second key point",
    "Third key point"
  ],
  "version_info": "Version numbers, release dates, or beta/RC info if mentioned (or null)",
  "freshness_hint": "Why this is fresh/new (e.g., 'new release', 'beta', 'RC', 'major update') or null"
}}

Important:
- Return ONLY valid JSON, no markdown, no code fences, no explanatory text
- The relevance score should be between 0.0 and 1.0
- If information is not available, use null for optional fields
- Be specific and accurate in your extraction

JSON Response:
"""
        
        # Step 5: Use Ollama LLM to analyze content
        logger.info(f"Analyzing content with Ollama model: {ollama_model}")
        llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.2  # Lower temperature for more focused extraction
        )
        
        raw_response = llm.invoke(prompt)
        
        if not raw_response or len(raw_response.strip()) < 50:
            return False, "LLM analysis returned insufficient content", raw_content, None, raw_response, None
        
        # Step 6: Parse JSON response
        structured_data = _parse_json_response(raw_response)
        
        # Step 7: Check keyword matches and article recommendation
        keyword_match = None
        if keywords and structured_data:
            logger.info(f"Checking keyword matches against {len(keywords)} keywords")
            keyword_match = _check_keyword_matches(
                structured_data=structured_data,
                content=content,
                article_title=article_title,
                article_description=article_description,
                keywords=keywords,
                ollama_model=ollama_model,
                ollama_base_url=ollama_base_url
            )
        
        if structured_data:
            logger.info(f"Successfully parsed structured data: {structured_data.get('topic', 'Unknown topic')}")
            # Return both filtered content and raw content for display
            return True, "Content analyzed successfully", raw_content, structured_data, raw_response, keyword_match
        else:
            # Fallback: return raw response if JSON parsing fails
            logger.warning("Failed to parse JSON response, returning raw response")
            return True, "Content analyzed (JSON parsing failed, showing raw response)", raw_content, None, raw_response, keyword_match
        
    except Exception as e:
        error_msg = f"Error analyzing article content: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg, None, None, None, None


def _parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling markdown code blocks and extra text.
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Parsed JSON dictionary, or None if parsing fails
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
                    # Found complete JSON object
                    json_str = response[start_idx:i+1]
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, dict) and 'topic' in data:
                            return _normalize_structured_data(data)
                    except json.JSONDecodeError:
                        pass
        
        # Strategy 2: Try direct JSON parse
        try:
            data = json.loads(response)
            if isinstance(data, dict) and 'topic' in data:
                return _normalize_structured_data(data)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Try to extract JSON from text using regex (fallback)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and 'topic' in data:
                    return _normalize_structured_data(data)
            except json.JSONDecodeError:
                pass
            
    except Exception as e:
        logger.warning(f"Error parsing JSON response: {str(e)}")
    
    return None


def _normalize_structured_data(data: Dict) -> Dict:
    """Normalize structured data with default values.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Normalized dictionary with all required fields
    """
    return {
        "topic": data.get("topic", "Unknown topic"),
        "summary": data.get("summary", "No summary available"),
        "tech_entities": data.get("tech_entities", []) if isinstance(data.get("tech_entities"), list) else [],
        "relevance": float(data.get("relevance", 0.5)) if data.get("relevance") is not None else 0.5,
        "category": data.get("category", "other"),
        "action": data.get("action", "monitor"),
        "key_points": data.get("key_points", []) if isinstance(data.get("key_points"), list) else [],
        "version_info": data.get("version_info"),
        "freshness_hint": data.get("freshness_hint")
    }


def _check_keyword_matches(
    structured_data: Dict,
    content: str,
    article_title: str,
    article_description: str,
    keywords: List[str],
    ollama_model: str = "mistral",
    ollama_base_url: str = "http://localhost:11434"
) -> Optional[Dict]:
    """Check if article content matches any keywords and if we should write an article.
    
    Args:
        structured_data: Structured analysis data from article
        content: Filtered article content
        article_title: Article title
        article_description: Article description
        keywords: List of keywords to check against
        ollama_model: Ollama model to use
        ollama_base_url: Ollama base URL
        
    Returns:
        Dictionary with keyword match information, or None if check fails
    """
    try:
        # Prepare keywords list
        keywords_str = "\n".join([f"- {kw}" for kw in keywords])
        
        # Prepare content summary
        topic = structured_data.get('topic', '')
        summary = structured_data.get('summary', '')
        tech_entities = structured_data.get('tech_entities', [])
        tech_entities_str = ", ".join(tech_entities) if tech_entities else "None"
        category = structured_data.get('category', 'other')
        relevance = structured_data.get('relevance', 0.5)
        freshness_hint = structured_data.get('freshness_hint', '')
        
        # Create prompt for keyword matching and article recommendation
        prompt = f"""You are a content strategist and SEO expert. Your task is to analyze if an article content matches any of the provided keywords and determine if we should write an article about it.

Article Analysis:
- Topic: {topic}
- Summary: {summary}
- Technical Entities: {tech_entities_str}
- Category: {category}
- Relevance Score: {relevance}
- Freshness Hint: {freshness_hint or 'None'}

Keywords to Match Against:
{keywords_str}

Article Content (first 5000 characters):
---
{content[:5000]}
---

Instructions:
1. Analyze if the article content is related to ANY of the provided keywords
2. Identify which keywords match (if any)
3. Determine if we should write an article about this topic based on:
   - Is it about new things coming to market (products, updates, releases)?
   - Is it about issues or problems that need discussion?
   - Is it about improvements or innovations?
   - Would writing about this early give us good SEO/traffic opportunities?
   - Is the content fresh/new (releases, betas, RCs, major updates)?
4. Provide a recommendation for article creation
5. If should_write_article is true, suggest AT LEAST 5 specific article titles that would be good targets for SEO and content creation

Return ONLY a valid JSON object with the following structure:

{{
  "matched_keywords": ["list", "of", "keywords", "that", "match"],
  "should_write_article": true,
  "article_recommendation": "Detailed recommendation on why we should or should not write an article",
  "article_titles": ["Title 1: Specific article title suggestion", "Title 2: Another specific article title", "Title 3: Another suggestion", "Title 4: Another suggestion", "Title 5: Another suggestion"],
  "seo_potential": "high|medium|low",
  "reasoning": "Why this content matches the keywords and why we should/shouldn't write about it"
}}

Important:
- Return ONLY valid JSON, no markdown, no code fences, no explanatory text
- If no keywords match, return empty array for matched_keywords
- **REQUIRED**: If should_write_article is true, article_titles MUST contain AT LEAST 5 specific, actionable article title suggestions (preferably more than 5)
- If should_write_article is false, article_titles can be empty array or contain fewer suggestions
- Article titles should be:
  * Specific and actionable (not generic)
  * SEO-friendly and optimized for search queries
  * Relevant to the content, matched keywords, and target audience
  * Compelling and clear about what the article will cover
  * Targeting specific search intent (how-to, comparison, guide, tutorial, etc.)
- Be specific about SEO potential and article recommendation
- Focus on early content opportunities (new products, updates, issues, improvements)
- Article titles should target specific search queries that people are likely to search for

JSON Response:
"""
        
        # Use Ollama LLM to check keyword matches
        logger.info(f"Checking keyword matches with Ollama model: {ollama_model}")
        llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.3  # Moderate temperature for balanced analysis
        )
        
        response = llm.invoke(prompt)
        
        if not response or len(response.strip()) < 50:
            logger.warning("Keyword match check returned insufficient content")
            return None
        
        # Parse JSON response
        keyword_match_data = _parse_keyword_match_response(response)
        
        if keyword_match_data:
            logger.info(f"Keyword match check completed: {len(keyword_match_data.get('matched_keywords', []))} keywords matched")
            return keyword_match_data
        else:
            logger.warning("Failed to parse keyword match response")
            return None
            
    except Exception as e:
        logger.error(f"Error checking keyword matches: {str(e)}", exc_info=True)
        return None


def _parse_keyword_match_response(response: str) -> Optional[Dict]:
    """Parse keyword match JSON response from LLM.
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Parsed keyword match dictionary, or None if parsing fails
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
                    # Found complete JSON object
                    json_str = response[start_idx:i+1]
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, dict):
                            return _normalize_keyword_match_data(data)
                    except json.JSONDecodeError:
                        pass
        
        # Strategy 2: Try direct JSON parse
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return _normalize_keyword_match_data(data)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Try to extract JSON from text using regex (fallback)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    return _normalize_keyword_match_data(data)
            except json.JSONDecodeError:
                pass
                
    except Exception as e:
        logger.warning(f"Error parsing keyword match response: {str(e)}")
    
    return None


def _normalize_keyword_match_data(data: Dict) -> Dict:
    """Normalize keyword match data with default values.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Normalized dictionary with all required fields
    """
    article_titles = data.get("article_titles", [])
    if not isinstance(article_titles, list):
        article_titles = []
    
    should_write = bool(data.get("should_write_article", False))
    
    return {
        "matched_keywords": data.get("matched_keywords", []) if isinstance(data.get("matched_keywords"), list) else [],
        "should_write_article": should_write,
        "article_recommendation": data.get("article_recommendation", "No recommendation provided"),
        "article_titles": article_titles,  # Can be empty or have fewer than 5 - UI will handle display
        "seo_potential": data.get("seo_potential", "low"),
        "reasoning": data.get("reasoning", "No reasoning provided")
    }

