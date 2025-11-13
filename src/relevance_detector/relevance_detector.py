"""Relevance Detector - Multi-factor content relevance scoring system.

This module provides a comprehensive relevance detection system that uses
multiple scoring factors to determine how relevant a webpage is to a search term.
"""

import re
import logging
from typing import Dict, Optional, Tuple, List
from urllib.parse import urlparse
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

# Try to import numpy and sklearn, use fallback if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy not available, using fallback for similarity calculation")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available, using fallback for similarity calculation")


class RelevanceDetector:
    """
    Multi-factor relevance detection system for evaluating webpage content.
    
    Uses a weighted scoring system with multiple factors:
    1. Title & URL Exactness (45%)
    2. Semantic Embedding Similarity (70%)
    3. Keyword Density (10%)
    4. Domain Trust/Authority (15%)
    
    Final score: weighted sum of all factors (0.0 to 1.0)
    """
    
    # Weight configuration
    WEIGHT_URL_TITLE = 0.45
    WEIGHT_SEMANTIC_SIMILARITY = 0.70
    WEIGHT_KEYWORD_DENSITY = 0.10
    WEIGHT_DOMAIN_AUTHORITY = 0.15
    
    # Domain authority scores
    DOMAIN_AUTHORITY_SCORES = {
        # Official domains
        '.com': 0.20,
        '.co': 0.18,
        '.io': 0.18,
        '.org': 0.16,
        '.net': 0.15,
        '.edu': 0.20,
        '.gov': 0.25,
        '.in': 0.15,
        # Trusted platforms
        'linkedin.com': 0.15,
        'crunchbase.com': 0.10,
        'wikipedia.org': 0.12,
        'github.com': 0.12,
        'medium.com': 0.08,
        'reddit.com': 0.05,
        # Negative scores for untrusted sources
        'blogspot.com': -0.03,
        'wordpress.com': -0.03,
        'tumblr.com': -0.05,
        't.co': -0.05,
    }
    
    # Negative keywords that reduce relevance
    NEGATIVE_KEYWORDS = [
        'scam', 'fraud', 'fake', 'spam', 'malware', 'virus',
        'complaint', 'lawsuit', 'warning', 'beware'
    ]
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """Initialize the Relevance Detector.
        
        Args:
            embedding_model: Name of the embedding model to use
            ollama_base_url: Base URL for Ollama API
        """
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.embeddings = None
        self._embeddings_initialized = False
    
    def _get_embeddings(self) -> OllamaEmbeddings:
        """Get or initialize embeddings model.
        
        Returns:
            OllamaEmbeddings instance
        """
        if not self._embeddings_initialized:
            try:
                self.embeddings = OllamaEmbeddings(
                    model=self.embedding_model,
                    base_url=self.ollama_base_url
                )
                # Test the connection
                self.embeddings.embed_query("test")
                self._embeddings_initialized = True
                logger.info(f"Embeddings model initialized: {self.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {str(e)}")
                raise
        return self.embeddings
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _cosine_similarity_manual(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity manually (fallback if sklearn not available).
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if len(vec1) != len(vec2):
            logger.warning("Vector dimensions don't match, returning 0.0")
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return similarity
    
    def _calculate_url_title_score(self, search_term: str, url: str, title: str = "") -> float:
        """Calculate URL and Title exactness score.
        
        Args:
            search_term: The search term (e.g., "PriorCoder")
            url: URL of the webpage
            title: Title of the webpage (optional)
            
        Returns:
            Score between 0.0 and 1.0
        """
        search_term_lower = self._normalize_text(search_term)
        url_lower = self._normalize_text(url)
        title_lower = self._normalize_text(title)
        
        score = 0.0
        
        # Extract domain from URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove 'www.' if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check if domain contains search term
            if search_term_lower in domain:
                score += 0.45
                logger.debug(f"   URL contains search term: +0.45")
            
            # Check if URL path contains search term
            path = parsed.path.lower()
            if search_term_lower in path:
                score += 0.15
                logger.debug(f"   URL path contains search term: +0.15")
        except Exception as e:
            logger.warning(f"Error parsing URL: {str(e)}")
        
        # Check if title contains search term
        if title and search_term_lower in title_lower:
            score += 0.25
            logger.debug(f"   Title contains search term: +0.25")
        
        # Check for partial matches (word boundaries)
        search_words = search_term_lower.split()
        if len(search_words) > 0:
            # Check if all words appear in domain or title
            domain_text = domain + " " + title_lower
            matches = sum(1 for word in search_words if word in domain_text)
            if matches == len(search_words):
                score += 0.10
                logger.debug(f"   All words match in domain/title: +0.10")
            elif matches > 0:
                partial_score = (matches / len(search_words)) * 0.10
                score += partial_score
                logger.debug(f"   Partial word matches: +{partial_score:.3f}")
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _calculate_semantic_similarity_score(
        self,
        search_term: str,
        content: str
    ) -> float:
        """Calculate semantic embedding similarity score.
        
        Args:
            search_term: The search term
            content: Content from the webpage
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            embeddings = self._get_embeddings()
            
            # Truncate content if too long (keep first 5000 chars for embedding)
            content_preview = content[:5000] if len(content) > 5000 else content
            
            # Compute embeddings
            query_embedding = embeddings.embed_query(search_term)
            content_embedding = embeddings.embed_query(content_preview)
            
            # Calculate cosine similarity
            if HAS_NUMPY and HAS_SKLEARN:
                # Use sklearn for efficient cosine similarity
                query_vec = np.array(query_embedding).reshape(1, -1)
                content_vec = np.array(content_embedding).reshape(1, -1)
                similarity = cosine_similarity(query_vec, content_vec)[0][0]
            else:
                # Fallback: manual cosine similarity calculation
                similarity = self._cosine_similarity_manual(query_embedding, content_embedding)
            
            # Normalize to 0-1 range (cosine similarity is already in -1 to 1, but typically 0 to 1 for normalized vectors)
            similarity = max(0.0, min(1.0, similarity))
            
            logger.debug(f"   Semantic similarity: {similarity:.3f}")
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            # Return 0.5 as a neutral score if embedding fails
            return 0.5
    
    def _calculate_keyword_density_score(self, search_term: str, content: str) -> float:
        """Calculate keyword density score.
        
        Args:
            search_term: The search term
            content: Content from the webpage
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not content or len(content.strip()) < 10:
            return 0.0
        
        search_term_lower = self._normalize_text(search_term)
        content_lower = self._normalize_text(content)
        
        # Count occurrences of search term in content
        matches = content_lower.count(search_term_lower)
        
        # Count total words
        words = re.findall(r'\b\w+\b', content_lower)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Calculate density
        density = matches / total_words if total_words > 0 else 0.0
        
        # Normalize density (typical keyword density is 0-5%, so normalize accordingly)
        # Use logarithmic normalization for better distribution
        normalized_density = min(1.0, density * 20)  # Scale: 5% density = 1.0
        
        # Also check for search term words individually
        search_words = search_term_lower.split()
        if len(search_words) > 1:
            word_matches = sum(1 for word in search_words if word in content_lower)
            word_score = word_matches / len(search_words)
            # Combine density and word match scores
            normalized_density = (normalized_density + word_score) / 2
        
        logger.debug(f"   Keyword density: {density:.4f}, normalized: {normalized_density:.3f}")
        
        return normalized_density
    
    def _calculate_domain_authority_score(self, url: str) -> float:
        """Calculate domain trust/authority score.
        
        Args:
            url: URL of the webpage
            
        Returns:
            Score between -0.1 and 0.3
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove 'www.' if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            score = 0.0
            
            # Check for exact domain matches
            if domain in self.DOMAIN_AUTHORITY_SCORES:
                score = self.DOMAIN_AUTHORITY_SCORES[domain]
                logger.debug(f"   Domain authority (exact match): {score:.3f}")
                return score
            
            # Check for TLD matches
            for tld, tld_score in self.DOMAIN_AUTHORITY_SCORES.items():
                if domain.endswith(tld):
                    score = tld_score
                    logger.debug(f"   Domain authority (TLD match): {score:.3f}")
                    return score
            
            # Check for subdomain matches
            domain_parts = domain.split('.')
            if len(domain_parts) >= 2:
                main_domain = '.'.join(domain_parts[-2:])
                if main_domain in self.DOMAIN_AUTHORITY_SCORES:
                    score = self.DOMAIN_AUTHORITY_SCORES[main_domain]
                    logger.debug(f"   Domain authority (subdomain match): {score:.3f}")
                    return score
            
            # Default: neutral score for unknown domains
            logger.debug(f"   Domain authority (default): 0.0")
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating domain authority: {str(e)}")
            return 0.0
    
    def _check_negative_keywords(self, content: str) -> float:
        """Check for negative keywords that reduce relevance.
        
        Args:
            content: Content from the webpage
            
        Returns:
            Penalty score (0.0 to -0.2)
        """
        content_lower = self._normalize_text(content)
        
        penalty = 0.0
        for keyword in self.NEGATIVE_KEYWORDS:
            if keyword in content_lower:
                penalty -= 0.05
                logger.debug(f"   Negative keyword found: '{keyword}', penalty: -0.05")
        
        # Cap penalty at -0.2
        return max(-0.2, penalty)
    
    def calculate_relevance_score(
        self,
        search_term: str,
        url: str,
        content: str,
        title: str = ""
    ) -> Dict[str, any]:
        """
        Calculate comprehensive relevance score for a webpage.
        
        Args:
            search_term: The search term (e.g., "PriorCoder")
            url: URL of the webpage
            content: Content extracted from the webpage
            title: Title of the webpage (optional)
            
        Returns:
            Dictionary with detailed scoring breakdown:
            {
                'final_score': float (0.0 to 1.0),
                'url_title_score': float,
                'semantic_similarity_score': float,
                'keyword_density_score': float,
                'domain_authority_score': float,
                'negative_penalty': float,
                'is_relevant': bool (True if final_score >= threshold),
                'confidence': str ('high', 'medium', 'low')
            }
        """
        logger.info(f"Calculating relevance score for: {url}")
        logger.info(f"   Search term: {search_term}")
        logger.info(f"   Content length: {len(content)} chars")
        logger.info(f"   Title: {title[:50] if title else 'N/A'}...")
        
        # Calculate individual scores
        url_title_score = self._calculate_url_title_score(search_term, url, title)
        logger.info(f"   ✅ URL/Title Score: {url_title_score:.3f}")
        
        semantic_similarity_score = self._calculate_semantic_similarity_score(search_term, content)
        logger.info(f"   ✅ Semantic Similarity Score: {semantic_similarity_score:.3f}")
        
        keyword_density_score = self._calculate_keyword_density_score(search_term, content)
        logger.info(f"   ✅ Keyword Density Score: {keyword_density_score:.3f}")
        
        domain_authority_score = self._calculate_domain_authority_score(url)
        logger.info(f"   ✅ Domain Authority Score: {domain_authority_score:.3f}")
        
        # Check for negative keywords
        negative_penalty = self._check_negative_keywords(content)
        if negative_penalty < 0:
            logger.warning(f"   ⚠️ Negative Penalty: {negative_penalty:.3f}")
        
        # Calculate weighted final score
        # Note: Domain authority can be negative, so we handle it separately
        final_score = (
            self.WEIGHT_URL_TITLE * url_title_score +
            self.WEIGHT_SEMANTIC_SIMILARITY * semantic_similarity_score +
            self.WEIGHT_KEYWORD_DENSITY * keyword_density_score +
            self.WEIGHT_DOMAIN_AUTHORITY * domain_authority_score +
            negative_penalty  # Apply penalty directly
        )
        
        # Ensure score is between 0.0 and 1.0
        final_score = max(0.0, min(1.0, final_score))
        
        # Determine relevance threshold (0.5 is a reasonable threshold)
        threshold = 0.5
        is_relevant = final_score >= threshold
        
        # Determine confidence level
        if final_score >= 0.7:
            confidence = 'high'
        elif final_score >= 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        logger.info(f"   📊 Final Relevance Score: {final_score:.3f}")
        logger.info(f"   📊 Is Relevant: {is_relevant} (threshold: {threshold})")
        logger.info(f"   📊 Confidence: {confidence}")
        
        return {
            'final_score': final_score,
            'url_title_score': url_title_score,
            'semantic_similarity_score': semantic_similarity_score,
            'keyword_density_score': keyword_density_score,
            'domain_authority_score': domain_authority_score,
            'negative_penalty': negative_penalty,
            'is_relevant': is_relevant,
            'confidence': confidence,
            'threshold': threshold
        }
    
    def rank_pages_by_relevance(
        self,
        search_term: str,
        pages: List[Dict[str, str]]
    ) -> List[Dict[str, any]]:
        """
        Rank multiple pages by relevance score.
        
        Args:
            search_term: The search term
            pages: List of page dictionaries with 'url', 'content', 'title' keys
            
        Returns:
            List of pages sorted by relevance score (highest first),
            each with added 'relevance_score' and 'relevance_details' keys
        """
        logger.info(f"Ranking {len(pages)} pages by relevance...")
        
        ranked_pages = []
        
        for i, page in enumerate(pages, 1):
            url = page.get('url', '')
            content = page.get('content', '')
            title = page.get('title', '')
            
            logger.info(f"   [{i}/{len(pages)}] Evaluating: {url}")
            
            # Calculate relevance score
            relevance_details = self.calculate_relevance_score(
                search_term=search_term,
                url=url,
                content=content,
                title=title
            )
            
            # Add relevance information to page
            page_copy = page.copy()
            page_copy['relevance_score'] = relevance_details['final_score']
            page_copy['relevance_details'] = relevance_details
            page_copy['is_relevant'] = relevance_details['is_relevant']
            page_copy['confidence'] = relevance_details['confidence']
            
            ranked_pages.append(page_copy)
        
        # Sort by relevance score (highest first)
        ranked_pages.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        
        logger.info(f"   ✅ Ranking completed")
        for i, page in enumerate(ranked_pages[:5], 1):
            logger.info(f"      {i}. {page.get('url', 'N/A')}: {page.get('relevance_score', 0.0):.3f} ({page.get('confidence', 'unknown')})")
        
        return ranked_pages
    
    def filter_relevant_pages(
        self,
        search_term: str,
        pages: List[Dict[str, str]],
        min_score: float = 0.5
    ) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
        """
        Filter pages by relevance score.
        
        Args:
            search_term: The search term
            pages: List of page dictionaries
            min_score: Minimum relevance score threshold (default: 0.5)
            
        Returns:
            Tuple of (relevant_pages, irrelevant_pages)
        """
        ranked_pages = self.rank_pages_by_relevance(search_term, pages)
        
        relevant_pages = [p for p in ranked_pages if p.get('relevance_score', 0.0) >= min_score]
        irrelevant_pages = [p for p in ranked_pages if p.get('relevance_score', 0.0) < min_score]
        
        logger.info(f"   ✅ Filtered: {len(relevant_pages)} relevant, {len(irrelevant_pages)} irrelevant (threshold: {min_score})")
        
        return relevant_pages, irrelevant_pages

