"""LAYER 1: Intent Router - Determines input type without asking LLM knowledge questions.

This layer acts like a traffic signal - it only decides what to do next.
It does NOT ask the LLM any knowledge questions.
"""

import re
import logging
from typing import Literal, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class IntentRouter:
    """Routes input to appropriate handler based on intent detection.
    
    This is Layer 1 of the learning system. It determines if the input is:
    - URL: A web address to process
    - Question: A knowledge question to answer
    - Command: An instruction to execute
    
    This layer uses pattern matching and heuristics, NOT LLM calls for routing.
    """
    
    # URL patterns
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
        re.IGNORECASE
    )
    
    # Known website domains that trigger specific processors
    WEBSITE_PROCESSORS = {
        'upwork.com': 'website_processor',
        'fiverr.com': 'website_processor',
        'linkedin.com': 'website_processor',
        'github.com': 'website_processor',
        'stackoverflow.com': 'website_processor',
    }
    
    # Command patterns (simple keywords)
    COMMAND_PATTERNS = [
        r'\b(analyze|process|extract|fetch|get|load|save|store|create|delete|update)\b',
    ]
    
    # Question indicators
    QUESTION_INDICATORS = [
        r'^what\s+',
        r'^who\s+',
        r'^when\s+',
        r'^where\s+',
        r'^why\s+',
        r'^how\s+',
        r'^is\s+',
        r'^are\s+',
        r'^can\s+',
        r'^could\s+',
        r'^should\s+',
        r'^will\s+',
        r'^does\s+',
        r'^do\s+',
        r'\?$',  # Ends with question mark
    ]
    
    def __init__(self):
        """Initialize the Intent Router."""
        self.url_pattern = self.URL_PATTERN
        self.command_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.COMMAND_PATTERNS]
        self.question_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.QUESTION_INDICATORS]
        
    def route(self, input_text: str) -> Dict[str, any]:
        """Route input to appropriate handler.
        
        Args:
            input_text: User input text
            
        Returns:
            Dictionary with:
                - intent: 'url', 'question', or 'command'
                - input: Original input text
                - metadata: Additional routing information
                - processor: Specific processor to use (if applicable)
        """
        if not input_text or not input_text.strip():
            return {
                'intent': 'unknown',
                'input': input_text,
                'metadata': {},
                'processor': None,
                'confidence': 0.0
            }
        
        input_text = input_text.strip()
        input_lower = input_text.lower()
        
        # Check for URL first (highest priority)
        url_match = self.url_pattern.search(input_text)
        if url_match:
            url = url_match.group(0)
            domain = self._extract_domain(url)
            processor = self._get_processor_for_domain(domain)
            
            logger.info(f"Intent: URL detected - {url}, Domain: {domain}, Processor: {processor}")
            
            return {
                'intent': 'url',
                'input': input_text,
                'url': url,
                'domain': domain,
                'metadata': {
                    'original_input': input_text,
                    'extracted_url': url,
                    'domain': domain
                },
                'processor': processor,
                'confidence': 1.0
            }
        
        # Check for commands
        if self._is_command(input_text):
            logger.info(f"Intent: Command detected - {input_text}")
            
            return {
                'intent': 'command',
                'input': input_text,
                'metadata': {
                    'original_input': input_text,
                    'command_type': self._extract_command_type(input_text)
                },
                'processor': 'command_processor',
                'confidence': 0.8
            }
        
        # Check for questions
        if self._is_question(input_text):
            logger.info(f"Intent: Question detected - {input_text}")
            
            return {
                'intent': 'question',
                'input': input_text,
                'metadata': {
                    'original_input': input_text,
                    'question_type': self._extract_question_type(input_text)
                },
                'processor': 'question_processor',
                'confidence': 0.9
            }
        
        # Default: treat as general input (might be a question without explicit indicators)
        logger.info(f"Intent: General input (default) - {input_text}")
        
        return {
            'intent': 'general',
            'input': input_text,
            'metadata': {
                'original_input': input_text
            },
            'processor': 'general_processor',
            'confidence': 0.5
        }
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL.
        
        Args:
            url: URL string
            
        Returns:
            Domain name or None
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            # Remove www. if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception as e:
            logger.warning(f"Error extracting domain from URL {url}: {str(e)}")
            return None
    
    def _get_processor_for_domain(self, domain: str) -> Optional[str]:
        """Get processor name for domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Processor name or None
        """
        # Check exact match
        if domain in self.WEBSITE_PROCESSORS:
            return self.WEBSITE_PROCESSORS[domain]
        
        # Check partial match (e.g., "subdomain.upwork.com" -> "website_processor")
        for known_domain, processor in self.WEBSITE_PROCESSORS.items():
            if known_domain in domain or domain in known_domain:
                return processor
        
        # Default processor for any URL
        return 'website_processor'
    
    def _is_command(self, text: str) -> bool:
        """Check if input is a command.
        
        Args:
            text: Input text
            
        Returns:
            True if command, False otherwise
        """
        text_lower = text.lower()
        
        # Check command patterns
        for pattern in self.command_patterns:
            if pattern.search(text):
                return True
        
        # Additional heuristic: if it starts with an action verb and is short
        action_verbs = ['analyze', 'process', 'extract', 'fetch', 'get', 'load', 'save', 'store', 
                       'create', 'delete', 'update', 'show', 'list', 'display']
        words = text_lower.split()
        if words and words[0] in action_verbs and len(words) <= 5:
            return True
        
        return False
    
    def _is_question(self, text: str) -> bool:
        """Check if input is a question.
        
        Args:
            text: Input text
            
        Returns:
            True if question, False otherwise
        """
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        
        # Check question patterns
        for pattern in self.question_patterns:
            if pattern.search(text_stripped):
                return True
        
        return False
    
    def _extract_command_type(self, text: str) -> str:
        """Extract command type from text.
        
        Args:
            text: Command text
            
        Returns:
            Command type
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['analyze', 'process']):
            return 'analyze'
        elif any(word in text_lower for word in ['fetch', 'get', 'load']):
            return 'fetch'
        elif any(word in text_lower for word in ['save', 'store']):
            return 'save'
        elif any(word in text_lower for word in ['create', 'make']):
            return 'create'
        elif any(word in text_lower for word in ['delete', 'remove']):
            return 'delete'
        elif any(word in text_lower for word in ['update', 'modify']):
            return 'update'
        else:
            return 'unknown'
    
    def _extract_question_type(self, text: str) -> str:
        """Extract question type from text.
        
        Args:
            text: Question text
            
        Returns:
            Question type
        """
        text_lower = text.lower().strip()
        
        if text_lower.startswith('what'):
            return 'what'
        elif text_lower.startswith('who'):
            return 'who'
        elif text_lower.startswith('when'):
            return 'when'
        elif text_lower.startswith('where'):
            return 'where'
        elif text_lower.startswith('why'):
            return 'why'
        elif text_lower.startswith('how'):
            return 'how'
        elif text_lower.startswith(('is', 'are', 'was', 'were')):
            return 'yes_no'
        elif text_lower.startswith(('can', 'could', 'should', 'will', 'would')):
            return 'modal'
        else:
            return 'general'

