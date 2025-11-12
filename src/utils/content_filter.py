"""Content filter for removing unwanted elements from scraped articles."""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Common unwanted patterns to filter out
UNWANTED_PATTERNS = [
    # Reddit-specific patterns
    r'Go to\s+\w+\s*r/\w+',
    r'r/\w+',
    r'\[D\]\s*\[R\]\s*\[P\]',
    r'Members\s+\w+',
    r'\[D\]\s+.*?Discussion',
    r'Read more\s+Share',
    r'Share\s+Save',
    r'Read more',
    r'Share',
    r'Save',
    r'Subscribe',
    r'Join',
    r'Follow',
    r'Upvote',
    r'Downvote',
    r'Comments',
    r'Edit',
    r'Delete',
    r'Report',
    
    # Navigation elements
    r'Home\s+About\s+Contact',
    r'Privacy\s+Policy',
    r'Terms\s+of\s+Service',
    r'Cookie\s+Policy',
    r'Accept\s+All\s+Cookies',
    r'Cookie\s+Settings',
    r'Skip\s+to\s+content',
    r'Skip\s+to\s+main\s+content',
    r'Menu',
    r'Navigation',
    r'Main\s+menu',
    
    # Social media buttons
    r'Share\s+on\s+Facebook',
    r'Share\s+on\s+Twitter',
    r'Share\s+on\s+LinkedIn',
    r'Share\s+on\s+Reddit',
    r'Tweet\s+this',
    r'Like\s+this',
    r'Follow\s+us\s+on',
    
    # Advertisement patterns
    r'Advertisement',
    r'Ad\s+-\s+Continue\s+reading',
    r'Sponsored\s+content',
    r'Promoted\s+content',
    r'This\s+content\s+is\s+sponsored',
    
    # Newsletter/Subscription
    r'Subscribe\s+to\s+our\s+newsletter',
    r'Sign\s+up\s+for\s+updates',
    r'Get\s+notifications',
    r'Email\s+address',
    
    # Footer/Header noise
    r'Copyright\s+\d{4}',
    r'All\s+rights\s+reserved',
    r'Powered\s+by',
    r'Built\s+with',
    
    # Reddit post metadata
    r'Posted\s+by\s+u/\w+',
    r'u/\w+',
    r'\d+\s+points',
    r'\d+\s+comments',
    r'\d+\s+hours?\s+ago',
    r'\d+\s+days?\s+ago',
    r'\d+\s+months?\s+ago',
    r'\d+\s+years?\s+ago',
    
    # Generic unwanted phrases
    r'Click\s+here',
    r'Read\s+more\s+here',
    r'Learn\s+more',
    r'See\s+more',
    r'Show\s+more',
    r'Hide\s+this',
    r'Close',
    r'×',
    r'Continue\s+reading',
    r'View\s+all',
    r'Load\s+more',
]

# Patterns that indicate navigation/sidebar content (lines to remove)
NAVIGATION_INDICATORS = [
    r'^Go to\s+',
    r'^r/',
    r'^u/',
    r'^\[D\]',
    r'^\[R\]',
    r'^\[P\]',
    r'^Members\s+',
    r'^Read more\s+Share',
    r'^Share\s+Save',
    r'^\d+\s+points',
    r'^\d+\s+comments',
    r'^Posted\s+by',
    r'^Subscribe',
    r'^Follow',
    r'^Join',
    r'^Menu',
    r'^Navigation',
    r'^Home\s+',
    r'^About\s+',
    r'^Contact\s+',
    r'^Privacy',
    r'^Terms',
    r'^Cookie',
]

# Unwanted phrases that should be removed from content
UNWANTED_PHRASES = [
    'Go to',
    'Read more Share',
    'Share Save',
    'Read more',
    'Share',
    'Save',
    'Subscribe',
    'Follow',
    'Join',
    'Upvote',
    'Downvote',
    'Comments',
    'Edit',
    'Delete',
    'Report',
    'Menu',
    'Navigation',
    'Home',
    'About',
    'Contact',
    'Privacy Policy',
    'Terms of Service',
    'Cookie Policy',
    'Accept All Cookies',
    'Cookie Settings',
    'Skip to content',
    'Skip to main content',
    'Main menu',
    'Share on Facebook',
    'Share on Twitter',
    'Share on LinkedIn',
    'Share on Reddit',
    'Tweet this',
    'Like this',
    'Follow us on',
    'Advertisement',
    'Ad - Continue reading',
    'Sponsored content',
    'Promoted content',
    'This content is sponsored',
    'Subscribe to our newsletter',
    'Sign up for updates',
    'Get notifications',
    'Email address',
    'Copyright',
    'All rights reserved',
    'Powered by',
    'Built with',
    'Posted by',
    'points',
    'comments',
    'hours ago',
    'days ago',
    'months ago',
    'years ago',
    'Click here',
    'Read more here',
    'Learn more',
    'See more',
    'Show more',
    'Hide this',
    'Close',
    'Continue reading',
    'View all',
    'Load more',
]


def filter_unwanted_content(content: str, aggressive: bool = True) -> str:
    """Filter out unwanted content from scraped article text.
    
    Args:
        content: Raw content from web scraper
        aggressive: If True, apply aggressive filtering (default: True)
        
    Returns:
        Filtered content with unwanted elements removed
    """
    if not content:
        return content
    
    # Step 1: Remove lines that match navigation indicators
    lines = content.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Skip lines that match navigation indicators
        is_navigation = False
        for pattern in NAVIGATION_INDICATORS:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                is_navigation = True
                break
        
        if not is_navigation:
            # Check if line contains unwanted patterns
            contains_unwanted = False
            for pattern in UNWANTED_PATTERNS:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    # Allow if it's part of a longer meaningful sentence
                    if len(line_stripped) < 100:  # Short lines are likely navigation
                        contains_unwanted = True
                        break
            
            if not contains_unwanted:
                filtered_lines.append(line)
    
    # Join filtered lines
    filtered_content = '\n'.join(filtered_lines)
    
    # Step 2: Remove unwanted patterns from content
    for pattern in UNWANTED_PATTERNS:
        filtered_content = re.sub(pattern, '', filtered_content, flags=re.IGNORECASE)
    
    # Step 3: Remove unwanted phrases (aggressive mode)
    if aggressive:
        for phrase in UNWANTED_PHRASES:
            # Remove phrase if it appears as standalone (not part of a word)
            pattern = r'\b' + re.escape(phrase) + r'\b'
            filtered_content = re.sub(pattern, '', filtered_content, flags=re.IGNORECASE)
    
    # Step 4: Clean up multiple spaces and empty lines
    filtered_content = re.sub(r'\s+', ' ', filtered_content)  # Multiple spaces to single
    filtered_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', filtered_content)  # Multiple newlines to double
    filtered_content = filtered_content.strip()
    
    # Step 5: Remove Reddit-specific noise
    # Remove lines that are just "r/subreddit" or "u/username"
    lines = filtered_content.split('\n')
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are just subreddit or username references
        if re.match(r'^(r/|u/)[\w-]+$', line_stripped, re.IGNORECASE):
            continue
        # Skip lines that are mostly navigation
        if len(line_stripped) < 20 and any(word in line_stripped.lower() for word in ['share', 'save', 'subscribe', 'follow', 'join', 'menu', 'home', 'about']):
            continue
        cleaned_lines.append(line)
    
    filtered_content = '\n'.join(cleaned_lines)
    
    # Step 6: Remove short lines that are likely navigation (aggressive mode)
    if aggressive:
        lines = filtered_content.split('\n')
        meaningful_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Keep lines that are meaningful (long enough or contain substantial content)
            if len(line_stripped) > 30:  # Keep longer lines
                meaningful_lines.append(line)
            elif len(line_stripped) > 10 and not any(word in line_stripped.lower() for word in ['share', 'save', 'subscribe', 'follow', 'join', 'menu', 'home', 'about', 'click', 'read more']):
                # Keep shorter lines if they don't contain navigation words
                meaningful_lines.append(line)
        
        filtered_content = '\n'.join(meaningful_lines)
    
    # Final cleanup
    filtered_content = re.sub(r'\s+', ' ', filtered_content)
    filtered_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', filtered_content)
    filtered_content = filtered_content.strip()
    
    return filtered_content


def extract_main_content(content: str, min_length: int = 100) -> str:
    """Extract main content by removing navigation and unwanted elements.
    
    Args:
        content: Raw content from web scraper
        min_length: Minimum length of content to keep (default: 100)
        
    Returns:
        Extracted main content
    """
    if not content:
        return content
    
    # Filter unwanted content
    filtered = filter_unwanted_content(content, aggressive=True)
    
    # If filtered content is too short, return original (might be a short article)
    if len(filtered) < min_length and len(content) > min_length:
        # Try less aggressive filtering
        filtered = filter_unwanted_content(content, aggressive=False)
    
    return filtered


def is_navigation_line(line: str) -> bool:
    """Check if a line is likely navigation/content.
    
    Args:
        line: Line of text to check
        
    Returns:
        True if line is likely navigation, False otherwise
    """
    line_stripped = line.strip()
    
    # Check navigation indicators
    for pattern in NAVIGATION_INDICATORS:
        if re.match(pattern, line_stripped, re.IGNORECASE):
            return True
    
    # Check for navigation words in short lines
    if len(line_stripped) < 50:
        nav_words = ['share', 'save', 'subscribe', 'follow', 'join', 'menu', 'home', 'about', 'contact', 'privacy', 'terms', 'cookie', 'click here', 'read more', 'learn more']
        if any(word in line_stripped.lower() for word in nav_words):
            return True
    
    return False

