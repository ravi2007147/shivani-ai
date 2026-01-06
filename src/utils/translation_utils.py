"""Shared translation utilities for book translation.

This module provides a centralized translation function that is used by both
the single-page translation API endpoint and the background translation job manager.
This ensures consistency - when translation logic is updated in one place,
it applies to all translation operations.
"""

import logging
from typing import Optional
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


def translate_text_with_ollama(
    llm: OllamaLLM,
    english_text: str,
    use_refinement: bool = True,
    page_display: Optional[str] = None
) -> str:
    """Translate English text to Hindi using Ollama LLM.
    
    This is the shared translation function used by both:
    - Single-page translation API endpoint
    - Background translation job manager
    
    Args:
        llm: Initialized OllamaLLM instance
        english_text: English text to translate
        use_refinement: If True, use two-step translation (initial + refinement)
        page_display: Optional page number for logging (e.g., "Page 5/100")
        
    Returns:
        Translated Hindi text (cleaned of prefixes)
    """
    log_prefix = f"[{page_display}] " if page_display else ""
    
    if use_refinement:
        # Two-step translation: initial translation + refinement
        logger.info(f"{log_prefix}Starting two-step translation (Step 1: Initial translation)...")
        
        # Step 1: Initial translation with context awareness
        initial_prompt = f"""You are a professional translator translating English to Hindi. Read the ENTIRE paragraph/section first to understand the full context before translating.

CRITICAL INSTRUCTIONS:
1. **Read the complete text first** - Understand the full context, meaning, and flow of the entire paragraph before translating
2. **Paragraph-level translation** - Translate based on the meaning of complete sentences and paragraphs, NOT word-by-word
3. **Keep English words when appropriate** - Keep technical terms, proper nouns, brand names, and words that are commonly used in English in India (like "computer", "internet", "software", "email", etc.) in English. This creates natural Hinglish that is easy to read
4. **Context-aware meaning** - Understand what the author is trying to say in the full context, then express that meaning naturally in Hindi
5. **Natural Hindi flow** - The translation should read naturally in Hindi, as if originally written in Hindi

ENGLISH TEXT TO TRANSLATE:
{english_text}

TRANSLATION RULES:
- Translate the MEANING and CONTEXT, not individual words
- Keep technical terms, proper nouns, and commonly used English words in English
- Use natural Hindi sentence structure and word order
- Preserve paragraph breaks and formatting
- Output ONLY the Hindi/Hinglish translation, no explanations

Hindi Translation:"""
        
        initial_translation = llm.invoke(initial_prompt).strip()
        logger.info(f"{log_prefix}Initial translation completed ({len(initial_translation)} chars)")
        
        # Step 2: Refinement with emphasis on context and natural flow
        logger.info(f"{log_prefix}Starting Step 2: Refinement...")
        refinement_prompt = f"""You are refining a Hindi/Hinglish translation to make it more natural, contextually accurate, and readable.

ORIGINAL ENGLISH TEXT (read this first to understand full context):
{english_text}

CURRENT HINDI/HINGLISH TRANSLATION:
{initial_translation}

REFINEMENT TASK:
1. **Context Check**: Read the original English text completely. Does the translation capture the full meaning and context of the paragraph? If not, improve it.
2. **Natural Flow**: Make the translation flow naturally in Hindi - it should read as if originally written in Hindi, not like a word-by-word translation
3. **Hinglish Balance**: Keep technical terms, proper nouns, and commonly used English words (like "computer", "internet", "software", "email", "website", "app", etc.) in English when they make more sense than Hindi equivalents
4. **Sentence Structure**: Use natural Hindi sentence structure - don't force English word order
5. **Meaning over Literal**: Prioritize conveying the correct meaning over literal word translation
6. **Readability**: Ensure the translation is easy to read and understand for Hindi speakers

IMPORTANT:
- The translation should make sense when read in Hindi
- Don't translate words that are commonly used in English in India
- Focus on the overall meaning and context of the paragraph, not individual words
- If a word doesn't have a good Hindi equivalent or sounds awkward in Hindi, keep it in English

Output ONLY the refined Hindi/Hinglish translation, no explanations:"""
        
        try:
            hindi_translation = llm.invoke(refinement_prompt).strip()
            logger.info(f"{log_prefix}Refinement completed ({len(hindi_translation)} chars)")
        except Exception as refine_error:
            logger.error(f"{log_prefix}Refinement FAILED: {refine_error}", exc_info=True)
            raise  # Re-raise to be caught by outer exception handler
    else:
        # Single-step translation with improved context awareness
        logger.info(f"{log_prefix}Starting single-step translation...")
        translation_prompt = f"""You are a professional translator with native-level proficiency in both English and Hindi. Your expertise includes understanding cultural nuances, idiomatic expressions, and context-dependent meanings.

CRITICAL: Read the ENTIRE text first to understand the full context, meaning, and flow before translating.

TRANSLATION APPROACH:
1. **Paragraph-Level Understanding**: Read the complete paragraph/section first. Understand the full context, the author's intent, and how sentences connect to each other
2. **Meaning-Based Translation**: Translate the MEANING and CONTEXT of the paragraph, NOT individual words. Think about what the author is trying to convey, then express that naturally in Hindi
3. **Natural Hindi Flow**: The translation should read naturally in Hindi, as if it were originally written in Hindi. Use appropriate Hindi sentence structures and word order
4. **Hinglish Style**: Keep technical terms, proper nouns, brand names, and commonly used English words in India (like "computer", "internet", "software", "email", "website", "app", "mobile", "laptop", etc.) in English. This creates natural Hinglish that is easy to read and understand
5. **Context-Aware Word Choice**: Choose Hindi words that fit the context of the entire paragraph, not just the immediate sentence
6. **Avoid Word-by-Word**: Don't translate each word separately. Instead, understand the complete thought and express it naturally in Hindi
7. **Cultural Adaptation**: Adapt idioms, expressions, and cultural references to be meaningful in Hindi context while keeping the original meaning

WORDS TO KEEP IN ENGLISH (examples):
- Technical terms: computer, internet, software, hardware, email, website, app, mobile, laptop, etc.
- Proper nouns: Names of people, places, companies, brands
- Words commonly used in English in India that don't have good Hindi equivalents
- Words that would sound awkward or unclear if translated to Hindi

ORIGINAL ENGLISH TEXT:
{english_text}

TRANSLATION REQUIREMENTS:
- Read the entire text first to understand full context
- Translate the MEANING of the paragraph, not individual words
- Keep technical terms and commonly used English words in English (Hinglish style)
- Use natural Hindi sentence structure and flow
- Output ONLY the Hindi/Hinglish translation
- No explanations, notes, or additional text
- Preserve paragraph breaks and formatting
- Use proper Devanagari script for Hindi words

Hindi/Hinglish Translation:"""
        
        hindi_translation = llm.invoke(translation_prompt).strip()
        logger.info(f"{log_prefix}Translation completed ({len(hindi_translation)} chars)")
    
    # Clean up translation (remove prefixes that LLM might add)
    prefixes_to_remove = [
        "Hindi translation:", "Translation:", "Here is the translation:",
        "The Hindi translation is:", "हिंदी अनुवाद:", "अनुवाद:",
        "Hindi Translation:", "TRANSLATION:"
    ]
    for prefix in prefixes_to_remove:
        if hindi_translation.lower().startswith(prefix.lower()):
            hindi_translation = hindi_translation[len(prefix):].strip()
    
    return hindi_translation

