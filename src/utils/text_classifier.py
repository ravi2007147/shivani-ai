"""Text classification utilities for categorizing content."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# PDF Category Labels - Predefined categories for classification
PDF_CATEGORIES = {
    "academic_research": "Academic Research Papers",
    "technical_documentation": "Technical Documentation",
    "business_report": "Business Report",
    "legal_document": "Legal Document",
    "medical_health": "Medical/Health Document",
    "finance_accounting": "Finance/Accounting",
    "marketing_sales": "Marketing/Sales Material",
    "news_article": "News Article",
    "educational_material": "Educational Material",
    "literature_book": "Literature/Book",
    "engineering_design": "Engineering/Design",
    "product_manual": "Product Manual",
    "general_document": "General Document",
}


class PDFClassifier:
    """Classify PDF content into predefined categories using LLM."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the classifier.
        
        Args:
            base_url: Base URL for Ollama API
        """
        self.base_url = base_url
        self.categories_list = list(PDF_CATEGORIES.values())
        self.categories_map = {v: k for k, v in PDF_CATEGORIES.items()}
    
    def classify_text(
        self, 
        text: str, 
        model: str = "mistral",
        max_length: int = 2000
    ) -> Dict[str, any]:
        """Classify text content into a category.
        
        Args:
            text: Text content to classify
            model: Ollama model to use for classification
            max_length: Maximum length of text to use for classification
            
        Returns:
            Dictionary containing classification results
        """
        if not text or not text.strip():
            return {
                "category": "general_document",
                "category_name": "General Document",
                "confidence": 0.0,
                "reasoning": "Empty or invalid text provided"
            }
        
        # Truncate text if too long to save on API calls
        text_sample = text[:max_length] if len(text) > max_length else text
        
        # Create classification prompt
        prompt = self._create_classification_prompt(text_sample)
        
        try:
            # Use Ollama LLM to classify
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=model, base_url=self.base_url)
            
            # Get classification response
            response = llm.invoke(prompt)
            
            # Parse the response
            result = self._parse_classification_response(response)
            return result
            
        except Exception as e:
            # Return error result
            return {
                "category": "general_document",
                "category_name": "General Document",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}",
                "error": str(e)
            }
    
    def _create_classification_prompt(self, text: str) -> str:
        """Create a prompt for text classification.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification prompt
        """
        categories_str = "\n".join([f"- {cat}" for cat in self.categories_list])
        
        prompt = f"""Analyze the following text and classify it into one of these categories:
{categories_str}

Text to classify:
---
{text[:1500]}...
---

Respond in JSON format with these fields:
{{
  "category": "exact_category_name_from_above_list",
  "confidence": <number between 0 and 1>,
  "reasoning": "brief explanation of why this category was chosen"
}}

Important: The category must be EXACTLY one of the category names from the list above.
"""
        return prompt
    
    def _parse_classification_response(self, response: str) -> Dict[str, any]:
        """Parse the LLM classification response.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed classification result
        """
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate and map category
                category_name = result.get("category", "General Document")
                if category_name not in self.categories_map:
                    # Try to find closest match
                    category_name = self._find_closest_category(category_name)
                
                category_key = self.categories_map.get(category_name, "general_document")
                
                return {
                    "category": category_key,
                    "category_name": category_name,
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", "No reasoning provided"),
                }
        except Exception as e:
            pass
        
        # Fallback: try to find category keyword in response
        category_name = self._find_category_from_text(response)
        
        return {
            "category": self.categories_map.get(category_name, "general_document"),
            "category_name": category_name,
            "confidence": 0.7,
            "reasoning": "Inferred from keywords",
        }
    
    def _find_closest_category(self, category_name: str) -> str:
        """Find the closest matching category name.
        
        Args:
            category_name: Category name to match
            
        Returns:
            Closest matching category name
        """
        category_lower = category_name.lower()
        
        # Check for keyword matches
        keywords_map = {
            "academic": "Academic Research Papers",
            "research": "Academic Research Papers",
            "technical": "Technical Documentation",
            "documentation": "Technical Documentation",
            "business": "Business Report",
            "legal": "Legal Document",
            "medical": "Medical/Health Document",
            "health": "Medical/Health Document",
            "finance": "Finance/Accounting",
            "accounting": "Finance/Accounting",
            "marketing": "Marketing/Sales Material",
            "sales": "Marketing/Sales Material",
            "news": "News Article",
            "educational": "Educational Material",
            "education": "Educational Material",
            "literature": "Literature/Book",
            "book": "Literature/Book",
            "engineering": "Engineering/Design",
            "design": "Engineering/Design",
            "manual": "Product Manual",
        }
        
        for keyword, category in keywords_map.items():
            if keyword in category_lower:
                return category
        
        return "General Document"
    
    def _find_category_from_text(self, text: str) -> str:
        """Find category from text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Category name
        """
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["academic", "research", "paper", "study"]):
            return "Academic Research Papers"
        elif any(kw in text_lower for kw in ["technical", "documentation", "guide", "api"]):
            return "Technical Documentation"
        elif any(kw in text_lower for kw in ["business", "corporate", "company"]):
            return "Business Report"
        elif any(kw in text_lower for kw in ["legal", "law", "court", "attorney"]):
            return "Legal Document"
        elif any(kw in text_lower for kw in ["medical", "health", "medicine", "patient", "clinical"]):
            return "Medical/Health Document"
        elif any(kw in text_lower for kw in ["finance", "accounting", "financial", "account"]):
            return "Finance/Accounting"
        elif any(kw in text_lower for kw in ["marketing", "sales", "advertising"]):
            return "Marketing/Sales Material"
        elif any(kw in text_lower for kw in ["news", "article", "report"]):
            return "News Article"
        elif any(kw in text_lower for kw in ["educational", "learning", "course", "lesson"]):
            return "Educational Material"
        elif any(kw in text_lower for kw in ["book", "novel", "literature", "fiction"]):
            return "Literature/Book"
        elif any(kw in text_lower for kw in ["engineering", "design", "architect"]):
            return "Engineering/Design"
        elif any(kw in text_lower for kw in ["manual", "instruction", "how to"]):
            return "Product Manual"
        else:
            return "General Document"
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available categories.
        
        Returns:
            List of category names
        """
        return self.categories_list


def classify_pdf_content(
    text: str,
    model: str = "mistral",
    base_url: str = "http://localhost:11434",
    max_length: int = 2000
) -> Dict[str, any]:
    """Convenience function to classify PDF content.
    
    Args:
        text: Text content to classify
        model: Ollama model to use
        base_url: Ollama base URL
        max_length: Maximum text length for classification
        
    Returns:
        Classification results dictionary
    """
    classifier = PDFClassifier(base_url=base_url)
    return classifier.classify_text(text, model=model, max_length=max_length)

