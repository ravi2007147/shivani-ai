"""Voice input processing for expense entry.
This module can be easily removed by deleting this file and removing the import/usage.
"""

import json
import re
from typing import Dict, Optional, Any
from datetime import date, datetime


def extract_expense_from_text(
    text: str,
    available_categories: list,
    available_accounts: list,
    ollama_model: str = "mistral",
    ollama_base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """Extract expense information from spoken/text input using LLM.
    
    Args:
        text: Transcribed text from voice input
        available_categories: List of available expense categories
        available_accounts: List of available accounts
        ollama_model: Ollama model to use for extraction
        ollama_base_url: Ollama base URL
        
    Returns:
        Dictionary with extracted expense data:
        - amount: float
        - category: str (category name)
        - account: str (account name)
        - date: str (ISO format date)
        - currency: str (currency code, default 'INR')
        - note: str (optional note)
    """
    try:
        from langchain_ollama import OllamaLLM
        
        llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.1
        )
        
        # Prepare category and account lists
        category_names = [cat['name'] for cat in available_categories] if available_categories else []
        account_names = [acc['name'] for acc in available_accounts] if available_accounts else []
        
        # Current date context
        today = date.today()
        current_month = today.strftime('%B %Y')
        
        prompt = f"""You are an expense data extraction system. Extract expense information from the spoken text and return ONLY a valid JSON object, nothing else.

Input text: "{text}"

Available expense categories: {', '.join(category_names) if category_names else 'None available'}
Available accounts: {', '.join(account_names) if account_names else 'None available'}
Today's date: {today.isoformat()} ({today.strftime('%B %d, %Y')})

Extract the following information and return as JSON:
- amount: Number (e.g., 500, 2500.50)
- category: One of these: {', '.join(category_names) if category_names else 'None'} (find closest match)
- account: One of these: {', '.join(account_names) if account_names else 'None'} (find closest match)
- date: YYYY-MM-DD format (default: {today.isoformat()} if not mentioned)
- currency: Currency code (INR, USD, EUR, etc.) - default: INR
- note: Additional details or empty string

Important: Return ONLY the JSON object. No markdown, no explanations, no code blocks. Example format:
{{"amount": 500, "category": "Food", "account": "Cash", "date": "2025-11-04", "currency": "INR", "note": "lunch expense"}}

If a field cannot be determined, use null. Return the JSON now:
"""
        
        response = llm.invoke(prompt)
        
        # Clean response - remove markdown code blocks if present
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        # Try to parse JSON
        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^}]+\}', cleaned_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Fallback: try to extract basic info manually
                data = _fallback_extraction(text, category_names, account_names, today)
        
        # Validate and clean the extracted data
        result = {
            "amount": _extract_amount(data.get("amount"), text),
            "category": _match_category(data.get("category"), category_names),
            "account": _match_account(data.get("account"), account_names),
            "date": _parse_date(data.get("date"), text, today),
            "currency": data.get("currency", "INR").upper(),
            "note": data.get("note", "").strip()
        }
        
        return result
        
    except Exception as e:
        # Fallback extraction if LLM fails
        return _fallback_extraction(text, 
                                    [cat['name'] for cat in available_categories] if available_categories else [],
                                    [acc['name'] for acc in available_accounts] if available_accounts else [],
                                    date.today())


def _extract_amount(amount_value: Any, text: str) -> Optional[float]:
    """Extract amount from value or text."""
    if amount_value:
        try:
            return float(amount_value)
        except (ValueError, TypeError):
            pass
    
    # Try to extract from text
    # Look for numbers followed by currency words or standalone numbers
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:rupees?|rs|inr|dollars?|usd|euros?|eur)',
        r'(?:rupees?|rs|inr|dollars?|usd|euros?|eur)\s*(\d+(?:\.\d+)?)',
        r'\b(\d+(?:\.\d+)?)\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return None


def _match_category(category: Optional[str], available: list) -> Optional[str]:
    """Match category to available list."""
    if not category or not available:
        return available[0] if available else None
    
    category_lower = category.lower()
    
    # Exact match
    for cat in available:
        if cat.lower() == category_lower:
            return cat
    
    # Partial match
    for cat in available:
        if category_lower in cat.lower() or cat.lower() in category_lower:
            return cat
    
    # Return first available if no match
    return available[0] if available else None


def _match_account(account: Optional[str], available: list) -> Optional[str]:
    """Match account to available list."""
    if not account or not available:
        return available[0] if available else None
    
    account_lower = account.lower()
    
    # Exact match
    for acc in available:
        if acc.lower() == account_lower:
            return acc
    
    # Partial match
    for acc in available:
        if account_lower in acc.lower() or acc.lower() in account_lower:
            return acc
    
    # Return first available if no match
    return available[0] if available else None


def _parse_date(date_str: Optional[str], text: str, default_date: date) -> str:
    """Parse date string or extract from text."""
    if date_str:
        try:
            # Try to parse ISO format
            parsed = datetime.fromisoformat(date_str).date()
            return parsed.isoformat()
        except (ValueError, TypeError):
            pass
    
    # Try to extract date from text
    text_lower = text.lower()
    
    if "today" in text_lower:
        return default_date.isoformat()
    elif "yesterday" in text_lower:
        from datetime import timedelta
        return (default_date - timedelta(days=1)).isoformat()
    
    # Look for date patterns
    date_patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
        r'(\d{1,2})\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{2,4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Try to parse
            try:
                # Simple date parsing (can be enhanced)
                return default_date.isoformat()
            except:
                pass
    
    return default_date.isoformat()


def _fallback_extraction(text: str, categories: list, accounts: list, default_date: date) -> Dict[str, Any]:
    """Fallback extraction when LLM fails."""
    return {
        "amount": _extract_amount(None, text),
        "category": categories[0] if categories else None,
        "account": accounts[0] if accounts else None,
        "date": default_date.isoformat(),
        "currency": "INR",
        "note": text.strip()
    }

