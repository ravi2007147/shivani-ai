"""FastAPI server for Expense Manager APIs."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime
from calendar import monthrange
import sys
from pathlib import Path
import io
import tempfile
import os
import zipfile
import json
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import ExpenseDB
from src.utils.books_db import BooksDBManager

# Create FastAPI app
app = FastAPI(
    title="Expense Manager API",
    description="REST API endpoints for Expense Manager resources",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = ExpenseDB()

# Initialize translation job manager
from src.api.translation_job_manager import TranslationJobManager
translation_job_manager = TranslationJobManager(BooksDBManager())


# Helper functions
def get_current_month_dates():
    """Get start and end dates for current month."""
    today = date.today()
    first_day = date(today.year, today.month, 1)
    last_day = date(today.year, today.month, monthrange(today.year, today.month)[1])
    return first_day.isoformat(), last_day.isoformat()


# Response Models
class CategoryResponse(BaseModel):
    id: int
    name: str
    type: str


class AccountResponse(BaseModel):
    id: int
    name: str


class SourceResponse(BaseModel):
    id: int
    name: str


class APIResponse(BaseModel):
    success: bool
    data: List
    message: Optional[str] = None


# Request Models
class IncomeRequest(BaseModel):
    date: str
    category_id: int
    amount: float
    currency: str = 'INR'
    note: Optional[str] = None
    source_id: Optional[int] = None


class ExpenseRequest(BaseModel):
    date: str
    category_id: int
    amount: float
    account_id: int
    currency: str = 'INR'
    note: Optional[str] = None


# Response Models for created records
class IncomeResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


class ExpenseResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


# Income Categories Endpoints
@app.get("/api/income-categories", response_model=APIResponse)
async def get_income_categories():
    """Get all income categories."""
    try:
        categories = db.get_categories(category_type='income')
        data = [
            {
                "id": cat["id"],
                "name": cat["name"],
                "type": cat["type"]
            }
            for cat in categories
        ]
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Expense Categories Endpoints
@app.get("/api/expense-categories", response_model=APIResponse)
async def get_expense_categories():
    """Get all expense categories."""
    try:
        categories = db.get_categories(category_type='expense')
        data = [
            {
                "id": cat["id"],
                "name": cat["name"],
                "type": cat["type"]
            }
            for cat in categories
        ]
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Accounts Endpoints
@app.get("/api/accounts", response_model=APIResponse)
async def get_accounts():
    """Get all expense accounts."""
    try:
        accounts = db.get_accounts()
        data = [
            {
                "id": acc["id"],
                "name": acc["name"]
            }
            for acc in accounts
        ]
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Sources Endpoints
@app.get("/api/sources", response_model=APIResponse)
async def get_sources():
    """Get all income sources."""
    try:
        sources = db.get_sources()
        data = [
            {
                "id": src["id"],
                "name": src["name"]
            }
            for src in sources
        ]
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Income Management Endpoints
@app.post("/api/income", response_model=IncomeResponse)
async def create_income(income: IncomeRequest):
    """Create a new income record."""
    try:
        # Validate date format (basic check)
        if not income.date:
            raise HTTPException(status_code=400, detail="Date is required")
        
        # Validate category exists
        categories = db.get_categories(category_type='income')
        category_ids = [cat["id"] for cat in categories]
        if income.category_id not in category_ids:
            raise HTTPException(status_code=400, detail=f"Invalid category_id: {income.category_id}")
        
        # Validate source if provided
        if income.source_id is not None:
            sources = db.get_sources()
            source_ids = [src["id"] for src in sources]
            if income.source_id not in source_ids:
                raise HTTPException(status_code=400, detail=f"Invalid source_id: {income.source_id}")
        
        # Validate amount
        if income.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        
        # Add income to database
        success, message = db.add_income(
            date=income.date,
            category_id=income.category_id,
            amount=income.amount,
            currency=income.currency,
            note=income.note,
            source_id=income.source_id
        )
        
        if success:
            return IncomeResponse(
                success=True,
                message=message,
                data={
                    "date": income.date,
                    "category_id": income.category_id,
                    "amount": income.amount,
                    "currency": income.currency,
                    "note": income.note,
                    "source_id": income.source_id
                }
            )
        else:
            raise HTTPException(status_code=400, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating income record: {str(e)}")


# Income List Endpoint
@app.get("/api/income", response_model=APIResponse)
async def list_income(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category_id: Optional[int] = None,
    source_id: Optional[int] = None
):
    """Get income records with optional filters. Defaults to current month if no dates provided."""
    try:
        # Default to current month if no dates provided
        if not start_date and not end_date:
            start_date, end_date = get_current_month_dates()
        
        income_records = db.get_income(
            start_date=start_date,
            end_date=end_date,
            category_id=category_id,
            source_id=source_id
        )
        
        return APIResponse(success=True, data=income_records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching income records: {str(e)}")


# Income Update Endpoint
@app.put("/api/income/{income_id}", response_model=IncomeResponse)
async def update_income(income_id: int, income: IncomeRequest):
    """Update an existing income record."""
    try:
        # Validate category exists
        categories = db.get_categories(category_type='income')
        category_ids = [cat["id"] for cat in categories]
        if income.category_id not in category_ids:
            raise HTTPException(status_code=400, detail=f"Invalid category_id: {income.category_id}")
        
        # Validate source if provided
        if income.source_id is not None:
            sources = db.get_sources()
            source_ids = [src["id"] for src in sources]
            if income.source_id not in source_ids:
                raise HTTPException(status_code=400, detail=f"Invalid source_id: {income.source_id}")
        
        # Validate amount
        if income.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        
        # Update income record
        success, message = db.update_income(
            income_id=income_id,
            date=income.date,
            category_id=income.category_id,
            amount=income.amount,
            currency=income.currency,
            note=income.note,
            source_id=income.source_id
        )
        
        if success:
            return IncomeResponse(
                success=True,
                message=message,
                data={
                    "id": income_id,
                    "date": income.date,
                    "category_id": income.category_id,
                    "amount": income.amount,
                    "currency": income.currency,
                    "note": income.note,
                    "source_id": income.source_id
                }
            )
        else:
            raise HTTPException(status_code=400, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating income record: {str(e)}")


# Income Delete Endpoint
@app.delete("/api/income/{income_id}", response_model=IncomeResponse)
async def delete_income(income_id: int):
    """Delete an income record."""
    try:
        success, message = db.delete_income(income_id)
        
        if success:
            return IncomeResponse(
                success=True,
                message=message,
                data={"id": income_id}
            )
        else:
            raise HTTPException(status_code=400, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting income record: {str(e)}")


# Expense Management Endpoints
@app.post("/api/expense", response_model=ExpenseResponse)
async def create_expense(expense: ExpenseRequest):
    """Create a new expense record."""
    try:
        # Validate date format (basic check)
        if not expense.date:
            raise HTTPException(status_code=400, detail="Date is required")
        
        # Validate category exists
        categories = db.get_categories(category_type='expense')
        category_ids = [cat["id"] for cat in categories]
        if expense.category_id not in category_ids:
            raise HTTPException(status_code=400, detail=f"Invalid category_id: {expense.category_id}")
        
        # Validate account exists
        accounts = db.get_accounts()
        account_ids = [acc["id"] for acc in accounts]
        if expense.account_id not in account_ids:
            raise HTTPException(status_code=400, detail=f"Invalid account_id: {expense.account_id}")
        
        # Validate amount
        if expense.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        
        # Add expense to database
        success, message = db.add_expense(
            date=expense.date,
            category_id=expense.category_id,
            amount=expense.amount,
            account_id=expense.account_id,
            currency=expense.currency,
            note=expense.note
        )
        
        if success:
            return ExpenseResponse(
                success=True,
                message=message,
                data={
                    "date": expense.date,
                    "category_id": expense.category_id,
                    "amount": expense.amount,
                    "account_id": expense.account_id,
                    "currency": expense.currency,
                    "note": expense.note
                }
            )
        else:
            raise HTTPException(status_code=400, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating expense record: {str(e)}")


# Expense List Endpoint
@app.get("/api/expense", response_model=APIResponse)
async def list_expense(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category_id: Optional[int] = None,
    account_id: Optional[int] = None
):
    """Get expense records with optional filters. Defaults to current month if no dates provided."""
    try:
        # Default to current month if no dates provided
        if not start_date and not end_date:
            start_date, end_date = get_current_month_dates()
        
        expense_records = db.get_expenses(
            start_date=start_date,
            end_date=end_date,
            category_id=category_id,
            account_id=account_id
        )
        
        return APIResponse(success=True, data=expense_records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching expense records: {str(e)}")


# Expense Update Endpoint
@app.put("/api/expense/{expense_id}", response_model=ExpenseResponse)
async def update_expense(expense_id: int, expense: ExpenseRequest):
    """Update an existing expense record."""
    try:
        # Validate category exists
        categories = db.get_categories(category_type='expense')
        category_ids = [cat["id"] for cat in categories]
        if expense.category_id not in category_ids:
            raise HTTPException(status_code=400, detail=f"Invalid category_id: {expense.category_id}")
        
        # Validate account exists
        accounts = db.get_accounts()
        account_ids = [acc["id"] for acc in accounts]
        if expense.account_id not in account_ids:
            raise HTTPException(status_code=400, detail=f"Invalid account_id: {expense.account_id}")
        
        # Validate amount
        if expense.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        
        # Update expense record
        success, message = db.update_expense(
            expense_id=expense_id,
            date=expense.date,
            category_id=expense.category_id,
            amount=expense.amount,
            account_id=expense.account_id,
            currency=expense.currency,
            note=expense.note
        )
        
        if success:
            return ExpenseResponse(
                success=True,
                message=message,
                data={
                    "id": expense_id,
                    "date": expense.date,
                    "category_id": expense.category_id,
                    "amount": expense.amount,
                    "account_id": expense.account_id,
                    "currency": expense.currency,
                    "note": expense.note
                }
            )
        else:
            raise HTTPException(status_code=400, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating expense record: {str(e)}")


# Expense Delete Endpoint
@app.delete("/api/expense/{expense_id}", response_model=ExpenseResponse)
async def delete_expense(expense_id: int):
    """Delete an expense record."""
    try:
        success, message = db.delete_expense(expense_id)
        
        if success:
            return ExpenseResponse(
                success=True,
                message=message,
                data={"id": expense_id}
            )
        else:
            raise HTTPException(status_code=400, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting expense record: {str(e)}")


# PDF Text Extraction Endpoint
class PDFTextExtractRequest(BaseModel):
    book_id: int
    page_number: int
    include_next_page: bool = True  # Include text from next page if sentence is incomplete
    include_previous_page: bool = False  # Include text from previous page to complete sentence


class PDFTextExtractResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    message: Optional[str] = None
    includes_next_page: bool = False
    includes_previous_page: bool = False


def is_sentence_complete(text: str) -> bool:
    """Check if the text ends with a complete sentence.
    
    Args:
        text: Text to check
        
    Returns:
        True if the last sentence appears complete, False otherwise
    """
    if not text or not text.strip():
        return True
    
    # Remove trailing whitespace
    text = text.rstrip()
    if not text:
        return True
    
    # Check if ends with sentence-ending punctuation
    sentence_endings = ['.', '!', '?', '。', '！', '？']
    if text[-1] in sentence_endings:
        # Check if it's not an abbreviation (simple heuristic)
        # Common abbreviations that end with period
        common_abbreviations = ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'etc.', 'vs.', 'e.g.', 'i.e.', 'a.m.', 'p.m.', 'Inc.', 'Ltd.', 'St.', 'Ave.']
        words = text.split()
        if words:
            last_word = words[-1]
            # If last word is a known abbreviation, check if there's more context
            if last_word in common_abbreviations:
                # If it's the only word or very short, might be incomplete
                if len(words) == 1 or len(text) < 50:
                    return False
        return True
    
    # Check if ends with other punctuation that might indicate continuation
    continuation_punctuation = [',', ';', ':', '-', '–', '—']
    if text[-1] in continuation_punctuation:
        return False
    
    # If text doesn't end with sentence punctuation, it's likely incomplete
    return False


def get_complete_sentence_text(text: str, next_page_text: Optional[str] = None) -> str:
    """Get text with completed sentences, optionally including next page text.
    
    Args:
        text: Current page text
        next_page_text: Optional text from next page to complete sentences
        
    Returns:
        Text with completed sentences
    """
    if not text:
        return ""
    
    # If no next page text or sentence is already complete, return as is
    if not next_page_text or is_sentence_complete(text):
        return text
    
    # Find the last complete sentence position
    sentence_endings = ['.', '!', '?', '。', '！', '？']
    last_complete_pos = -1
    
    # Look backwards for sentence-ending punctuation
    for i in range(len(text) - 1, -1, -1):
        if text[i] in sentence_endings:
            # Make sure it's not part of an abbreviation (simple check)
            if i > 0 and text[i-1].isalpha():
                last_complete_pos = i
                break
    
    # Split text into complete and incomplete parts
    if last_complete_pos >= 0:
        complete_text = text[:last_complete_pos + 1].rstrip()
        incomplete_part = text[last_complete_pos + 1:].strip()
    else:
        # No complete sentences found - entire text might be one incomplete sentence
        complete_text = ""
        incomplete_part = text.strip()
    
    # Try to complete the sentence from next page
    if incomplete_part and next_page_text:
        import re
        # Get first sentence from next page
        # Find first sentence ending in next page text
        first_sentence_match = re.search(r'^([^.!?。！？]*[.!?。！？])', next_page_text.strip(), re.MULTILINE)
        
        if first_sentence_match:
            first_sentence = first_sentence_match.group(1).strip()
        else:
            # No sentence ending found, take first line or first 200 chars
            first_line = next_page_text.strip().split('\n')[0]
            first_sentence = first_line[:200].strip()
        
        # Combine incomplete part with first sentence from next page
        if incomplete_part:
            # Remove any trailing punctuation from incomplete part (might be a hyphen or dash)
            incomplete_clean = incomplete_part.rstrip(' -–—')
            # Add space if needed
            if incomplete_clean and not incomplete_clean[-1].isspace():
                completed = incomplete_clean + " " + first_sentence
            else:
                completed = incomplete_clean + first_sentence
        else:
            completed = first_sentence
        
        # Combine with complete text
        if complete_text:
            return complete_text + "\n\n" + completed
        else:
            return completed
    
    return text


def extract_pdf_page_text_accurate(pdf_path: Path, page_number: int, cached_doc=None) -> Optional[str]:
    """Extract text from a specific PDF page with formatting preserved.
    This is a copy of the function from the translation page for API use."""
    try:
        import fitz  # pymupdf
        
        if cached_doc is not None:
            doc = cached_doc
            should_close = False
        else:
            doc = fitz.open(pdf_path)
            should_close = True
        
        if page_number >= len(doc):
            if should_close:
                doc.close()
            return None
        
        page = doc[page_number]
        
        # Method 1: Use "dict" format to preserve layout with spacing and newlines
        try:
            text_dict = page.get_text("dict")
            if text_dict and "blocks" in text_dict:
                result_lines = []
                prev_y = None
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                line_text_parts = []
                                prev_x = None
                                
                                for span in line["spans"]:
                                    span_text = span.get("text", "")
                                    if not span_text:
                                        continue
                                    
                                    bbox = span.get("bbox", [])
                                    if len(bbox) >= 4:
                                        x0 = bbox[0]
                                        
                                        if prev_x is not None and x0 > prev_x + 10:
                                            tabs_needed = int((x0 - prev_x) / 50)
                                            if tabs_needed > 0:
                                                line_text_parts.append("\t" * min(tabs_needed, 4))
                                        
                                        prev_x = x0 + (bbox[2] - bbox[0])
                                    
                                    line_text_parts.append(span_text)
                                
                                if line_text_parts:
                                    line_text = "".join(line_text_parts)
                                    if line_text.strip():
                                        result_lines.append(line_text)
                                        
                                        bbox = line.get("bbox", [])
                                        if len(bbox) >= 4:
                                            current_y = bbox[1]
                                            if prev_y is not None:
                                                y_gap = current_y - prev_y
                                                if y_gap > 20:
                                                    result_lines.append("")
                                            prev_y = current_y
                
                if result_lines:
                    result = "\n".join(result_lines)
                    if result.strip():
                        if should_close:
                            doc.close()
                        return result
        except Exception:
            pass
        
        # Method 2: Use "text" format - preserves newlines
        text = page.get_text("text")
        if text and text.strip():
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                cleaned = line.rstrip()
                if cleaned or cleaned_lines:
                    cleaned_lines.append(cleaned)
            result = "\n".join(cleaned_lines)
            if result.strip():
                if should_close:
                    doc.close()
                return result
        
        # Method 3: Use blocks extraction
        blocks = page.get_text("blocks")
        if blocks:
            text_lines = []
            for block in blocks:
                if len(block) > 4:
                    block_text = block[4]
                    if block_text.strip():
                        text_lines.append(block_text.rstrip())
                        text_lines.append("")
            
            if text_lines:
                result = "\n".join(text_lines)
                if result.strip():
                    if should_close:
                        doc.close()
                    return result
        
        if should_close:
            doc.close()
        return None
        
    except ImportError:
        return None
    except Exception:
        return None


@app.post("/api/pdf/extract-text", response_model=PDFTextExtractResponse)
async def extract_pdf_page_text(request: PDFTextExtractRequest):
    """Extract text from a specific PDF page with formatting preserved.
    Can optionally include text from next/previous page to complete sentences."""
    try:
        import fitz  # pymupdf
        
        # Get book from database
        books_db = BooksDBManager()
        book = books_db.get_book(request.book_id)
        
        if not book:
            raise HTTPException(status_code=404, detail=f"Book with ID {request.book_id} not found")
        
        pdf_path = Path(book['file_path'])
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
        
        # Open PDF document
        doc = fitz.open(pdf_path)
        
        # Validate page number
        if request.page_number < 0 or request.page_number >= len(doc):
            doc.close()
            raise HTTPException(
                status_code=400, 
                detail=f"Page number {request.page_number} is out of range. PDF has {len(doc)} pages."
            )
        
        # Extract text from current page
        try:
            extracted_text = extract_pdf_page_text_accurate(pdf_path, request.page_number, cached_doc=doc)
            includes_next = False
            includes_prev = False
            
            # If include_previous_page is True, get text from previous page
            if request.include_previous_page and request.page_number > 0:
                prev_text = extract_pdf_page_text_accurate(pdf_path, request.page_number - 1, cached_doc=doc)
                if prev_text and not is_sentence_complete(prev_text):
                    # Previous page has incomplete sentence, get the incomplete part
                    # Find last complete sentence in previous page
                    sentence_endings = ['.', '!', '?', '。', '！', '？']
                    last_complete_pos = -1
                    for i in range(len(prev_text) - 1, -1, -1):
                        if prev_text[i] in sentence_endings:
                            if i > 0 and prev_text[i-1].isalpha():
                                last_complete_pos = i
                                break
                    
                    if last_complete_pos >= 0:
                        incomplete_from_prev = prev_text[last_complete_pos + 1:].strip()
                        if incomplete_from_prev:
                            extracted_text = incomplete_from_prev + " " + (extracted_text or "")
                            includes_prev = True
            
            # If include_next_page is True and sentence is incomplete, get text from next page
            if request.include_next_page and extracted_text and not is_sentence_complete(extracted_text):
                if request.page_number < len(doc) - 1:
                    next_text = extract_pdf_page_text_accurate(pdf_path, request.page_number + 1, cached_doc=doc)
                    if next_text:
                        extracted_text = get_complete_sentence_text(extracted_text, next_text)
                        includes_next = True
            
        finally:
            doc.close()
        
        if extracted_text and extracted_text.strip():
            return PDFTextExtractResponse(
                success=True,
                text=extracted_text,
                message=f"Successfully extracted text from page {request.page_number + 1}" + 
                       (f" (includes next page)" if includes_next else "") +
                       (f" (includes previous page)" if includes_prev else ""),
                includes_next_page=includes_next,
                includes_previous_page=includes_prev
            )
        else:
            return PDFTextExtractResponse(
                success=False,
                text=None,
                message="No text found on this page. The page might be image-only."
            )
            
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"PDF library not available: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF text: {str(e)}")


# Translation Save Endpoint
class TranslationSaveRequest(BaseModel):
    book_id: int
    page_number: int
    translated_text: Optional[str] = None
    original_text: Optional[str] = None


class TranslationSaveResponse(BaseModel):
    success: bool
    message: str


@app.post("/api/pdf/save-translation", response_model=TranslationSaveResponse)
async def save_translation(request: TranslationSaveRequest):
    """Save translation for a specific PDF page."""
    try:
        # Get book from database
        books_db = BooksDBManager()
        book = books_db.get_book(request.book_id)
        
        if not book:
            raise HTTPException(status_code=404, detail=f"Book with ID {request.book_id} not found")
        
        # Save translation
        success = books_db.save_translation(
            book_id=request.book_id,
            page_number=request.page_number,
            original_text=request.original_text,
            translated_text=request.translated_text
        )
        
        if success:
            return TranslationSaveResponse(
                success=True,
                message=f"Translation saved for page {request.page_number + 1}"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to save translation")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving translation: {str(e)}")


class AutoTranslateRequest(BaseModel):
    book_id: int
    page_number: int
    ollama_model: str = "aya:8b"
    ollama_base_url: str = "http://localhost:11434"
    use_refinement: bool = True
    temperature: float = 0.2


class AutoTranslateResponse(BaseModel):
    success: bool
    message: str
    translated_text: Optional[str] = None
    page_number: int


@app.post("/api/pdf/auto-translate-page", response_model=AutoTranslateResponse)
async def auto_translate_page(request: AutoTranslateRequest):
    """Auto-translate a single page using Ollama."""
    import logging
    import time as time_module
    logger = logging.getLogger(__name__)
    
    start_time = time_module.time()
    page_num = request.page_number + 1
    
    logger.info(f"{'='*60}")
    logger.info(f"🔄 [API] Auto-translate request received: Book ID={request.book_id}, Page={page_num}")
    logger.info(f"📋 [API] Settings: model={request.ollama_model}, refinement={request.use_refinement}, temp={request.temperature}")
    logger.info(f"{'='*60}")
    
    try:
        from langchain_ollama import OllamaLLM
        
        # Get book from database
        books_db = BooksDBManager()
        book = books_db.get_book(request.book_id)
        
        if not book:
            raise HTTPException(status_code=404, detail=f"Book with ID {request.book_id} not found")
        
        # Get English text from the page
        pdf_path = Path(book['file_path'])
        if not pdf_path.exists():
            logger.error(f"❌ [API] PDF file not found: {pdf_path}")
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
        
        logger.info(f"📄 [Page {page_num}] Step 1: Extracting text...")
        extract_start = time_module.time()
        # Extract text from page
        english_text = extract_pdf_page_text_accurate(pdf_path, request.page_number)
        extract_time = time_module.time() - extract_start
        logger.info(f"⏱️  [Page {page_num}] Step 1 completed: Text extraction in {extract_time:.2f}s")
        
        # Handle blank pages - save empty translation and continue
        if not english_text or not english_text.strip():
            logger.warning(f"⚠️  [Page {page_num}] No text found - this is a blank page")
            logger.info(f"💾 [Page {page_num}] Saving empty translation for blank page...")
            
            # Save empty translation for blank page
            save_start = time_module.time()
            success = books_db.save_translation(
                book_id=request.book_id,
                page_number=request.page_number,
                original_text="",
                translated_text=""
            )
            save_time = time_module.time() - save_start
            
            if success:
                total_time = time_module.time() - start_time
                logger.info(f"✅ [Page {page_num}] Blank page saved successfully in {save_time:.2f}s")
                logger.info(f"✅ [Page {page_num}] COMPLETE - Blank page processed in {total_time:.2f}s")
                logger.info(f"{'='*60}")
                return AutoTranslateResponse(
                    success=True,
                    message=f"Blank page {page_num} - saved empty translation",
                    translated_text="",
                    page_number=request.page_number
                )
            else:
                total_time = time_module.time() - start_time
                logger.error(f"❌ [Page {page_num}] Failed to save blank page after {save_time:.2f}s")
                logger.error(f"❌ [Page {page_num}] FAILED after {total_time:.2f}s")
                logger.info(f"{'='*60}")
                return AutoTranslateResponse(
                    success=False,
                    message=f"Failed to save blank page {page_num}",
                    page_number=request.page_number
                )
        
        text_length = len(english_text)
        logger.info(f"✅ [Page {page_num}] Extracted {text_length} characters")
        
        # Initialize Ollama LLM
        try:
            logger.info(f"🤖 [Page {page_num}] Step 2: Initializing Ollama LLM ({request.ollama_model})...")
            init_start = time_module.time()
            llm = OllamaLLM(
                model=request.ollama_model,
                base_url=request.ollama_base_url,
                temperature=request.temperature,
                top_p=0.9,
                num_ctx=4096
            )
            init_time = time_module.time() - init_start
            logger.info(f"✅ [Page {page_num}] Step 2 completed: Ollama initialized in {init_time:.2f}s")
        except Exception as e:
            total_time = time_module.time() - start_time
            logger.error(f"❌ [Page {page_num}] Step 2 FAILED: Ollama initialization error after {total_time:.2f}s: {str(e)}")
            logger.error(f"❌ [Page {page_num}] FAILED during Ollama initialization")
            logger.info(f"{'='*60}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Ollama model: {str(e)}")
        
        # Translation prompt
        if request.use_refinement:
            logger.info(f"🔄 [Page {page_num}] Step 3: Starting two-step translation (refinement enabled)")
            
            # Step 1: Initial translation
            initial_prompt = f"""Translate the following English text to Hindi. Focus on accuracy and meaning.

English text:
{english_text}

Provide the Hindi translation:"""
            
            logger.info(f"📝 [Page {page_num}] Step 3a: Calling Ollama for initial translation (may take 10-30s)...")
            translate_start = time_module.time()
            initial_translation = llm.invoke(initial_prompt).strip()
            translate_time = time_module.time() - translate_start
            logger.info(f"✅ [Page {page_num}] Step 3a completed: Initial translation in {translate_time:.2f}s ({len(initial_translation)} chars)")
            
            # Step 2: Refinement
            refinement_prompt = f"""You are refining a Hindi translation to make it more natural, fluent, and accurate.

ORIGINAL ENGLISH TEXT:
{english_text}

CURRENT HINDI TRANSLATION:
{initial_translation}

TASK: Refine this translation to:
1. Make it sound more natural and fluent in Hindi
2. Improve word choice and expressions
3. Ensure proper grammar and sentence structure
4. Enhance readability and flow
5. Fix any awkward phrasings or literal translations

Output ONLY the refined Hindi translation, no explanations:"""
            
            logger.info(f"✨ [Page {page_num}] Step 3b: Calling Ollama for refinement (may take 10-30s)...")
            refine_start = time_module.time()
            hindi_translation = llm.invoke(refinement_prompt).strip()
            refine_time = time_module.time() - refine_start
            logger.info(f"✅ [Page {page_num}] Step 3b completed: Refinement in {refine_time:.2f}s ({len(hindi_translation)} chars)")
        else:
            # Single-step translation
            translation_prompt = f"""You are a professional translator with native-level proficiency in both English and Hindi. Your expertise includes understanding cultural nuances, idiomatic expressions, and context-dependent meanings.

TASK: Translate the following English text into natural, fluent Hindi that reads as if it were originally written in Hindi.

TRANSLATION GUIDELINES:
1. **Context Understanding**: Analyze the full context, not individual words. Understand the intended meaning, tone, and purpose.
2. **Natural Flow**: The translation should flow naturally in Hindi, using appropriate sentence structures and word order.
3. **Cultural Adaptation**: Adapt cultural references, idioms, and expressions to be meaningful in Hindi context.
4. **Vocabulary Selection**: Choose the most appropriate Hindi words that convey the exact meaning and register (formal/informal).
5. **Grammar & Syntax**: Use correct Hindi grammar, including proper use of matras (diacritics), verb conjugations, and case markers.
6. **Technical Terms**: For technical terms, use standard Hindi translations when available, or transliterate appropriately.
7. **Proper Nouns**: Keep names, places, and brand names as-is or use common Hindi transliterations.
8. **Tone Preservation**: Maintain the original tone (formal, casual, technical, narrative, etc.).

ORIGINAL ENGLISH TEXT:
{english_text}

TRANSLATION REQUIREMENTS:
- Output ONLY the Hindi translation
- No explanations, notes, or additional text
- Preserve paragraph breaks and formatting
- Use proper Devanagari script

Hindi Translation:"""
            
            logger.info(f"📝 [Page {page_num}] Step 3: Calling Ollama for translation (may take 10-30s)...")
            translate_start = time_module.time()
            hindi_translation = llm.invoke(translation_prompt).strip()
            translate_time = time_module.time() - translate_start
            logger.info(f"✅ [Page {page_num}] Step 3 completed: Translation in {translate_time:.2f}s ({len(hindi_translation)} chars)")
        
        # Clean up translation (remove prefixes)
        logger.info(f"🧹 [Page {page_num}] Step 4: Cleaning up translation text...")
        prefixes_to_remove = [
            "Hindi translation:", "Translation:", "Here is the translation:",
            "The Hindi translation is:", "हिंदी अनुवाद:", "अनुवाद:",
            "Hindi Translation:", "TRANSLATION:"
        ]
        for prefix in prefixes_to_remove:
            if hindi_translation.lower().startswith(prefix.lower()):
                hindi_translation = hindi_translation[len(prefix):].strip()
        logger.info(f"✅ [Page {page_num}] Step 4 completed: Translation cleaned ({len(hindi_translation)} chars)")
        
        # Save translation
        logger.info(f"💾 [Page {page_num}] Step 5: Saving translation to database...")
        save_start = time_module.time()
        success = books_db.save_translation(
            book_id=request.book_id,
            page_number=request.page_number,
            original_text=english_text,
            translated_text=hindi_translation
        )
        save_time = time_module.time() - save_start
        
        if success:
            total_time = time_module.time() - start_time
            logger.info(f"✅ [Page {page_num}] Step 5 completed: Translation saved in {save_time:.2f}s")
            logger.info(f"✅ [Page {page_num}] COMPLETE - Processed successfully in {total_time:.2f}s")
            if request.use_refinement:
                logger.info(f"   📊 Breakdown: Extract={extract_time:.2f}s, Translate={translate_time:.2f}s, Refine={refine_time:.2f}s, Save={save_time:.2f}s")
            else:
                logger.info(f"   📊 Breakdown: Extract={extract_time:.2f}s, Translate={translate_time:.2f}s, Save={save_time:.2f}s")
            logger.info(f"{'='*60}")
            return AutoTranslateResponse(
                success=True,
                message=f"Page {page_num} translated and saved",
                translated_text=hindi_translation,
                page_number=request.page_number
            )
        else:
            total_time = time_module.time() - start_time
            logger.error(f"❌ [Page {page_num}] Step 5 FAILED: Save error after {save_time:.2f}s")
            logger.error(f"❌ [Page {page_num}] FAILED during save after {total_time:.2f}s")
            logger.info(f"{'='*60}")
            raise HTTPException(status_code=400, detail="Failed to save translation")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error auto-translating page: {str(e)}")


# Background Translation Job Endpoints
class StartTranslationJobRequest(BaseModel):
    book_id: int
    ollama_model: str = "aya:8b"
    ollama_base_url: str = "http://localhost:11434"
    use_refinement: bool = True
    temperature: float = 0.2
    parallel_workers: int = 10


class StartTranslationJobResponse(BaseModel):
    success: bool
    message: str
    job_id: int


class TranslationJobProgressResponse(BaseModel):
    success: bool
    job_id: int
    book_id: int
    status: str
    total_pages: int
    completed_pages: int
    failed_pages: int
    avg_time_per_page: float
    estimated_time_remaining: float
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


@app.post("/api/pdf/start-translation-job", response_model=StartTranslationJobResponse)
async def start_translation_job(request: StartTranslationJobRequest):
    """Start a background translation job for a book."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        books_db = BooksDBManager()
        
        # Check if book exists
        book = books_db.get_book(request.book_id)
        if not book:
            raise HTTPException(status_code=404, detail=f"Book with ID {request.book_id} not found")
        
        # Check if there's already an active job for this book
        active_job = books_db.get_active_job_for_book(request.book_id)
        if active_job:
            return StartTranslationJobResponse(
                success=True,
                message=f"Translation job already running (Job ID: {active_job['id']})",
                job_id=active_job['id']
            )
        
        # Get total pages
        pdf_path = Path(book['file_path'])
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
        
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        if total_pages == 0:
            raise HTTPException(status_code=400, detail="PDF has no pages")
        
        # Create job in database
        job_id = books_db.create_translation_job(
            book_id=request.book_id,
            total_pages=total_pages,
            ollama_model=request.ollama_model,
            ollama_base_url=request.ollama_base_url,
            use_refinement=request.use_refinement,
            temperature=request.temperature,
            parallel_workers=request.parallel_workers
        )
        
        # Start background job
        translation_job_manager.start_translation_job(
            job_id=job_id,
            book_id=request.book_id,
            total_pages=total_pages,
            ollama_model=request.ollama_model,
            ollama_base_url=request.ollama_base_url,
            use_refinement=request.use_refinement,
            temperature=request.temperature,
            parallel_workers=request.parallel_workers
        )
        
        logger.info(f"✅ Started translation job {job_id} for book {request.book_id} ({total_pages} pages)")
        
        return StartTranslationJobResponse(
            success=True,
            message=f"Translation job started (Job ID: {job_id})",
            job_id=job_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting translation job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting translation job: {str(e)}")


@app.get("/api/pdf/translation-job-progress/{job_id}", response_model=TranslationJobProgressResponse)
async def get_translation_job_progress(job_id: int):
    """Get progress for a translation job."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        progress = translation_job_manager.get_job_progress(job_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail=f"Translation job {job_id} not found")
        
        return TranslationJobProgressResponse(**progress)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting job progress: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting job progress: {str(e)}")


@app.get("/api/pdf/translation-job-for-book/{book_id}", response_model=TranslationJobProgressResponse)
async def get_translation_job_for_book(book_id: int):
    """Get active translation job for a book."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        books_db = BooksDBManager()
        job = books_db.get_active_job_for_book(book_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"No active translation job found for book {book_id}")
        
        progress = translation_job_manager.get_job_progress(job['id'])
        
        if not progress:
            # Job exists in DB but not in memory - check if it's stuck
            is_actually_running = translation_job_manager.is_job_running(job['id'])
            if not is_actually_running and job['status'] in ['running', 'pending']:
                # Job is stuck - mark status as 'stuck' for frontend
                status = 'stuck'
            else:
                status = job['status']
            
            return TranslationJobProgressResponse(
                success=True,
                job_id=job['id'],
                book_id=job['book_id'],
                status=status,
                total_pages=job['total_pages'],
                completed_pages=job.get('completed_pages', 0),
                failed_pages=job.get('failed_pages', 0),
                avg_time_per_page=job.get('avg_time_per_page') or 0.0,
                estimated_time_remaining=job.get('estimated_time_remaining') or 0.0,
                started_at=str(job.get('started_at')) if job.get('started_at') else None,
                completed_at=str(job.get('completed_at')) if job.get('completed_at') else None,
                error_message=job.get('error_message')
            )
        
        # Check if job is actually running
        is_actually_running = translation_job_manager.is_job_running(job['id'])
        if not is_actually_running and progress['status'] in ['running', 'pending']:
            progress['status'] = 'stuck'
        
        return TranslationJobProgressResponse(**progress)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting job for book: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting job for book: {str(e)}")


@app.post("/api/pdf/resume-translation-job/{job_id}")
async def resume_translation_job(job_id: int):
    """Resume a stuck translation job."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        success = translation_job_manager.resume_job(job_id)
        if success:
            return {"success": True, "message": f"Translation job {job_id} resumed successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Could not resume job {job_id}. It may already be running or have an invalid status.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error resuming translation job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error resuming translation job: {str(e)}")


@app.post("/api/pdf/restart-translation-job/{job_id}")
async def restart_translation_job(job_id: int):
    """Restart a translation job from the beginning, clearing all existing translations."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        books_db = BooksDBManager()
        
        # Get job info
        job = books_db.get_translation_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        book_id = job['book_id']
        
        # Cancel/stop the current job if running
        translation_job_manager.cancel_job(job_id)
        
        # Delete all existing translations for this book
        logger.info(f"🗑️ Deleting all translations for book {book_id}")
        import sqlite3
        conn = sqlite3.connect(books_db.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM book_translations WHERE book_id = ?", (book_id,))
        conn.commit()
        conn.close()
        
        # Mark old job as cancelled
        books_db.complete_job(job_id, 'cancelled', 'Restarted by user')
        
        # Create a new job
        new_job_id = books_db.create_translation_job(
            book_id=book_id,
            total_pages=job['total_pages'],
            ollama_model=job['ollama_model'],
            ollama_base_url=job['ollama_base_url'],
            use_refinement=job['use_refinement'],
            temperature=job['temperature'],
            parallel_workers=job.get('parallel_workers', 10)
        )
        
        # Start the new job
        translation_job_manager.start_translation_job(
            job_id=new_job_id,
            book_id=book_id,
            total_pages=job['total_pages'],
            ollama_model=job['ollama_model'],
            ollama_base_url=job['ollama_base_url'],
            use_refinement=job['use_refinement'],
            temperature=job['temperature'],
            parallel_workers=job.get('parallel_workers', 10)
        )
        
        logger.info(f"✅ Restarted translation job: old={job_id}, new={new_job_id} for book {book_id}")
        
        return {
            "success": True,
            "message": f"Translation job restarted from beginning",
            "old_job_id": job_id,
            "new_job_id": new_job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error restarting translation job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error restarting translation job: {str(e)}")


@app.post("/api/pdf/cancel-translation-job/{job_id}")
async def cancel_translation_job(job_id: int):
    """Cancel a running translation job."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        success = translation_job_manager.cancel_job(job_id)
        
        if success:
            return {"success": True, "message": f"Job {job_id} cancelled"}
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not running")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error cancelling job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")


class PDFGenerateRequest(BaseModel):
    book_id: int


@app.post("/api/pdf/generate-translated-pdf")
async def generate_translated_pdf(request: PDFGenerateRequest):
    """Generate a PDF file with all translated text for a book."""
    try:
        # Get book from database
        books_db = BooksDBManager()
        book = books_db.get_book(request.book_id)
        
        if not book:
            raise HTTPException(status_code=404, detail=f"Book with ID {request.book_id} not found")
        
        # Get all translations
        translations = books_db.get_all_translations(request.book_id)
        
        if not translations:
            raise HTTPException(status_code=400, detail="No translations found for this book")
        
        # Filter translations that have translated_text
        translations_with_text = [
            t for t in translations 
            if t.get('translated_text') and t.get('translated_text').strip()
        ]
        
        if not translations_with_text:
            raise HTTPException(status_code=400, detail="No translated text found for this book")
        
        # Sort by page number
        translations_with_text.sort(key=lambda x: x.get('page_number', 0))
        
        book_title = book.get('title', f"Book {request.book_id}")
        safe_title = "".join(c for c in book_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title}_translated.pdf"
        
        # Try different PDF libraries in order of preference for Hindi/Unicode support
        USE_WEASYPRINT = False
        USE_XHTML2PDF = False
        USE_REPORTLAB = False
        
        # First try weasyprint - best Unicode/Devanagari support via HTML/CSS rendering
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            USE_WEASYPRINT = True
        except ImportError:
            pass
        
        # Second try xhtml2pdf (pisa) - good Unicode support
        if not USE_WEASYPRINT:
            try:
                from xhtml2pdf import pisa
                USE_XHTML2PDF = True
            except ImportError:
                pass
        
        # Last resort: reportlab
        if not USE_WEASYPRINT and not USE_XHTML2PDF:
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
                from reportlab.lib.enums import TA_LEFT, TA_CENTER
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont
                from reportlab.pdfbase.cidfonts import UnicodeCIDFont
                USE_REPORTLAB = True
            except ImportError:
                USE_REPORTLAB = False
        
        if USE_WEASYPRINT:
            # Generate PDF using weasyprint - excellent Unicode/Devanagari support
            # Build HTML content with proper CSS for Hindi text
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    @page {{
                        size: A4;
                        margin: 2cm;
                    }}
                    body {{
                        font-family: 'Noto Sans Devanagari', 'Lohit Devanagari', 'DejaVu Sans', sans-serif;
                        font-size: 11pt;
                        line-height: 1.6;
                        color: #000;
                    }}
                    h1 {{
                        font-size: 18pt;
                        text-align: center;
                        margin-bottom: 20px;
                        color: #000;
                    }}
                    h2 {{
                        font-size: 14pt;
                        color: #333;
                        margin-top: 30px;
                        margin-bottom: 10px;
                    }}
                    .page-content {{
                        margin-bottom: 30px;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }}
                    .page-header {{
                        font-size: 14pt;
                        font-weight: bold;
                        color: #333;
                        margin-top: 20px;
                        margin-bottom: 12px;
                    }}
                </style>
            </head>
            <body>
                <h1>{book_title}</h1>
                <h2>Translated Text</h2>
            """
            
            # Add translations page by page
            for trans in translations_with_text:
                page_num = trans.get('page_number', 0) + 1
                translated_text = trans.get('translated_text', '').strip()
                
                if translated_text:
                    # Escape HTML but preserve Unicode
                    import html as html_escape
                    # Convert markdown to HTML
                    import re
                    formatted_text = translated_text
                    # Convert **bold** to <strong>bold</strong>
                    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_text)
                    # Convert *italic* to <em>italic</em>
                    formatted_text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', formatted_text)
                    # Convert line breaks
                    formatted_text = formatted_text.replace('\n', '<br/>')
                    # Escape HTML special characters in text (but keep our tags)
                    parts = re.split(r'(<[^>]+>)', formatted_text)
                    escaped_parts = []
                    for part in parts:
                        if part.startswith('<') and part.endswith('>'):
                            escaped_parts.append(part)
                        else:
                            escaped_parts.append(html_escape.escape(part))
                    formatted_text = ''.join(escaped_parts)
                    
                    html_content += f"""
                    <div class="page-header">Page {page_num}</div>
                    <div class="page-content">{formatted_text}</div>
                    """
            
            html_content += """
            </body>
            </html>
            """
            
            # Generate PDF
            buffer = io.BytesIO()
            HTML(string=html_content).write_pdf(buffer)
            buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(buffer.read()),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
        
        elif USE_XHTML2PDF:
            # Generate PDF using xhtml2pdf (pisa) - good Unicode support
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    @page {{
                        size: A4;
                        margin: 2cm;
                    }}
                    body {{
                        font-family: 'Noto Sans Devanagari', 'Lohit Devanagari', 'DejaVu Sans', sans-serif;
                        font-size: 11pt;
                        line-height: 1.6;
                        color: #000;
                    }}
                    h1 {{
                        font-size: 18pt;
                        text-align: center;
                        margin-bottom: 20px;
                    }}
                    h2 {{
                        font-size: 14pt;
                        color: #333;
                        margin-top: 30px;
                        margin-bottom: 10px;
                    }}
                    .page-content {{
                        margin-bottom: 30px;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }}
                    .page-header {{
                        font-size: 14pt;
                        font-weight: bold;
                        color: #333;
                        margin-top: 20px;
                        margin-bottom: 12px;
                    }}
                </style>
            </head>
            <body>
                <h1>{book_title}</h1>
                <h2>Translated Text</h2>
            """
            
            # Add translations page by page
            for trans in translations_with_text:
                page_num = trans.get('page_number', 0) + 1
                translated_text = trans.get('translated_text', '').strip()
                
                if translated_text:
                    # Convert markdown to HTML
                    import re
                    import html as html_escape
                    formatted_text = translated_text
                    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_text)
                    formatted_text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', formatted_text)
                    formatted_text = formatted_text.replace('\n', '<br/>')
                    # Escape HTML
                    parts = re.split(r'(<[^>]+>)', formatted_text)
                    escaped_parts = []
                    for part in parts:
                        if part.startswith('<') and part.endswith('>'):
                            escaped_parts.append(part)
                        else:
                            escaped_parts.append(html_escape.escape(part))
                    formatted_text = ''.join(escaped_parts)
                    
                    html_content += f"""
                    <div class="page-header">Page {page_num}</div>
                    <div class="page-content">{formatted_text}</div>
                    """
            
            html_content += """
            </body>
            </html>
            """
            
            # Generate PDF
            buffer = io.BytesIO()
            pisa.CreatePDF(html_content, dest=buffer, encoding='utf-8')
            buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(buffer.read()),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
        
        elif USE_REPORTLAB:
            # Generate PDF using reportlab with Unicode support
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, 
                                   rightMargin=72, leftMargin=72,
                                   topMargin=72, bottomMargin=72)
            
            # Register Unicode fonts for Hindi/Devanagari support with proper matra positioning
            # We need a font specifically designed for Devanagari script
            try:
                import platform
                system = platform.system()
                
                hindi_font_registered = False
                font_name = 'Helvetica'  # Default fallback
                
                if system == "Linux":
                    # Try common Hindi fonts on Linux - prioritize Noto Sans Devanagari for best matra support
                    # Noto Sans Devanagari is specifically designed for proper matra positioning
                    font_paths = []
                    
                    # First, try to find NotoSansDevanagari-Regular.ttf (best for Hindi matras)
                    import subprocess
                    try:
                        result = subprocess.run(
                            ['find', '/usr/share/fonts', '-name', 'NotoSansDevanagari-Regular.ttf'],
                            capture_output=True, text=True, timeout=2
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            font_paths.append(result.stdout.strip().split('\n')[0])
                    except:
                        pass
                    
                    # Add other common paths
                    font_paths.extend([
                        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf",
                        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
                        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    ])
                    
                    # Also check in user's home directory
                    home_font_paths = [
                        os.path.expanduser("~/.fonts/NotoSansDevanagari-Regular.ttf"),
                        os.path.expanduser("~/.local/share/fonts/NotoSansDevanagari-Regular.ttf"),
                    ]
                    font_paths.extend(home_font_paths)
                    
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            try:
                                pdfmetrics.registerFont(TTFont('HindiFont', font_path))
                                hindi_font_registered = True
                                font_name = 'HindiFont'
                                print(f"Successfully registered Hindi font: {font_path}")
                                break
                            except Exception as e:
                                print(f"Failed to register font {font_path}: {e}")
                                continue
                
                # If no system font found, try UnicodeCIDFont which has better Devanagari support
                if not hindi_font_registered:
                    try:
                        # Try different CID fonts that support Devanagari
                        cid_fonts = ['HeiseiKakuGo-W5', 'HeiseiMin-W3', 'KozMinPro-Regular']
                        for cid_font in cid_fonts:
                            try:
                                pdfmetrics.registerFont(UnicodeCIDFont(cid_font))
                                hindi_font_registered = True
                                font_name = cid_font
                                print(f"Successfully registered CID font: {cid_font}")
                                break
                            except:
                                continue
                    except Exception as e:
                        print(f"Failed to register CID font: {e}")
                
                # Final fallback - use Helvetica (won't support Hindi well, but won't crash)
                if not hindi_font_registered:
                    print("Warning: No Hindi-supporting font found. PDF may not display Hindi correctly.")
                    font_name = 'Helvetica'
                    
            except Exception as e:
                # Fallback to basic font
                font_name = 'Helvetica'
                print(f"Warning: Could not register Hindi font: {e}")
            
            # Container for the 'Flowable' objects
            story = []
            
            # Define styles with Unicode font
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=18,
                textColor='#000000',
                spaceAfter=30,
                alignment=TA_CENTER
            )
            page_header_style = ParagraphStyle(
                'PageHeader',
                parent=styles['Heading2'],
                fontName=font_name,
                fontSize=14,
                textColor='#333333',
                spaceAfter=12,
                spaceBefore=20
            )
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=11,
                leading=14,
                alignment=TA_LEFT,
                spaceAfter=12
            )
            
            # Add title
            book_title = book.get('title', f"Book {request.book_id}")
            story.append(Paragraph(book_title, title_style))
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Translated Text", styles['Heading2']))
            story.append(Spacer(1, 0.3*inch))
            
            # Add translations page by page
            for trans in translations_with_text:
                page_num = trans.get('page_number', 0) + 1  # Convert to 1-indexed
                translated_text = trans.get('translated_text', '').strip()
                
                if translated_text:
                    # Add page header
                    story.append(Paragraph(f"Page {page_num}", page_header_style))
                    
                    # Convert markdown-like formatting to HTML for reportlab
                    # Simple markdown to HTML conversion
                    import re
                    from xml.sax.saxutils import escape
                    
                    formatted_text = translated_text
                    
                    # First convert markdown to HTML tags (before escaping)
                    # Convert **bold** to <b>bold</b>
                    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_text)
                    # Convert *italic* to <i>italic</i> (but not if it's part of **)
                    formatted_text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', formatted_text)
                    
                    # Convert line breaks
                    formatted_text = formatted_text.replace('\n', '<br/>')
                    
                    # Now escape HTML special characters in the text content (but preserve our HTML tags)
                    # We need to escape &, <, > but keep our HTML tags intact
                    # Split by HTML tags, escape text parts, then rejoin
                    parts = re.split(r'(<[^>]+>)', formatted_text)
                    escaped_parts = []
                    for part in parts:
                        if part.startswith('<') and part.endswith('>'):
                            # It's an HTML tag, keep it as is
                            escaped_parts.append(part)
                        else:
                            # It's text content, escape HTML special chars but preserve Unicode
                            # Use escape() which preserves Unicode characters
                            escaped_parts.append(escape(part))
                    formatted_text = ''.join(escaped_parts)
                    
                    # Ensure the text is properly encoded as UTF-8 for reportlab
                    # reportlab's Paragraph should handle Unicode, but we ensure it's clean
                    try:
                        # Verify it's valid UTF-8
                        formatted_text.encode('utf-8')
                    except UnicodeEncodeError:
                        # If encoding fails, replace problematic characters
                        formatted_text = formatted_text.encode('utf-8', 'replace').decode('utf-8')
                    
                    # Add translated text - Paragraph handles Unicode automatically
                    # The font we registered (Noto Sans Devanagari) should properly render matras
                    story.append(Paragraph(formatted_text, body_style))
                    story.append(Spacer(1, 0.2*inch))
                    story.append(PageBreak())
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(buffer.read()),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
        else:
            # Fallback: Generate simple text-based PDF using fpdf2 (Unicode support)
            try:
                # Try fpdf2 which has better Unicode support
                try:
                    from fpdf import FPDF
                    USE_FPDF2 = False
                except:
                    from fpdf2 import FPDF
                    USE_FPDF2 = True
                
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                
                # Add pages
                book_title = book.get('title', f"Book {request.book_id}")
                
                for trans in translations_with_text:
                    page_num = trans.get('page_number', 0) + 1
                    translated_text = trans.get('translated_text', '').strip()
                    
                    if translated_text:
                        pdf.add_page()
                        
                        # Try to use a font that supports Hindi
                        try:
                            # fpdf2 supports Unicode better
                            if USE_FPDF2:
                                pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
                                pdf.set_font('DejaVu', '', 11)
                            else:
                                pdf.set_font("Arial", "", 11)
                        except:
                            # Fallback to default font
                            try:
                                pdf.set_font("Arial", "", 11)
                            except:
                                pass
                        
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(0, 10, f"Page {page_num}", ln=1)
                        pdf.ln(5)
                        
                        try:
                            if USE_FPDF2:
                                pdf.set_font('DejaVu', '', 11)
                            else:
                                pdf.set_font("Arial", "", 11)
                        except:
                            pdf.set_font("Arial", "", 11)
                        
                        # Split text into lines and add them
                        lines = translated_text.split('\n')
                        for line in lines:
                            if line.strip():
                                # For Unicode support, use unicode encoding
                                try:
                                    if USE_FPDF2:
                                        pdf.multi_cell(0, 5, line)
                                    else:
                                        # Try to handle Unicode
                                        pdf.multi_cell(0, 5, line.encode('utf-8', 'replace').decode('utf-8', 'replace'))
                                except:
                                    # Last resort: replace unsupported characters
                                    safe_line = line.encode('ascii', 'replace').decode('ascii')
                                    pdf.multi_cell(0, 5, safe_line)
                                pdf.ln(2)
                
                # Save to buffer
                buffer = io.BytesIO()
                pdf.output(buffer)
                buffer.seek(0)
                
                safe_title = "".join(c for c in book_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{safe_title}_translated.pdf"
                
                return StreamingResponse(
                    io.BytesIO(buffer.read()),
                    media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'}
                )
            except ImportError:
                raise HTTPException(
                    status_code=500, 
                    detail="PDF generation libraries not available. Please install one of: 'weasyprint' (recommended for Hindi), 'xhtml2pdf', or 'reportlab'. Install with: pip install weasyprint"
                )
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}\n{traceback.format_exc()}")


class PWAPackageRequest(BaseModel):
    book_id: int
    base_url: Optional[str] = "https://blog.priorcoder.com/books"


@app.post("/api/pwa/generate-package")
async def generate_pwa_package(request: PWAPackageRequest):
    """Generate a PWA package (HTML/JS/CSS/JSON) for reading translated books."""
    try:
        # Get book from database
        books_db = BooksDBManager()
        book = books_db.get_book(request.book_id)
        
        if not book:
            raise HTTPException(status_code=404, detail=f"Book with ID {request.book_id} not found")
        
        # Get all translations
        translations = books_db.get_all_translations(request.book_id)
        
        if not translations:
            raise HTTPException(status_code=400, detail="No translations found for this book")
        
        # Filter and sort translations
        translations_with_text = [
            t for t in translations 
            if t.get('translated_text') and t.get('translated_text').strip()
        ]
        
        if not translations_with_text:
            raise HTTPException(status_code=400, detail="No translated text found for this book")
        
        translations_with_text.sort(key=lambda x: x.get('page_number', 0))
        
        book_title = book.get('title', f"Book {request.book_id}")
        book_author = book.get('author', '')
        
        # Create safe folder name
        safe_folder_name = re.sub(r'[^a-zA-Z0-9_-]', '_', book_title).strip('_')[:50]
        if not safe_folder_name:
            safe_folder_name = f"book_{request.book_id}"
        
        # Prepare book data as JSON
        book_data = {
            "id": request.book_id,
            "title": book_title,
            "author": book_author,
            "total_pages": len(translations_with_text),
            "pages": []
        }
        
        for trans in translations_with_text:
            page_num = trans.get('page_number', 0)
            translated_text = trans.get('translated_text', '').strip()
            if translated_text:
                book_data["pages"].append({
                    "page_number": page_num,
                    "content": translated_text
                })
        
        # Create ZIP file in memory
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add book data JSON
            zip_file.writestr('book-data.json', json.dumps(book_data, ensure_ascii=False, indent=2))
            
            # Note: dashboard is now index.html in the root books folder, not in individual book folders
            # Individual book folders only contain the reader (index.html)
            
            # Add index.html (reader)
            index_html = generate_reader_html(book_title, book_author, request.book_id, safe_folder_name)
            zip_file.writestr('index.html', index_html)
            
            # Add manifest.json
            manifest_json = generate_manifest_json(book_title, safe_folder_name)
            zip_file.writestr('manifest.json', manifest_json)
            
            # Add service-worker.js
            service_worker_js = generate_service_worker_js()
            zip_file.writestr('service-worker.js', service_worker_js)
            
            # Add styles.css
            styles_css = generate_reader_css()
            zip_file.writestr('styles.css', styles_css)
            
            # Add app.js
            app_js = generate_app_js()
            zip_file.writestr('app.js', app_js)
            
            # Add book-info.json for book metadata (used by dashboard to discover books)
            book_info = {
                "id": request.book_id,
                "title": book_title,
                "author": book_author,
                "folder": safe_folder_name,
                "total_pages": len(translations_with_text),
                "created_at": datetime.now().isoformat()
            }
            zip_file.writestr('book-info.json', json.dumps(book_info, ensure_ascii=False, indent=2))
            
            # Add README.txt
            readme_txt = f"""PWA Book Reader Package
========================

Book: {book_title}
Author: {book_author}

Installation:
1. Extract this ZIP file to your web server
2. Place it in: {request.base_url}/{safe_folder_name}/
3. Access via: {request.base_url}/{safe_folder_name}/index.html

Files:
- index.html: Book reader (individual book)
- manifest.json: PWA manifest
- service-worker.js: Service worker for offline support
- styles.css: Reader styling
- app.js: Main application logic
- book-data.json: Book content (page-wise)
- book-info.json: Book metadata (for auto-discovery)

The app will automatically:
- Store book data in IndexedDB for fast reading
- Save reading progress in browser storage
- Work offline after first load
- Show reading progress on dashboard
- Auto-discover new books from server
"""
            zip_file.writestr('README.txt', readme_txt)
        
        buffer.seek(0)
        filename = f"{safe_folder_name}_pwa.zip"
        
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error generating PWA package: {str(e)}\n{traceback.format_exc()}")


@app.get("/api/pwa/list-books")
async def list_pwa_books(base_url: str = "https://blog.priorcoder.com/books"):
    """List all available PWA books by scanning the books directory.
    
    This endpoint should be called from the dashboard to auto-discover books.
    It returns books from the database that have been generated as PWA packages (have translations).
    """
    try:
        books_db = BooksDBManager()
        all_books = books_db.list_books()
        
        # Filter books that likely have PWA packages (have translations)
        books_with_pwa = []
        for book in all_books:
            translations = books_db.get_all_translations(book['id'])
            if translations and any(t.get('translated_text') for t in translations):
                # Create safe folder name (same logic as in generate_pwa_package)
                book_title = book.get('title', f"Book {book['id']}")
                safe_folder_name = re.sub(r'[^a-zA-Z0-9_-]', '_', book_title).strip('_')[:50]
                if not safe_folder_name:
                    safe_folder_name = f"book_{book['id']}"
                
                books_with_pwa.append({
                    "id": book['id'],
                    "title": book_title,
                    "author": book.get('author', ''),
                    "folder": safe_folder_name,
                    "url": f"{base_url}/{safe_folder_name}/",
                    "total_pages": len([t for t in translations if t.get('translated_text')])
                })
        
        return {
            "success": True,
            "books": books_with_pwa,
            "base_url": base_url
        }
            
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error listing books: {str(e)}\n{traceback.format_exc()}")


@app.get("/api/pwa/generate-dashboard")
async def generate_dashboard_file(base_url: str = "https://blog.priorcoder.com/books"):
    """Generate a standalone index.html file for the books root folder.
    
    This dashboard will be placed at: {base_url}/index.html
    It will automatically discover and list all available books by scanning
    subdirectories for book-info.json files (no API server required).
    
    Also generates a books-list.json file with all known books (optional, for faster loading).
    """
    try:
        dashboard_html = generate_dashboard_html(base_url)
        
        # Also generate a books-list.json file with all books (optional, for faster loading)
        books_db = BooksDBManager()
        all_books = books_db.list_books()
        
        books_with_info = []
        for book in all_books:
            translations = books_db.get_all_translations(book['id'])
            translations_with_text = [
                t for t in translations 
                if t.get('translated_text') and t.get('translated_text').strip()
            ]
            
            if translations_with_text:
                book_title = book.get('title', f"Book {book['id']}")
                safe_folder_name = re.sub(r'[^a-zA-Z0-9_-]', '_', book_title).strip('_')[:50]
                if not safe_folder_name:
                    safe_folder_name = f"book_{book['id']}"
                
                books_with_info.append({
                    "id": book['id'],
                    "title": book_title,
                    "author": book.get('author', ''),
                    "folder": safe_folder_name,
                    "url": f"{base_url}/{safe_folder_name}/",
                    "total_pages": len(translations_with_text)
                })
        
        books_list_json = json.dumps({
            "success": True,
            "books": books_with_info,
            "base_url": base_url,
            "generated_at": datetime.now().isoformat()
        }, ensure_ascii=False, indent=2)
        
        # Generate manifest.json for dashboard
        dashboard_manifest_json = json.dumps({
            "name": "Book Reader - Library",
            "short_name": "Book Library",
            "description": "Library dashboard for reading translated books",
            "start_url": "./index.html",
            "display": "standalone",
            "background_color": "#667eea",
            "theme_color": "#2c3e50",
            "orientation": "portrait",
            "icons": [
                {
                    "src": "icon-192.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "icon-512.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]
        }, indent=2)
        
        # Generate service worker for dashboard (dashboard-specific)
        dashboard_service_worker = """const CACHE_NAME = 'book-library-dashboard-v1';
const urlsToCache = [
    './',
    './index.html',
    './manifest.json',
    './books-list.json'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                return response || fetch(event.request);
            })
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});"""
        
        # Return all files as a ZIP
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('index.html', dashboard_html)
            zip_file.writestr('books-list.json', books_list_json)
            zip_file.writestr('manifest.json', dashboard_manifest_json)
            zip_file.writestr('service-worker.js', dashboard_service_worker)
            zip_file.writestr('README.txt', f"""Dashboard Package
==================

Files:
- index.html: Main dashboard (lists all books)
- books-list.json: Pre-generated book list (optional, for faster loading)
- manifest.json: PWA manifest file
- service-worker.js: Service worker for offline support

Installation:
1. Extract all files to your web server
2. Place them in: {base_url}/
3. Access via: {base_url}/index.html

The dashboard will:
- First try to load books from books-list.json (if available)
- Fall back to scanning subdirectories for book-info.json files
- Work completely offline with static files only (no API server needed)

To add new books:
- Upload book folders to subdirectories
- Each folder should contain book-info.json
- Click "Refresh Books" on the dashboard to discover new books

Note: You can optionally add icon-192.png and icon-512.png for PWA icons.
""")
        
        buffer.seek(0)
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="dashboard.zip"'}
        )
            
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}\n{traceback.format_exc()}")


def generate_dashboard_html(base_url: str) -> str:
    """Generate dashboard HTML that lists available books."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#2c3e50">
    <title>Book Reader - Dashboard</title>
    <link rel="manifest" href="manifest.json">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .books-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .book-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }
        
        .book-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        
        .book-title {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        
        .book-author {
            color: #7f8c8d;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        
        .book-progress {
            margin-top: 15px;
        }
        
        .progress-bar {
            height: 6px;
            background: #ecf0f1;
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 5px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
        }
        
        .progress-text {
            font-size: 0.85em;
            color: #7f8c8d;
        }
        
        .no-books {
            text-align: center;
            color: white;
            padding: 40px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            margin-top: 30px;
        }
        
        .loading {
            text-align: center;
            color: white;
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📚 My Library</h1>
            <p class="subtitle">Continue reading your books</p>
        </header>
        
        <div id="books-container" class="loading">
            <p>Loading books...</p>
        </div>
        
        <div style="text-align: center; margin-top: 30px; display: flex; gap: 10px; justify-content: center;">
            <button onclick="loadBooks()" style="padding: 10px 20px; background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3); border-radius: 8px; cursor: pointer; font-size: 0.9em;">
                🔄 Refresh Books
            </button>
            <button onclick="clearCache()" style="padding: 10px 20px; background: rgba(255,152,0,0.3); color: white; border: 1px solid rgba(255,152,0,0.5); border-radius: 8px; cursor: pointer; font-size: 0.9em;">
                🗑️ Clear Cache
            </button>
        </div>
    </div>
    
    <script>
        // Define base_url for this dashboard
        const base_url = '{BASE_URL_PLACEHOLDER}';
        
        // List of known book folders (can be updated by scanning)
        let knownBookFolders = [];
        
        // Auto-detect books from folder structure (static file discovery)
        async function loadBooks() {
            const container = document.getElementById('books-container');
            container.innerHTML = '<p class="loading">Loading books...</p>';
            
            try {
                // Always scan folders for fresh book-info.json files first
                // This ensures we get the latest page counts even after updates
                let books = await scanForBooks();
                
                // If no books found from scanning, try books-list.json as fallback
                if (books.length === 0) {
                    try {
                        const listResponse = await fetch('./books-list.json');
                        if (listResponse.ok) {
                            const listData = await listResponse.json();
                            if (listData.books && Array.isArray(listData.books)) {
                                books = listData.books;
                                console.log('Loaded books from books-list.json (fallback)');
                            }
                        }
                    } catch (e) {
                        console.log('books-list.json not found');
                    }
                }
                
                if (books.length === 0) {
                    container.innerHTML = `
                        <div class="no-books">
                            <h2>No books found</h2>
                            <p>Books will appear here once they are generated and uploaded to the server.</p>
                            <p style="margin-top: 20px; font-size: 0.9em; opacity: 0.8;">
                                Books should be placed in subdirectories under: ${base_url}/
                            </p>
                            <p style="margin-top: 10px; font-size: 0.85em; opacity: 0.7;">
                                Each book should be in its own folder with a book-info.json file
                            </p>
                        </div>
                    `;
                    return;
                }
                
                // Store known books in localStorage for offline access
                localStorage.setItem('knownBooks', JSON.stringify(books));
                
                let html = '<div class="books-grid">';
                
                for (const book of books) {
                    const progress = await getBookProgress(book.id);
                    // Use fresh total_pages from book-info.json, fallback to progress.totalPages
                    const totalPages = book.total_pages || progress.totalPages || 0;
                    const currentPage = progress.currentPage || 0;
                    const progressPercent = totalPages > 0 
                        ? Math.round((currentPage / totalPages) * 100) 
                        : 0;
                    
                    // Determine book URL - use folder if available, otherwise construct from base_url
                    const bookUrl = book.url || `${base_url}/${book.folder}/`;
                    
                    html += `
                        <div class="book-card" onclick="openBook('${book.folder}', '${bookUrl}')">
                            <div class="book-title">${escapeHtml(book.title)}</div>
                            <div class="book-author">${escapeHtml(book.author || 'Unknown Author')}</div>
                            <div class="book-progress">
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${progressPercent}%"></div>
                                </div>
                                <div class="progress-text">
                                    Page ${currentPage + 1} of ${totalPages} (${progressPercent}%)
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                html += '</div>';
                container.innerHTML = html;
                
            } catch (error) {
                console.error('Error loading books:', error);
                
                // Fallback: try to load from localStorage
                const knownBooks = JSON.parse(localStorage.getItem('knownBooks') || '[]');
                
                if (knownBooks.length > 0) {
                    let html = '<div class="books-grid">';
                    for (const book of knownBooks) {
                        const progress = await getBookProgress(book.id);
                        // Use book.total_pages if progress doesn't have it
                        const totalPages = progress.totalPages || book.total_pages || 0;
                        const currentPage = progress.currentPage || 0;
                        const progressPercent = totalPages > 0 
                            ? Math.round((currentPage / totalPages) * 100) 
                            : 0;
                        
                        const bookUrl = book.url || `${base_url}/${book.folder}/`;
                        html += `
                            <div class="book-card" onclick="openBook('${book.folder}', '${bookUrl}')">
                                <div class="book-title">${escapeHtml(book.title)}</div>
                                <div class="book-author">${escapeHtml(book.author || 'Unknown Author')}</div>
                                <div class="book-progress">
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: ${progressPercent}%"></div>
                                    </div>
                                    <div class="progress-text">
                                        Page ${currentPage + 1} of ${totalPages} (${progressPercent}%)
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    html += '</div>';
                    container.innerHTML = html + '<p style="text-align: center; color: #ff9800; margin-top: 20px;">⚠️ Using cached data. Click Refresh to scan for new books.</p>';
                } else {
                    container.innerHTML = `
                        <div class="no-books">
                            <h2>Error loading books</h2>
                            <p>${error.message}</p>
                            <p style="margin-top: 20px; font-size: 0.9em; opacity: 0.8;">
                                Make sure book folders are properly uploaded to the server.
                            </p>
                        </div>
                    `;
                }
            }
        }
        
        // Scan subdirectories for book-info.json files
        async function scanForBooks() {
            const books = [];
            // Common book folder names to try (based on typical naming)
            // We'll try to fetch book-info.json from likely folder names
            const commonFolders = [
                'the-atomic-habit',
                'the-unconscious-mind',
                'atomic-habits',
                'book-1',
                'book-2'
            ];
            
            // Try to get folder list from a directory listing or known folders
            // Since we can't list directories via HTTP, we'll try common patterns
            // and also check localStorage for previously discovered folders
            
            const cachedFolders = JSON.parse(localStorage.getItem('discoveredBookFolders') || '[]');
            let foldersToCheck = [...new Set([...commonFolders, ...cachedFolders])];
            
            // Also try to load from books-list.json to discover folder names
            try {
                const listResponse = await fetch('./books-list.json?t=' + Date.now());
                if (listResponse.ok) {
                    const listData = await listResponse.json();
                    if (listData.books && Array.isArray(listData.books)) {
                        // Extract folder names from books-list.json
                        listData.books.forEach(book => {
                            if (book.folder && !foldersToCheck.includes(book.folder)) {
                                foldersToCheck.push(book.folder);
                            }
                        });
                    }
                }
            } catch (e) {
                // Ignore if books-list.json doesn't exist
            }
            
            // Fetch fresh book-info.json from each folder with cache-busting
            for (const folder of foldersToCheck) {
                try {
                    // Add cache-busting parameter to ensure fresh data
                    const response = await fetch(`./${folder}/book-info.json?t=${Date.now()}`);
                    if (response.ok) {
                        const bookInfo = await response.json();
                        books.push({
                            id: bookInfo.id || books.length + 1,
                            title: bookInfo.title || folder,
                            author: bookInfo.author || '',
                            folder: bookInfo.folder || folder,
                            url: `${base_url}/${folder}/`,
                            total_pages: bookInfo.total_pages || 0
                        });
                        // Cache this folder for future scans
                        if (!cachedFolders.includes(folder)) {
                            cachedFolders.push(folder);
                            localStorage.setItem('discoveredBookFolders', JSON.stringify(cachedFolders));
                        }
                    }
                } catch (e) {
                    // Folder doesn't exist or no book-info.json, skip it
                    continue;
                }
            }
            
            return books;
        }
        
        async function getBookProgress(bookId) {
            const db = await openDB();
            return new Promise((resolve, reject) => {
                const transaction = db.transaction(['progress'], 'readonly');
                const store = transaction.objectStore('progress');
                const request = store.get(bookId);
                
                request.onsuccess = () => {
                    const progress = request.result;
                    if (progress) {
                        resolve({
                            currentPage: progress.currentPage || 0,
                            totalPages: progress.totalPages || 0
                        });
                    } else {
                        resolve({ currentPage: 0, totalPages: 0 });
                    }
                };
                
                request.onerror = () => {
                    resolve({ currentPage: 0, totalPages: 0 });
                };
            });
        }
        
        function openBook(folder, bookUrl) {
            // Use bookUrl if provided, otherwise construct from folder
            if (bookUrl) {
                window.location.href = bookUrl + 'index.html';
            } else {
                window.location.href = `${folder}/index.html`;
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function openDB() {
            return new Promise((resolve, reject) => {
                const request = indexedDB.open('BookReaderDB', 1);
                request.onerror = () => reject(request.error);
                request.onsuccess = () => resolve(request.result);
                request.onupgradeneeded = (event) => {
                    const db = event.target.result;
                    if (!db.objectStoreNames.contains('books')) {
                        db.createObjectStore('books', { keyPath: 'id' });
                    }
                    if (!db.objectStoreNames.contains('progress')) {
                        db.createObjectStore('progress', { keyPath: 'bookId' });
                    }
                };
            });
        }
        
        async function clearCache() {
            if (!confirm('Clear all cached data?\\n\\nThis will:\\n- Clear cached book data\\n- Clear localStorage cache\\n- Clear service worker cache\\n\\nYour reading progress will be preserved.')) {
                return;
            }
            
            try {
                // Clear IndexedDB 'books' store (but keep 'progress' store)
                const db = await openDB();
                const transaction = db.transaction(['books'], 'readwrite');
                const store = transaction.objectStore('books');
                const clearRequest = store.clear();
                
                await new Promise((resolve, reject) => {
                    clearRequest.onsuccess = () => resolve();
                    clearRequest.onerror = () => reject(clearRequest.error);
                });
                
                console.log('Cleared IndexedDB books store');
                
                // Clear localStorage cache (except discoveredBookFolders for convenience)
                const discoveredFolders = localStorage.getItem('discoveredBookFolders');
                localStorage.clear();
                if (discoveredFolders) {
                    localStorage.setItem('discoveredBookFolders', discoveredFolders);
                }
                console.log('Cleared localStorage cache');
                
                // Clear service worker cache
                if ('serviceWorker' in navigator && 'caches' in window) {
                    const cacheNames = await caches.keys();
                    await Promise.all(
                        cacheNames.map(cacheName => caches.delete(cacheName))
                    );
                    console.log('Cleared service worker cache');
                }
                
                // Unregister service worker to force fresh registration
                if ('serviceWorker' in navigator) {
                    const registrations = await navigator.serviceWorker.getRegistrations();
                    await Promise.all(
                        registrations.map(registration => registration.unregister())
                    );
                    console.log('Unregistered service workers');
                }
                
                alert('Cache cleared successfully!\\n\\nPage will reload to fetch fresh data.');
                
                // Reload the page to fetch fresh data
                window.location.reload(true);
                
            } catch (error) {
                console.error('Error clearing cache:', error);
                alert('Error clearing cache: ' + error.message);
            }
        }
        
        // Load books on page load
        loadBooks();
        
        // Register service worker
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('service-worker.js')
                .then(reg => console.log('Service Worker registered'))
                .catch(err => console.log('Service Worker registration failed:', err));
        }
    </script>
</body>
</html>""".replace('{BASE_URL_PLACEHOLDER}', base_url)


def generate_reader_html(book_title: str, book_author: str, book_id: int, folder_name: str) -> str:
    """Generate the main reader HTML."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#2c3e50">
    <title>{book_title} - Reader</title>
    <link rel="manifest" href="manifest.json">
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div id="app">
        <header>
            <button id="back-btn" class="icon-btn" title="Back to Library">←</button>
            <div class="header-info">
                <h1 id="book-title">{book_title}</h1>
                <p id="book-author">{book_author}</p>
            </div>
            <button id="settings-btn" class="icon-btn" title="Settings">⚙️</button>
        </header>
        
        <main id="reader">
            <div id="content" class="content"></div>
        </main>
        
        <nav class="reader-nav">
            <button id="prev-btn" class="nav-btn">◀ Previous</button>
            <div class="page-info">
                <span id="page-indicator">Page 1 of 1</span>
            </div>
            <button id="next-btn" class="nav-btn">Next ▶</button>
        </nav>
        
        <div id="settings-panel" class="settings-panel hidden">
            <h3>Reading Settings</h3>
            <div class="setting-item">
                <label>Font Size</label>
                <input type="range" id="font-size" min="14" max="24" value="18">
                <span id="font-size-value">18px</span>
            </div>
            <div class="setting-item">
                <label>Line Height</label>
                <input type="range" id="line-height" min="1.4" max="2.2" step="0.1" value="1.8">
                <span id="line-height-value">1.8</span>
            </div>
            <div class="setting-item">
                <label>Theme</label>
                <select id="theme-select">
                    <option value="light">Light</option>
                    <option value="sepia">Sepia</option>
                    <option value="dark">Dark</option>
                </select>
            </div>
            <div class="setting-item" style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(0,0,0,0.1);">
                <button id="clear-cache-btn" class="close-btn" style="background: #ff9800; color: white; width: 100%;">🗑️ Clear Cache</button>
                <p style="font-size: 0.85em; color: #666; margin-top: 10px; text-align: center;">
                    Clears cached book data but preserves reading progress
                </p>
            </div>
            <button id="close-settings" class="close-btn">Close</button>
        </div>
    </div>
    
    <script src="app.js"></script>
    <script>
        // Initialize app
        const app = new BookReaderApp({{
            bookId: {book_id},
            folderName: '{folder_name}'
        }});
        app.init();
    </script>
</body>
</html>"""


def generate_manifest_json(book_title: str, folder_name: str) -> str:
    """Generate PWA manifest."""
    return json.dumps({
        "name": f"{book_title} - Reader",
        "short_name": book_title[:30],
        "description": "E-reader for translated books",
        "start_url": f"./{folder_name}/index.html",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#2c3e50",
        "orientation": "portrait",
        "icons": [
            {
                "src": "icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "icon-512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }, indent=2)


def generate_service_worker_js() -> str:
    """Generate service worker for offline support."""
    return """const CACHE_NAME = 'book-reader-v1';
const urlsToCache = [
    './',
    './index.html',
    './styles.css',
    './app.js',
    './manifest.json',
    './book-data.json'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                return response || fetch(event.request);
            })
    );
});"""


def generate_reader_css() -> str:
    """Generate elegant e-reader CSS."""
    return """/* E-Reader Styles - Elegant and Easy on the Eyes */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --bg-light: #fefefe;
    --bg-sepia: #f4e8d0;
    --bg-dark: #1a1a1a;
    --text-light: #2c3e50;
    --text-sepia: #3d2817;
    --text-dark: #e0e0e0;
    --accent: #667eea;
}

body {
    font-family: 'Noto Sans Devanagari', 'Lohit Devanagari', 'DejaVu Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-light);
    color: var(--text-light);
    line-height: 1.8;
    font-size: 18px;
    transition: background 0.3s, color 0.3s;
    overflow-x: hidden;
}

body.theme-sepia {
    background: var(--bg-sepia);
    color: var(--text-sepia);
}

body.theme-dark {
    background: var(--bg-dark);
    color: var(--text-dark);
}

#app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

header {
    background: white;
    padding: 15px 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
    gap: 15px;
    position: sticky;
    top: 0;
    z-index: 100;
}

body.theme-dark header {
    background: #2c2c2c;
}

.icon-btn {
    background: none;
    border: none;
    font-size: 1.5em;
    cursor: pointer;
    padding: 5px 10px;
    color: var(--accent);
    transition: opacity 0.2s;
}

.icon-btn:hover {
    opacity: 0.7;
}

.header-info {
    flex: 1;
    text-align: center;
}

.header-info h1 {
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 2px;
}

.header-info p {
    font-size: 0.85em;
    opacity: 0.7;
}

#reader {
    flex: 1;
    display: flex;
    justify-content: center;
    padding: 40px 20px;
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
}

.content {
    width: 100%;
    max-width: 100%;
    word-wrap: break-word;
    white-space: pre-wrap;
    font-size: 18px;
    line-height: 1.8;
    text-align: justify;
    hyphens: auto;
    -webkit-hyphens: auto;
    -moz-hyphens: auto;
}

.content p {
    margin-bottom: 1.2em;
    text-indent: 1.5em;
}

.content p:first-child {
    text-indent: 0;
}

.reader-nav {
    background: white;
    padding: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
    gap: 15px;
}

body.theme-dark .reader-nav {
    background: #2c2c2c;
}

.nav-btn {
    padding: 12px 24px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1em;
    font-weight: 500;
    transition: opacity 0.2s, transform 0.2s;
    flex: 0 0 auto;
}

.nav-btn:hover:not(:disabled) {
    opacity: 0.9;
    transform: scale(1.05);
}

.nav-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.page-info {
    flex: 1;
    text-align: center;
    font-size: 0.9em;
    opacity: 0.8;
}

.settings-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 300px;
    height: 100vh;
    background: white;
    box-shadow: -2px 0 8px rgba(0,0,0,0.2);
    padding: 30px;
    transform: translateX(0);
    transition: transform 0.3s;
    z-index: 200;
    overflow-y: auto;
}

body.theme-dark .settings-panel {
    background: #2c2c2c;
}

.settings-panel.hidden {
    transform: translateX(100%);
}

.settings-panel h3 {
    margin-bottom: 20px;
    font-size: 1.3em;
}

.setting-item {
    margin-bottom: 25px;
}

.setting-item label {
    display: block;
    margin-bottom: 8px;
    font-weight: 500;
}

.setting-item input[type="range"] {
    width: 100%;
    margin-bottom: 5px;
}

.setting-item select {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 1em;
}

.close-btn {
    width: 100%;
    padding: 12px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1em;
    margin-top: 20px;
}

@media (max-width: 768px) {
    .content {
        font-size: 16px;
        padding: 20px 10px;
    }
    
    .reader-nav {
        flex-direction: column;
        gap: 10px;
    }
    
    .nav-btn {
        width: 100%;
    }
    
    .settings-panel {
        width: 100%;
    }
}"""


def generate_app_js() -> str:
    """Generate main application JavaScript."""
    return """// Book Reader App
class BookReaderApp {
    constructor(config) {
        this.bookId = config.bookId;
        this.folderName = config.folderName;
        this.currentPage = 0;
        this.bookData = null;
        this.db = null;
    }
    
    async init() {
        await this.initDB();
        await this.loadBookData();
        await this.loadProgress();
        this.setupEventListeners();
        this.loadPage(this.currentPage);
        this.registerServiceWorker();
    }
    
    async initDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('BookReaderDB', 1);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('books')) {
                    db.createObjectStore('books', { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains('progress')) {
                    db.createObjectStore('progress', { keyPath: 'bookId' });
                }
            };
        });
    }
    
    async loadBookData() {
        // Always fetch fresh book-data.json from server to get latest pages
        // Use cache-busting to ensure we get the latest version
        try {
            const response = await fetch(`book-data.json?t=${Date.now()}`);
            if (!response.ok) {
                throw new Error('Failed to fetch book-data.json');
            }
            const data = await response.json();
            this.bookData = data;
            
            // Update IndexedDB with fresh data
            const writeTransaction = this.db.transaction(['books'], 'readwrite');
            const writeStore = writeTransaction.objectStore('books');
            writeStore.put(data);
            
            // Update progress totalPages to match current book data
            const totalPages = data.pages ? data.pages.length : 0;
            const progressTransaction = this.db.transaction(['progress'], 'readwrite');
            const progressStore = progressTransaction.objectStore('progress');
            const progressRequest = progressStore.get(this.bookId);
            
            progressRequest.onsuccess = () => {
                const existingProgress = progressRequest.result || { bookId: this.bookId, currentPage: 0 };
                // Update totalPages to match current book data
                const updatedProgress = {
                    ...existingProgress,
                    totalPages: totalPages
                };
                progressStore.put(updatedProgress);
            };
        } catch (error) {
            // Fallback: try to load from IndexedDB if fetch fails
            console.warn('Failed to fetch fresh book-data.json, using cached version:', error);
            const transaction = this.db.transaction(['books'], 'readonly');
            const store = transaction.objectStore('books');
            const request = store.get(this.bookId);
            
            return new Promise((resolve, reject) => {
                request.onsuccess = () => {
                    if (request.result) {
                        this.bookData = request.result;
                        resolve();
                    } else {
                        reject(new Error('No book data available'));
                    }
                };
                request.onerror = () => reject(request.error);
            });
        }
    }
    
    async loadProgress() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['progress'], 'readonly');
            const store = transaction.objectStore('progress');
            const request = store.get(this.bookId);
            
            request.onsuccess = () => {
                if (request.result) {
                    this.currentPage = request.result.currentPage || 0;
                } else {
                    // No progress found, start from page 0
                    this.currentPage = 0;
                }
                resolve();
            };
            
            request.onerror = () => {
                // On error, default to page 0
                this.currentPage = 0;
                reject(request.error);
            };
        });
    }
    
    async saveProgress() {
        const transaction = this.db.transaction(['progress'], 'readwrite');
        const store = transaction.objectStore('progress');
        store.put({
            bookId: this.bookId,
            currentPage: this.currentPage,
            totalPages: this.bookData.pages.length,
            lastRead: new Date().toISOString()
        });
    }
    
    async clearCache() {
        if (!confirm('Clear all cached data?\\n\\nThis will:\\n- Clear cached book data\\n- Clear localStorage cache\\n- Clear service worker cache\\n\\nYour reading progress will be preserved.')) {
            return;
        }
        
        try {
            // Clear IndexedDB 'books' store (but keep 'progress' store)
            const transaction = this.db.transaction(['books'], 'readwrite');
            const store = transaction.objectStore('books');
            const clearRequest = store.clear();
            
            await new Promise((resolve, reject) => {
                clearRequest.onsuccess = () => resolve();
                clearRequest.onerror = () => reject(clearRequest.error);
            });
            
            console.log('Cleared IndexedDB books store');
            
            // Clear localStorage cache (except reading preferences)
            const fontSize = localStorage.getItem('fontSize');
            const lineHeight = localStorage.getItem('lineHeight');
            const theme = localStorage.getItem('theme');
            localStorage.clear();
            if (fontSize) localStorage.setItem('fontSize', fontSize);
            if (lineHeight) localStorage.setItem('lineHeight', lineHeight);
            if (theme) localStorage.setItem('theme', theme);
            console.log('Cleared localStorage cache');
            
            // Clear service worker cache
            if ('caches' in window) {
                const cacheNames = await caches.keys();
                await Promise.all(
                    cacheNames.map(cacheName => caches.delete(cacheName))
                );
                console.log('Cleared service worker cache');
            }
            
            // Unregister service worker to force fresh registration
            if ('serviceWorker' in navigator) {
                const registrations = await navigator.serviceWorker.getRegistrations();
                await Promise.all(
                    registrations.map(registration => registration.unregister())
                );
                console.log('Unregistered service workers');
            }
            
            alert('Cache cleared successfully!\\n\\nPage will reload to fetch fresh data.');
            
            // Reload the page to fetch fresh data
            window.location.reload(true);
            
        } catch (error) {
            console.error('Error clearing cache:', error);
            alert('Error clearing cache: ' + error.message);
        }
    }
    
    loadPage(pageIndex) {
        if (!this.bookData || pageIndex < 0 || pageIndex >= this.bookData.pages.length) {
            return;
        }
        
        const page = this.bookData.pages[pageIndex];
        const content = document.getElementById('content');
        
        // Convert markdown-like formatting to HTML
        let html = this.formatText(page.content);
        content.innerHTML = html;
        
        // Update page indicator
        const indicator = document.getElementById('page-indicator');
        indicator.textContent = `Page ${pageIndex + 1} of ${this.bookData.pages.length}`;
        
        // Update navigation buttons
        document.getElementById('prev-btn').disabled = pageIndex === 0;
        document.getElementById('next-btn').disabled = pageIndex === this.bookData.pages.length - 1;
        
        // Save progress
        this.currentPage = pageIndex;
        this.saveProgress();
        
        // Scroll to top
        window.scrollTo(0, 0);
    }
    
    formatText(text) {
        // Convert markdown to HTML
        let html = text;
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        html = html.replace(/\\n/g, '<br>');
        
        // Wrap in paragraphs
        const paragraphs = html.split('<br><br>');
        return paragraphs.map(p => `<p>${p}</p>`).join('');
    }
    
    setupEventListeners() {
        document.getElementById('prev-btn').addEventListener('click', () => {
            if (this.currentPage > 0) {
                this.loadPage(this.currentPage - 1);
            }
        });
        
        document.getElementById('next-btn').addEventListener('click', () => {
            if (this.currentPage < this.bookData.pages.length - 1) {
                this.loadPage(this.currentPage + 1);
            }
        });
        
        document.getElementById('back-btn').addEventListener('click', () => {
            window.location.href = '../index.html';
        });
        
        document.getElementById('settings-btn').addEventListener('click', () => {
            document.getElementById('settings-panel').classList.remove('hidden');
        });
        
        document.getElementById('close-settings').addEventListener('click', () => {
            document.getElementById('settings-panel').classList.add('hidden');
        });
        
        document.getElementById('clear-cache-btn').addEventListener('click', () => {
            app.clearCache();
        });
        
        // Font size
        const fontSizeSlider = document.getElementById('font-size');
        const fontSizeValue = document.getElementById('font-size-value');
        fontSizeSlider.addEventListener('input', (e) => {
            const size = e.target.value;
            fontSizeValue.textContent = size + 'px';
            document.getElementById('content').style.fontSize = size + 'px';
            localStorage.setItem('fontSize', size);
        });
        const savedFontSize = localStorage.getItem('fontSize') || '18';
        fontSizeSlider.value = savedFontSize;
        fontSizeValue.textContent = savedFontSize + 'px';
        document.getElementById('content').style.fontSize = savedFontSize + 'px';
        
        // Line height
        const lineHeightSlider = document.getElementById('line-height');
        const lineHeightValue = document.getElementById('line-height-value');
        lineHeightSlider.addEventListener('input', (e) => {
            const height = e.target.value;
            lineHeightValue.textContent = height;
            document.getElementById('content').style.lineHeight = height;
            localStorage.setItem('lineHeight', height);
        });
        const savedLineHeight = localStorage.getItem('lineHeight') || '1.8';
        lineHeightSlider.value = savedLineHeight;
        lineHeightValue.textContent = savedLineHeight;
        document.getElementById('content').style.lineHeight = savedLineHeight;
        
        // Theme
        const themeSelect = document.getElementById('theme-select');
        themeSelect.addEventListener('change', (e) => {
            document.body.className = 'theme-' + e.target.value;
            localStorage.setItem('theme', e.target.value);
        });
        const savedTheme = localStorage.getItem('theme') || 'light';
        themeSelect.value = savedTheme;
        document.body.className = 'theme-' + savedTheme;
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft' && this.currentPage > 0) {
                this.loadPage(this.currentPage - 1);
            } else if (e.key === 'ArrowRight' && this.currentPage < this.bookData.pages.length - 1) {
                this.loadPage(this.currentPage + 1);
            }
        });
    }
    
    registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('service-worker.js')
                .then(reg => console.log('Service Worker registered'))
                .catch(err => console.log('Service Worker registration failed:', err));
        }
    }
}"""


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Expense Manager API"}


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Expense Manager API",
        "version": "1.0.0",
        "endpoints": {
            "income_categories": {"GET": "/api/income-categories"},
            "expense_categories": {"GET": "/api/expense-categories"},
            "accounts": {"GET": "/api/accounts"},
            "sources": {"GET": "/api/sources"},
            "income": {
                "GET": "/api/income?start_date=&end_date=&category_id=&source_id=",
                "POST": "/api/income",
                "PUT": "/api/income/{income_id}",
                "DELETE": "/api/income/{income_id}"
            },
            "expense": {
                "GET": "/api/expense?start_date=&end_date=&category_id=&account_id=",
                "POST": "/api/expense",
                "PUT": "/api/expense/{expense_id}",
                "DELETE": "/api/expense/{expense_id}"
            },
            "health": {"GET": "/api/health"},
            "pdf_extract_text": {"POST": "/api/pdf/extract-text"},
            "pdf_save_translation": {"POST": "/api/pdf/save-translation"},
            "pdf_generate_translated": {"POST": "/api/pdf/generate-translated-pdf"},
            "pwa_generate_package": {"POST": "/api/pwa/generate-package"},
            "pwa_list_books": {"GET": "/api/pwa/list-books"},
            "pwa_generate_dashboard": {"GET": "/api/pwa/generate-dashboard"},
            "docs": {"GET": "/docs"}
        }
    }
