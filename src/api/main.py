"""FastAPI server for Expense Manager APIs."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime
from calendar import monthrange
import sys
from pathlib import Path

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
            "docs": {"GET": "/docs"}
        }
    }
