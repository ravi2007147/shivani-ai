"""FastAPI server for Expense Manager APIs."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import ExpenseDB

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
            "income": {"POST": "/api/income"},
            "expense": {"POST": "/api/expense"},
            "health": {"GET": "/api/health"},
            "docs": {"GET": "/docs"}
        }
    }
