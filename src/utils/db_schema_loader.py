"""Utility to load database schema as a knowledge base document."""

import sys
from pathlib import Path
from datetime import date
from calendar import monthrange

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.expense_db import ExpenseDB


def get_database_schema_document() -> str:
    """Generate a comprehensive database schema document for the knowledge base.
    
    Returns:
        String containing the database schema documentation
    """
    db = ExpenseDB()
    
    # Get current month info
    today = date.today()
    first_day = date(today.year, today.month, 1)
    last_day = date(today.year, today.month, monthrange(today.year, today.month)[1])
    
    # Get sample data counts
    try:
        income_count = len(db.get_income())
        expense_count = len(db.get_expenses())
        categories = db.get_categories()
        accounts = db.get_accounts()
        sources = db.get_sources()
    except Exception:
        income_count = 0
        expense_count = 0
        categories = []
        accounts = []
        sources = []
    
    # Build comprehensive schema document
    schema_doc = f"""EXPENSE AND INCOME MANAGEMENT DATABASE SCHEMA

This document describes the expense and income management database system. This database stores personal financial transactions including income and expenses with categories, accounts, and sources.

DATABASE OVERVIEW:
- Purpose: Track personal income and expenses
- Current Status: {income_count} income records, {expense_count} expense records
- Currency: INR (Indian Rupees) by default
- Current Date Context: Today is {today.isoformat()} ({today.strftime('%B %d, %Y')})
- Current Month: {first_day.isoformat()} to {last_day.isoformat()} ({today.strftime('%B %Y')})

TABLE STRUCTURES:

1. INCOME TABLE
   - Purpose: Stores income transaction records
   - Fields:
     * id (INTEGER): Unique identifier for each income record
     * date (DATE): Transaction date in YYYY-MM-DD format
     * category_id (INTEGER): References an income category (foreign key)
     * amount (REAL): Income amount (must be greater than 0)
     * currency (TEXT): Currency code, default is 'INR'
     * note (TEXT): Optional description or notes about the income
     * source_id (INTEGER): References an income source (foreign key, optional)
     * created_at (TIMESTAMP): Record creation timestamp
   - Relationships:
     * category_id → categories.id (where categories.type = 'income')
     * source_id → sources.id

2. EXPENSE TABLE
   - Purpose: Stores expense transaction records
   - Fields:
     * id (INTEGER): Unique identifier for each expense record
     * date (DATE): Transaction date in YYYY-MM-DD format
     * category_id (INTEGER): References an expense category (foreign key)
     * amount (REAL): Expense amount (must be greater than 0)
     * currency (TEXT): Currency code, default is 'INR'
     * note (TEXT): Optional description or notes about the expense
     * account_id (INTEGER): References a payment account (foreign key)
     * created_at (TIMESTAMP): Record creation timestamp
   - Relationships:
     * category_id → categories.id (where categories.type = 'expense')
     * account_id → accounts.id

3. CATEGORIES TABLE
   - Purpose: Defines transaction categories for both income and expenses
   - Fields:
     * id (INTEGER): Unique identifier
     * name (TEXT): Category name (e.g., 'Salary', 'Food', 'Transportation')
     * type (TEXT): Either 'income' or 'expense'
   - Available Income Categories: {', '.join([c['name'] for c in categories if c.get('type') == 'income'])} and more
   - Available Expense Categories: {', '.join([c['name'] for c in categories if c.get('type') == 'expense'])} and more

4. ACCOUNTS TABLE
   - Purpose: Defines payment accounts used for expenses
   - Fields:
     * id (INTEGER): Unique identifier
     * name (TEXT): Account name (e.g., 'Cash', 'Bank Accounts', 'Card')
   - Available Accounts: {', '.join([a['name'] for a in accounts])} and more

5. SOURCES TABLE
   - Purpose: Defines income sources
   - Fields:
     * id (INTEGER): Unique identifier
     * name (TEXT): Source name (e.g., 'Employer', 'Freelance', 'Investment')
   - Available Sources: {', '.join([s['name'] for s in sources])} and more

QUERY GUIDELINES:

When users ask questions about expenses or income, they are referring to this database system. Common queries include:

- "What are my expenses this month?" - Query expense table filtered by current month date range
- "Show me my income by category" - Group income records by category and sum amounts
- "What's my total income last month?" - Sum income amounts for previous month
- "List expenses over 5000" - Filter expenses where amount > 5000
- "Show expenses by category this month" - Group current month expenses by category
- "What's my balance?" - Calculate income total minus expense total
- "Show me expenses from last month" - Filter expenses for previous month date range

DATE RANGES:
- "This month" means: {first_day.isoformat()} to {last_day.isoformat()}
- "Last month" means: Previous month's first day to last day
- "This year" means: January 1 to December 31 of current year

IMPORTANT NOTES:
- All monetary amounts are stored as REAL (floating point) numbers
- Dates must be in YYYY-MM-DD format (ISO format)
- When displaying results, always show category names, account names, and source names (not IDs)
- Currency is typically INR (Indian Rupees)
- Amounts should be formatted with commas for readability (e.g., 50,000.00)
- Totals should be calculated using SUM() aggregation
- Date filtering uses: date >= 'YYYY-MM-DD' AND date <= 'YYYY-MM-DD'

RELATIONSHIPS SUMMARY:
- Income records link to categories (type='income') and sources
- Expense records link to categories (type='expense') and accounts
- Categories are shared between income and expense tables but distinguished by type field

This database schema is integrated into the knowledge base to help the AI understand financial queries and provide accurate responses about income and expense data.
"""
    
    return schema_doc


def is_expense_related_query(query: str) -> bool:
    """Determine if a query is related to expenses, income, or financial data.
    
    Args:
        query: User query text
        
    Returns:
        True if query is expense/income related, False otherwise
    """
    query_lower = query.lower()
    
    # Keywords that indicate expense/income queries
    expense_keywords = [
        'expense', 'expenses', 'spending', 'spent', 'cost', 'costs',
        'income', 'earning', 'earnings', 'salary', 'wage', 'revenue',
        'money', 'financial', 'finance', 'budget', 'balance',
        'category', 'categories', 'account', 'accounts', 'source', 'sources',
        'transaction', 'transactions', 'payment', 'payments',
        'month', 'monthly', 'this month', 'last month', 'year', 'yearly',
        'total', 'sum', 'amount', 'amounts', 'rupee', 'rupees', 'inr',
        'food', 'transportation', 'household', 'apparel', 'education',
        'allowance', 'petty cash', 'bonus', 'freelance', 'investment'
    ]
    
    # Check if query contains expense-related keywords
    return any(keyword in query_lower for keyword in expense_keywords)
