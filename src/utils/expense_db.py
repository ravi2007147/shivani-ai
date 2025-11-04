"""SQLite database management for expense tracker."""

import sqlite3
import os
from typing import List, Dict, Optional
from datetime import datetime

class ExpenseDB:
    """Manages SQLite database for expense tracking."""
    
    def __init__(self, db_path: str = None):
        """Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file (default: expense_manager.db in .chroma_db/expense_manager/)
        """
        if db_path is None:
            db_dir = os.path.join(".chroma_db", "expense_manager")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "expense_manager.db")
        
        self.db_path = db_path
        self._initialize_database()
    
    def _get_connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _initialize_database(self):
        """Initialize database tables and default data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Categories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('income', 'expense')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, type)
            )
        """)
        
        # Accounts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sources table (for income sources)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Income table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS income (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                category_id INTEGER NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                currency TEXT NOT NULL DEFAULT 'INR',
                note TEXT,
                source_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE RESTRICT,
                FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE RESTRICT
            )
        """)
        
        # Expense table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expense (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                category_id INTEGER NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                currency TEXT NOT NULL DEFAULT 'INR',
                note TEXT,
                account_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE RESTRICT,
                FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE RESTRICT
            )
        """)
        
        # Migrate existing data: add currency column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE income ADD COLUMN currency TEXT DEFAULT 'INR'")
            cursor.execute("UPDATE income SET currency = 'INR' WHERE currency IS NULL")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE expense ADD COLUMN currency TEXT DEFAULT 'INR'")
            cursor.execute("UPDATE expense SET currency = 'INR' WHERE currency IS NULL")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Migrate existing data: add source_id column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE income ADD COLUMN source_id INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Insert default categories and accounts if they don't exist
        default_income_cats = ['Allowance', 'Salary', 'Petty cash', 'Bonus', 'Other']
        default_expense_cats = ['Food', 'Transportation', 'Household', 'Apparel', 'Education']
        default_accounts = ['Cash', 'Bank Accounts', 'Card']
        default_sources = ['Employer', 'Freelance', 'Investment', 'Gift', 'Other']
        
        for cat in default_income_cats:
            try:
                cursor.execute("INSERT INTO categories (name, type) VALUES (?, ?)", (cat, 'income'))
            except sqlite3.IntegrityError:
                pass
        
        for cat in default_expense_cats:
            try:
                cursor.execute("INSERT INTO categories (name, type) VALUES (?, ?)", (cat, 'expense'))
            except sqlite3.IntegrityError:
                pass
        
        for acc in default_accounts:
            try:
                cursor.execute("INSERT INTO accounts (name) VALUES (?)", (acc,))
            except sqlite3.IntegrityError:
                pass
        
        for src in default_sources:
            try:
                cursor.execute("INSERT INTO sources (name) VALUES (?)", (src,))
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
    
    # Category management
    def add_category(self, name: str, category_type: str) -> tuple[bool, str]:
        """Add a new category."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO categories (name, type) VALUES (?, ?)", (name, category_type))
            conn.commit()
            conn.close()
            return True, f"✅ Category '{name}' added"
        except sqlite3.IntegrityError:
            conn.close()
            return False, f"❌ Category '{name}' already exists"
        except Exception as e:
            conn.close()
            return False, f"❌ Error adding category: {str(e)}"
    
    def get_categories(self, category_type: str = None) -> List[Dict]:
        """Get all categories, optionally filtered by type."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if category_type:
            cursor.execute("SELECT * FROM categories WHERE type = ? ORDER BY name", (category_type,))
        else:
            cursor.execute("SELECT * FROM categories ORDER BY name")
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def delete_category(self, category_id: int) -> tuple[bool, str]:
        """Delete a category."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM categories WHERE id = ?", (category_id,))
            conn.commit()
            conn.close()
            return True, "✅ Category deleted"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting category: {str(e)}"
    
    # Account management
    def add_account(self, name: str) -> tuple[bool, str]:
        """Add a new account."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO accounts (name) VALUES (?)", (name,))
            conn.commit()
            conn.close()
            return True, f"✅ Account '{name}' added"
        except sqlite3.IntegrityError:
            conn.close()
            return False, f"❌ Account '{name}' already exists"
        except Exception as e:
            conn.close()
            return False, f"❌ Error adding account: {str(e)}"
    
    def get_accounts(self) -> List[Dict]:
        """Get all accounts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM accounts ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def delete_account(self, account_id: int) -> tuple[bool, str]:
        """Delete an account."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
            conn.commit()
            conn.close()
            return True, "✅ Account deleted"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting account: {str(e)}"
    
    # Source management
    def add_source(self, name: str) -> tuple[bool, str]:
        """Add a new source."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO sources (name) VALUES (?)", (name,))
            conn.commit()
            conn.close()
            return True, f"✅ Source '{name}' added"
        except sqlite3.IntegrityError:
            conn.close()
            return False, f"❌ Source '{name}' already exists"
        except Exception as e:
            conn.close()
            return False, f"❌ Error adding source: {str(e)}"
    
    def get_sources(self) -> List[Dict]:
        """Get all sources."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sources ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def delete_source(self, source_id: int) -> tuple[bool, str]:
        """Delete a source."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            conn.commit()
            conn.close()
            return True, "✅ Source deleted"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting source: {str(e)}"
    
    # Income management
    def add_income(self, date: str, category_id: int, amount: float, currency: str = 'INR', note: str = None, source_id: int = None) -> tuple[bool, str]:
        """Add income record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO income (date, category_id, amount, currency, note, source_id) VALUES (?, ?, ?, ?, ?, ?)",
                (date, category_id, amount, currency, note, source_id)
            )
            conn.commit()
            income_id = cursor.lastrowid
            conn.close()
            return True, f"✅ Income record added (ID: {income_id})"
        except Exception as e:
            conn.close()
            return False, f"❌ Error adding income: {str(e)}"
    
    def get_income(self, start_date: str = None, end_date: str = None, category_id: int = None) -> List[Dict]:
        """Get income records with filters."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT i.id, i.date, i.amount, i.currency, i.note, c.name as category_name, s.name as source_name
            FROM income i
            JOIN categories c ON i.category_id = c.id
            LEFT JOIN sources s ON i.source_id = s.id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND i.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND i.date <= ?"
            params.append(end_date)
        if category_id:
            query += " AND i.category_id = ?"
            params.append(category_id)
        
        query += " ORDER BY i.date DESC, i.id DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def update_income(self, income_id: int, date: str, category_id: int, amount: float, currency: str = 'INR', note: str = None, source_id: int = None) -> tuple[bool, str]:
        """Update income record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE income SET date = ?, category_id = ?, amount = ?, currency = ?, note = ?, source_id = ? WHERE id = ?",
                (date, category_id, amount, currency, note, source_id, income_id)
            )
            conn.commit()
            conn.close()
            return True, "✅ Income record updated"
        except Exception as e:
            conn.close()
            return False, f"❌ Error updating income: {str(e)}"
    
    def delete_income(self, income_id: int) -> tuple[bool, str]:
        """Delete income record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM income WHERE id = ?", (income_id,))
            conn.commit()
            conn.close()
            return True, "✅ Income record deleted"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting income: {str(e)}"
    
    # Expense management
    def add_expense(self, date: str, category_id: int, amount: float, account_id: int, currency: str = 'INR', note: str = None) -> tuple[bool, str]:
        """Add expense record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO expense (date, category_id, amount, currency, note, account_id) VALUES (?, ?, ?, ?, ?, ?)",
                (date, category_id, amount, currency, note, account_id)
            )
            conn.commit()
            expense_id = cursor.lastrowid
            conn.close()
            return True, f"✅ Expense record added (ID: {expense_id})"
        except Exception as e:
            conn.close()
            return False, f"❌ Error adding expense: {str(e)}"
    
    def get_expenses(self, start_date: str = None, end_date: str = None, category_id: int = None, account_id: int = None) -> List[Dict]:
        """Get expense records with filters."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT e.id, e.date, e.amount, e.currency, e.note, c.name as category_name, a.name as account_name
            FROM expense e
            JOIN categories c ON e.category_id = c.id
            JOIN accounts a ON e.account_id = a.id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND e.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND e.date <= ?"
            params.append(end_date)
        if category_id:
            query += " AND e.category_id = ?"
            params.append(category_id)
        if account_id:
            query += " AND e.account_id = ?"
            params.append(account_id)
        
        query += " ORDER BY e.date DESC, e.id DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def update_expense(self, expense_id: int, date: str, category_id: int, amount: float, account_id: int, currency: str = 'INR', note: str = None) -> tuple[bool, str]:
        """Update expense record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE expense SET date = ?, category_id = ?, amount = ?, currency = ?, note = ?, account_id = ? WHERE id = ?",
                (date, category_id, amount, currency, note, account_id, expense_id)
            )
            conn.commit()
            conn.close()
            return True, "✅ Expense record updated"
        except Exception as e:
            conn.close()
            return False, f"❌ Error updating expense: {str(e)}"
    
    def delete_expense(self, expense_id: int) -> tuple[bool, str]:
        """Delete expense record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM expense WHERE id = ?", (expense_id,))
            conn.commit()
            conn.close()
            return True, "✅ Expense record deleted"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting expense: {str(e)}"
    
    # Summary
    def get_summary(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get summary of income and expenses."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        income_query = "SELECT SUM(amount) as total FROM income WHERE 1=1"
        expense_query = "SELECT SUM(amount) as total FROM expense WHERE 1=1"
        income_params = []
        expense_params = []
        
        if start_date:
            income_query += " AND date >= ?"
            income_params.append(start_date)
            expense_query += " AND date >= ?"
            expense_params.append(start_date)
        
        if end_date:
            income_query += " AND date <= ?"
            income_params.append(end_date)
            expense_query += " AND date <= ?"
            expense_params.append(end_date)
        
        cursor.execute(income_query, income_params)
        income_result = cursor.fetchone()
        total_income = income_result['total'] if income_result['total'] else 0.0
        
        cursor.execute(expense_query, expense_params)
        expense_result = cursor.fetchone()
        total_expense = expense_result['total'] if expense_result['total'] else 0.0
        
        conn.close()
        
        return {
            'total_income': total_income,
            'total_expense': total_expense,
            'balance': total_income - total_expense
        }
