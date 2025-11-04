"""SQL RAG for expense database querying using LangChain Agent with persistent context."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import date
from calendar import monthrange

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain_community.agent_toolkits import create_sql_agent

from src.utils.expense_db import ExpenseDB
from src.utils.db_schema_loader import get_database_schema_document


class ExpenseSQLRAG:
    """SQL RAG class for querying expense database using natural language with persistent context."""
    
    def __init__(self, ollama_model: str = "mistral", ollama_base_url: str = "http://localhost:11434"):
        """Initialize the SQL RAG system with persistent schema context.
        
        Args:
            ollama_model: Name of the LLM model to use
            ollama_base_url: Base URL for Ollama API
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.db = ExpenseDB()
        self.sql_db = None
        self.agent = None
        self.llm = None
        self.schema_context = None  # Persistent schema context
        self._initialize_database()
        self._load_schema_context()
        self._initialize_agent()
    
    def _initialize_database(self):
        """Initialize SQLDatabase connection."""
        # Get the database path from ExpenseDB
        db_path = self.db.db_path
        
        # Create SQLDatabase instance with schema info
        # Include sample rows for better understanding
        self.sql_db = SQLDatabase.from_uri(
            f"sqlite:///{db_path}",
            include_tables=[
                'income',
                'expense',
                'categories',
                'accounts',
                'sources'
            ],
            sample_rows_in_table_info=2  # Reduced for efficiency
        )
    
    def _load_schema_context(self):
        """Load database schema context once and store it persistently."""
        # Get comprehensive schema document
        self.schema_context = get_database_schema_document()
        
        # Get current month info
        today = date.today()
        first_day = date(today.year, today.month, 1)
        last_day = date(today.year, today.month, monthrange(today.year, today.month)[1])
        
        # Add current context
        self.schema_context += f"""

CURRENT CONTEXT (automatically updated):
- Today: {today.isoformat()} ({today.strftime('%B %d, %Y')})
- Current Month Start: {first_day.isoformat()}
- Current Month End: {last_day.isoformat()}
- Current Month Name: {today.strftime('%B %Y')}

When users say "this month", it refers to {first_day.isoformat()} to {last_day.isoformat()}.
"""
    
    def _initialize_agent(self):
        """Initialize the SQL agent with persistent schema context."""
        # Initialize LLM
        self.llm = OllamaLLM(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=0.1  # Lower temperature for more accurate SQL generation
        )
        
        # Create system prompt with persistent schema context
        system_prompt = f"""You are an expert SQL query generator for an expense and income management database.

{self.schema_context}

Your task is to:
1. Understand natural language questions about income, expenses, and financial data
2. Generate accurate SQL queries to retrieve the requested information
3. Execute queries safely (only SELECT statements, no modifications)
4. Format results in a clear, human-readable way
5. Remember previous context in the conversation

**Important Guidelines:**
- Always use JOINs to include category names, account names, and source names (not IDs)
- For "this month" queries, use the current month date range from the context above
- For date filtering, use: date >= 'YYYY-MM-DD' AND date <= 'YYYY-MM-DD'
- Always aggregate amounts when calculating totals (use SUM)
- Include currency information in responses
- Format large numbers with commas for readability (e.g., 50,000.00)
- Group results logically (by category, by month, by account, etc.)
- If user asks follow-up questions like "and last month?" or "break it down by category", use context from previous queries

**Query Examples:**
- "What are my expenses this month?" → SELECT expense records with date filter for current month
- "Show me income by category" → SELECT with GROUP BY category, JOIN to get category names
- "What's my total income last month?" → SELECT SUM with date filter for previous month
- "List expenses over 5000" → SELECT with amount > 5000 filter
- Follow-up: "And last month?" → Use previous query structure but change date range to last month

ONLY generate SELECT queries. NEVER generate INSERT, UPDATE, DELETE, or DROP statements.
"""
        
        # Create SQL agent with custom prompt
        # The create_sql_agent handles the tool creation and agent setup
        try:
            self.agent = create_sql_agent(
                llm=self.llm,
                db=self.sql_db,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=5,
                agent_executor_kwargs={
                    "return_intermediate_steps": False,
                    "handle_parsing_errors": True
                }
            )
        except (TypeError, AttributeError):
            # Fallback if parameters not supported
            try:
                self.agent = create_sql_agent(
                    llm=self.llm,
                    db=self.sql_db,
                    verbose=False,
                    handle_parsing_errors=True
                )
            except (TypeError, AttributeError):
                # Final fallback
                self.agent = create_sql_agent(
                    llm=self.llm,
                    db=self.sql_db,
                    verbose=False
                )
    
    def query(self, question: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """Query the database using natural language with persistent context.
        
        Args:
            question: Natural language question about expenses/income
            conversation_history: Optional list of previous questions/answers for context
            
        Returns:
            Dictionary with 'answer', 'success', and optionally 'error'
        """
        try:
            # Refresh current month context if needed (for date-related queries)
            # This ensures "this month" always refers to the current month
            self._refresh_date_context()
            
            # Build enhanced query with schema context and optional conversation history
            # Note: create_sql_agent may not use custom system prompts, so we include context in the input
            # The SQLDatabase already provides table schemas, but we add our enhanced context
            enhanced_query = f"""Additional Database Context:
{self.schema_context}

"""
            
            if conversation_history:
                enhanced_query += f"""Previous conversation:
{chr(10).join(f"- {msg}" for msg in conversation_history[-3:])}  # Last 3 messages

"""
            
            enhanced_query += f"""Current question: {question}

Please generate an SQL query to answer this question using the schema context above. Use JOINs to get category names, account names, and source names. Format numbers with commas for readability.
"""
            
            # Run the agent
            result = self.agent.invoke({"input": enhanced_query})
            
            # Extract answer - the result structure may vary
            if isinstance(result, dict):
                answer = result.get("output", result.get("answer", str(result)))
            else:
                answer = str(result)
            
            if not answer or answer.strip() == "":
                answer = "I couldn't process your query. Please try rephrasing."
            
            return {
                "success": True,
                "answer": answer,
                "query": question
            }
            
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error messages
            if "no such table" in error_msg.lower():
                return {
                    "success": False,
                    "answer": "I encountered an error accessing the database. Please ensure the database is properly initialized.",
                    "error": error_msg,
                    "query": question
                }
            elif "syntax error" in error_msg.lower() or "sql" in error_msg.lower():
                return {
                    "success": False,
                    "answer": "I had trouble understanding your question. Could you please rephrase it? For example: 'What are my expenses this month?' or 'Show me my income by category.'",
                    "error": error_msg,
                    "query": question
                }
            else:
                return {
                    "success": False,
                    "answer": f"I encountered an error while processing your query: {error_msg}. Please try rephrasing your question.",
                    "error": error_msg,
                    "query": question
                }
    
    def _refresh_date_context(self):
        """Refresh date context in schema for current date awareness."""
        # Update the date portion of schema context
        today = date.today()
        first_day = date(today.year, today.month, 1)
        last_day = date(today.year, today.month, monthrange(today.year, today.month)[1])
        
        # Update schema context's date section
        # This ensures "this month" always refers to current month
        date_section = f"""
CURRENT CONTEXT (automatically updated):
- Today: {today.isoformat()} ({today.strftime('%B %d, %Y')})
- Current Month Start: {first_day.isoformat()}
- Current Month End: {last_day.isoformat()}
- Current Month Name: {today.strftime('%B %Y')}

When users say "this month", it refers to {first_day.isoformat()} to {last_day.isoformat()}.
"""
        
        # Replace the date context section in schema_context
        if self.schema_context:
            # Find and replace the CURRENT CONTEXT section
            lines = self.schema_context.split('\n')
            new_lines = []
            skip_until_empty = False
            for line in lines:
                if "CURRENT CONTEXT" in line:
                    skip_until_empty = True
                    new_lines.append(date_section.strip())
                    continue
                if skip_until_empty:
                    if line.strip() == "":
                        skip_until_empty = False
                        new_lines.append(line)
                    continue
                new_lines.append(line)
            self.schema_context = '\n'.join(new_lines)
    
    def get_database_summary(self) -> str:
        """Get a summary of the database contents."""
        try:
            # Get counts
            income_count = len(self.db.get_income())
            expense_count = len(self.db.get_expenses())
            
            # Get current month summary
            today = date.today()
            first_day = date(today.year, today.month, 1)
            last_day = date(today.year, today.month, monthrange(today.year, today.month)[1])
            
            summary = self.db.get_summary(start_date=first_day.isoformat(), end_date=last_day.isoformat())
            
            return f"""
Database Summary:
- Total Income Records: {income_count}
- Total Expense Records: {expense_count}
- Current Month Income: {summary.get('total_income', 0):.2f} {summary.get('currency', 'INR')}
- Current Month Expenses: {summary.get('total_expense', 0):.2f} {summary.get('currency', 'INR')}
- Current Month Balance: {summary.get('balance', 0):.2f} {summary.get('currency', 'INR')}
"""
        except Exception as e:
            return f"Could not generate database summary: {str(e)}"
    
    def update_model(self, ollama_model: str, ollama_base_url: str = None):
        """Update the LLM model being used and refresh schema context.
        
        Args:
            ollama_model: New model name
            ollama_base_url: New base URL (optional)
        """
        self.ollama_model = ollama_model
        if ollama_base_url:
            self.ollama_base_url = ollama_base_url
        
        # Refresh schema context (in case database has changed)
        self._load_schema_context()
        
        # Reinitialize agent with new model and refreshed schema
        self._initialize_agent()
    
    def refresh_schema(self):
        """Manually refresh the database schema context."""
        self._load_schema_context()
        # Reinitialize agent to use new schema
        self._initialize_agent()
