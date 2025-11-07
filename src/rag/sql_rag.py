"""SQL RAG for expense database querying using LangChain Agent with persistent schema context (Stable SQLite + Ollama Version)."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import date
from calendar import monthrange
import re

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# LangChain imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_ollama import OllamaLLM

# Local imports
from src.utils.expense_db import ExpenseDB
from src.utils.db_schema_loader import get_database_schema_document


class ExpenseSQLRAG:
    """SQL RAG system for querying an expense and income database via natural language."""

    def __init__(self, ollama_model: str = "mistral", ollama_base_url: str = "http://localhost:11434"):
        """Initialize with persistent schema context."""
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.db = ExpenseDB()
        self.sql_db = None
        self.agent = None
        self.llm = None
        self.schema_context = None

        self._initialize_database()
        self._load_schema_context()
        self._initialize_agent()

    # ---------------------------------------------
    # Database setup
    # ---------------------------------------------
    def _initialize_database(self):
        """Connect SQLite DB for use with LangChain SQL agent."""
        db_path = self.db.db_path
        self.sql_db = SQLDatabase.from_uri(
            f"sqlite:///{db_path}",
            include_tables=["income", "expense", "categories", "accounts", "sources"],
            sample_rows_in_table_info=2,
        )

    # ---------------------------------------------
    # Schema context
    # ---------------------------------------------
    def _load_schema_context(self):
        """Load and build schema context with current month awareness."""
        schema_doc = get_database_schema_document()
        today = date.today()
        first_day = date(today.year, today.month, 1)
        last_day = date(today.year, today.month, monthrange(today.year, today.month)[1])

        self.schema_context = f"""{schema_doc}

CURRENT CONTEXT (automatically updated):
- Today: {today.isoformat()} ({today.strftime('%B %d, %Y')})
- Current Month Start: {first_day.isoformat()}
- Current Month End: {last_day.isoformat()}
- Current Month Name: {today.strftime('%B %Y')}

When users say "this month", it refers to {first_day.isoformat()} to {last_day.isoformat()}.
IMPORTANT: Table names are singular — use "expense" and "income", not "expenses" or "incomes".
"""

    # ---------------------------------------------
    # Agent initialization
    # ---------------------------------------------
    def _initialize_agent(self):
        """Initialize SQL agent with Ollama and SQLite-safe rules."""
        self.llm = OllamaLLM(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=0.1,
        )

        system_prompt = f"""
You are an expert SQL assistant for a personal finance tracking system.

{self.schema_context}

SQL DIALECT & EXECUTION RULES:
- The database is SQLite. Do NOT use unsupported features (e.g., INTERVAL, DATE_ADD, EXTRACT).
- Date filtering: use BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD' (inclusive).
- Only SELECT statements are allowed. Never modify or delete data.
- One tool call must contain only ONE SELECT statement (no multiple SELECTs separated by semicolons).
- Use JOINs only when retrieving category/account/source names. Never join income and expense together.

BALANCE / NET CALCULATION RULES:
- When calculating remaining balance (income - expenses), NEVER JOIN income and expense.
- Instead, use independent scalar subqueries:
  SELECT
    (SELECT COALESCE(SUM(amount),0) FROM income WHERE date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD')
    -
    (SELECT COALESCE(SUM(amount),0) FROM expense WHERE date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD')
  AS balance;
- This avoids double-counting that happens with cartesian joins like 'INNER JOIN ON 1=1'.
- If there are multiple income or expense entries, subqueries ensure correct arithmetic.

STYLE & OUTPUT:
- Use table names exactly: income, expense, categories, accounts, sources.
- For "this month", use CURRENT CONTEXT above.
- Return amounts in INR with commas and two decimals (e.g., ₹11,500.00).
- Answer naturally and directly without meta-commentary like "Based on the context", "According to the database", "Based on the query results", or similar phrases.
- Do not include markdown, code blocks, or technical commentary in final answer.
- Provide a clear, conversational answer as if you're directly answering the user's question.
"""

        self.agent = create_sql_agent(
            llm=self.llm,
            db=self.sql_db,
            agent_type="zero-shot-react-description",
            verbose=True,
            system_message=system_prompt.strip(),
            handle_parsing_errors=True,
        )

    # ---------------------------------------------
    # Query execution
    # ---------------------------------------------
    def query(self, question: str, conversation_history: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a natural language question through the SQL RAG agent."""
        from langchain_core.exceptions import OutputParserException
        SQLITE_MULTI_STMT_ERR = "You can only execute one statement at a time"

        try:
            self._refresh_date_context()

            enhanced_query = f"""
{self.schema_context}

User Question: {question}

Please follow:
- SQLite supports only one SELECT per tool call.
- If you need both income and expense totals, query them separately and compute in reasoning.
- Do not use JOIN income + expense. Use scalar subqueries for balance.
- Use INR currency formatting.
"""

            if conversation_history:
                enhanced_query += "\nConversation History:\n" + "\n".join(conversation_history[-3:])

            result = self.agent.invoke({"input": enhanced_query})
            answer = self._extract_answer_text(result)

            # Sanity check (cartesian join detection)
            answer = self._sanity_check_balance(answer)

            return {"success": True, "answer": answer, "query": question}

        except OutputParserException as e:
            raw_text = getattr(e, "llm_output", None) or str(e)
            return {"success": True, "answer": raw_text.strip(), "query": question}

        except Exception as e:
            err = str(e)

            if SQLITE_MULTI_STMT_ERR in err:
                try:
                    retry_query = enhanced_query + "\n⚠️ Important: Only ONE SELECT per SQL tool call!"
                    result = self.agent.invoke({"input": retry_query})
                    answer = self._extract_answer_text(result)
                    answer = self._sanity_check_balance(answer)
                    return {"success": True, "answer": answer, "query": question, "note": "Retried safely."}
                except Exception as e2:
                    return {
                        "success": False,
                        "answer": "Retry failed due to multi-statement error again.",
                        "error": str(e2),
                        "query": question,
                    }

            if "no such table" in err.lower():
                return {"success": False, "answer": "Database schema mismatch. Ensure tables exist.", "error": err}
            if "OUTPUT_PARSING_FAILURE" in err:
                return {"success": False, "answer": "Model output was unclear, but SQL likely worked.", "error": err}
            return {"success": False, "answer": f"Unexpected error: {err}", "error": err}

    # ---------------------------------------------
    # Helpers
    # ---------------------------------------------
    def _extract_answer_text(self, result: Any) -> str:
        if isinstance(result, dict):
            return result.get("output") or result.get("final_answer") or str(result)
        return str(result)

    def _sanity_check_balance(self, answer: str) -> str:
        """Detect impossible or double-counted results."""
        nums = re.findall(r"\d[\d,]*\.?\d*", answer)
        if len(nums) >= 2:
            try:
                vals = [float(x.replace(",", "")) for x in nums]
                if max(vals) > 10 * min(vals):
                    return answer + "\n(Note: Possible calculation issue detected.)"
            except:
                pass
        return answer

    # ---------------------------------------------
    # Context refresh
    # ---------------------------------------------
    def _refresh_date_context(self):
        """Keep schema context synced with today's date."""
        today = date.today()
        first_day = date(today.year, today.month, 1)
        last_day = date(today.year, today.month, monthrange(today.year, today.month)[1])
        date_section = f"""
CURRENT CONTEXT (automatically updated):
- Today: {today.isoformat()} ({today.strftime('%B %d, %Y')})
- Current Month Start: {first_day.isoformat()}
- Current Month End: {last_day.isoformat()}
- Current Month Name: {today.strftime('%B %Y')}

When users say "this month", it refers to {first_day.isoformat()} to {last_day.isoformat()}.
"""
        if self.schema_context:
            lines = self.schema_context.splitlines()
            new_lines, skip = [], False
            for line in lines:
                if "CURRENT CONTEXT" in line:
                    new_lines.append(date_section.strip())
                    skip = True
                    continue
                if skip and line.strip() == "":
                    skip = False
                    continue
                if not skip:
                    new_lines.append(line)
            self.schema_context = "\n".join(new_lines)

    # ---------------------------------------------
    # Summary & refresh
    # ---------------------------------------------
    def get_database_summary(self) -> str:
        try:
            income_count = len(self.db.get_income())
            expense_count = len(self.db.get_expenses())
            today = date.today()
            first_day = date(today.year, today.month, 1)
            last_day = date(today.year, today.month, monthrange(today.year, today.month)[1])
            summary = self.db.get_summary(start_date=first_day.isoformat(), end_date=last_day.isoformat())

            return f"""
Database Summary:
- Income Records: {income_count}
- Expense Records: {expense_count}
- This Month Income: ₹{summary.get('total_income', 0):,.2f}
- This Month Expense: ₹{summary.get('total_expense', 0):,.2f}
- Balance: ₹{summary.get('balance', 0):,.2f}
"""
        except Exception as e:
            return f"Could not fetch summary: {e}"

    def update_model(self, ollama_model: str, ollama_base_url: Optional[str] = None):
        self.ollama_model = ollama_model
        if ollama_base_url:
            self.ollama_base_url = ollama_base_url
        self._load_schema_context()
        self._initialize_agent()

    def refresh_schema(self):
        self._load_schema_context()
        self._initialize_agent()
