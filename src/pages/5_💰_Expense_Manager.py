"""Expense Manager page for tracking income and expenses."""

import streamlit as st
import sys
import calendar
from pathlib import Path
from datetime import datetime, date
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_ollama import OllamaLLM
from src.utils import ExpenseDB
from src.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_LLM_MODEL
from src.utils import fetch_ollama_models, get_default_model

# Currency support
CURRENCIES = {
    'INR': '₹',
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CNY': '¥',
    'AUD': 'A$',
    'CAD': 'C$',
}

DEFAULT_CURRENCY = 'INR'

def get_currency_symbol(currency_code: str) -> str:
    """Get currency symbol for a currency code."""
    return CURRENCIES.get(currency_code, currency_code)

# Page configuration
st.set_page_config(
    page_title="Expense Manager",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Expense Manager")
st.markdown("Track your income and expenses with categories and accounts")

# Initialize database
# Note: Not using cache to avoid stale instances when database schema changes
db = ExpenseDB()

# Sidebar for AI configuration
with st.sidebar:
    st.header("🤖 AI Configuration")
    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama API endpoint"
    )
    
    # Load models
    if 'ollama_models_expense' not in st.session_state:
        st.session_state.ollama_models_expense = fetch_ollama_models(ollama_base_url)
    
    # Model selection
    if st.session_state.ollama_models_expense:
        default_model = get_default_model(st.session_state.ollama_models_expense, DEFAULT_LLM_MODEL)
        ollama_model = st.selectbox(
            "AI Model for Note Generation",
            options=st.session_state.ollama_models_expense,
            index=st.session_state.ollama_models_expense.index(default_model) if default_model in st.session_state.ollama_models_expense else 0,
            help="LLM model used for generating notes"
        )
    else:
        ollama_model = st.text_input(
            "AI Model for Note Generation",
            value=DEFAULT_LLM_MODEL,
            help="LLM model used for generating notes"
        )
    
    # Refresh models button
    if st.button("🔄 Refresh Models"):
        with st.spinner("Loading models..."):
            fetched_models = fetch_ollama_models(ollama_base_url)
            st.session_state.ollama_models_expense = fetched_models
            if fetched_models:
                st.success(f"✅ Loaded {len(fetched_models)} model(s)")
            else:
                st.warning("⚠️ Could not fetch models. Make sure Ollama is running.")

def generate_note_with_ai(data_type: str, date: str, category: str, amount: float, currency: str = 'INR', account: str = None, source: str = None, ollama_model: str = None, ollama_base_url: str = None) -> str:
    """Generate a descriptive note using AI based on income/expense data.
    
    The note is designed to be descriptive and useful for RAG queries later.
    
    Args:
        data_type: 'income' or 'expense'
        date: Date of the transaction
        category: Category name
        amount: Amount of the transaction
        currency: Currency code (default: 'INR')
        account: Account name (only for expenses)
        source: Source name (only for income)
        ollama_model: Ollama model name
        ollama_base_url: Ollama base URL
        
    Returns:
        Generated note text
    """
    try:
        llm = OllamaLLM(
            model=ollama_model or DEFAULT_LLM_MODEL,
            base_url=ollama_base_url or DEFAULT_OLLAMA_BASE_URL,
        )
        
        if data_type == 'income':
            currency_symbol = get_currency_symbol(currency)
            formatted_amount = f"{currency_symbol}{amount:,.2f}"
            
            # Build descriptive prompt with all details
            source_details = f" received from {source}" if source else ""
            prompt = f"""Create a descriptive, factual note for an income transaction that will be used for personal finance tracking.

Transaction Details:
- Transaction Type: Income
- Category: {category}
- Source: {source if source else 'Not specified'}
- Amount: {formatted_amount} ({currency})
- Date: {date}

Instructions:
Generate a clear, descriptive note (1-2 sentences) that naturally incorporates:
1. The category type ({category}) - this describes what kind of income it is
2. The source ({source if source else 'the income source'}) - this describes where the income came from
3. The amount ({formatted_amount}) - you MUST include this specific amount
4. The date ({date}) - you can reference this if relevant

Be descriptive and natural. Combine the category and source in a way that reads naturally. For example, if category is "Salary" and source is "Freelance", describe it as "Freelance salary payment" or "Salary income from freelance work". If category is "Bonus" and source is "Company XYZ", describe it as "Bonus received from Company XYZ".

DO NOT invent any details, locations, or specifics that were not provided. Only use the information given above.

Note:"""
        else:  # expense
            currency_symbol = get_currency_symbol(currency)
            formatted_amount = f"{currency_symbol}{amount:,.2f}"
            prompt = f"""Create a descriptive, factual note for an expense transaction that will be used for personal finance tracking.

Transaction Details:
- Transaction Type: Expense
- Category: {category}
- Account: {account}
- Amount: {formatted_amount} ({currency})
- Date: {date}

Instructions:
Generate a clear, descriptive note (1-2 sentences) that naturally incorporates:
1. The category type ({category}) - this describes what the expense was for
2. The account ({account}) - this describes which payment method/account was used
3. The amount ({formatted_amount}) - you MUST include this specific amount
4. The date ({date}) - you can reference this if relevant

Be descriptive and natural. Combine the category and account in a way that reads naturally. For example, if category is "Food" and account is "Cash", describe it as "Food expense paid with cash" or "Cash payment for food expenses". If category is "Transportation" and account is "Card", describe it as "Transportation expense charged to card" or "Card payment for transportation".

DO NOT invent any details, store names, locations, or specifics that were not provided. Only use the information given above.

Note:"""
        
        result = llm.invoke(prompt)
        # Clean up the response - remove any leading/trailing whitespace and any markdown formatting
        cleaned_result = result.strip()
        # Remove common markdown formatting if present
        if cleaned_result.startswith('**'):
            cleaned_result = cleaned_result.replace('**', '')
        if cleaned_result.startswith('*'):
            cleaned_result = cleaned_result.replace('*', '')
        return cleaned_result
    except Exception as e:
        return f"Error generating note: {str(e)}"

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["💰 Income", "💸 Expenses", "📊 Summary", "⚙️ Settings"])

# Tab 1: Income
with tab1:
    st.header("💰 Income Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add Income")
        
        # Initialize generated note key if not exists
        if 'ai_generated_income_note' not in st.session_state:
            st.session_state['ai_generated_income_note'] = ''
        
        # Initialize income_note key if not exists
        if 'income_note' not in st.session_state:
            st.session_state['income_note'] = ''
        
        # Update income_note session state if we have a generated note (and clear it after use)
        if st.session_state.get('ai_generated_income_note'):
            st.session_state['income_note'] = st.session_state['ai_generated_income_note']
            st.session_state['ai_generated_income_note'] = ''  # Clear after using
        
        with st.form("add_income_form", clear_on_submit=False):
            income_date = st.date_input("Date", value=date.today(), key="income_date")
            income_categories = db.get_categories('income')
            income_category_options = {cat['name']: cat['id'] for cat in income_categories}
            if income_category_options:
                income_category_name = st.selectbox("Category", options=list(income_category_options.keys()), key="income_category")
                income_category_id = income_category_options[income_category_name]
            else:
                st.warning("No income categories available. Add categories in Settings tab.")
                income_category_id = None
            
            # Source dropdown
            sources = db.get_sources()
            source_options = {src['name']: src['id'] for src in sources}
            if source_options:
                income_source_name = st.selectbox("Source", options=['None'] + list(source_options.keys()), key="income_source")
                income_source_id = source_options[income_source_name] if income_source_name != 'None' else None
            else:
                st.warning("No sources available. Add sources in Settings tab.")
                income_source_id = None
            
            col_amount, col_currency = st.columns([3, 1])
            with col_amount:
                income_amount = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f", key="income_amount")
            with col_currency:
                income_currency = st.selectbox("Currency", options=list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index(DEFAULT_CURRENCY), key="income_currency")
            
            # Note field with AI generation
            note_col1, note_col2 = st.columns([4, 1])
            with note_col1:
                income_note = st.text_area("Note (Optional)", key="income_note", height=100)
            with note_col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                generate_income_note = st.form_submit_button("🤖 AI", help="Generate note using AI")
            
            submit_income = st.form_submit_button("➕ Add Income")
            
            # Handle AI note generation
            if generate_income_note:
                # Validate required fields
                if not income_category_id:
                    st.error("⚠️ Please select a category first")
                elif not income_amount or income_amount <= 0:
                    st.error("⚠️ Please enter an amount first")
                else:
                    with st.spinner("🤖 Generating note with AI..."):
                        # Use the current form values (category name, amount, date, source)
                        category_name = income_category_name  # Use the selected category name from form
                        source_name = income_source_name if income_source_name != 'None' else None
                        generated_note = generate_note_with_ai(
                            data_type='income',
                            date=income_date.isoformat(),
                            category=category_name,
                            amount=float(income_amount),
                            currency=income_currency,
                            source=source_name,
                            ollama_model=ollama_model,
                            ollama_base_url=ollama_base_url
                        )
                        # Store generated note in separate key
                        st.session_state['ai_generated_income_note'] = generated_note
                        st.rerun()
            
            # Handle form submission
            if submit_income and income_category_id:
                success, message = db.add_income(
                    date=income_date.isoformat(),
                    category_id=income_category_id,
                    amount=float(income_amount),
                    currency=income_currency,
                    note=income_note if income_note else None,
                    source_id=income_source_id
                )
                if success:
                    st.success(message)
                    # Clear form by rerunning (will reset all form fields)
                    st.rerun()
                else:
                    st.error(message)
    
    with col2:
        st.subheader("Filters")
        
        # Initialize date preset in session state if not exists (default to Current Month)
        if 'income_date_preset' not in st.session_state:
            st.session_state['income_date_preset'] = "Current Month"
        
        # Date range presets
        income_date_preset = st.selectbox(
            "Date Range",
            options=["Current Month", "Last Month", "This Year", "All Time", "Custom"],
            index=["Current Month", "Last Month", "This Year", "All Time", "Custom"].index(st.session_state.get('income_date_preset', "Current Month")),
            key="income_date_preset"
        )
        
        # Calculate default dates based on preset
        today = date.today()
        first_day_current_month = date(today.year, today.month, 1)
        
        if income_date_preset == "Current Month":
            income_default_start = first_day_current_month
            income_default_end = today
        elif income_date_preset == "Last Month":
            if today.month == 1:
                income_default_start = date(today.year - 1, 12, 1)
                last_day_last_month = date(today.year - 1, 12, 31)
            else:
                income_default_start = date(today.year, today.month - 1, 1)
                # Get last day of last month
                last_day = calendar.monthrange(today.year, today.month - 1)[1]
                last_day_last_month = date(today.year, today.month - 1, last_day)
            income_default_end = last_day_last_month
        elif income_date_preset == "This Year":
            income_default_start = date(today.year, 1, 1)
            income_default_end = today
        elif income_date_preset == "All Time":
            income_default_start = None
            income_default_end = None
        else:  # Custom
            income_default_start = None
            income_default_end = None
        
        # Date inputs - only show if Custom
        if income_date_preset == "Custom":
            filter_income_start = st.date_input("Start Date", value=None, key="filter_income_start")
            filter_income_end = st.date_input("End Date", value=None, key="filter_income_end")
        else:
            # Use preset dates
            filter_income_start = income_default_start
            filter_income_end = income_default_end
            # Show selected range
            if filter_income_start and filter_income_end:
                st.info(f"📅 Showing: {filter_income_start.strftime('%b %d')} - {filter_income_end.strftime('%b %d, %Y')}")
        
        all_income_cats = db.get_categories('income')
        # Put 'All' first in the list
        income_filter_options = {'All': None}
        income_filter_options.update({cat['name']: cat['id'] for cat in all_income_cats})
        
        # Initialize default to "All" if not set
        if 'filter_income_category' not in st.session_state:
            st.session_state['filter_income_category'] = 'All'
        
        # Ensure the selected value exists in options (in case categories changed)
        if st.session_state.get('filter_income_category') not in income_filter_options:
            st.session_state['filter_income_category'] = 'All'
        
        filter_income_category_name = st.selectbox(
            "Category",
            options=list(income_filter_options.keys()),
            key="filter_income_category"
        )
        filter_income_category_id = income_filter_options.get(filter_income_category_name)
    
    st.markdown("---")
    st.subheader("Income Records")
    
    # Get filtered income
    income_records = db.get_income(
        start_date=filter_income_start.isoformat() if filter_income_start else None,
        end_date=filter_income_end.isoformat() if filter_income_end else None,
        category_id=filter_income_category_id
    )
    
    if income_records:
        # Display in a table format
        for record in income_records:
            currency = record.get('currency', DEFAULT_CURRENCY)
            currency_symbol = get_currency_symbol(currency)
            with st.expander(f"📅 {record['date']} - {record['category_name']} - {currency_symbol}{record['amount']:.2f}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    currency = record.get('currency', DEFAULT_CURRENCY)
                    currency_symbol = get_currency_symbol(currency)
                    st.write(f"**Date:** {record['date']}")
                    st.write(f"**Category:** {record['category_name']}")
                    if record.get('source_name'):
                        st.write(f"**Source:** {record['source_name']}")
                    st.write(f"**Amount:** {currency_symbol}{record['amount']:.2f} ({currency})")
                    if record['note']:
                        st.write(f"**Note:** {record['note']}")
                
                with col2:
                    if st.button("✏️ Edit", key=f"edit_income_{record['id']}"):
                        st.session_state[f'editing_income_{record["id"]}'] = True
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_income_{record['id']}"):
                        success, message = db.delete_income(record['id'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                # Edit form
                if st.session_state.get(f'editing_income_{record["id"]}', False):
                    with st.form(f"edit_income_form_{record['id']}", clear_on_submit=False):
                        edit_income_date = st.date_input("Date", value=datetime.fromisoformat(record['date']).date(), key=f"edit_income_date_{record['id']}")
                        edit_income_categories = db.get_categories('income')
                        edit_income_category_options = {cat['name']: cat['id'] for cat in edit_income_categories}
                        # Find current category name
                        current_cat_name = record['category_name']
                        edit_income_category_name = st.selectbox(
                            "Category", 
                            options=list(edit_income_category_options.keys()),
                            index=list(edit_income_category_options.keys()).index(current_cat_name) if current_cat_name in edit_income_category_options else 0,
                            key=f"edit_income_category_{record['id']}"
                        )
                        edit_income_category_id = edit_income_category_options[edit_income_category_name]
                        
                        # Source dropdown in edit form
                        edit_sources = db.get_sources()
                        edit_source_options = {src['name']: src['id'] for src in edit_sources}
                        if edit_source_options:
                            current_source_name = record.get('source_name', 'None')
                            source_options_list = ['None'] + list(edit_source_options.keys())
                            edit_income_source_name = st.selectbox(
                                "Source",
                                options=source_options_list,
                                index=source_options_list.index(current_source_name) if current_source_name in source_options_list else 0,
                                key=f"edit_income_source_{record['id']}"
                            )
                            edit_income_source_id = edit_source_options[edit_income_source_name] if edit_income_source_name != 'None' else None
                        else:
                            edit_income_source_id = None
                        
                        edit_col_amount, edit_col_currency = st.columns([3, 1])
                        with edit_col_amount:
                            edit_income_amount = st.number_input("Amount", min_value=0.01, step=0.01, value=float(record['amount']), format="%.2f", key=f"edit_income_amount_{record['id']}")
                        with edit_col_currency:
                            current_currency = record.get('currency', DEFAULT_CURRENCY)
                            edit_income_currency = st.selectbox(
                                "Currency",
                                options=list(CURRENCIES.keys()),
                                index=list(CURRENCIES.keys()).index(current_currency) if current_currency in CURRENCIES else 0,
                                key=f"edit_income_currency_{record['id']}"
                            )
                        edit_income_note = st.text_area("Note", value=record['note'] if record['note'] else "", key=f"edit_income_note_{record['id']}")
                        
                        col_submit, col_cancel = st.columns(2)
                        with col_submit:
                            if st.form_submit_button("💾 Save"):
                                success, message = db.update_income(
                                    income_id=record['id'],
                                    date=edit_income_date.isoformat(),
                                    category_id=edit_income_category_id,
                                    amount=float(edit_income_amount),
                                    currency=edit_income_currency,
                                    note=edit_income_note if edit_income_note else None,
                                    source_id=edit_income_source_id
                                )
                                if success:
                                    st.success(message)
                                    st.session_state[f'editing_income_{record["id"]}'] = False
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        with col_cancel:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state[f'editing_income_{record["id"]}'] = False
                                st.rerun()
        
        # Total
        total_income = sum(record['amount'] for record in income_records)
        # Group by currency for total
        totals_by_currency = {}
        for record in income_records:
            curr = record.get('currency', DEFAULT_CURRENCY)
            if curr not in totals_by_currency:
                totals_by_currency[curr] = 0
            totals_by_currency[curr] += record['amount']
        
        if len(totals_by_currency) == 1:
            curr = list(totals_by_currency.keys())[0]
            curr_symbol = get_currency_symbol(curr)
            st.info(f"**Total Income:** {curr_symbol}{total_income:.2f}")
        else:
            st.info("**Total Income by Currency:**")
            for curr, total in totals_by_currency.items():
                curr_symbol = get_currency_symbol(curr)
                st.write(f"- {curr}: {curr_symbol}{total:.2f}")
    else:
        st.info("No income records found. Add your first income using the form above.")

# Tab 2: Expenses
with tab2:
    st.header("💸 Expense Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add Expense")
        
        # Initialize generated note key if not exists
        if 'ai_generated_expense_note' not in st.session_state:
            st.session_state['ai_generated_expense_note'] = ''
        
        # Initialize expense_note key if not exists
        if 'expense_note' not in st.session_state:
            st.session_state['expense_note'] = ''
        
        # Update expense_note session state if we have a generated note (and clear it after use)
        if st.session_state.get('ai_generated_expense_note'):
            st.session_state['expense_note'] = st.session_state['ai_generated_expense_note']
            st.session_state['ai_generated_expense_note'] = ''  # Clear after using
        
        with st.form("add_expense_form", clear_on_submit=False):
            expense_date = st.date_input("Date", value=date.today(), key="expense_date")
            expense_categories = db.get_categories('expense')
            expense_category_options = {cat['name']: cat['id'] for cat in expense_categories}
            if expense_category_options:
                expense_category_name = st.selectbox("Category", options=list(expense_category_options.keys()), key="expense_category")
                expense_category_id = expense_category_options[expense_category_name]
            else:
                st.warning("No expense categories available. Add categories in Settings tab.")
                expense_category_id = None
            
            exp_col_amount, exp_col_currency = st.columns([3, 1])
            with exp_col_amount:
                expense_amount = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f", key="expense_amount")
            with exp_col_currency:
                expense_currency = st.selectbox("Currency", options=list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index(DEFAULT_CURRENCY), key="expense_currency")
            
            accounts = db.get_accounts()
            account_options = {acc['name']: acc['id'] for acc in accounts}
            if account_options:
                expense_account_name = st.selectbox("Account", options=list(account_options.keys()), key="expense_account")
                expense_account_id = account_options[expense_account_name]
            else:
                st.warning("No accounts available. Add accounts in Settings tab.")
                expense_account_id = None
            
            # Note field with AI generation
            exp_note_col1, exp_note_col2 = st.columns([4, 1])
            with exp_note_col1:
                expense_note = st.text_area("Note (Optional)", key="expense_note", height=100)
            with exp_note_col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                generate_expense_note = st.form_submit_button("🤖 AI", help="Generate note using AI")
            
            submit_expense = st.form_submit_button("➕ Add Expense")
            
            # Handle AI note generation
            if generate_expense_note:
                # Validate required fields
                if not expense_category_id:
                    st.error("⚠️ Please select a category first")
                elif not expense_account_id:
                    st.error("⚠️ Please select an account first")
                elif not expense_amount or expense_amount <= 0:
                    st.error("⚠️ Please enter an amount first")
                else:
                    with st.spinner("🤖 Generating note with AI..."):
                        # Use the current form values (category name, account name, amount, date)
                        category_name = expense_category_name  # Use the selected category name from form
                        account_name = expense_account_name  # Use the selected account name from form
                        generated_note = generate_note_with_ai(
                            data_type='expense',
                            date=expense_date.isoformat(),
                            category=category_name,
                            amount=float(expense_amount),
                            currency=expense_currency,
                            account=account_name,
                            ollama_model=ollama_model,
                            ollama_base_url=ollama_base_url
                        )
                        # Store generated note in separate key
                        st.session_state['ai_generated_expense_note'] = generated_note
                        st.rerun()
            
            # Handle form submission
            if submit_expense and expense_category_id and expense_account_id:
                success, message = db.add_expense(
                    date=expense_date.isoformat(),
                    category_id=expense_category_id,
                    amount=float(expense_amount),
                    account_id=expense_account_id,
                    currency=expense_currency,
                    note=expense_note if expense_note else None
                )
                if success:
                    st.success(message)
                    # Clear form by rerunning (will reset all form fields)
                    st.rerun()
                else:
                    st.error(message)
    
    with col2:
        st.subheader("Filters")
        
        # Initialize date preset in session state if not exists (default to Current Month)
        if 'expense_date_preset' not in st.session_state:
            st.session_state['expense_date_preset'] = "Current Month"
        
        # Date range presets
        date_preset = st.selectbox(
            "Date Range",
            options=["Current Month", "Last Month", "This Year", "All Time", "Custom"],
            index=["Current Month", "Last Month", "This Year", "All Time", "Custom"].index(st.session_state.get('expense_date_preset', "Current Month")),
            key="expense_date_preset"
        )
        
        # Calculate default dates based on preset
        today = date.today()
        first_day_current_month = date(today.year, today.month, 1)
        
        if date_preset == "Current Month":
            default_start = first_day_current_month
            default_end = today
        elif date_preset == "Last Month":
            if today.month == 1:
                default_start = date(today.year - 1, 12, 1)
                last_day_last_month = date(today.year - 1, 12, 31)
            else:
                default_start = date(today.year, today.month - 1, 1)
                # Get last day of last month
                last_day = calendar.monthrange(today.year, today.month - 1)[1]
                last_day_last_month = date(today.year, today.month - 1, last_day)
            default_end = last_day_last_month
        elif date_preset == "This Year":
            default_start = date(today.year, 1, 1)
            default_end = today
        elif date_preset == "All Time":
            default_start = None
            default_end = None
        else:  # Custom
            default_start = None
            default_end = None
        
        # Date inputs - only show if Custom
        if date_preset == "Custom":
            filter_expense_start = st.date_input("Start Date", value=None, key="filter_expense_start")
            filter_expense_end = st.date_input("End Date", value=None, key="filter_expense_end")
        else:
            # Use preset dates
            filter_expense_start = default_start
            filter_expense_end = default_end
            # Show selected range
            if filter_expense_start and filter_expense_end:
                st.info(f"📅 Showing: {filter_expense_start.strftime('%b %d')} - {filter_expense_end.strftime('%b %d, %Y')}")
        
        all_expense_cats = db.get_categories('expense')
        # Put 'All' first in the list
        expense_filter_options = {'All': None}
        expense_filter_options.update({cat['name']: cat['id'] for cat in all_expense_cats})
        
        # Initialize default to "All" if not set
        if 'filter_expense_category' not in st.session_state:
            st.session_state['filter_expense_category'] = 'All'
        
        # Ensure the selected value exists in options (in case categories changed)
        if st.session_state.get('filter_expense_category') not in expense_filter_options:
            st.session_state['filter_expense_category'] = 'All'
        
        filter_expense_category_name = st.selectbox(
            "Category",
            options=list(expense_filter_options.keys()),
            key="filter_expense_category"
        )
        filter_expense_category_id = expense_filter_options.get(filter_expense_category_name)
        
        all_accounts = db.get_accounts()
        # Put 'All' first in the list
        account_filter_options = {'All': None}
        account_filter_options.update({acc['name']: acc['id'] for acc in all_accounts})
        
        # Initialize default to "All" if not set
        if 'filter_expense_account' not in st.session_state:
            st.session_state['filter_expense_account'] = 'All'
        
        # Ensure the selected value exists in options (in case accounts changed)
        if st.session_state.get('filter_expense_account') not in account_filter_options:
            st.session_state['filter_expense_account'] = 'All'
        
        filter_expense_account_name = st.selectbox(
            "Account",
            options=list(account_filter_options.keys()),
            key="filter_expense_account"
        )
        filter_expense_account_id = account_filter_options.get(filter_expense_account_name)
    
    st.markdown("---")
    st.subheader("Expense Records")
    
    # Get filtered expenses
    expense_records = db.get_expenses(
        start_date=filter_expense_start.isoformat() if filter_expense_start else None,
        end_date=filter_expense_end.isoformat() if filter_expense_end else None,
        category_id=filter_expense_category_id,
        account_id=filter_expense_account_id
    )
    
    if expense_records:
        # Display in a table format
        for record in expense_records:
            currency = record.get('currency', DEFAULT_CURRENCY)
            currency_symbol = get_currency_symbol(currency)
            with st.expander(f"📅 {record['date']} - {record['category_name']} - {currency_symbol}{record['amount']:.2f} ({record['account_name']})"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    currency = record.get('currency', DEFAULT_CURRENCY)
                    currency_symbol = get_currency_symbol(currency)
                    st.write(f"**Date:** {record['date']}")
                    st.write(f"**Category:** {record['category_name']}")
                    st.write(f"**Amount:** {currency_symbol}{record['amount']:.2f} ({currency})")
                    st.write(f"**Account:** {record['account_name']}")
                    if record['note']:
                        st.write(f"**Note:** {record['note']}")
                
                with col2:
                    if st.button("✏️ Edit", key=f"edit_expense_{record['id']}"):
                        st.session_state[f'editing_expense_{record["id"]}'] = True
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_expense_{record['id']}"):
                        success, message = db.delete_expense(record['id'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                # Edit form
                if st.session_state.get(f'editing_expense_{record["id"]}', False):
                    with st.form(f"edit_expense_form_{record['id']}", clear_on_submit=False):
                        edit_expense_date = st.date_input("Date", value=datetime.fromisoformat(record['date']).date(), key=f"edit_expense_date_{record['id']}")
                        edit_expense_categories = db.get_categories('expense')
                        edit_expense_category_options = {cat['name']: cat['id'] for cat in edit_expense_categories}
                        current_cat_name = record['category_name']
                        edit_expense_category_name = st.selectbox(
                            "Category", 
                            options=list(edit_expense_category_options.keys()),
                            index=list(edit_expense_category_options.keys()).index(current_cat_name) if current_cat_name in edit_expense_category_options else 0,
                            key=f"edit_expense_category_{record['id']}"
                        )
                        edit_expense_category_id = edit_expense_category_options[edit_expense_category_name]
                        edit_exp_col_amount, edit_exp_col_currency = st.columns([3, 1])
                        with edit_exp_col_amount:
                            edit_expense_amount = st.number_input("Amount", min_value=0.01, step=0.01, value=float(record['amount']), format="%.2f", key=f"edit_expense_amount_{record['id']}")
                        with edit_exp_col_currency:
                            current_currency = record.get('currency', DEFAULT_CURRENCY)
                            edit_expense_currency = st.selectbox(
                                "Currency",
                                options=list(CURRENCIES.keys()),
                                index=list(CURRENCIES.keys()).index(current_currency) if current_currency in CURRENCIES else 0,
                                key=f"edit_expense_currency_{record['id']}"
                            )
                        
                        edit_accounts = db.get_accounts()
                        edit_account_options = {acc['name']: acc['id'] for acc in edit_accounts}
                        current_acc_name = record['account_name']
                        edit_expense_account_name = st.selectbox(
                            "Account",
                            options=list(edit_account_options.keys()),
                            index=list(edit_account_options.keys()).index(current_acc_name) if current_acc_name in edit_account_options else 0,
                            key=f"edit_expense_account_{record['id']}"
                        )
                        edit_expense_account_id = edit_account_options[edit_expense_account_name]
                        
                        edit_expense_note = st.text_area("Note", value=record['note'] if record['note'] else "", key=f"edit_expense_note_{record['id']}")
                        
                        col_submit, col_cancel = st.columns(2)
                        with col_submit:
                            if st.form_submit_button("💾 Save"):
                                success, message = db.update_expense(
                                    expense_id=record['id'],
                                    date=edit_expense_date.isoformat(),
                                    category_id=edit_expense_category_id,
                                    amount=float(edit_expense_amount),
                                    account_id=edit_expense_account_id,
                                    currency=edit_expense_currency,
                                    note=edit_expense_note if edit_expense_note else None
                                )
                                if success:
                                    st.success(message)
                                    st.session_state[f'editing_expense_{record["id"]}'] = False
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        with col_cancel:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state[f'editing_expense_{record["id"]}'] = False
                                st.rerun()
        
        # Total
        totals_by_currency = {}
        for record in expense_records:
            curr = record.get('currency', DEFAULT_CURRENCY)
            if curr not in totals_by_currency:
                totals_by_currency[curr] = 0
            totals_by_currency[curr] += record['amount']
        
        if len(totals_by_currency) == 1:
            curr = list(totals_by_currency.keys())[0]
            curr_symbol = get_currency_symbol(curr)
            total_expense = totals_by_currency[curr]
            st.info(f"**Total Expense:** {curr_symbol}{total_expense:.2f}")
        else:
            st.info("**Total Expense by Currency:**")
            for curr, total in totals_by_currency.items():
                curr_symbol = get_currency_symbol(curr)
                st.write(f"- {curr}: {curr_symbol}{total:.2f}")
    else:
        st.info("No expense records found. Add your first expense using the form above.")

# Tab 3: Summary
with tab3:
    st.header("📊 Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        summary_start_date = st.date_input("Start Date", value=None, key="summary_start")
    with col2:
        summary_end_date = st.date_input("End Date", value=None, key="summary_end")
    
    summary = db.get_summary(
        start_date=summary_start_date.isoformat() if summary_start_date else None,
        end_date=summary_end_date.isoformat() if summary_end_date else None
    )
    
    # Display summary cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Total Income", f"₹{summary['total_income']:.2f}")
    with col2:
        st.metric("💸 Total Expense", f"₹{summary['total_expense']:.2f}")
    with col3:
        balance_color = "normal" if summary['balance'] >= 0 else "inverse"
        st.metric("💵 Balance", f"₹{summary['balance']:.2f}")
    
    st.markdown("---")
    
    # Category-wise breakdown
    st.subheader("📈 Category-wise Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Income by Category")
        income_cats = db.get_categories('income')
        income_by_category = {}
        for cat in income_cats:
            cat_income = db.get_income(
                start_date=summary_start_date.isoformat() if summary_start_date else None,
                end_date=summary_end_date.isoformat() if summary_end_date else None,
                category_id=cat['id']
            )
            total = sum(record['amount'] for record in cat_income)
            if total > 0:
                income_by_category[cat['name']] = total
        
        if income_by_category:
            for cat_name, amount in sorted(income_by_category.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{cat_name}:** ₹{amount:.2f}")
        else:
            st.info("No income data")
    
    with col2:
        st.markdown("#### Expense by Category")
        expense_cats = db.get_categories('expense')
        expense_by_category = {}
        for cat in expense_cats:
            cat_expense = db.get_expenses(
                start_date=summary_start_date.isoformat() if summary_start_date else None,
                end_date=summary_end_date.isoformat() if summary_end_date else None,
                category_id=cat['id']
            )
            total = sum(record['amount'] for record in cat_expense)
            if total > 0:
                expense_by_category[cat['name']] = total
        
        if expense_by_category:
            for cat_name, amount in sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{cat_name}:** ₹{amount:.2f}")
        else:
            st.info("No expense data")

# Tab 4: Settings
with tab4:
    st.header("⚙️ Settings")
    
    # Category Management
    st.subheader("📁 Category Management")
    
    cat_tab1, cat_tab2 = st.tabs(["💰 Income Categories", "💸 Expense Categories"])
    
    with cat_tab1:
        st.markdown("### Add Income Category")
        with st.form("add_income_category_form", clear_on_submit=True):
            new_income_cat_name = st.text_input("Category Name", key="new_income_cat")
            submit_income_cat = st.form_submit_button("➕ Add Category")
            
            if submit_income_cat and new_income_cat_name:
                success, message = db.add_category(new_income_cat_name, 'income')
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown("### Existing Income Categories")
        income_categories = db.get_categories('income')
        if income_categories:
            for cat in income_categories:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📁 {cat['name']}")
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_income_cat_{cat['id']}"):
                        success, message = db.delete_category(cat['id'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("No income categories")
    
    with cat_tab2:
        st.markdown("### Add Expense Category")
        with st.form("add_expense_category_form", clear_on_submit=True):
            new_expense_cat_name = st.text_input("Category Name", key="new_expense_cat")
            submit_expense_cat = st.form_submit_button("➕ Add Category")
            
            if submit_expense_cat and new_expense_cat_name:
                success, message = db.add_category(new_expense_cat_name, 'expense')
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown("### Existing Expense Categories")
        expense_categories = db.get_categories('expense')
        if expense_categories:
            for cat in expense_categories:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📁 {cat['name']}")
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_expense_cat_{cat['id']}"):
                        success, message = db.delete_category(cat['id'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("No expense categories")
    
    st.markdown("---")
    
    # Account Management
    st.subheader("🏦 Account Management")
    
    st.markdown("### Add Account")
    with st.form("add_account_form", clear_on_submit=True):
        new_account_name = st.text_input("Account Name", key="new_account")
        submit_account = st.form_submit_button("➕ Add Account")
        
        if submit_account and new_account_name:
            success, message = db.add_account(new_account_name)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    st.markdown("### Existing Accounts")
    accounts = db.get_accounts()
    if accounts:
        for acc in accounts:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"🏦 {acc['name']}")
            with col2:
                if st.button("🗑️ Delete", key=f"delete_account_{acc['id']}"):
                    success, message = db.delete_account(acc['id'])
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    else:
        st.info("No accounts") 
    
    st.markdown("---")
    
    # Source Management
    st.subheader("📥 Source Management")
    
    st.markdown("### Add Source")
    with st.form("add_source_form", clear_on_submit=True):
        new_source_name = st.text_input("Source Name", key="new_source")
        submit_source = st.form_submit_button("➕ Add Source")
        
        if submit_source and new_source_name:
            success, message = db.add_source(new_source_name)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    st.markdown("### Existing Sources")
    sources = db.get_sources()
    if sources:
        for src in sources:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📥 {src['name']}")
            with col2:
                if st.button("🗑️ Delete", key=f"delete_source_{src['id']}"):
                    success, message = db.delete_source(src['id'])
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    else:
        st.info("No sources") 