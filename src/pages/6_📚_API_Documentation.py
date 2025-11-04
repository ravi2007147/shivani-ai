"""API Documentation page for Expense Manager APIs."""

import streamlit as st
import json
import requests
from src.utils import ExpenseDB, is_api_server_running, get_api_server_url

st.set_page_config(
    page_title="API Documentation",
    page_icon="📚",
    layout="wide"
)

st.title("📚 API Documentation")
st.markdown("Interactive API documentation for Expense Manager REST APIs")

# Check API server status
api_base_url = "http://127.0.0.1:8000"
api_running = is_api_server_running()

if not api_running:
    st.error("❌ API server is not running. Please wait a moment for it to start, or check the main page sidebar for status.")
    st.info("The API server should start automatically when you launch Streamlit. If it's not running, you may need to restart the Streamlit app.")
    st.stop()

st.success(f"✅ API Server is running at {api_base_url}")

# Fetch OpenAPI schema
@st.cache_data(ttl=60)
def fetch_openapi_schema(base_url: str):
    """Fetch OpenAPI schema from the API server."""
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Failed to fetch OpenAPI schema: {str(e)}")
        return None

openapi_schema = fetch_openapi_schema(api_base_url)

if not openapi_schema:
    st.error("Could not load API documentation. Make sure the API server is running.")
    st.stop()

# Display API info
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("API Title", openapi_schema.get("info", {}).get("title", "N/A"))
with col2:
    st.metric("Version", openapi_schema.get("info", {}).get("version", "N/A"))
with col3:
    st.metric("Endpoints", len(openapi_schema.get("paths", {})))

st.markdown("---")

# Sidebar for endpoint navigation
st.sidebar.title("🔗 Endpoints")
endpoints = list(openapi_schema.get("paths", {}).keys())

if endpoints:
    # Add "All Endpoints" option
    endpoint_options = ["All Endpoints"] + endpoints
    selected_option = st.sidebar.selectbox(
        "Filter by endpoint:",
        options=endpoint_options,
        format_func=lambda x: f"{x}"
    )
    
    # Determine which endpoints to show
    if selected_option == "All Endpoints":
        paths_to_show = endpoints
    else:
        paths_to_show = [selected_option]
else:
    paths_to_show = []
    st.sidebar.warning("No endpoints found")

# Helper function to make API call
def make_api_call(method: str, url: str, json_body: dict = None):
    """Make an API call and return the response."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
            return response.status_code, response.json() if response.content else {}
        elif method.upper() == "POST":
            headers = {"Content-Type": "application/json"} if json_body else {}
            response = requests.post(url, json=json_body, headers=headers, timeout=10)
            return response.status_code, response.json() if response.content else {}
        else:
            return None, {"error": f"Method {method} not supported in this tester"}
    except requests.exceptions.RequestException as e:
        return None, {"error": str(e)}

# Display each endpoint
for path in paths_to_show:
    path_data = openapi_schema["paths"][path]
    
    for method, details in path_data.items():
        if method not in ["get", "post", "put", "delete", "patch"]:
            continue
        
        method_upper = method.upper()
        method_colors = {
            "GET": "🟢",
            "POST": "🔵",
            "PUT": "🟡",
            "DELETE": "🔴",
            "PATCH": "🟠"
        }
        method_emoji = method_colors.get(method_upper, "⚪")
        
        # Create expander for each endpoint
        endpoint_id = f"{method_upper}_{path}"
        with st.expander(f"{method_emoji} **{method_upper}** `{path}`", expanded=(len(paths_to_show) == 1)):
            # Description
            description = details.get("summary", "") or details.get("description", "No description available")
            st.markdown(f"**Description:** {description}")
            
            # Parameters
            parameters = details.get("parameters", [])
            if parameters:
                st.markdown("#### 📋 Parameters")
                params_data = []
                for param in parameters:
                    params_data.append({
                        "Name": param.get("name", ""),
                        "Location": param.get("in", ""),
                        "Required": "Yes" if param.get("required", False) else "No",
                        "Type": param.get("schema", {}).get("type", "string"),
                        "Description": param.get("description", "")
                    })
                st.table(params_data)
            
            # Request Body (for POST, PUT, etc.)
            request_body = details.get("requestBody", {})
            if request_body:
                st.markdown("#### 📤 Request Body")
                content = request_body.get("content", {})
                for content_type, schema_info in content.items():
                    st.markdown(f"**Content Type:** `{content_type}`")
                    schema = schema_info.get("schema", {})
                    if schema:
                        st.json(schema)
            
            # Response Schema
            responses = details.get("responses", {})
            if responses:
                st.markdown("#### 📥 Responses")
                for status_code, response_info in responses.items():
                    st.markdown(f"**{status_code}** - {response_info.get('description', '')}")
                    content = response_info.get("content", {})
                    if content:
                        for content_type, schema_info in content.items():
                            schema = schema_info.get("schema", {})
                            if schema:
                                with st.expander(f"View {status_code} response schema"):
                                    st.json(schema)
            
            # Try it out section
            st.markdown("---")
            st.markdown("#### 🧪 Try it out")
            
            # Build full URL
            full_url = f"{api_base_url}{path}"
            
            # For POST/PUT requests, show JSON input
            json_input = None
            if method_upper in ["POST", "PUT", "PATCH"]:
                request_body = details.get("requestBody", {})
                if request_body:
                    # Generate example JSON from schema
                    content = request_body.get("content", {})
                    if "application/json" in content:
                        schema = content["application/json"].get("schema", {})
                        example_json = {}
                        
                        # Try to extract properties from schema
                        if "properties" in schema:
                            properties = schema.get("properties", {})
                            for prop_name, prop_schema in properties.items():
                                prop_type = prop_schema.get("type", "string")
                                default_value = prop_schema.get("default")
                                
                                # Use default if available
                                if default_value is not None:
                                    example_json[prop_name] = default_value
                                elif prop_type == "string":
                                    # Provide better examples based on field name
                                    if "date" in prop_name.lower():
                                        from datetime import date
                                        example_json[prop_name] = date.today().isoformat()
                                    elif "currency" in prop_name.lower():
                                        example_json[prop_name] = "INR"
                                    else:
                                        example_json[prop_name] = ""
                                elif prop_type == "integer":
                                    if "id" in prop_name.lower():
                                        example_json[prop_name] = 1  # Example ID
                                    else:
                                        example_json[prop_name] = 0
                                elif prop_type == "number":
                                    if "amount" in prop_name.lower():
                                        example_json[prop_name] = 1000.0  # Example amount
                                    else:
                                        example_json[prop_name] = 0.0
                                elif prop_type == "boolean":
                                    example_json[prop_name] = False
                                else:
                                    example_json[prop_name] = None
                        
                        # Show JSON input
                        json_input_text = st.text_area(
                            "Request Body (JSON):",
                            value=json.dumps(example_json, indent=2),
                            height=150,
                            key=f"json_input_{endpoint_id}",
                            help="Edit the JSON payload for this request"
                        )
                        
                        try:
                            json_input = json.loads(json_input_text)
                        except json.JSONDecodeError as e:
                            st.error(f"Invalid JSON: {str(e)}")
                            json_input = None
            
            # Show URL and send button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(full_url, language="bash")
            with col2:
                button_disabled = (method_upper in ["POST", "PUT", "PATCH"] and json_input is None)
                if st.button("▶️ Send Request", key=f"test_{endpoint_id}", type="primary", disabled=button_disabled):
                    with st.spinner("Calling API..."):
                        status_code, response_data = make_api_call(method_upper, full_url, json_input)
                        
                        if status_code:
                            if status_code >= 200 and status_code < 300:
                                st.success(f"✅ Status Code: {status_code}")
                            else:
                                st.warning(f"⚠️ Status Code: {status_code}")
                            st.markdown("**Response:**")
                            st.json(response_data)
                        else:
                            st.error("❌ Request Failed")
                            st.json(response_data)

# Also show actual current data from database
st.markdown("---")
st.markdown("## 📊 Current Data")

tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Income Categories",
    "💸 Expense Categories", 
    "💳 Accounts",
    "📥 Sources"
])

# Initialize database
@st.cache_resource
def get_expense_db():
    return ExpenseDB()

db = get_expense_db()

with tab1:
    income_categories = db.get_categories(category_type='income')
    if income_categories:
        categories_data = [{"ID": cat["id"], "Name": cat["name"], "Type": cat["type"]} for cat in income_categories]
        st.dataframe(categories_data, use_container_width=True)
    else:
        st.info("No income categories found.")

with tab2:
    expense_categories = db.get_categories(category_type='expense')
    if expense_categories:
        categories_data = [{"ID": cat["id"], "Name": cat["name"], "Type": cat["type"]} for cat in expense_categories]
        st.dataframe(categories_data, use_container_width=True)
    else:
        st.info("No expense categories found.")

with tab3:
    accounts = db.get_accounts()
    if accounts:
        accounts_data = [{"ID": acc["id"], "Name": acc["name"]} for acc in accounts]
        st.dataframe(accounts_data, use_container_width=True)
    else:
        st.info("No accounts found.")

with tab4:
    sources = db.get_sources()
    if sources:
        sources_data = [{"ID": src["id"], "Name": src["name"]} for src in sources]
        st.dataframe(sources_data, use_container_width=True)
    else:
        st.info("No sources found.")

# Footer
st.markdown("---")
st.markdown("### ℹ️ About")
st.info("""
This page displays the API documentation directly from the running API server using the OpenAPI specification. 
You can test endpoints directly from this page without needing to visit external URLs.

**Features:**
- ✅ Complete API documentation from OpenAPI schema
- ✅ Interactive endpoint testing
- ✅ View actual current data from the database
- ✅ All functionality within Streamlit - no external URLs needed
""")
