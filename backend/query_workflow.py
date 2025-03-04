import os
import logging
import pandas as pd
from google.cloud import bigquery
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langgraph.graph import StateGraph
from query_processing import get_query_context

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing! Set it in your environment.")

# Load Sentence Transformer Model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize BigQuery client
bq_client = bigquery.Client(project="your_project_id")

# Load FAISS Index (Handle errors)
try:
    vector_db = FAISS.load_local("faiss_table_index", embedding_model, allow_dangerous_deserialization=True)
    logging.info("✅ FAISS index loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading FAISS index: {e}")
    vector_db = None

# Initialize OpenAI LLM
llm = OpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Define State Class
class QueryState:
    def __init__(self, query_text):
        self.query_text = query_text
        self.query_type = None
        self.query_context = None
        self.sql_query = None
        self.results = None

# Step 1: Classify Query
def classify_query(state):
    """Classifies the user query as either 'General Query' or 'DB Query'."""
    query_text = state.query_text.lower()
    if any(word in query_text for word in ["warehouse", "inventory", "item", "stock"]):
        state.query_type = "DB Query"
    else:
        state.query_type = "General Query"
    return state

# Step 2: Fetch Query Context from FAISS
def get_query_context_wrapper(state):
    """Wrapper to fetch query context from FAISS using query_processing.py."""
    if vector_db:
        state.query_context = get_query_context(state.query_text)
    else:
        state.query_context = "FAISS index unavailable."
    return state

# Step 3: Generate SQL Query
def generate_sql_query(state):
    """Uses LLM to generate SQL query based on FAISS context."""
    if state.query_context == "FAISS index unavailable.":
        state.sql_query = "No query generated due to missing FAISS context."
        return state

    prompt = f"Using this context: {state.query_context}, generate an SQL query to fetch relevant information."
    try:
        state.sql_query = llm(prompt)
    except Exception as e:
        logging.error(f" Error generating SQL query: {e}")
        state.sql_query = "SQL generation failed."
    return state

# Step 4: Execute Query in BigQuery
def execute_query(state):
    """Runs the generated SQL query in BigQuery."""
    if "failed" in state.sql_query.lower() or "No query generated" in state.sql_query:
        state.results = "No valid SQL query to execute."
        return state

    try:
        query_job = bq_client.query(state.sql_query)
        df = query_job.to_dataframe()
        state.results = df
    except Exception as e:
        logging.error(f" Error executing SQL query: {e}")
        state.results = "SQL execution failed."
    return state

# Step 5: Handle General Queries via LLM
def llm_response(state):
    """Returns an LLM-generated response for general queries."""
    try:
        state.results = llm(state.query_text)
    except Exception as e:
        logging.error(f" LLM response error: {e}")
        state.results = "Error retrieving response."
    return state

# Define Workflow using LangGraph
workflow = StateGraph(QueryState)

# Add Nodes
workflow.add_node("classify_query", classify_query)
workflow.add_node("get_query_context", get_query_context_wrapper)
workflow.add_node("generate_sql_query", generate_sql_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("llm_response", llm_response)

# Conditional Routing
def classify_edge(state):
    """Determines the next step based on query type."""
    return "llm_response" if state.query_type == "General Query" else "get_query_context"

workflow.add_conditional_edges("classify_query", classify_edge)

# Connect Steps
workflow.add_edge("get_query_context", "generate_sql_query")
workflow.add_edge("generate_sql_query", "execute_query")

# Set Workflow Entry & Termination Nodes
workflow.set_entry_point("classify_query")
workflow.set_finish_edge("execute_query")  # Corrected termination condition
workflow.set_finish_edge("llm_response")   # Corrected termination condition

# Compile Workflow
executor = workflow.compile()

# Function to Process Queries
def process_query(user_input):
    """Processes user input through LangGraph workflow and returns results."""
    query_state = QueryState(user_input)
    final_state = executor.invoke(query_state)
    return final_state.results
