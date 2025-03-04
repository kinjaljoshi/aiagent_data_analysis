import os
import logging
import pandas as pd
from google.cloud import bigquery
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langgraph.graph import StateGraph, END
from query_processing import get_query_context

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(" OPENAI_API_KEY is missing! Set it in your environment.")

# Load Sentence Transformer Model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize BigQuery client
bq_client = bigquery.Client(project="your_project_id")

# Load FAISS Index (Handle errors)
try:
    vector_db = FAISS.load_local("faiss_table_index", embedding_model, allow_dangerous_deserialization=True)
    logging.info(" FAISS index loaded successfully.")
except Exception as e:
    logging.error(f" Error loading FAISS index: {e}")
    vector_db = None

# Initialize OpenAI LLM
llm = OpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Define StateGraph Workflow
workflow = StateGraph(dict)  #  LangGraph requires a dict-based state

# Step 1: Classify Query
def classify_query(state):
    """Classifies the user query using LLM as either 'General Query' or 'Database Query'."""
    print("++++++++++ Entering classify_query ++++++++++")
    
    classification_prompt = f"""
    Classify the following user query as either 'General Query' or 'Database Query':

    Query: "{state['query_text']}"

    Return ONLY 'General Query' or 'Database Query'.
    """

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are an AI that classifies queries."},
            {"role": "user", "content": classification_prompt},
        ])
        query_type = response.strip()  # Ensure we capture only the classification
        if query_type not in ["General Query", "Database Query"]:
            query_type = "General Query"  # Default fallback
    except Exception as e:
        logging.error(f" Error classifying query with LLM: {e}")
        query_type = "General Query"  # Default to general if LLM fails

    print(f"State after classify_query: {state}, Query Type: {query_type}")
    print("++++++++++ Exiting classify_query ++++++++++")
    return {"query_type": query_type}

# Step 2: Fetch Query Context from FAISS
def get_query_context_wrapper(state):
    """Fetch query context from FAISS using query_processing.py."""
    print("++++++++++ Entering get_query_context_wrapper ++++++++++")
    query_context = get_query_context(state["query_text"]) if vector_db else "FAISS index unavailable."
    print(f"State after get_query_context_wrapper: {state}, Query Context: {query_context}")
    print("++++++++++ Exiting get_query_context_wrapper ++++++++++")
    return {"query_context": query_context}

# Step 3: Generate SQL Query
def generate_sql_query(state):
    """Uses LLM to generate SQL query based on FAISS context."""
    print("++++++++++ Entering generate_sql_query ++++++++++")
    if state["query_context"] == "FAISS index unavailable.":
        print("++++++++++ Exiting generate_sql_query (No FAISS context) ++++++++++")
        return {"sql_query": "No query generated due to missing FAISS context."}

    prompt = f"Using this context: {state['query_context']}, generate an SQL query to fetch relevant information."
    try:
        sql_query = llm(prompt)
    except Exception as e:
        logging.error(f" Error generating SQL query: {e}")
        sql_query = "SQL generation failed."

    print(f"State after generate_sql_query: {state}, SQL Query: {sql_query}")
    print("++++++++++ Exiting generate_sql_query ++++++++++")
    return {"sql_query": sql_query}

# Step 4: Execute Query in BigQuery
def execute_query(state):
    """Runs the generated SQL query in BigQuery."""
    print("++++++++++ Entering execute_query ++++++++++")
    if "failed" in state["sql_query"].lower() or "No query generated" in state["sql_query"]:
        print("++++++++++ Exiting execute_query (No valid SQL) ++++++++++")
        return {"results": "No valid SQL query to execute."}

    try:
        query_job = bq_client.query(state["sql_query"])
        df = query_job.to_dataframe()
        results = df
    except Exception as e:
        logging.error(f" Error executing SQL query: {e}")
        results = "SQL execution failed."

    print(f"State after execute_query: {state}, Results: {results}")
    print("++++++++++ Exiting execute_query ++++++++++")
    return {"results": results}

# Step 5: Handle General Queries via LLM
def llm_response(state):
    """Returns an LLM-generated response for general queries."""
    print("++++++++++ Entering llm_response ++++++++++")
    try:
        results = llm(state["query_text"])
    except Exception as e:
        logging.error(f" LLM response error: {e}")
        results = "Error retrieving response."

    print(f"State after llm_response: {state}, LLM Response: {results}")
    print("++++++++++ Exiting llm_response ++++++++++")
    return {"results": results}

# Add Nodes to Workflow
workflow.add_node("classify_query", classify_query)
workflow.add_node("get_query_context", get_query_context_wrapper)
workflow.add_node("generate_sql_query", generate_sql_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("llm_response", llm_response)

# Define Conditional Routing
def classify_edge(state):
    """Determines the next step based on LLM classification."""
    return "llm_response" if state["query_type"] == "General Query" else "get_query_context"

workflow.add_conditional_edges("classify_query", classify_edge)

# Connect Steps
workflow.add_edge("get_query_context", "generate_sql_query")
workflow.add_edge("generate_sql_query", "execute_query")

#  Use END node for workflow termination
workflow.add_edge("execute_query", END)
workflow.add_edge("llm_response", END)

# Set Workflow Entry Point
workflow.set_entry_point("classify_query")

# Compile Workflow
executor = workflow.compile()

# Function to Process Queries
def process_query(user_input):
    """Processes user input through LangGraph workflow and returns results."""
    print("++++++++++ Starting process_query ++++++++++")
    initial_state = {"query_text": user_input}
    final_state = executor.invoke(initial_state)
    print(f"++++++++++ Final State in process_query: {final_state} ++++++++++")
    print("++++++++++ Exiting process_query ++++++++++")
    return final_state["results"]  #  Extract results from the dict
