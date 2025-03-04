import os
import logging
import pandas as pd
from google.cloud import bigquery
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langgraph.graph import StateGraph, END
from query_processing import get_query_context
from openai import OpenAI

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(" OPENAI_API_KEY is missing! Set it in your environment.")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

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

# Define StateGraph Workflow
workflow = StateGraph(dict)  #  LangGraph requires a dict-based state

def classify_query(state):
    """Classifies the user query using LLM as either 'General Query' or 'Database Query'."""
    print("++++++++++ Entering classify_query ++++++++++")

    classification_prompt = f"""
    Classify the following user query as either 'General Query' or 'Database Query':

    Query: "{state['query_text']}"

    Return ONLY 'General Query' or 'Database Query'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that classifies queries."},
                {"role": "user", "content": classification_prompt},
            ]
        )
        query_type = response.choices[0].message.content.strip()

        if query_type not in ["General Query", "Database Query"]:
            print(" Invalid LLM response, defaulting to General Query")
            query_type = "General Query"

    except Exception as e:
        print(f" Error classifying query with LLM: {e}")
        query_type = "General Query"

    print(f"State after classify_query: {state}, Query Type: {query_type}")
    print("++++++++++ Exiting classify_query ++++++++++")
    return {"query_type": query_type}

# Add Nodes to Workflow
workflow.add_node("classify_query", classify_query)

# Conditional Routing Using LLM Classification
workflow.add_conditional_edges("classify_query", classify_edge)

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
    return final_state["results"]
