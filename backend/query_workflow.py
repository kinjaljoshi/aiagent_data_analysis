import os
import logging
import pandas as pd
import re
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
    raise ValueError("❌ OPENAI_API_KEY is missing! Set it in your environment.")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load Sentence Transformer Model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize BigQuery client
bq_client = bigquery.Client(project="llm-text-to-sql-445914")

# Load FAISS Index (Handle errors)
try:
    vector_db = FAISS.load_local("faiss_table_index", embedding_model, allow_dangerous_deserialization=True)
    logging.info("✅ FAISS index loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading FAISS index: {e}")
    vector_db = None

# Function to replace table names with project & dataset in BigQuery SQL
def replace_table_name_with_project_and_dataset(raw_query, project_id="llm-text-to-sql-445914", dataset_id="llm_text_to_sql"):
    """Replaces table names in the query with the format `project_id.dataset_id.table_name`."""
    print("++++++++++ Entering replace_table_name_with_project_and_dataset ++++++++++")
    
    table_pattern = re.compile(r"\bFROM\s+(\w+)\b|\bJOIN\s+(\w+)\b", re.IGNORECASE)

    def replace_table(match):
        table_name = match.group(1) or match.group(2)
        return match.group(0).replace(table_name, f"`{project_id}.{dataset_id}.{table_name}`")

    modified_query = table_pattern.sub(replace_table, raw_query)

    print(f"Modified Query: {modified_query}")
    print("++++++++++ Exiting replace_table_name_with_project_and_dataset ++++++++++")
    return modified_query

# Function to classify user query using OpenAI
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that classifies queries."},
                {"role": "user", "content": classification_prompt},
            ]
        )
        query_type = response.choices[0].message.content.strip()

        if query_type not in ["General Query", "Database Query"]:
            print("❌ Invalid LLM response, defaulting to General Query")
            query_type = "General Query"

    except Exception as e:
        print(f"❌ Error classifying query with LLM: {e}")
        query_type = "General Query"

    print(f"State after classify_query: {state}, Query Type: {query_type}")
    updated_state = {**state, "query_type": query_type}  
    print("++++++++++ Exiting classify_query ++++++++++")
    return updated_state

# Function to fetch Query Context from FAISS
def get_query_context_wrapper(state):
    """Fetch query context from FAISS using query_processing.py."""
    print("++++++++++ Entering get_query_context_wrapper ++++++++++")
    query_context = get_query_context(state["query_text"]) if vector_db else "FAISS index unavailable."
    print(f"State after get_query_context_wrapper: {state}, Query Context: {query_context}")
    print("++++++++++ Exiting get_query_context_wrapper ++++++++++")
    return {"query_context": query_context}

# Function to generate SQL query using OpenAI
def generate_sql_query(state):
    """Uses LLM to generate an SQL query based on FAISS context, ensuring no explanations or markdown."""
    final_context = 'Question:' + state["query_text"] + 'SQL Context:' + state['query_context']
    print("++++++++++ Entering generate_sql_query ++++++++++", state['query_context'])
    if state["query_context"] == "FAISS index unavailable.":
        print("++++++++++ Exiting generate_sql_query (No FAISS context) ++++++++++")
        return {**state, "sql_query": "No query generated due to missing FAISS context."}
    
    # Updated prompt to ensure only SQL is returned
    prompt = f"""
    You are an expert SQL generator. 

    Based on the provided context, generate only a valid SQL query. Generate unique alias for each table used.
    Only use tables needed for answering user question.
    Include condition for aggregation and filtering as needed

    Do not include any explanations, formatting, markdown, or comments. 

    Context:
    {final_context}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI specialized in generating precise SQL queries."},
                {"role": "user", "content": prompt},
            ]
        )
        sql_query = response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"❌ Error generating SQL query: {e}")
        sql_query = "SQL generation failed."

    print(f"Generated SQL Query: {sql_query}")
    print("++++++++++ Exiting generate_sql_query ++++++++++")
    return {**state, "sql_query": sql_query}

# Function to execute query in BigQuery
def execute_query(state):
    """Runs the generated SQL query in BigQuery after formatting table names."""
    print("++++++++++ Entering execute_query ++++++++++")

    if "failed" in state["sql_query"].lower() or "No query generated" in state["sql_query"]:
        print("++++++++++ Exiting execute_query (No valid SQL) ++++++++++")
        return {"results": "No valid SQL query to execute."}

    try:
        formatted_sql = replace_table_name_with_project_and_dataset(state["sql_query"])
        query_job = bq_client.query(formatted_sql)
        df = query_job.to_dataframe()
        results = df

    except Exception as e:
        logging.error(f"❌ Error executing SQL query: {e}")
        results = "SQL execution failed."

    print(f"State after execute_query: {state}, Results: {results}")
    print("++++++++++ Exiting execute_query ++++++++++")
    return {"results": results}

# Define Conditional Routing Using LLM Classification
def classify_edge(state):
    """Determines the next step based on LLM classification."""
    return "llm_response" if state["query_type"] == "General Query" else "get_query_context"

# Define LangGraph Workflow
workflow = StateGraph(dict)
workflow.add_node("classify_query", classify_query)
workflow.add_node("get_query_context", get_query_context_wrapper)
workflow.add_node("generate_sql_query", generate_sql_query)
workflow.add_node("execute_query", execute_query)
workflow.add_conditional_edges("classify_query", classify_edge)
workflow.add_edge("get_query_context", "generate_sql_query")
workflow.add_edge("generate_sql_query", "execute_query")
workflow.add_edge("execute_query", END)

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
