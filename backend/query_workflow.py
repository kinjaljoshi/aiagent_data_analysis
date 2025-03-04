import os
import faiss
import pandas as pd
from google.cloud import bigquery
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langgraph.graph import StateGraph
from query_processing import get_query_context


# Load Sentence Transformer Model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize BigQuery client
bq_client = bigquery.Client(project="your_project_id")

# Load FAISS Index
vector_db = FAISS.load_local("faiss_table_index", embedding_model, allow_dangerous_deserialization=True)

# Initialize LLM
llm = OpenAI(model_name="gpt-4")  # Ensure API key is set

class QueryState:
    def __init__(self, query_text):
        self.query_text = query_text
        self.query_type = None
        self.query_context = None
        self.sql_query = None
        self.results = None

def classify_query(state):
    """ Classifies the user query as either 'General Query' or 'DB Query'. """
    query_text = state.query_text.lower()
    if any(word in query_text for word in ["warehouse", "inventory", "item", "stock"]):
        state.query_type = "DB Query"
    else:
        state.query_type = "General Query"
    return state

def get_query_context_wrapper(state):
    """Wrapper to fetch query context from FAISS using query_processing.py."""
    state.query_context = get_query_context(state.query_text)
    return state


def generate_sql_query(state):
    """ Uses LLM to generate SQL query based on FAISS context. """
    context = state.query_context
    prompt = f"Using this context: {context}, generate an SQL query to fetch relevant information."
    state.sql_query = llm(prompt)
    return state

def execute_query(state):
    """ Runs the generated SQL query in BigQuery. """
    query_job = bq_client.query(state.sql_query)
    df = query_job.to_dataframe()
    state.results = df
    return state

def llm_response(state):
    """ Returns an LLM-generated response for general queries. """
    state.results = llm(state.query_text)
    return state

# Define Workflow using LangGraph
workflow = StateGraph(QueryState)
workflow.add_node("classify_query", classify_query)
workflow.add_node("get_query_context", get_query_context_wrapper)
workflow.add_node("generate_sql_query", generate_sql_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("llm_response", llm_response)

# Define Workflow Edges
workflow.add_edge("classify_query", "llm_response", condition=lambda state: state.query_type == "General Query")
workflow.add_edge("classify_query", "get_query_context", condition=lambda state: state.query_type == "DB Query")
workflow.add_edge("get_query_context", "generate_sql_query")
workflow.add_edge("generate_sql_query", "execute_query")

# Compile Workflow
workflow.set_entry_point("classify_query")
workflow.set_termination_nodes(["execute_query", "llm_response"])
executor = workflow.compile()

def process_query(user_input):
    """ Processes user input through LangGraph workflow and returns results. """
    query_state = QueryState(user_input)
    final_state = executor.invoke(query_state)
    return final_state.results
