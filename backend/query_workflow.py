import os
import logging
import pandas as pd
import re
import matplotlib.pyplot as plt
from google.cloud import bigquery
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langgraph.graph import StateGraph, END
from query_processing import get_query_context
from openai import OpenAI
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
import streamlit as st


# ---------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing! Set it in your environment.")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load Sentence Transformer Model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize BigQuery client
bq_client = bigquery.Client(project="llm-text-to-sql-445914")

# Load FAISS Index
try:
    vector_db = FAISS.load_local("faiss_table_index", embedding_model, allow_dangerous_deserialization=True)
    logging.info("FAISS index loaded successfully.")
except Exception as e:
    logging.error(f"Error loading FAISS index: {e}")
    vector_db = None

# ---------------------------------------------------------------------
# Utility: Replace table names in SQL with project.dataset.table
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Classify user query using OpenAI
# ---------------------------------------------------------------------
def classify_query(state):
    """Classifies the user query using LLM as either 'General Query' or 'Database Query'."""
    print("++++++++++ Entering classify_query ++++++++++")
    classification_prompt = f"""
    Classify the following user query as either 'General Query' 'General Query with DB context' or 'Database Query' or 'Plot Requested':
    Query: "{state['query_text']}"
    Return ONLY 'General Query' 'General Query with DB context' or 'Database Query'.
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
        if query_type not in ["General Query", "Database Query", "General Query with DB context","Plot Requested"]:
            print("Invalid LLM response, defaulting to General Query")
            query_type = "General Query"
    except Exception as e:
        print(f"Error classifying query with LLM: {e}")
        query_type = "General Query"

    print(f"{'++++' * 10}State after classify_query: {state}, Query Type: {query_type} {'++++' * 10}")
    updated_state = {**state, "query_type": query_type}  
    print("++++++++++ Exiting classify_query ++++++++++")
    return updated_state

# ---------------------------------------------------------------------
# Fetch Query Context from FAISS
# ---------------------------------------------------------------------
def get_query_context_wrapper(state):
    """Fetch query context from FAISS using query_processing.py."""
    print("++++++++++ Entering get_query_context_wrapper ++++++++++")
    query_context = get_query_context(state["query_text"]) if vector_db else "FAISS index unavailable."
    print(f"State after get_query_context_wrapper: {state}, Query Context: {query_context}")
    print("++++++++++ Exiting get_query_context_wrapper ++++++++++")
    updated_state = {**state, "query_context": query_context}
    return updated_state

# ---------------------------------------------------------------------
# Generate SQL Query using OpenAI
# ---------------------------------------------------------------------
def generate_sql_query(state):
    """Uses LLM to generate an SQL query based on FAISS context."""
    final_context = 'Question:' + state['query_text'] + 'SQL Context:' + state['query_context']
    print('++++++++++++++++++++++++++++Final Context ++++++++++++', final_context)
    print("++++++++++ Entering generate_sql_query ++++++++++", state['query_context'])

    if state["query_context"] == "FAISS index unavailable.":
        print("++++++++++ Exiting generate_sql_query (No FAISS context) ++++++++++")
        return {**state, "sql_query": "No query generated due to missing FAISS context."}
    
    prompt = f"""
    You are an expert SQL generator. 
    Based on the provided context, generate only a valid SQL query. Generate a unique alias for each table used.
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
        logging.error(f"Error generating SQL query: {e}")
        sql_query = "SQL generation failed."

    print(f"Generated SQL Query: {sql_query}")
    print("++++++++++ Exiting generate_sql_query ++++++++++")
    return {**state, "sql_query": sql_query}

def llm_response(state):
    """Uses LLM to generate an answer that doesn't necessarily require DB context."""
    final_context = 'Question:' + state['query_text']
    print('++++++++++++++++++++++++++++Final Context ++++++++++++', final_context)
    prompt = f"""
    You are a helpful assistant. Answer the user question:
    {final_context}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant. Never mention OpenAI or GPT. Always concise."},
                {"role": "user", "content": prompt},
            ]
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        answer = "Response generation failed."

    print(f"Generated Answer: {answer}")
    print("++++++++++ Exiting llm_response ++++++++++")
    return {**state, "results": answer}

def llm_sql_response(state):
    """Uses LLM to generate an answer for queries classified as 'General Query with DB context'."""
    final_context = 'Question:' + state['query_text']
    print('++++++++++++++++++++++++++++Final Context ++++++++++++', final_context)
    prompt = f"""
    Answer the user question with the context provided:
    {final_context}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant. Never mention OpenAI or GPT. Always concise."},
                {"role": "user", "content": prompt},
            ]
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        answer = "Response generation failed."

    print(f"Generated Answer: {answer}")
    print("++++++++++ Exiting llm_sql_response ++++++++++")
    return {**state, "results": answer}

# ---------------------------------------------------------------------
# Execute SQL query in BigQuery
# ---------------------------------------------------------------------
def execute_query(state):
    """Runs the generated SQL query in BigQuery after formatting table names."""
    print("++++++++++ Entering execute_query ++++++++++")
    sql_query = state.get("sql_query", "")

    if "failed" in sql_query.lower() or "No query generated" in sql_query:
        print("++++++++++ Exiting execute_query (No valid SQL) ++++++++++")
        return {**state, "results": "No valid SQL query to execute."}

    try:
        formatted_sql = replace_table_name_with_project_and_dataset(sql_query)
        query_job = bq_client.query(formatted_sql)
        df = query_job.to_dataframe()
        # Store DataFrame in state so the plot node can access it
        # Save as parquet file
        os.makedirs("parquet_cache", exist_ok=True)
        parquet_path = "parquet_cache/df_file.parquet"
        df.to_parquet(parquet_path)
        updated_state = {**state, "results": df, "df": df}
    except Exception as e:
        logging.error(f"Error executing SQL query: {e}")
        updated_state = {**state, "results": "SQL execution failed.", "df": None}

    print(f"State after execute_query: {updated_state}, Results: {updated_state['results']}")
    print("++++++++++ Exiting execute_query ++++++++++")
    return updated_state

# ---------------------------------------------------------------------
# Decide Next Step Based on Classification
# ---------------------------------------------------------------------
def classify_edge(state):
    """Determines the next step based on LLM classification."""
    if state["query_type"] == "General Query with DB context":
        return "llm_sql_response"
    elif state["query_type"] == "General Query":
        return "llm_response"
    elif state["query_type"]== "Plot Requested":
        return "plot_chart" 
    else:
        # 'Database Query'
        return "get_query_context"

# ---------------------------------------------------------------------
# Chart-request detection
# ---------------------------------------------------------------------
def is_chart_requested(context: dict) -> bool:
    """
    Checks whether the user request indicates they want a chart.
    Simple keyword-based approach.
    """
    user_request = state['query_text'].lower()
    trigger_words = ["plot", "chart", "visualize", "bar chart", "line chart", "graph"]
    return any(word in user_request for word in trigger_words)

def is_not_chart_requested(context: dict) -> bool:
    """Returns the negation of is_chart_requested."""
    return not is_chart_requested(context)

# ---------------------------------------------------------------------
# Mock function for generating matplotlib code via an LLM
# ---------------------------------------------------------------------
def call_llm_for_plot_code(prompt: str) -> str:
    """
    In production, you'd call your real LLM here. 
    For illustration, returns a simple snippet that plots the first 2 columns of 'df'.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that writes Python matplotlib code. Do not use seaborn."},
                {"role": "user", "content": prompt},
            ],
            # Possibly define temperature or other params
        )
        # The LLM's returned code
        generated_code = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling LLM for plot code: {e}")
        generated_code = (
            "import matplotlib.pyplot as plt\n"
            "plt.figure()\n"
            "plt.text(0.5, 0.5, 'Failed to get LLM code', ha='center', va='center')\n"
            "plt.show()\n"
        )
    return generated_code

# ---------------------------------------------------------------------
# The new plot_chart node
# ---------------------------------------------------------------------
def plot_chart(state: dict) -> dict:
    """
    This node:
      1) Takes a DataFrame in context['df'].
      2) Takes a user’s query text (context['query_text']) to see what they want to plot.
      3) Calls LLM to generate Python code for a matplotlib chart.
      4) Executes that code in a local environment (where 'df' is available).
      5) Returns updated context with an indicator that the plot was displayed (or an error).
    """
    print("++++++++++ Entering plot_chart ++++++++++")
    #df = state.get("df")
    df = pd.read_parquet("parquet_cache/df_file.parquet")
    user_chart_request = state.get("query_text", "")

    if df is None or not isinstance(df, pd.DataFrame):
        logging.warning("No valid DataFrame found to plot.")
        return {**state, "plot_error": "No DataFrame available for plotting."}

    # Build prompt for the LLM
        prompt = f"""
You are given a pandas DataFrame named `df` with columns: {list(df.columns)}.
Generate Python code using matplotlib to visualize this request:
\"\"\"{user_request}\"\"\"

Requirements:
- Do NOT import pandas or redefine df.
- Do NOT include plt.show().
- Use matplotlib (no seaborn).
- Only output pure Python code — no comments, explanations, or markdown.
"""

    # Call LLM
    generated_code = call_llm_for_plot_code(prompt)
    st.code(generated_code, language="python")  # Optional: display the generated code

    # Prepare REPL
    repl = PythonREPL()
    repl.globals["df"] = df
    repl.globals["plt"] = plt

    # Clear previous figures
    plt.clf()

    try:
        # Run the code (it shouldn't include plt.show())
        repl.run(generated_code)

        # Now display the last matplotlib figure
        st.pyplot(plt.gcf())  # ✅ This renders the figure in Streamlit

    except Exception as e:
        return {**state, "plot_error": str(e)}

    return {**state, "chart_plotted": True}


# Define a routing function that returns the next node name
def chart_edge(state):
    return "plot_chart" if is_chart_requested(state) else END



# ---------------------------------------------------------------------
# Define the LangGraph Workflow
# ---------------------------------------------------------------------
workflow = StateGraph(dict)
workflow.add_node("classify_query", classify_query)
workflow.add_node("llm_sql_response", llm_sql_response)
workflow.add_node("get_query_context", get_query_context_wrapper)
workflow.add_node("generate_sql_query", generate_sql_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("llm_response", llm_response)
# NEW node
workflow.add_node("plot_chart", plot_chart)

# add edges
workflow.add_conditional_edges("classify_query", classify_edge)
workflow.add_edge("get_query_context", "generate_sql_query")
workflow.add_edge("generate_sql_query", "execute_query")

# If user wants a chart, go to plot_chart; else go to END
#workflow.add_conditional_edges("execute_query", "plot_chart", is_chart_requested)
#workflow.add_conditional_edges("execute_query", END, is_not_chart_requested)
workflow.add_edge("execute_query", END)

workflow.add_edge("plot_chart", END)
workflow.add_edge("llm_response", END)
workflow.add_edge("llm_sql_response", END)

# Set Workflow Entry Point
workflow.set_entry_point("classify_query")

# Compile Workflow
executor = workflow.compile()

# ---------------------------------------------------------------------
# Function to Process Queries
# ---------------------------------------------------------------------
def process_query(user_input):
    """Processes user input through LangGraph workflow and returns results."""
    print("++++++++++ Starting process_query ++++++++++")
    initial_state = {"query_text": user_input}
    final_state = executor.invoke(initial_state)
    print(f"++++++++++ Final State in process_query: {final_state} ++++++++++")
    print("++++++++++ Exiting process_query ++++++++++")
    return final_state.get("results", None)
