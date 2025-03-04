import sys
import os

# Ensure backend folder is added to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from query_workflow import process_query
import streamlit as st
import pandas as pd

# Streamlit UI Configuration
st.set_page_config(page_title="AI Query Assistant", layout="wide")

# App Title
st.title("📊 AI Query Assistant with FAISS & BigQuery")

# User Input
user_query = st.text_input("Enter your query:", placeholder="Get warehouse and item inventory")

if st.button("Search"):
    if user_query.strip():
        with st.spinner("Processing your query..."):
            results = process_query(user_query)

            # Display Results
            if isinstance(results, pd.DataFrame):
                st.success("Query executed successfully!")
                st.dataframe(results)
            elif isinstance(results, str):
                st.info(results)
            else:
                st.warning("No relevant results found.")
    else:
        st.error("Please enter a valid query.")
