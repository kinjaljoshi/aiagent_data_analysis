import sys
import os

# Ensure backend folder is added to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from query_workflow import process_query
import streamlit as st
import pandas as pd
from langgraph.graph import END 

# Streamlit UI Configuration
st.set_page_config(page_title="AI Query Assistant", layout="wide")

# App Title
st.title("📊 AI Query Assistant with FAISS & BigQuery")

# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_query = st.text_input("Ask a question:", placeholder="Ask .....")

# Chat History Display
st.subheader("Chat History")
for entry in st.session_state.chat_history:
    st.markdown(f"**🧑 User:** {entry['user_query']}")
    
    # If the response is a DataFrame, display it properly
    if isinstance(entry["response"], pd.DataFrame):
        st.dataframe(entry["response"])
    else:
        st.markdown(f"**🤖 AI:** {entry['response']}")

# Process Query on Button Click
if st.button("Ask AI"):
    if user_query.strip():
        with st.spinner("Processing your query..."):
            response = process_query(user_query)

            # Store query & response in chat history
            st.session_state.chat_history.append({"user_query": user_query, "response": response})

            # Refresh UI to display updated chat history
            st.experimental_rerun()
    else:
        st.error("Please enter a valid query.")
