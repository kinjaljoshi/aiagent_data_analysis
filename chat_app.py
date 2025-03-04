import sys
import os

# Ensure backend folder is added to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from query_workflow import process_query
import streamlit as st
import pandas as pd
from langgraph.graph import END 

# Streamlit UI Configuration
st.set_page_config(page_title="Personalized Query Assistant", layout="wide")

# App Title
st.title("Personalized Query Assistant")

# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Display
#st.subheader("💬 Conversation History")
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f" {entry['user_query']}")
    
    with st.chat_message("assistant"):
        if isinstance(entry["response"], pd.DataFrame):
            st.dataframe(entry["response"])  # Display DataFrame properly
        else:
            st.markdown(f" {entry['response']}")

# User Input
user_query = st.chat_input("Ask your question...")

# Process Query on Input
if user_query:
    with st.chat_message("user"):
        st.markdown(f" {user_query}")

    with st.spinner(" Thinking..."):
        response = process_query(user_query)

        # Store query & response in chat history
        st.session_state.chat_history.append({"user_query": user_query, "response": response})

        # Display AI Response
        with st.chat_message("assistant"):
            if isinstance(response, pd.DataFrame):
                st.dataframe(response)
            else:
                st.markdown(f" {response}")

# Clear Chat Button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
