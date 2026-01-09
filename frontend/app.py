
import streamlit as st
import requests
import pandas as pd
import io

API_URL = "http://localhost:8000/api"

st.set_page_config(layout="wide", page_title="Data Analyst Agent")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and not st.session_state.session_id:
        with st.spinner("Uploading and analyzing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            try:
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_id = data["session_id"]
                    st.success("File uploaded successfully!")
                    st.json(data["columns"])
                else:
                    st.error(f"Upload failed: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.title("Data Analyst AI Agent 🤖")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "code" in msg:
            with st.expander("View Code"):
                st.code(msg["code"], language="python")
        if "plot_url" in msg:
            st.image(msg["plot_url"])

# Chat input
if query := st.chat_input("Ask about your data..."):
    if not st.session_state.session_id:
        st.error("Please upload a CSV file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"session_id": st.session_state.session_id, "query": query}
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        reasoning = data["response"]
                        code = data["code"]
                        plot_id = data.get("plot_id")
                        
                        st.markdown(reasoning)
                        with st.expander("View Code"):
                            st.code(code, language="python")
                        
                        plot_url = None
                        if plot_id is not None:
                            plot_url = f"{API_URL}/plots/{st.session_state.session_id}/{plot_id}"
                            st.image(plot_url)
                        
                        msg_data = {
                            "role": "assistant",
                            "content": reasoning,
                            "code": code
                        }
                        if plot_url:
                            msg_data["plot_url"] = plot_url
                            
                        st.session_state.messages.append(msg_data)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
