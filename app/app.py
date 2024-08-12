import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from src.document_processor import query_to_answer


st.title("Recipes Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


if query := st.chat_input("How can I help you?"):
    
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = query_to_answer(query=query)
    
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    #print(st.session_state.messages)
