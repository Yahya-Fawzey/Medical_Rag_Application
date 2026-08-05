import streamlit as st
from rag_pipeline import create_rag_chain

st.set_page_config(page_title="Medical RAG Assistant", layout="centered")

@st.cache_resource
def load_chain():
    return create_rag_chain()

rag_answer = load_chain()

st.title("Medical Symptom RAG Checker")
st.write("Describe your symptoms below to check potential conditions based on clinical datasets.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("e.g., I have a skin rash and mild fever..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing symptoms locally..."):
            response = rag_answer(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})