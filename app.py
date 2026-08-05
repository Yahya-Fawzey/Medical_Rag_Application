import streamlit as st
from rag_pipeline import create_rag_chain

# Initialize the RAG chain and cache it to prevent reloading
@st.cache_resource
def load_chain():
    return create_rag_chain()

answer_func = load_chain()

st.title("Medical Symptom RAG Checker")

# 1. The UI Safety Disclaimer
st.warning("⚠️ **Disclaimer:** This AI triage system is for educational demonstration only and is not a substitute for professional medical diagnosis.")

# 2. Demo Buttons State Handler
if 'demo_query' not in st.session_state:
    st.session_state.demo_query = ""

def set_query(text):
    st.session_state.demo_query = text

st.markdown("**Quick Test Prompts:**")
col1, col2 = st.columns(2)
with col1:
    st.button("Test: Migraine", on_click=set_query, args=("I feel slow, my head is spinning, and I have a severe throbbing pain.",))
with col2:
    st.button("Test: Hypertension", on_click=set_query, args=("I have a pounding headache, chest pain, and I feel dizzy.",))

# 3. Input Guardrail Function
def is_safe_medical_query(query: str) -> bool:
    # Block common prompt injection attempts
    injection_keywords = ["ignore", "system prompt", "instructions", "bypass", "rule", "developer"]
    if any(word in query.lower() for word in injection_keywords):
        return False
        
    # Require the query to be related to symptoms or health
    medical_keywords = ["pain", "hurt", "feel", "ache", "dizzy", "blood", "doctor", "symptom", "fever", "sick", "swollen", "slow", "head", "chest"]
    
    return any(word in query.lower() for word in medical_keywords)

# 4. User Input & Execution
user_query = st.text_input("Describe your symptoms below:", value=st.session_state.demo_query)

if st.button("Analyze Symptoms") and user_query:
    
    # The Safety Layer triggers first!
    if not is_safe_medical_query(user_query):
        st.error("⚠️ **Blocked:** Please restrict queries to medical symptoms. System prompt instructions cannot be modified.")
    
    else:
        # Only run the heavy ML inference pipeline if the query passes the safety check
        with st.spinner("Analyzing clinical datasets..."):
            rag_answer = answer_func(user_query)
            st.write(rag_answer)
