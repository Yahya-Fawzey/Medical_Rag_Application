import random
import streamlit as st
from rag_pipeline import create_rag_chain

# Initialize the RAG chain and cache it to prevent reloading
@st.cache_resource
def load_chain():
    return create_rag_chain()

answer_func = load_chain()

st.title("Medical Symptom RAG Checker")

# 1. The UI Safety Disclaimer
st.warning("⚠️ **Disclaimer:** This AI triage system analyzes symptoms to suggest potential conditions, but it does not replace professional medical advice. This tool does not compensate for a real clinical diagnosis—please visit a doctor if you feel unwell or require medical attention.")

# 2. Random Demo Button State Handler
if 'demo_query' not in st.session_state:
    st.session_state.demo_query = ""

def set_random_query():
    prompts = [
        "I feel slow, my head is spinning, and I have a severe throbbing pain.",
        "I have a pounding headache, chest pain, and I feel dizzy.",
        "I have a severe dry cough, a high fever, and my muscles ache all over."
    ]
    st.session_state.demo_query = random.choice(prompts)

st.markdown("**Quick Test:**")
st.button("🎲 Load Test Example", on_click=set_random_query)

# 3. Smart Filtering Layers
def is_emergency_query(query: str) -> bool:
    emergency_keywords = [
        "chest pain", "heart attack", "can't breathe", "cannot breathe",
        "difficulty breathing", "severe bleeding", "unconscious", "suicide",
        "suicidal", "kill myself", "overdose", "stroke", "seizure",
        "loss of consciousness", "not breathing", "stopped breathing",
        "severe head injury", "anaphylaxis", "allergic reaction",
        "ألم في الصدر", "نزيف حاد", "لا أستطيع التنفس", "انتحار"
    ]
    q_lower = query.lower()
    return any(kw in q_lower for kw in emergency_keywords)

def is_out_of_scope(query: str) -> bool:
    # Catches non-medical prompts like recipes, coding, etc.
    unrelated_topics = [
        "recipe", "cook", "food", "python", "code", "movie", 
        "game", "football", "weather", "javascript", "sql", "bake"
    ]
    q_lower = query.lower()
    return any(topic in q_lower for topic in unrelated_topics)

# 4. User Input & Execution Flow
user_query = st.text_input("Describe your symptoms below:", value=st.session_state.demo_query)

if st.button("Analyze Symptoms") and user_query:
    
    # Check 1: Immediate Emergency Warning
    if is_emergency_query(user_query):
        st.markdown(
            "<h3 style='color: red;'>🚨 URGENT: Please seek immediate medical attention or call emergency services (123). Do not rely on AI triage for critical conditions.</h3>", 
            unsafe_allow_html=True
        )
        
    # Check 2: Relevance / Out-of-Scope Check (e.g., recipes, programming)
    elif is_out_of_scope(user_query):
        st.error("⚠️ **Not Relevant:** This is a medical symptom checker. Please enter health-related queries or symptoms rather than non-medical topics.")
    
    # Check 3: Let the ML Pipeline / AI Handle Everything Else
    else:
        with st.spinner("Analyzing clinical datasets..."):
            rag_answer = answer_func(user_query)
            st.write(rag_answer)
