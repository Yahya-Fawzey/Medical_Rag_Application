import random
import streamlit as st
from rag_pipeline import create_rag_chain
import re

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
def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s'ء-ي]", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_emergency_query(query: str) -> bool:
    emergency_keywords = [
        # cardiac / respiratory
        "chest pain", "heart attack", "can't breathe", "cannot breathe",
        "cant breath", "cant breathe", "breath correctly", "breathe correctly",
        "difficulty breathing", "not breathing", "stopped breathing",
        "gasping for air", "cant catch my breath", "cant catch breath",
        "tightness in my chest", "crushing chest pain",

        # bleeding / trauma
        "severe bleeding", "heavy bleeding", "bleeding heavily", "bleeding a lot",
        "won't stop bleeding", "wont stop bleeding", "i'm bleeding", "im bleeding",
        "bleeding out", "severe head injury", "deep wound", "coughing up blood",
        "vomiting blood",

        # neuro
        "stroke", "seizure", "unconscious", "loss of consciousness",
        "passed out", "not waking up", "unresponsive", "slurred speech",
        "face is drooping", "cant feel my", "can't feel my", "cant move my",
        "can't move my",

        # allergic / toxic
        "anaphylaxis", "allergic reaction", "throat closing", "overdose",
        "poisoned", "poisoning", "swallowed something",

        # self-harm / crisis
        "suicide", "suicidal", "kill myself", "want to die", "end my life",
        "hurting myself", "self harm", "self-harm",

        # generic distress catch-alls — deliberately broad, false positives are cheap here
        "im dying", "i'm dying", "dying right now", "call 911", "call an ambulance",
        "need an ambulance", "this is an emergency", "help me now",

        # Arabic
        "ألم في الصدر", "نزيف حاد", "لا أستطيع التنفس", "انتحار",
        "فقدان الوعي", "سكتة دماغية", "نوبة قلبية", "لا أستطيع الحركة",
    ]
    q = _normalize(query)
    return any(kw in q for kw in emergency_keywords)

def is_out_of_scope(query: str) -> bool:
    unrelated_topics = [
        "recipe", "cook", "food", "python", "code", "movie",
        "game", "football", "weather", "javascript", "sql", "bake"
    ]
    medical_exceptions = [
        "food poisoning", "food allergy", "food allergic", "allergic to food",
        "cant keep food down", "can't keep food down", "vomiting food",
    ]
    q_lower = query.lower()

    if any(exc in q_lower for exc in medical_exceptions):
        return False

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
