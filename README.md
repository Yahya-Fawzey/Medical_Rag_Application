# Medical Symptom RAG Checker

An AI-powered clinical triage assistant built using Retrieval-Augmented Generation (RAG) to analyze symptoms, match them with clinical datasets, and provide structured risk assessments with built-in safety guardrails.

## 🌐 Live Demo
* **Streamlit Web App:** [https://medicalragapplication-37s2am8bdad564675hvsap.streamlit.app/](https://medicalragapplication-37s2am8bdad564675hvsap.streamlit.app/)

## ✨ Key Features
* **RAG Pipeline:** Utilizes ChromaDB and HuggingFace embeddings for precise symptom-to-disease retrieval.
* **Local LLM Inference:** Powered by `llama-cpp-python` running Qwen2.5-0.5B-Instruct.
* **Safety Guardrails:** Cascading filters for immediate emergency detection and out-of-scope query blocking.
* **Streamlit Interface:** Interactive UI with automated test scenarios and clinical disclaimers.

## 🚀 Local Installation & Quick Start

```bash
# Clone the repository
git clone [https://github.com/Yahya-Fawzey/Medical_Rag_Application.git](https://github.com/Yahya-Fawzey/Medical_Rag_Application.git)
cd Medical_Rag_Application

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
