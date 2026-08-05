import os
os.environ["USE_TF"] = "0"  # Bypasses TensorFlow checks

import pandas as pd
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH = "data/Symptom2Disease.csv"
DB_DIR = "vector_db"

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    loader = CSVLoader(file_path=DATA_PATH, source_column="label")
    docs = loader.load()
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vectorstore

def get_llm():
    # Downloads the model directly to the cloud server on boot
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
    )
    
    return LlamaCpp(
        model_path=model_path,
        temperature=0.0,
        n_ctx=2048,
        n_batch=512,
        verbose=False
        # n_gpu_layers removed because Streamlit Cloud uses CPUs
    )

def create_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        "<|im_start|>system\n"
        "You are an expert clinical triage assistant. Analyze the context and identify potential conditions that match the query.\n\n"
        "STRICT RULES:\n"
        "1. Do NOT output row numbers, record IDs, or dataset indices.\n"
        "2. Express confidence strictly as a percentage ending with '%'.\n"
        "3. Do NOT show reasoning or preambles.\n\n"
        "EXAMPLE OUTPUT FORMAT:\n"
        "**Most Likely Conditions:**\n"
        "* **Hypertension** - 85%\n"
        "* **Migraine** - 70%\n\n"
        "**Brief Explanation:**\n"
        "The patient presents with symptoms aligning with cardiovascular and neurological stress.<|im_end|>\n"
        "<|im_start|>user\n"
        "Context:\n{context}\n\n"
        "Patient Query: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    def answer(query: str):
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = prompt.format(context=context, question=query)
        response = llm.invoke(formatted_prompt)
        return response

    return answer
