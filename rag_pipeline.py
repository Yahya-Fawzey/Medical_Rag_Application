import os
import pandas as pd
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate

MODEL_PATH = hf_hub_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF", 
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
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
    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.0,
        n_ctx=2048,
        n_batch=512,
        verbose=False
    )

def create_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        "You are a medical assistant. Use ONLY the given patient symptom records to diagnose or explain the condition.\n"
        "If you do not know based on the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Patient Query: {question}"
    )
    
    def answer(query: str):
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = prompt.format(context=context, question=query)
        response = llm.invoke(formatted_prompt)
        return response

    return answer