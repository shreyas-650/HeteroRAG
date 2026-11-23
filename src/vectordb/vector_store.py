from langchain_community.vectorstores import FAISS
from embeddings.embedder import get_HF_embed_model,get_Ollama_embed_model

def vector_store(chunks):
    return FAISS.from_documents(documents=chunks,embedding=get_HF_embed_model)