from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embedding=OllamaEmbeddings(model='qwen3-embedding:0.6b') # nomic-embed-text
def Vector_store(chunks):
    return FAISS.from_documents(documents=chunks,embedding=embedding)