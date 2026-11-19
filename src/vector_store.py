from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embedding=OllamaEmbeddings(model='nomic-embed-text')
def Vector_store(chunks):
    return FAISS.from_documents(documents=chunks,embedding=embedding)