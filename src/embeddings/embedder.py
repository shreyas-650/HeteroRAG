from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

# Returns Embedding Model to Vector Store

def get_Ollama_embed_model():
    return OllamaEmbeddings(model='qwen3-embedding:0.6b') # nomic-embed-text
def get_HF_embed_model():
    return HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')