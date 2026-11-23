from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

# Returing the LLM based on the user needs
def deepseek(temp=0):
    LLM_DeepSeek = ChatDeepSeek(model='deepseek-flash',temperature=temp)
    return LLM_DeepSeek
def ollama(temp=0):
    LLM_ChatOllama = ChatOllama(model='llama3.2')
    return LLM_ChatOllama