from ingestion.loader import Loader
from chunking.chunker import Splitter
from vectordb.vector_store import Vector_store
from retrieval.retriever import Retriever
from prompts.prompt_templates import get_prompt
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

docs = Loader('BNS.pdf')
chunks = Splitter(docs=docs)
vector_store = Vector_store(chunks)
retriever = Retriever(vector_store,10)

# LLM = ChatDeepSeek(model='deepseek-flash')
LLM = ChatOllama(model='llama3.2')

chat_history = {}

def get_history(session_id: str):
    if session_id not in chat_history:
        chat_history[session_id]=ChatMessageHistory()
    return chat_history[session_id]
        
prompt = get_prompt()

chain = RunnableParallel({
    'retrieved_context': itemgetter('query') | retriever,
    'query': itemgetter('query'),
    'chat_history': itemgetter('chat_history')
}) | prompt | LLM

chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key='query',
    history_messages_key='chat_history',
)

response = chatbot.invoke({'query':'POCSO'},
                config={'configurable':{'session_id':'shreyas'}})

print(response)

