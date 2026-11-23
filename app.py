from ingestion.loader import loader
from chunking.chunker import splitter
from vectordb.vector_store import vector_store
from retrieval.retriever import retriever
from chatbot.prompt import get_prompt
from chatbot.llm import deepseek,ollama
from chatbot.history import get_history
from chatbot.prompt import get_prompt

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import streamlit as st
from dotenv import load_dotenv
load_dotenv()




docs = loader('BNS.pdf')
chunks = splitter(docs=docs)
vectorstore = vector_store(chunks)
retriever = retriever(vectorstore,10)
LLM = ollama()        
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

