from src.ingestion.loader import loader
from src.chunking.chunker import get_chunks
from src.vectordb.vector_store import vector_store
from src.retrieval.retriever import retriever
from src.chatbot.prompt import get_prompt
from src.chatbot.llm import deepseek,ollama
from src.chatbot.history import get_history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.title('Hetero RAG')
st.sidebar.title('Upload The PDF')
st.sidebar.file_uploader('Upload PDF')



docs = loader('BNS.pdf')
chunks = get_chunks(docs=docs)
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

