from src.ingestion.loader import loader
from src.chunking.chunker import get_chunks
from src.vectordb.vector_store import vector_store
from src.retrieval.retriever import retriever
from src.chatbot.prompt import get_prompt
from src.chatbot.llm import ollama
from src.chatbot.history import get_history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from dotenv import load_dotenv
load_dotenv()

# Document Loader
docs = loader('./tests/BNS.pdf')
# Splitted into Chunks
chunks = get_chunks(docs=docs)
# Add and Convert Chunks into Vector Embeddings
vectorstore = vector_store(chunks)
# Retriver
# retriever = retriever(vectorstore,10)

# LLM
LLM = ollama()

# Prompt        
prompt = get_prompt()

# Chain
chain = RunnableParallel({
    'retrieved_context': itemgetter('query') | retriever,
    'query': itemgetter('query'),
    'chat_history': itemgetter('chat_history')
}) | prompt | LLM

# Chatbot
chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key='query',
    history_messages_key='chat_history',
)

# Response
response = chatbot.invoke({'query':'Punishment for fraud'},
                config={'configurable':{'session_id':'shreyas'}})

print(response)

