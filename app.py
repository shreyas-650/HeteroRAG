from src.loader import Loader
from src.splitter import Splitter
from src.vector_store import Vector_store
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv()

docs = Loader('BNS.pdf')
chunks = Splitter(docs=docs)
store = Vector_store(chunks)
retriever = store.as_retriever(search_Kwargs={'k':1})

LLM = ChatDeepSeek(model='deepseek-flash')
res = retriever.invoke('Rape Case')
print(res)