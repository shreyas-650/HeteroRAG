from src.splitter import Loader

docs = Loader('BNS.pdf')
print(doc for doc in docs)