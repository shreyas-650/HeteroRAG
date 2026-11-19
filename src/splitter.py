from langchain_text_splitters import RecursiveCharacterTextSplitter

def Splitter(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators='\n')
    return splitter.split_documents(docs)
    
