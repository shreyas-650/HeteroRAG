from langchain_text_splitters import RecursiveCharacterTextSplitter

# Convert large documents into chunks

def get_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators='\n')
    return splitter.split_documents(docs)
    
