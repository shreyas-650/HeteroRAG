def retriever(vector_store,k=10):
    return vector_store.as_retriever(search_Kwargs={'k':k})
