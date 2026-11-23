from langchain_community.document_loaders import PyMuPDFLoader


def Loader(path):
    return PyMuPDFLoader(path).load()