from langchain_community.document_loaders import PyMuPDFLoader


def loader(path):
    return PyMuPDFLoader(path).load()