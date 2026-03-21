# app/document_loader.py

from langchain_community.document_loaders import UnstructuredFileLoader


def load_document(file_path):

    loader = UnstructuredFileLoader(file_path)

    documents = loader.load()

    return "\n".join([doc.page_content for doc in documents])