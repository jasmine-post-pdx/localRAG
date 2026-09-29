# ingest.py
#
# Ingestion pipeline: load source documents from data/documents/, split them
# into chunks, generate embeddings, and persist them to the local vector store
# in data/vectorstore/.
# rag_local.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader # Or UnstructuredPDFLoader
load_dotenv() # Optional: Loads environment variables from.env file
DATA_PATH = "data/"
PDF_FILENAME = "mydocument.pdf" # Replace with your PDF filename
def load_documents():
    """Loads documents from the specified data path."""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    # loader = UnstructuredPDFLoader(pdf_path) # Alternative
    documents = loader.load()
    print(f"Loaded{len(documents)} page(s) from{pdf_path}")
    return documents
# documents = load_documents() # Call this later