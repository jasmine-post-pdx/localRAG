"""Ingestion pipeline for the local RAG system.

Loads source documents from data/documents/, splits them into chunks,
generates embeddings, and persists them to the local vector store in
data/vectorstore/.

Attributes:
    DATA_PATH (str): Directory containing the source PDF documents, relative
        to the current working directory.
    PDF_FILENAME (str): Name of the PDF file inside DATA_PATH to load.
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # Or UnstructuredPDFLoader

load_dotenv() # Optional: Loads environment variables from.env file
DATA_PATH = "data/documents/"
PDF_FILENAME = "document.pdf" # Replace with your PDF filename

def load_documents():
    """Loads a PDF from the data directory as LangChain documents.

    Reads the file at DATA_PATH/PDF_FILENAME with PyPDFLoader, producing one
    Document per page.

    Returns:
        list[Document]: One Document per PDF page, with the page text in
        ``page_content`` and the source path and page number in ``metadata``.

    Raises:
        ValueError: If the PDF path does not point to an existing file.
    """
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    # loader = UnstructuredPDFLoader(pdf_path) # Alternative
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from {pdf_path}")
    return documents

documents = load_documents()