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
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def split_documents(documents):
    """Splits documents into overlapping text chunks for embedding.

    Uses RecursiveCharacterTextSplitter, which breaks text on paragraph, line,
    and word boundaries in that order, keeping each chunk at or under 1000
    characters. Consecutive chunks share up to 200 characters so context isn't
    lost at chunk boundaries.

    Args:
        documents (list[Document]): Documents to split, typically the per-page
            output of ``load_documents``.

    Returns:
        list[Document]: The chunks, each carrying a copy of its source
        document's ``metadata`` (source path and page number).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    return all_splits

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

