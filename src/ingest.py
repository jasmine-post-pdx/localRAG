"""Ingestion pipeline for the local RAG system.

Loads source documents from data/documents/, splits them into chunks,
generates embeddings, and persists them to the local vector store in
data/vectorstore/.

Attributes:
    DATA_PATH (str): Directory containing the source PDF documents, relative
        to the current working directory.
    PDF_FILENAME (str): Name of the PDF file inside DATA_PATH to load.
    CHROMA_PATH (str): Directory where the Chroma vector store is saved,
        relative to the current working directory.
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # Or UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv() # Optional: Loads environment variables from.env file
DATA_PATH = "data/documents/"
PDF_FILENAME = "document.pdf" # Replace with your PDF filename
CHROMA_PATH = "data/vectorstore"

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

def get_embedding_function(model_name="nomic-embed-text"):
    """Creates the embedding model used to vectorize document chunks.

    Returns a LangChain wrapper around a model served by the local Ollama
    server. Creating it doesn't contact Ollama; the server must be running and
    the model pulled (e.g. ``ollama pull nomic-embed-text``) before any text is
    actually embedded.

    Args:
        model_name (str): Name of the Ollama embedding model to use. Defaults
            to ``"nomic-embed-text"``.

    Returns:
        OllamaEmbeddings: An embedding function exposing ``embed_documents``
        and ``embed_query``, suitable for passing to a vector store.
    """
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    return embeddings

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    """Opens the persistent Chroma vector store.

    Loads the store saved in ``persist_directory``, or creates a new empty one
    there if none exists yet. This doesn't add any documents; chunks must be
    added to the returned store separately.

    Args:
        embedding_function (Embeddings): Model used to embed documents added
            to the store and queries run against it, typically the output of
            ``get_embedding_function``. It must be the same model that embedded
            any vectors already saved in the store.
        persist_directory (str): Directory where Chroma saves its data.
            Defaults to ``CHROMA_PATH``.

    Returns:
        Chroma: The vector store, which writes changes to ``persist_directory``.
    """
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )
    print(f"Vector store initialized/loaded from: {persist_directory}")
    return vectorstore

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Embeds document chunks and saves them in Chroma.

    Sends each chunk's text to the embedding model and stores the resulting
    vectors, along with the chunk text and metadata, in the Chroma store at
    ``persist_directory``. The chunks are added to any store already saved
    there rather than replacing it, and each gets a new random ID, so indexing
    the same chunks twice stores duplicates.

    Args:
        chunks (list[Document]): Chunks to index, typically the output of
            ``split_documents``.
        embedding_function (Embeddings): Model used to embed the chunks,
            typically the output of ``get_embedding_function``. For an Ollama
            model, the server must be running and the model pulled.
        persist_directory (str): Directory where Chroma saves its data.
            Defaults to ``CHROMA_PATH``.

    Returns:
        Chroma: The vector store containing the newly indexed chunks.
    """
    print(f"Indexing {len(chunks)} chunks...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory,
    )

    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vector_store


# if __name__ == "__main__":
#     documents = load_documents()
#     chunks = split_documents(documents)
#     embedding_function = get_embedding_function()
#     index_documents(chunks, embedding_function)