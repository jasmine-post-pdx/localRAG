"""Ingestion pipeline for the local RAG system.

Loads source documents from data/documents/, splits them into chunks,
generates embeddings, and persists them to the local vector store in
data/vectorstore/.

Attributes:
    DATA_PATH (str): Directory containing the source documents, relative to
        the current working directory. Subdirectories are searched too.
    CHROMA_PATH (str): Directory where the Chroma vector store is saved,
        relative to the current working directory.
    LOADERS (dict[str, Callable[[str], BaseLoader]]): Maps a lowercase file
        extension to a factory that builds a LangChain loader for a file path.
        Files with other extensions are ignored.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv() # Optional: Loads environment variables from.env file
DATA_PATH = "data/documents/"
CHROMA_PATH = "data/vectorstore"

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda path: TextLoader(path, autodetect_encoding=True),
    ".md": lambda path: TextLoader(path, autodetect_encoding=True),
}

def load_documents(data_path=DATA_PATH):
    """Loads every supported file in the data directory as LangChain documents.

    Searches ``data_path`` recursively and loads each file whose extension is
    in LOADERS. PDFs produce one Document per page; text and Markdown files
    produce one Document per file.

    Args:
        data_path (str): Directory to search. Defaults to ``DATA_PATH``.

    Returns:
        list[Document]: The loaded documents, with the text in
        ``page_content`` and the source path in ``metadata["source"]``. PDF
        pages also have a zero-based ``metadata["page"]``.
    """
    documents = []
    file_count = 0
    for path in sorted(Path(data_path).rglob("*")):
        make_loader = LOADERS.get(path.suffix.lower())
        if make_loader is None or not path.is_file():
            continue
        documents.extend(make_loader(str(path)).load())
        file_count += 1
    print(f"Loaded {len(documents)} document(s) from {file_count} file(s) in {data_path}")
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
        document's ``metadata`` (source path, plus page number for PDFs).
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

def assign_chunk_ids(chunks):
    """Gives each chunk a deterministic ID based on where it came from.

    The ID has the form ``"<source>:<page>:<chunk_index>"``, where
    ``chunk_index`` counts chunks within the same source and page. Text files
    have no page, so their page part is ``0``. Running the same files through
    ``load_documents`` and ``split_documents`` again yields the same IDs, which
    is what lets ``index_documents`` skip chunks it has already stored.

    Args:
        chunks (list[Document]): Chunks in the order produced by
            ``split_documents``. Each chunk's ``metadata`` is modified in
            place to add an ``"id"`` key.

    Returns:
        list[Document]: The same chunks, now with ``metadata["id"]`` set.
    """
    counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)
        index = counts.get((source, page), 0)
        counts[(source, page)] = index + 1
        chunk.metadata["id"] = f"{source}:{page}:{index}"
    return chunks

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Embeds document chunks and saves the new ones in Chroma.

    Assigns each chunk a deterministic ID with ``assign_chunk_ids``, then
    embeds and stores only the chunks whose IDs aren't already in the store at
    ``persist_directory``. Re-running ingestion on the same files therefore
    adds nothing, and adding a new file only embeds that file's chunks.

    Because existing IDs are skipped, editing a file that was already indexed
    doesn't update its stored chunks. To pick up edits, delete
    ``persist_directory`` and ingest again.

    Args:
        chunks (list[Document]): Chunks to index, typically the output of
            ``split_documents``.
        embedding_function (Embeddings): Model used to embed the chunks,
            typically the output of ``get_embedding_function``. For an Ollama
            model, the server must be running and the model pulled.
        persist_directory (str): Directory where Chroma saves its data.
            Defaults to ``CHROMA_PATH``.

    Returns:
        Chroma: The vector store, including any newly indexed chunks.
    """
    vector_store = get_vector_store(embedding_function, persist_directory)
    existing_ids = set(vector_store.get(include=[])["ids"])

    assign_chunk_ids(chunks)
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    print(
        f"{len(existing_ids)} chunk(s) already stored; "
        f"indexing {len(new_chunks)} new of {len(chunks)} total..."
    )

    if new_chunks:
        vector_store.add_documents(
            new_chunks,
            ids=[c.metadata["id"] for c in new_chunks],
        )

    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vector_store


# if __name__ == "__main__":
#     documents = load_documents()
#     chunks = split_documents(documents)
#     embedding_function = get_embedding_function()
#     index_documents(chunks, embedding_function)