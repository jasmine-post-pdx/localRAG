"""Ingestion pipeline for the local RAG system.

Loads a data store's source documents, splits them into chunks, generates
embeddings, and persists them to the store's local vector store.

Run from the project root with ``python -m src.ingest``, adding
``--store <name>`` to ingest into a store other than the default.

Attributes:
    LOADERS (dict[str, Callable[[str], BaseLoader]]): Maps a lowercase file
        extension to a factory that builds a LangChain loader for a file path.
        Files with other extensions are ignored.
"""
import argparse
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import ResponseError
from rich.panel import Panel
from rich.text import Text

from src import config
from src.console import console
from src.manifest import find_mismatches, has_vectors, read_manifest, write_manifest

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda path: TextLoader(path, autodetect_encoding=True),
    ".md": lambda path: TextLoader(path, autodetect_encoding=True),
}

def find_document_files(data_path):
    """Lists every supported file in the data directory.

    Args:
        data_path (str | Path): Directory to search recursively.

    Returns:
        list[Path]: Files whose extension is in LOADERS, sorted by path.
    """
    return [
        path for path in sorted(Path(data_path).rglob("*"))
        if path.suffix.lower() in LOADERS and path.is_file()
    ]

def load_documents(data_path):
    """Loads every supported file in the data directory as LangChain documents.

    Searches ``data_path`` recursively and loads each file whose extension is
    in LOADERS. PDFs produce one Document per page; text and Markdown files
    produce one Document per file.

    Args:
        data_path (str | Path): Directory to search, typically a store's
            ``documents_path``.

    Returns:
        list[Document]: The loaded documents, with the text in
        ``page_content`` and the source path in ``metadata["source"]``. PDF
        pages also have a zero-based ``metadata["page"]``.

    Raises:
        RuntimeError: If the data directory doesn't exist or contains no
            supported files.
    """
    if not Path(data_path).is_dir():
        raise RuntimeError(
            f"Documents folder {data_path} doesn't exist. "
            "Create it and add documents to ingest."
        )
    files = find_document_files(data_path)
    if not files:
        supported = ", ".join(sorted(LOADERS))
        raise RuntimeError(
            f"No documents found in {data_path}. "
            f"Add files with one of these extensions: {supported}."
        )
    documents = []
    for path in files:
        documents.extend(LOADERS[path.suffix.lower()](str(path)).load())
    console.print(
        f"Loaded {len(documents)} document(s) from {len(files)} file(s) in {data_path}",
        style="info",
    )
    return documents

def split_documents(documents, chunk_size, chunk_overlap):
    """Splits documents into overlapping text chunks for embedding.

    Uses RecursiveCharacterTextSplitter, which breaks text on paragraph, line,
    and word boundaries in that order, keeping each chunk at or under
    ``chunk_size`` characters. Consecutive chunks share up to
    ``chunk_overlap`` characters so context isn't lost at chunk boundaries.

    Args:
        documents (list[Document]): Documents to split, typically the per-page
            output of ``load_documents``.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Characters shared between consecutive chunks.
            Must be smaller than ``chunk_size``.

    Returns:
        list[Document]: The chunks, each carrying a copy of its source
        document's ``metadata`` (source path, plus page number for PDFs).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    console.print(f"Split into {len(all_splits)} chunks", style="info")
    return all_splits

def get_embedding_function(model_name, ollama_host):
    """Creates the embedding model used to vectorize document chunks.

    Returns a LangChain wrapper around a model served by the Ollama server at
    ``ollama_host``. Creating it doesn't contact Ollama; the server must be
    running and the model pulled (e.g. ``ollama pull nomic-embed-text``)
    before any text is actually embedded.

    Args:
        model_name (str): Name of the Ollama embedding model to use,
            typically a store's ``ingest.embedding_model``.
        ollama_host (str): Address of the Ollama server.

    Returns:
        OllamaEmbeddings: An embedding function exposing ``embed_documents``
        and ``embed_query``, suitable for passing to a vector store.
    """
    embeddings = OllamaEmbeddings(model=model_name, base_url=ollama_host)
    console.print(f"Initialized Ollama embeddings with model: {model_name}", style="info")
    return embeddings

def check_ollama(embedding_function):
    """Verifies that Ollama is reachable and can run the embedding model.

    Embeds a short test string so problems surface before any documents are
    loaded or indexed, rather than partway through ingestion.

    Args:
        embedding_function (OllamaEmbeddings): The embedding function to test,
            typically the output of ``get_embedding_function``.

    Raises:
        SystemExit: If the Ollama server can't be reached or the model isn't
            available.
    """
    model = embedding_function.model
    try:
        embedding_function.embed_query("ping")
    except ConnectionError:
        console.print(Panel(
            f"Can't reach Ollama at {embedding_function.base_url}.\n"
            "Start it with `ollama serve` (or open the Ollama app) and run ingestion again.",
            title="Error", border_style="red",
        ))
        raise SystemExit(1) from None
    except ResponseError as e:
        console.print(Panel(
            f"Ollama can't use model {model!r}: {e.error}.\n"
            f"Try `ollama pull {model}`.",
            title="Error", border_style="red",
        ))
        raise SystemExit(1) from None

def get_vector_store(embedding_function, persist_directory):
    """Opens the persistent Chroma vector store.

    Loads the store saved in ``persist_directory``, or creates a new empty one
    there if none exists yet. This doesn't add any documents; chunks must be
    added to the returned store separately.

    Args:
        embedding_function (Embeddings): Model used to embed documents added
            to the store and queries run against it, typically the output of
            ``get_embedding_function``. It must be the same model that embedded
            any vectors already saved in the store.
        persist_directory (str | Path): Directory where Chroma saves its
            data, typically a store's ``vectorstore_path``.

    Returns:
        Chroma: The vector store, which writes changes to ``persist_directory``.
    """
    vectorstore = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
    )
    console.print(f"Vector store initialized/loaded from: {persist_directory}", style="info")
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

def index_documents(chunks, embedding_function, persist_directory, batch_size):
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
        persist_directory (str | Path): Directory where Chroma saves its
            data, typically a store's ``vectorstore_path``.
        batch_size (int): Chunks sent to the embedding model per request.

    Returns:
        Chroma: The vector store, including any newly indexed chunks.
    """
    vector_store = get_vector_store(embedding_function, persist_directory)
    existing_ids = set(vector_store.get(include=[])["ids"])

    assign_chunk_ids(chunks)
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    console.print(
        f"{len(existing_ids)} chunk(s) already stored; "
        f"indexing {len(new_chunks)} new of {len(chunks)} total...",
        style="info",
    )

    if new_chunks:
        for start in range(0, len(new_chunks), batch_size):
            batch = new_chunks[start:start + batch_size]
            for attempt in range(3):
                try:
                    vector_store.add_documents(batch, ids=[c.metadata["id"] for c in batch])
                    break
                except ResponseError:
                    if attempt == 2:
                        raise
                    time.sleep(2)

    console.print(f"Indexing complete. Data saved to: {persist_directory}", style="success")
    return vector_store

def offer_to_delete_documents(data_path):
    """Asks whether to delete the ingested files and deletes them if confirmed.

    Only files with a supported extension are deleted; anything ingestion
    skipped stays in place. Any answer other than yes, including just pressing
    Enter, keeps the files.

    Args:
        data_path (str | Path): Directory whose supported files may be deleted.
    """
    files = find_document_files(data_path)
    if not files:
        return
    console.rule("Cleanup")
    try:
        answer = console.input(Text(
            "All of the files in the documents folder have been added to the AI system.\n"
            "Do you want those files deleted from the documents folder? [y/N] ",
            style="question",
        ))
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        return
    for path in files:
        path.unlink()
    console.print("All files in the documents folder have been deleted", style="success")


def check_store_settings(store):
    """Stops ingestion if the store's vectors were built with different settings.

    Compares the store's embedding model and chunk settings with its
    manifest. Adding chunks made with different settings would mix vectors
    that can't be compared, or reuse chunk IDs for different text.

    Args:
        store (StoreConfig): The store about to be ingested into.

    Raises:
        SystemExit: If any fixed setting differs from the manifest.
    """
    manifest = read_manifest(store)
    if manifest is None:
        if has_vectors(store):
            console.print(
                "This store's vectors were built before settings were recorded, so "
                "they can't be checked. They will be recorded after this run.",
                style="warning",
            )
        return
    mismatches = find_mismatches(store, manifest)
    if mismatches:
        console.print(Panel(
            "\n".join(mismatches) + "\n\n"
            "Change the settings back, or delete the contents of "
            f"{store.vectorstore_path} and ingest again to rebuild the store with "
            "the new settings.",
            title="Settings changed since this store was built", border_style="red",
        ))
        raise SystemExit(1)


def ingest_store(store, app_settings):
    """Loads, splits, embeds, and indexes a data store's documents.

    Refuses to add to a store whose vectors were built with different fixed
    settings, and records the settings in the store's manifest afterwards.

    Args:
        store (StoreConfig): The store to ingest into.
        app_settings (AppSettings): Machine-wide settings, for the Ollama
            address.
    """
    console.rule("Setup")
    console.print(f"Data store: {store.name} ({store.config_path})", style="info")
    check_store_settings(store)
    embedding_function = get_embedding_function(
        store.ingest.embedding_model, app_settings.ollama_host,
    )
    check_ollama(embedding_function)
    console.rule("Loading documents")
    try:
        documents = load_documents(store.documents_path)
    except RuntimeError as error:
        console.print(Panel(str(error), title="Error", border_style="red"))
        raise SystemExit(1) from None
    chunks = split_documents(documents, store.ingest.chunk_size, store.ingest.chunk_overlap)
    console.rule("Indexing")
    vector_store = index_documents(
        chunks, embedding_function, store.vectorstore_path, store.ingest.embedding_batch_size,
    )
    chunk_count = len(vector_store.get(include=[])["ids"])
    if chunk_count:
        write_manifest(store, chunk_count)
    offer_to_delete_documents(store.documents_path)
    console.rule("Done")


def main(argv=None):
    """Ingests the documents of the store named by ``--store``, or ``RAG_STORE``."""
    parser = argparse.ArgumentParser(description="Ingest a data store's documents.")
    parser.add_argument("--store", help="data store to ingest into (default: RAG_STORE)")
    args = parser.parse_args(argv)

    app_settings = config.load_app_settings()
    store = config.load_store(args.store or app_settings.default_store)
    ingest_store(store, app_settings)


if __name__ == "__main__":
    main()