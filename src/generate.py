"""Generation step for the local RAG system.

Combines context retrieved from the vector store with the user's question
into a prompt and sends it to a local LLM served by Ollama to produce an
answer. Retrieved chunks are numbered and labeled with their source so the
model can cite them, and the chunks themselves are returned alongside the
answer so callers can show exactly which documents were used.
"""
from operator import itemgetter
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from ollama import Client, ResponseError

from src.console import console


def check_chat_model(llm_model, ollama_host):
    """Verifies that Ollama is reachable and the chat model is available.

    Asks the server for the model's details, so a missing model or a stopped
    server surfaces before the user asks a question, rather than as a raw
    error partway through the first answer.

    Args:
        llm_model (str): The chat model to check, typically a store's
            ``query.llm_model``.
        ollama_host (str): Address of the Ollama server.

    Raises:
        RuntimeError: If the Ollama server can't be reached or the model
            isn't pulled.
    """
    try:
        Client(host=ollama_host).show(llm_model)
    except ConnectionError:
        raise RuntimeError(
            f"Can't reach Ollama at {ollama_host}.\n"
            "Start it with `ollama serve` (or open the Ollama app) and run this again."
        ) from None
    except ResponseError as error:
        raise RuntimeError(
            f"Ollama can't use model {llm_model!r}: {error.error}.\n"
            f"Try `ollama pull {llm_model}`."
        ) from None


def source_label(doc):
    """Builds a human-readable label for a chunk: file name, plus page for PDFs.

    Pages in chunk metadata are zero-based (as PDF loaders report them); the
    label shows them one-based, as PDF readers do.

    Args:
        doc (Document): A retrieved chunk with ``metadata["source"]`` and,
            for PDFs, ``metadata["page"]``.

    Returns:
        str: e.g. ``"report.pdf, page 3"``, or ``"notes.txt"`` when the
        chunk has no page.
    """
    label = Path(doc.metadata.get("source", "unknown source")).name
    page = doc.metadata.get("page")
    if page is not None:
        label += f", page {page + 1}"
    return label


def format_docs(docs):
    """Formats retrieved chunks as a numbered, source-labeled context block.

    Each chunk becomes ``[n] <source label>`` followed by its text, so the
    chat model can cite chunks by number in its answer.

    Args:
        docs (list[Document]): The chunks retrieved for a question.

    Returns:
        str: The chunks joined by blank lines, numbered from 1 in the order
            retrieved.
    """
    return "\n\n".join(
        f"[{number}] {source_label(doc)}\n{doc.page_content}"
        for number, doc in enumerate(docs, 1)
    )


def format_sources(docs):
    """Lists the sources of retrieved chunks, deduplicated, in retrieval order.

    This is the deterministic companion to any citations the model makes:
    it always reflects exactly which chunks were retrieved, even if the
    model's inline citations are missing or wrong.

    Args:
        docs (list[Document]): The chunks retrieved for a question.

    Returns:
        str: Source labels joined by "; ", e.g.
            ``"report.pdf, page 3; notes.txt"``. Empty when no chunks were
            retrieved.
    """
    return "; ".join(dict.fromkeys(source_label(doc) for doc in docs))


def create_rag_chain(vector_store, query_settings, prompt_template, ollama_host):
    """Creates a retrieval-augmented generation chain backed by Ollama.

    The chain retrieves the ``query_settings.retrieval_k`` most similar
    documents from the vector store, formats them with ``format_docs`` into
    ``prompt_template`` alongside the user's question, sends the prompt to
    the Ollama chat model at ``ollama_host``, and parses the response into a
    plain string.

    Args:
        vector_store: A LangChain vector store that supports
            ``as_retriever()`` and holds the indexed document chunks.
        query_settings (QuerySettings): The store's chat model, its context
            window and temperature, and the number of chunks to retrieve.
            The model must already be pulled (e.g. ``ollama pull qwen3:8b``).
        prompt_template (str): The prompt, with ``{context}`` and
            ``{question}`` placeholders, typically a store's
            ``prompt_template``.
        ollama_host (str): Address of the Ollama server.

    Returns:
        A LangChain ``Runnable`` that takes a question string via
        ``invoke()`` and returns ``{"answer": str, "docs": list[Document]}``:
        the generated answer and the chunks it was grounded in.

    Example:
        >>> chain = create_rag_chain(
        ...     vector_store, store.query, store.prompt_template, app_settings.ollama_host,
        ... )
        >>> chain.invoke("What is the refund policy?")
        {'answer': 'Refunds are available within 30 days [1].', 'docs': [...]}
    """

    # Initialize the LLM.
    llm = ChatOllama(
        model=query_settings.llm_model,
        base_url=ollama_host,
        temperature=query_settings.temperature,
        num_ctx=query_settings.context_window,
    )
    console.print(
        f"Initialized ChatOllama with model: {query_settings.llm_model}, "
        f"context window: {query_settings.context_window}, "
        f"temperature: {query_settings.temperature}",
        style="info",
    )

    # Create the retriever.
    k = query_settings.retrieval_k
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    console.print(f"Retriever initialized (k={k}).", style="info")

    prompt = ChatPromptTemplate.from_template(prompt_template)
    console.print("Prompt template created.", style="info")

    # Define the RAG chain using LCEL. The retrieved chunks flow through the
    # whole chain so the answer and its sources can be returned together.
    rag_chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=lambda inputs: format_docs(inputs["docs"]))
        | {
            "answer": prompt | llm | StrOutputParser(),
            "docs": itemgetter("docs"),
        }
    )
    console.print("RAG chain created.", style="success")

    return rag_chain
