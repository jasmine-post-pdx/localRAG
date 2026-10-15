"""Generation step for the local RAG system.

Combines context retrieved from the vector store with the user's question
into a prompt and sends it to a local LLM served by Ollama to produce an
answer.
"""
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


def create_rag_chain(vector_store, query_settings, prompt_template, ollama_host):
    """Creates a retrieval-augmented generation chain backed by Ollama.

    The chain retrieves the ``query_settings.retrieval_k`` most similar
    documents from the vector store, inserts them into ``prompt_template``
    alongside the user's question, sends the prompt to the Ollama chat model
    at ``ollama_host``, and parses the response into a plain string.

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
        ``invoke()`` and returns the generated answer as a string.

    Example:
        >>> chain = create_rag_chain(
        ...     vector_store, store.query, store.prompt_template, app_settings.ollama_host,
        ... )
        >>> chain.invoke("What is the refund policy?")
        'Refunds are available within 30 days...'
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

    # Define the RAG chain using LCEL.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    console.print("RAG chain created.", style="success")

    return rag_chain
