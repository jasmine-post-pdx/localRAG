"""Generation step for the local RAG system.

Combines context retrieved from the vector store with the user's question
into a prompt and sends it to a local LLM served by Ollama to produce an
answer.
"""
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src import config
from src.console import console


def create_rag_chain(
    vector_store,
    llm_model_name=config.LLM_MODEL,
    context_window=config.LLM_CONTEXT_WINDOW,
    temperature=config.LLM_TEMPERATURE,
    k=config.RETRIEVAL_K,
):
    """Creates a retrieval-augmented generation chain backed by Ollama.

    The chain retrieves the ``k`` most similar documents from the vector
    store, inserts them into a prompt alongside the user's question, sends
    the prompt to the Ollama chat model at ``config.OLLAMA_HOST``, and parses
    the response into a plain string.

    Args:
        vector_store: A LangChain vector store that supports
            ``as_retriever()`` and holds the indexed document chunks.
        llm_model_name: Name of the Ollama model to use for generation.
            Must already be pulled locally (e.g. ``ollama pull qwen3:8b``).
            Defaults to ``config.LLM_MODEL``.
        context_window: Size of the model's context window in tokens,
            passed to Ollama as ``num_ctx``. Defaults to
            ``config.LLM_CONTEXT_WINDOW``.
        temperature: Sampling temperature for the model. Defaults to
            ``config.LLM_TEMPERATURE``.
        k: Number of chunks to retrieve for each question. Defaults to
            ``config.RETRIEVAL_K``.

    Returns:
        A LangChain ``Runnable`` that takes a question string via
        ``invoke()`` and returns the generated answer as a string.

    Example:
        >>> chain = create_rag_chain(vector_store)
        >>> chain.invoke("What is the refund policy?")
        'Refunds are available within 30 days...'
    """

    # Initialize the LLM.
    llm = ChatOllama(
        model=llm_model_name,
        base_url=config.OLLAMA_HOST,
        temperature=temperature,
        num_ctx=context_window,
    )
    console.print(
        f"Initialized ChatOllama with model: {llm_model_name}, "
        f"context window: {context_window}, temperature: {temperature}",
        style="info",
    )

    # Create the retriever.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    console.print(f"Retriever initialized (k={k}).", style="info")

    # Define the prompt template.
    template = """Answer the question based ONLY on the following context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
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
