"""Retrieval step for the local RAG system.

Turns a vector store into a retriever configured from the store's query
settings, and exposes a direct retrieval helper so retrieval quality can be
tested without invoking the LLM generation step.
"""
from src.console import console


def create_retriever(vector_store, query_settings):
    """Creates a similarity retriever for a store's configured top-k chunks.

    Args:
        vector_store: A LangChain vector store that supports ``as_retriever()``.
        query_settings (QuerySettings): The store's query settings, including
            ``retrieval_k``.

    Returns:
        Runnable: A retriever that returns the most similar chunks for a query.
    """
    k = query_settings.retrieval_k
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    console.print(f"Retriever initialized (k={k}).", style="info")
    return retriever


def retrieve_docs(vector_store, query_settings, question):
    """Returns the chunks most relevant to a question.

    This is the direct retrieval entry point for debugging and evaluation,
    where callers need retrieved chunks without running prompt construction or
    LLM generation.

    Args:
        vector_store: A LangChain vector store that supports ``as_retriever()``.
        query_settings (QuerySettings): The store's query settings.
        question (str): The user's question.

    Returns:
        list[Document]: The chunks returned by the vector store retriever.
    """
    return create_retriever(vector_store, query_settings).invoke(question)
