"""Generation step for the local RAG system.

Combines context retrieved from the vector store with the user's question
into a prompt and sends it to a local LLM served by Ollama to produce an
answer.
"""
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def create_rag_chain(
    vector_store,
    llm_model_name="qwen3:8b",
    context_window=8192,
):
    """Creates a retrieval-augmented generation chain backed by Ollama.

    The chain retrieves the three most similar documents from the vector
    store, inserts them into a prompt alongside the user's question, sends
    the prompt to a local Ollama chat model, and parses the response into a
    plain string.

    Args:
        vector_store: A LangChain vector store that supports
            ``as_retriever()`` and holds the indexed document chunks.
        llm_model_name: Name of the Ollama model to use for generation.
            Must already be pulled locally (e.g. ``ollama pull qwen3:8b``).
        context_window: Size of the model's context window in tokens,
            passed to Ollama as ``num_ctx``.

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
        temperature=0,
        num_ctx=context_window,
    )
    print(
        f"Initialized ChatOllama with model: {llm_model_name}, "
        f"context window: {context_window}"
    )

    # Create the retriever.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    print("Retriever initialized.")

    # Define the prompt template.
    template = """Answer the question based ONLY on the following context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")

    # Define the RAG chain using LCEL.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain created.")

    return rag_chain
