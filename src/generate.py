"""Generation step for the local RAG system.

Combines context retrieved from the vector store with the user's question
into a prompt and sends it to a local LLM served by Ollama to produce an
answer.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


def create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192):
    """Creates the RAG chain that answers questions from indexed documents.

    Builds a LangChain Expression Language (LCEL) pipeline that, given a
    question, retrieves the 3 most similar chunks from ``vector_store``, fills
    them into a prompt telling the model to answer only from that context,
    sends the prompt to an Ollama chat model, and returns the reply as plain
    text. Creating the chain doesn't contact Ollama; the server must be
    running and the model pulled (e.g. ``ollama pull qwen3:8b``) before the
    chain is invoked.

    Args:
        vector_store (VectorStore): Store holding the indexed document chunks,
            typically the output of ``get_vector_store``. Its embedding
            function must be the same model that embedded the saved chunks.
        llm_model_name (str): Name of the Ollama chat model to use. Defaults
            to ``"qwen3:8b"``.
        context_window (int): Context window size, in tokens, passed to
            Ollama as ``num_ctx``. It must be large enough to hold the prompt
            plus the retrieved chunks. Defaults to 8192.

    Returns:
        Runnable: A chain that takes the question as a string and returns the
        model's answer as a string, e.g.
        ``rag_chain.invoke("What is this document about?")``.
    """
    # Initialize the LLM
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0, # Lower temperature for more factual RAG answers
        num_ctx=context_window # IMPORTANT: Set context window size
    )
    print(f"Initialized ChatOllama with model:{llm_model_name}, context window:{context_window}")
    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", # Or "mmr"
        search_kwargs={'k': 3} # Retrieve top 3 relevant chunks
    )
    print("Retriever initialized.")
    # Define the prompt template
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")
    # Define the RAG chain using LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain
#... (previous function calls)
vector_store = get_vector_store(embedding_function) # Assuming DB is already indexed
rag_chain = create_rag_chain(vector_store) # Call this later

