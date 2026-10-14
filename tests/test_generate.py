# test_generate.py
#
# Tests for src/generate.py: prompt construction and LLM response handling.
# An in-memory vector store and a fake chat model stand in for Ollama.
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from src.config import QuerySettings
from src.generate import create_rag_chain

PROMPT = "Context: {context}\nQuestion: {question}\n"
OLLAMA_HOST = "http://ollama:11434"


@pytest.fixture
def vector_store():
    """An in-memory vector store holding two small documents."""
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    store.add_documents([
        Document(page_content="Refunds are available within 30 days.", metadata={"source": "a.txt"}),
        Document(page_content="Shipping takes 5 business days.", metadata={"source": "b.txt"}),
    ])
    return store


@pytest.fixture
def fake_llm(monkeypatch):
    """Replaces ChatOllama with a model that records each prompt and gives a fixed answer."""
    prompts = []

    def respond(prompt_value):
        prompts.append(prompt_value.to_string())
        return AIMessage("Within 30 days.")

    monkeypatch.setattr("src.generate.ChatOllama", lambda **kwargs: RunnableLambda(respond))
    return prompts


def make_chain(vector_store, query_settings=None):
    query_settings = query_settings or QuerySettings("qwen3:8b", 0.2, 4096, 3)
    return create_rag_chain(vector_store, query_settings, PROMPT, OLLAMA_HOST)


def test_chat_model_uses_query_settings_and_host(vector_store):
    chain = make_chain(vector_store)

    llm = chain.steps[2]
    assert llm.model == "qwen3:8b"
    assert llm.base_url == OLLAMA_HOST
    assert llm.temperature == 0.2
    assert llm.num_ctx == 4096


def test_retriever_fetches_retrieval_k_chunks(vector_store):
    chain = make_chain(vector_store, QuerySettings("qwen3:8b", 0.0, 8192, 1))

    retriever = chain.steps[0].steps__["context"]
    assert retriever.search_type == "similarity"
    assert retriever.search_kwargs == {"k": 1}


def test_chain_returns_answer_as_string(vector_store, fake_llm):
    chain = make_chain(vector_store)

    assert chain.invoke("What is the refund policy?") == "Within 30 days."


def test_prompt_contains_question_and_retrieved_context(vector_store, fake_llm):
    chain = make_chain(vector_store)

    chain.invoke("What is the refund policy?")

    [prompt] = fake_llm
    assert "Question: What is the refund policy?" in prompt
    assert "Refunds are available within 30 days." in prompt
    assert "Shipping takes 5 business days." in prompt
