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

from ollama import ResponseError

from src.config import QuerySettings
from src.generate import (
    NoRetrievedDocumentsError,
    check_chat_model,
    create_rag_chain,
    format_docs,
    format_sources,
    source_label,
)

PROMPT = "Context: {context}\nQuestion: {question}\n"
OLLAMA_HOST = "http://ollama:11434"


@pytest.fixture
def vector_store():
    """An in-memory vector store holding two small documents."""
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    store.add_documents([
        Document(page_content="Refunds are available within 30 days.", metadata={"source": "a.txt"}),
        Document(
            page_content="Shipping takes 5 business days.",
            metadata={"source": "docs/b.pdf", "page": 1},
        ),
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

    llm = chain.steps[3].steps__["answer"].steps[1]
    assert llm.model == "qwen3:8b"
    assert llm.base_url == OLLAMA_HOST
    assert llm.temperature == 0.2
    assert llm.num_ctx == 4096


def test_chain_returns_answer_and_retrieved_chunks(vector_store, fake_llm):
    chain = make_chain(vector_store)

    result = chain.invoke("What is the refund policy?")

    assert result["answer"] == "Within 30 days."
    assert {doc.metadata["source"] for doc in result["docs"]} == {"a.txt", "docs/b.pdf"}


def test_prompt_contains_question_and_numbered_labeled_context(vector_store, fake_llm):
    chain = make_chain(vector_store)

    chain.invoke("What is the refund policy?")

    [prompt] = fake_llm
    assert "Question: What is the refund policy?" in prompt
    assert "Refunds are available within 30 days." in prompt
    assert "Shipping takes 5 business days." in prompt
    assert "[1] " in prompt and "[2] " in prompt
    assert "a.txt" in prompt
    assert "b.pdf, page 2" in prompt


def test_chain_rejects_empty_retrieval_without_calling_llm(fake_llm):
    vector_store = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    chain = make_chain(vector_store)

    with pytest.raises(NoRetrievedDocumentsError, match="No relevant chunks"):
        chain.invoke("What is the refund policy?")

    assert fake_llm == []


# --- source_label, format_docs, format_sources --------------------------------


def test_source_label_uses_file_name_and_one_based_page():
    doc = Document("text", metadata={"source": "stores/research/documents/manual.pdf", "page": 2})

    assert source_label(doc) == "manual.pdf, page 3"


def test_source_label_omits_page_when_missing():
    assert source_label(Document("text", metadata={"source": "notes.txt"})) == "notes.txt"


def test_format_docs_numbers_chunks_in_order():
    docs = [
        Document("First.", metadata={"source": "a.txt"}),
        Document("Second.", metadata={"source": "b.pdf", "page": 0}),
    ]

    assert format_docs(docs) == "[1] a.txt\nFirst.\n\n[2] b.pdf, page 1\nSecond."


def test_format_sources_deduplicates_and_keeps_retrieval_order():
    docs = [
        Document("a", metadata={"source": "manual.pdf", "page": 0}),
        Document("b", metadata={"source": "notes.txt"}),
        Document("c", metadata={"source": "manual.pdf", "page": 0}),
    ]

    assert format_sources(docs) == "manual.pdf, page 1; notes.txt"


# --- check_chat_model ---------------------------------------------------------

def fake_client(monkeypatch, show):
    """Replaces the Ollama client with one whose show() behaves as given."""
    monkeypatch.setattr(
        "src.generate.Client",
        lambda host: type("FakeClient", (), {"show": staticmethod(show)})(),
    )


def test_check_chat_model_passes_when_model_is_pulled(monkeypatch):
    fake_client(monkeypatch, lambda model: None)

    check_chat_model("qwen3:8b", OLLAMA_HOST)


def test_check_chat_model_reports_unreachable_server(monkeypatch):
    def show(model):
        raise ConnectionError("refused")

    fake_client(monkeypatch, show)
    with pytest.raises(RuntimeError, match="Can't reach Ollama"):
        check_chat_model("qwen3:8b", OLLAMA_HOST)


def test_check_chat_model_reports_missing_model(monkeypatch):
    def show(model):
        raise ResponseError("model 'qwen3:8b' not found")

    fake_client(monkeypatch, show)
    with pytest.raises(RuntimeError, match="Try `ollama pull qwen3:8b`"):
        check_chat_model("qwen3:8b", OLLAMA_HOST)
