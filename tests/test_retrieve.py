# test_retrieve.py
#
# Tests for src/retrieve.py: querying the vector store and returning relevant chunks.
from langchain_core.documents import Document

from src.config import QuerySettings
from src.retrieve import create_retriever, retrieve_docs


QUERY_SETTINGS = QuerySettings("qwen3:8b", 0.0, 8192, 2)


class FakeRetriever:
    """A minimal retriever double that records the question it receives."""

    def __init__(self, docs):
        self.docs = docs
        self.question = None

    def invoke(self, question):
        self.question = question
        return self.docs


class FakeVectorStore:
    """A minimal vector store double that records retriever configuration."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.search_type = None
        self.search_kwargs = None

    def as_retriever(self, search_type, search_kwargs):
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        return self.retriever


def test_create_retriever_uses_similarity_search_and_retrieval_k():
    retriever = FakeRetriever([])
    vector_store = FakeVectorStore(retriever)

    result = create_retriever(vector_store, QUERY_SETTINGS)

    assert result is retriever
    assert vector_store.search_type == "similarity"
    assert vector_store.search_kwargs == {"k": 2}


def test_retrieve_docs_invokes_retriever_with_question():
    docs = [
        Document("Refunds are available within 30 days.", metadata={"source": "policy.txt"}),
        Document("Shipping takes 5 business days.", metadata={"source": "shipping.txt"}),
    ]
    retriever = FakeRetriever(docs)
    vector_store = FakeVectorStore(retriever)

    result = retrieve_docs(vector_store, QUERY_SETTINGS, "What is the refund policy?")

    assert result == docs
    assert retriever.question == "What is the refund policy?"
