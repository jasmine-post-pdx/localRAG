# conftest.py
#
# Shared fixtures: a temporary stores folder with complete defaults, and a
# factory for adding stores to it.
import pytest

DEFAULTS = """\
documents_path = "documents"
vectorstore_path = "vectorstore"

[ingest]
embedding_model = "default-embed"
chunk_size = 1000
chunk_overlap = 200
embedding_batch_size = 32

[query]
llm_model = "default-llm"
temperature = 0
context_window = 8192
retrieval_k = 3
"""

DEFAULT_PROMPT = "Context: {context}\nQuestion: {question}\n"


@pytest.fixture
def stores_dir(tmp_path, monkeypatch):
    """A stores folder with defaults.toml and prompt.txt, also the working directory."""
    (tmp_path / "defaults.toml").write_text(DEFAULTS)
    (tmp_path / "prompt.txt").write_text(DEFAULT_PROMPT)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def make_store(stores_dir):
    """Returns a function that adds a store with the given store.toml text."""
    def make(name, text=""):
        folder = stores_dir / name
        folder.mkdir()
        (folder / "store.toml").write_text(text)
        return folder
    return make
