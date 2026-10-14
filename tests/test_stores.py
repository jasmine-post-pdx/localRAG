# test_stores.py
#
# Tests for src/stores.py: creating stores and choosing the default store.
import tomllib

import pytest
from dotenv import dotenv_values

from src.stores import create_store, use_store


def read_toml(path):
    return tomllib.loads(path.read_text())


def test_create_pins_settings_from_defaults(stores_dir):
    store = create_store("research", description="Papers", stores_dir=stores_dir)

    settings = read_toml(stores_dir / "research" / "store.toml")
    assert settings["description"] == "Papers"
    assert settings["ingest"] == {
        "embedding_model": "default-embed",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_batch_size": 32,
    }
    assert settings["query"]["llm_model"] == "default-llm"
    assert (stores_dir / "research" / "documents").is_dir()
    assert store.sources["EMBEDDING_MODEL"].endswith("store.toml")


def test_create_applies_model_overrides(stores_dir):
    store = create_store(
        "research", embedding_model="mxbai-embed-large", llm_model="qwen3:4b",
        stores_dir=stores_dir,
    )

    assert store.ingest.embedding_model == "mxbai-embed-large"
    assert store.query.llm_model == "qwen3:4b"


def test_create_copies_settings_and_prompt_but_not_paths(stores_dir, make_store):
    source = make_store(
        "papers",
        'documents_path = "/somewhere/else"\n'
        '[ingest]\nchunk_size = 500\n[query]\nretrieval_k = 8\n',
    )
    (source / "prompt.txt").write_text("Papers: {context}\nQ: {question}")

    store = create_store("papers-v2", copy_from="papers", stores_dir=stores_dir)

    assert store.ingest.chunk_size == 500
    assert store.query.retrieval_k == 8
    assert store.prompt_template == "Papers: {context}\nQ: {question}"
    assert "documents_path" not in read_toml(stores_dir / "papers-v2" / "store.toml")


@pytest.mark.parametrize("name", ["", "-leading-dash", "has space", "../escape"])
def test_create_rejects_invalid_names(stores_dir, name):
    with pytest.raises(RuntimeError, match="isn't a valid store name"):
        create_store(name, stores_dir=stores_dir)


def test_create_rejects_existing_store(stores_dir, make_store):
    make_store("research")

    with pytest.raises(RuntimeError, match="already exists"):
        create_store("research", stores_dir=stores_dir)


def test_failed_create_leaves_nothing_behind(stores_dir):
    with pytest.raises(RuntimeError, match="'missing' doesn't exist"):
        create_store("research", copy_from="missing", stores_dir=stores_dir)

    assert not (stores_dir / "research").exists()


def test_use_saves_default_store_in_env(stores_dir, make_store, tmp_path):
    make_store("research")
    env_path = tmp_path / ".env"
    env_path.write_text("OLLAMA_HOST=http://other:11434\n")

    use_store("research", env_path=env_path, stores_dir=stores_dir)

    assert dotenv_values(env_path) == {
        "OLLAMA_HOST": "http://other:11434",
        "RAG_STORE": "research",
    }


def test_use_rejects_unknown_store(stores_dir, make_store, tmp_path):
    make_store("research")

    with pytest.raises(RuntimeError, match="'nope' doesn't exist. Available stores: research"):
        use_store("nope", env_path=tmp_path / ".env", stores_dir=stores_dir)
