# test_app.py
#
# Tests for src/app.py: checking a store is ready, answering questions,
# switching stores, and the question loop. Ollama, Chroma, and the keyboard
# are replaced with fakes, and stores load from a temporary stores folder.
import json

import pytest
from chromadb.errors import InvalidArgumentError
from langchain_core.runnables import RunnableLambda

from src import app, config
from src.app import (
    DEFAULT_QUESTION,
    build_rag_chain,
    check_store_ready,
    main,
    query_rag,
    switch_store,
)
from src.config import AppSettings
from src.manifest import manifest_path, write_manifest

APP_SETTINGS = AppSettings(ollama_host="http://ollama:11434", default_store="research")


@pytest.fixture
def stores(stores_dir, make_store, monkeypatch):
    """Stores research and notes, which src.app loads from the temporary stores folder."""
    make_store("research", '[ingest]\nembedding_model = "nomic-embed-text"\n')
    make_store("notes")
    load_store, list_stores = config.load_store, config.list_stores
    monkeypatch.setattr(config, "load_store", lambda name: load_store(name, stores_dir, shell_env={}))
    monkeypatch.setattr(config, "list_stores", lambda folder=stores_dir: list_stores(folder))
    monkeypatch.setattr(config, "load_app_settings", lambda: APP_SETTINGS)


@pytest.fixture
def store(stores):
    return config.load_store("research")


def chain_answering(answer):
    return RunnableLambda(lambda question: answer)


def chain_raising(error):
    def fail(question):
        raise error
    return RunnableLambda(fail)


# --- check_store_ready --------------------------------------------------------

def test_empty_store_warns_nothing_ingested(store, capsys):
    check_store_ready(store)

    assert "Nothing has been ingested into research yet" in capsys.readouterr().out


def test_store_built_with_same_model_is_ready(store, capsys):
    write_manifest(store, chunk_count=1)

    check_store_ready(store)

    assert capsys.readouterr().out == ""


def test_store_built_with_other_model_is_rejected(store):
    manifest = write_manifest(store, chunk_count=1)
    manifest["embedding_model"] = "mxbai-embed-large"
    manifest_path(store).write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="Can't search research: ingest.embedding_model"):
        check_store_ready(store)


# --- build_rag_chain ----------------------------------------------------------

def test_build_rag_chain_wires_store_settings_together(store, monkeypatch):
    monkeypatch.setattr(app, "check_chat_model", lambda model, host: None)
    monkeypatch.setattr(app, "get_embedding_function", lambda model, host: ("embed", model, host))
    monkeypatch.setattr(app, "get_vector_store", lambda embed, path: ("vectors", embed, path))
    monkeypatch.setattr(app, "create_rag_chain", lambda *args: args)

    vector_store, query, prompt, host = build_rag_chain(store, APP_SETTINGS)

    embed = ("embed", "nomic-embed-text", "http://ollama:11434")
    assert vector_store == ("vectors", embed, store.vectorstore_path)
    assert query == store.query
    assert prompt == store.prompt_template
    assert host == "http://ollama:11434"


# --- query_rag ----------------------------------------------------------------

def test_query_rag_returns_and_prints_answer(store, capsys):
    answer = query_rag(chain_answering("Within 30 days."), "Refunds?", store)

    assert answer == "Within 30 days."
    assert "Within 30 days." in capsys.readouterr().out


def test_wrong_embedding_dimension_is_reported(store, capsys):
    error = InvalidArgumentError("Collection expecting embedding with dimension of 768, got 1024")

    answer = query_rag(chain_raising(error), "Refunds?", store)

    assert answer is None
    assert "Embedding dimensions are wrong" in capsys.readouterr().out


def test_other_chroma_errors_are_raised(store):
    with pytest.raises(InvalidArgumentError, match="bad query"):
        query_rag(chain_raising(InvalidArgumentError("bad query")), "Refunds?", store)


# --- switch_store -------------------------------------------------------------

def test_use_without_name_lists_stores(store, capsys):
    assert switch_store("", store, APP_SETTINGS) is None

    assert "Data stores: notes, research (current)" in capsys.readouterr().out


def test_use_with_name_loads_store_and_builds_chain(store, monkeypatch):
    monkeypatch.setattr(app, "build_rag_chain", lambda store, settings: f"chain for {store.name}")

    new_store, chain = switch_store("notes", store, APP_SETTINGS)

    assert new_store.name == "notes"
    assert chain == "chain for notes"


def test_use_with_unknown_store_keeps_current(store, capsys):
    assert switch_store("nope", store, APP_SETTINGS) is None

    assert "Still using research" in capsys.readouterr().out


# --- main ---------------------------------------------------------------------

@pytest.fixture
def session(stores, monkeypatch):
    """Runs main with typed answers; returns the (chain, question) pairs that were queried."""
    queries = []
    monkeypatch.setattr(app, "build_rag_chain", lambda store, settings: f"chain for {store.name}")
    monkeypatch.setattr(app, "query_rag", lambda chain, question, store: queries.append((chain, question)))

    def run(*answers, argv=()):
        typed = iter(answers)
        monkeypatch.setattr(app, "ask", lambda prompt: next(typed))
        main(list(argv))
        return queries
    return run


def test_enter_first_asks_default_question_then_enter_quits(session):
    assert session("", "") == [("chain for research", DEFAULT_QUESTION)]


def test_questions_go_to_store_named_on_command_line(session):
    assert session("Refunds?", "", argv=["--store", "notes"]) == [("chain for notes", "Refunds?")]


def test_use_command_switches_store_for_later_questions(session):
    assert session("/use notes", "Refunds?", "") == [("chain for notes", "Refunds?")]


def test_unknown_command_is_reported_and_not_asked(session, capsys):
    assert session("/help", "Refunds?", "") == [("chain for research", "Refunds?")]

    assert "Unknown command /help" in capsys.readouterr().out


def test_setup_error_exits_with_status_1(stores, monkeypatch, capsys):
    def fail(store, settings):
        raise RuntimeError("Can't search research")
    monkeypatch.setattr(app, "build_rag_chain", fail)

    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 1
    assert "Can't search research" in capsys.readouterr().out
