# test_config.py
#
# Tests for src/config.py: loading machine-wide settings and per-store
# settings from temporary files.
from pathlib import Path

import pytest

from conftest import DEFAULT_PROMPT, DEFAULTS
from src.config import (
    SHELL,
    IngestSettings,
    QuerySettings,
    list_stores,
    load_app_settings,
    load_store,
    model_with_tag,
)


def load(stores_dir, name="research", shell_env=None):
    return load_store(name, stores_dir, shell_env or {})


# --- load_store ---------------------------------------------------------------

def test_loads_store_into_typed_config(stores_dir, make_store):
    make_store("research", 'description = "Papers"\n')

    store = load(stores_dir)

    assert store.name == "research"
    assert store.description == "Papers"
    assert store.root == Path("research")
    assert store.config_path == Path("research", "store.toml")
    assert store.ingest == IngestSettings("default-embed", 1000, 200, 32)
    assert store.query == QuerySettings("default-llm", 0.0, 8192, 3)


def test_store_file_overrides_defaults(stores_dir, make_store):
    make_store("research", '[query]\nllm_model = "qwen3:4b"\nretrieval_k = 6\n')

    store = load(stores_dir)

    assert store.query.llm_model == "qwen3:4b"
    assert store.query.retrieval_k == 6
    assert store.sources["LLM_MODEL"] == str(Path("research", "store.toml"))


def test_settings_missing_from_store_file_come_from_defaults(stores_dir, make_store):
    make_store("research")

    store = load(stores_dir)

    assert store.ingest.embedding_model == "default-embed"
    assert store.sources["EMBEDDING_MODEL"] == "defaults.toml"


def test_shell_overrides_store_file(stores_dir, make_store):
    make_store("research", "[query]\nretrieval_k = 6\n")

    store = load(stores_dir, shell_env={"RETRIEVAL_K": "9"})

    assert store.query.retrieval_k == 9
    assert store.sources["RETRIEVAL_K"] == SHELL


def test_empty_shell_value_counts_as_unset(stores_dir, make_store):
    make_store("research", '[query]\nllm_model = "qwen3:4b"\n')

    store = load(stores_dir, shell_env={"LLM_MODEL": "  "})

    assert store.query.llm_model == "qwen3:4b"


def test_whole_number_is_accepted_for_float_setting(stores_dir, make_store):
    make_store("research", "[query]\ntemperature = 1\n")

    store = load(stores_dir)

    assert store.query.temperature == 1.0
    assert isinstance(store.query.temperature, float)


def test_relative_paths_resolve_from_store_folder(stores_dir, make_store):
    make_store("research", 'documents_path = "papers"\n')

    store = load(stores_dir)

    assert store.documents_path == Path("research", "papers")
    assert store.vectorstore_path == Path("research", "vectorstore")


def test_absolute_and_home_paths_are_kept(stores_dir, make_store):
    elsewhere = stores_dir / "elsewhere"
    make_store("research", f"documents_path = '{elsewhere}'\nvectorstore_path = '~/vectors'\n")

    store = load(stores_dir)

    assert store.documents_path == elsewhere
    assert store.vectorstore_path == Path.home() / "vectors"


def test_get_returns_setting_by_environment_name(stores_dir, make_store):
    make_store("research", '[query]\nllm_model = "qwen3:4b"\n')

    store = load(stores_dir)

    assert store.get("LLM_MODEL") == "qwen3:4b"
    assert store.get("CHUNK_SIZE") == 1000
    assert store.get("DOCUMENTS_PATH") == Path("research", "documents")


def test_unknown_key_is_rejected(stores_dir, make_store):
    make_store("research", '[query]\nllm_modle = "qwen3:4b"\n')

    with pytest.raises(RuntimeError, match=r"Unknown setting\(s\).*query\.llm_modle"):
        load(stores_dir)


@pytest.mark.parametrize("value", ['"big"', "true", "1.5"])
def test_wrong_type_in_store_file_is_rejected(stores_dir, make_store, value):
    make_store("research", f"[ingest]\nchunk_size = {value}\n")

    with pytest.raises(RuntimeError, match="CHUNK_SIZE .* must be a whole number"):
        load(stores_dir)


def test_wrong_type_in_shell_is_rejected(stores_dir, make_store):
    make_store("research")

    with pytest.raises(RuntimeError, match="CHUNK_SIZE from the shell must be a whole number"):
        load(stores_dir, shell_env={"CHUNK_SIZE": "abc"})


def test_missing_store_lists_available_stores(stores_dir, make_store):
    make_store("research")
    make_store("notes")

    with pytest.raises(RuntimeError, match="'nope' doesn't exist.*Available stores: notes, research"):
        load(stores_dir, name="nope")


def test_invalid_toml_is_rejected(stores_dir, make_store):
    make_store("research", "llm_model = \n")

    with pytest.raises(RuntimeError, match="isn't valid TOML"):
        load(stores_dir)


def test_setting_missing_everywhere_is_rejected(stores_dir, make_store):
    (stores_dir / "defaults.toml").write_text(DEFAULTS.replace('llm_model = "default-llm"\n', ""))
    make_store("research")

    with pytest.raises(RuntimeError, match="query.llm_model isn't set for store 'research'"):
        load(stores_dir)


def test_committed_default_store_is_valid():
    store = load_store("default", shell_env={})

    assert store.name == "default"


# --- Prompt templates -----------------------------------------------------------

def test_store_without_prompt_uses_shared_prompt(stores_dir, make_store):
    make_store("research")

    store = load(stores_dir)

    assert store.prompt_template == DEFAULT_PROMPT
    assert store.prompt_path == Path("prompt.txt")


def test_store_prompt_overrides_shared_prompt(stores_dir, make_store):
    folder = make_store("research")
    (folder / "prompt.txt").write_text("Use {context} to answer {question} briefly.")

    store = load(stores_dir)

    assert store.prompt_template == "Use {context} to answer {question} briefly."
    assert store.prompt_path == Path("research", "prompt.txt")


def test_prompt_may_contain_escaped_braces(stores_dir, make_store):
    folder = make_store("research")
    (folder / "prompt.txt").write_text('Reply as {{"answer": ...}}. {context} {question}')

    assert load(stores_dir).prompt_template.startswith('Reply as {{"answer"')


@pytest.mark.parametrize("text, found", [
    ("Answer {question}", r"\{question\}"),
    ("{context} {question} {topic}", r"\{context\}, \{question\}, \{topic\}"),
    ("{context} {question} {}", r"\{\}"),
])
def test_prompt_with_wrong_placeholders_is_rejected(stores_dir, make_store, text, found):
    folder = make_store("research")
    (folder / "prompt.txt").write_text(text)

    with pytest.raises(RuntimeError, match=f"found .*{found}"):
        load(stores_dir)


def test_prompt_with_unbalanced_brace_is_rejected(stores_dir, make_store):
    folder = make_store("research")
    (folder / "prompt.txt").write_text("{context} {question} {")

    with pytest.raises(RuntimeError, match="isn't a valid prompt template"):
        load(stores_dir)


def test_committed_shared_prompt_is_valid():
    assert "{context}" in load_store("default", shell_env={}).prompt_template


# --- list_stores and model_with_tag --------------------------------------------

def test_list_stores_returns_folders_with_store_file(stores_dir, make_store):
    make_store("research")
    make_store("notes")
    (stores_dir / "not-a-store").mkdir()

    assert list_stores(stores_dir) == ["notes", "research"]


def test_model_with_tag_adds_latest_only_when_untagged():
    assert model_with_tag("nomic-embed-text") == "nomic-embed-text:latest"
    assert model_with_tag("qwen3:8b") == "qwen3:8b"


# --- load_app_settings --------------------------------------------------------

@pytest.fixture
def env_files(tmp_path):
    """Paths for a .env and a .env.default with every machine-wide setting."""
    env_path = tmp_path / ".env"
    default_env_path = tmp_path / ".env.default"
    default_env_path.write_text("OLLAMA_HOST=http://default:11434\nRAG_STORE=default\n")
    return env_path, default_env_path


def load_app(env_files, shell_env=None):
    env_path, default_env_path = env_files
    return load_app_settings(shell_env or {}, env_path, default_env_path)


def test_app_settings_come_from_env_default(env_files):
    settings = load_app(env_files)

    assert settings.ollama_host == "http://default:11434"
    assert settings.default_store == "default"


def test_env_overrides_env_default(env_files):
    env_files[0].write_text("RAG_STORE=research\n")

    assert load_app(env_files).default_store == "research"


def test_shell_overrides_env(env_files):
    env_files[0].write_text("RAG_STORE=research\n")

    assert load_app(env_files, shell_env={"RAG_STORE": "notes"}).default_store == "notes"


def test_missing_app_setting_is_rejected(env_files):
    env_files[1].write_text("RAG_STORE=default\n")

    with pytest.raises(RuntimeError, match="OLLAMA_HOST is not set"):
        load_app(env_files)


def test_per_store_settings_in_env_are_ignored_with_warning(env_files, capsys):
    env_files[0].write_text("LLM_MODEL=qwen3:4b\n")

    load_app(env_files)

    assert "Ignoring LLM_MODEL in .env" in capsys.readouterr().out
