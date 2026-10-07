# test_config.py
#
# Tests for src/config.py: resolving per-store settings from temporary stores.
import os
from pathlib import Path

import pytest

from src.config import SHELL, STORE_SETTINGS, load_store_settings

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


@pytest.fixture
def stores_dir(tmp_path, monkeypatch):
    """A stores folder with a complete defaults.toml, also the working directory."""
    (tmp_path / "defaults.toml").write_text(DEFAULTS)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def make_store(stores_dir, name, text=""):
    folder = stores_dir / name
    folder.mkdir()
    (folder / "store.toml").write_text(text)


def load(stores_dir, name="research", shell_env=None):
    return load_store_settings(name, stores_dir, shell_env or {})


def test_store_file_overrides_defaults(stores_dir):
    make_store(stores_dir, "research", '[query]\nllm_model = "qwen3:4b"\nretrieval_k = 6\n')

    settings, sources = load(stores_dir)

    assert settings["LLM_MODEL"] == "qwen3:4b"
    assert settings["RETRIEVAL_K"] == 6
    assert sources["LLM_MODEL"] == os.path.join("research", "store.toml")


def test_settings_missing_from_store_file_come_from_defaults(stores_dir):
    make_store(stores_dir, "research")

    settings, sources = load(stores_dir)

    assert settings["EMBEDDING_MODEL"] == "default-embed"
    assert sources["EMBEDDING_MODEL"] == "defaults.toml"
    assert set(settings) == set(STORE_SETTINGS)


def test_shell_overrides_store_file(stores_dir):
    make_store(stores_dir, "research", "[query]\nretrieval_k = 6\n")

    settings, sources = load(stores_dir, shell_env={"RETRIEVAL_K": "9"})

    assert settings["RETRIEVAL_K"] == 9
    assert sources["RETRIEVAL_K"] == SHELL


def test_empty_shell_value_counts_as_unset(stores_dir):
    make_store(stores_dir, "research", '[query]\nllm_model = "qwen3:4b"\n')

    settings, _ = load(stores_dir, shell_env={"LLM_MODEL": "  "})

    assert settings["LLM_MODEL"] == "qwen3:4b"


def test_whole_number_is_accepted_for_float_setting(stores_dir):
    make_store(stores_dir, "research", "[query]\ntemperature = 1\n")

    settings, _ = load(stores_dir)

    assert settings["LLM_TEMPERATURE"] == 1.0
    assert isinstance(settings["LLM_TEMPERATURE"], float)


def test_relative_paths_resolve_from_store_folder(stores_dir):
    make_store(stores_dir, "research", 'documents_path = "papers"\n')

    settings, _ = load(stores_dir)

    assert settings["DOCUMENTS_PATH"] == os.path.join("research", "papers")
    assert settings["VECTORSTORE_PATH"] == os.path.join("research", "vectorstore")


def test_absolute_and_home_paths_are_kept(stores_dir):
    elsewhere = stores_dir / "elsewhere"
    make_store(
        stores_dir, "research",
        f"documents_path = '{elsewhere}'\nvectorstore_path = '~/vectors'\n",
    )

    settings, _ = load(stores_dir)

    assert settings["DOCUMENTS_PATH"] == str(elsewhere)
    assert settings["VECTORSTORE_PATH"] == str(Path.home() / "vectors")


def test_unknown_key_is_rejected(stores_dir):
    make_store(stores_dir, "research", '[query]\nllm_modle = "qwen3:4b"\n')

    with pytest.raises(RuntimeError, match=r"Unknown setting\(s\).*query\.llm_modle"):
        load(stores_dir)


@pytest.mark.parametrize("value", ['"big"', "true", "1.5"])
def test_wrong_type_in_store_file_is_rejected(stores_dir, value):
    make_store(stores_dir, "research", f"[ingest]\nchunk_size = {value}\n")

    with pytest.raises(RuntimeError, match="CHUNK_SIZE .* must be a whole number"):
        load(stores_dir)


def test_wrong_type_in_shell_is_rejected(stores_dir):
    make_store(stores_dir, "research")

    with pytest.raises(RuntimeError, match="CHUNK_SIZE from the shell must be a whole number"):
        load(stores_dir, shell_env={"CHUNK_SIZE": "abc"})


def test_missing_store_lists_available_stores(stores_dir):
    make_store(stores_dir, "research")
    make_store(stores_dir, "notes")

    with pytest.raises(RuntimeError, match="'nope' doesn't exist.*Available stores: notes, research"):
        load(stores_dir, name="nope")


def test_invalid_toml_is_rejected(stores_dir):
    make_store(stores_dir, "research", "llm_model = \n")

    with pytest.raises(RuntimeError, match="isn't valid TOML"):
        load(stores_dir)


def test_setting_missing_everywhere_is_rejected(stores_dir):
    (stores_dir / "defaults.toml").write_text(DEFAULTS.replace('llm_model = "default-llm"\n', ""))
    make_store(stores_dir, "research")

    with pytest.raises(RuntimeError, match="query.llm_model isn't set for store 'research'"):
        load(stores_dir)


def test_committed_default_store_is_valid():
    settings, _ = load_store_settings("default", shell_env={})

    assert set(settings) == set(STORE_SETTINGS)
