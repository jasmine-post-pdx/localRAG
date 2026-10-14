# test_util.py
#
# Tests for src/util.py: saving model choices to a store's store.toml.
import tomllib

from src.util import save_store_setting


def test_save_changes_only_the_setting(tmp_path):
    path = tmp_path / "store.toml"
    path.write_text(
        "# About this store\n\n[query]\n# Chat model\n"
        'llm_model = "qwen3:8b"\nretrieval_k = 3\n'
    )

    save_store_setting("LLM_MODEL", "qwen3:4b", path)

    assert path.read_text() == (
        "# About this store\n\n[query]\n# Chat model\n"
        'llm_model = "qwen3:4b"\nretrieval_k = 3\n'
    )


def test_save_adds_missing_table(tmp_path):
    path = tmp_path / "store.toml"
    path.write_text('description = "notes"\n')

    save_store_setting("EMBEDDING_MODEL", "mxbai-embed-large", path)

    assert tomllib.loads(path.read_text()) == {
        "description": "notes",
        "ingest": {"embedding_model": "mxbai-embed-large"},
    }
