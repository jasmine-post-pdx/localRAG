# test_manifest.py
#
# Tests for src/manifest.py: recording a store's fixed settings and spotting
# changes to them.
import json

import pytest

from src.config import load_store
from src.manifest import (
    find_mismatches,
    has_vectors,
    manifest_path,
    read_manifest,
    write_manifest,
)


@pytest.fixture
def store(stores_dir, make_store):
    make_store("research", '[ingest]\nembedding_model = "nomic-embed-text"\n')
    return load_store("research", stores_dir, shell_env={})


def test_store_without_manifest_reads_as_none(store):
    assert read_manifest(store) is None
    assert not has_vectors(store)


def test_written_manifest_records_fixed_settings(store):
    write_manifest(store, chunk_count=42)

    manifest = read_manifest(store)

    assert manifest["embedding_model"] == "nomic-embed-text"
    assert manifest["chunk_size"] == 1000
    assert manifest["chunk_overlap"] == 200
    assert manifest["chunk_count"] == 42
    assert "updated_at" in manifest


def test_unchanged_settings_have_no_mismatches(store):
    manifest = write_manifest(store, chunk_count=1)

    assert find_mismatches(store, manifest) == []


def test_latest_tag_doesnt_count_as_a_different_model(store):
    manifest = write_manifest(store, chunk_count=1)
    manifest["embedding_model"] = "nomic-embed-text:latest"

    assert find_mismatches(store, manifest) == []


def test_changed_settings_are_reported_with_their_source(store):
    manifest = write_manifest(store, chunk_count=1)
    manifest["embedding_model"] = "mxbai-embed-large"
    manifest["chunk_size"] = 500

    mismatches = find_mismatches(store, manifest)

    assert len(mismatches) == 2
    assert "ingest.embedding_model: vectors were built with 'mxbai-embed-large'" in mismatches[0]
    assert "set in research/store.toml" in mismatches[0]
    assert "ingest.chunk_size" in mismatches[1]
    assert "set in defaults.toml" in mismatches[1]


def test_only_requested_settings_are_compared(store):
    manifest = write_manifest(store, chunk_count=1)
    manifest["chunk_size"] = 500

    assert find_mismatches(store, manifest, settings=("embedding_model",)) == []


def test_invalid_manifest_is_rejected(store):
    path = manifest_path(store)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")

    with pytest.raises(RuntimeError, match="isn't valid JSON"):
        read_manifest(store)


def test_manifest_is_plain_json(store):
    write_manifest(store, chunk_count=3)

    assert json.loads(manifest_path(store).read_text())["chunk_count"] == 3
