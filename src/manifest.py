"""Records the settings a data store's vectors were built with.

Ingestion writes ``manifest.json`` into the store's vector store folder.
Before more documents are ingested or questions are answered, the manifest
is compared with the store's current settings, so a changed embedding model
or chunking is reported clearly instead of causing a confusing error or
silently mixing incompatible chunks. Deleting the vector store's contents to
rebuild it deletes the manifest too.

Attributes:
    MANIFEST_FILENAME (str): Name of the manifest file in the vector store
        folder.
    FIXED_SETTINGS (tuple[str, ...]): ``[ingest]`` settings that can't change
        once a store has vectors.
"""
import json
from datetime import datetime

from src.config import STORE_SETTINGS, model_with_tag

MANIFEST_FILENAME = "manifest.json"
FIXED_SETTINGS = ("embedding_model", "chunk_size", "chunk_overlap")

_ENV_NAMES = {(table, key): name for name, (table, key, _) in STORE_SETTINGS.items()}


def manifest_path(store):
    """Returns the path of a store's manifest."""
    return store.vectorstore_path / MANIFEST_FILENAME


def has_vectors(store):
    """Reports whether a store's vector store has been created."""
    return (store.vectorstore_path / "chroma.sqlite3").exists()


def read_manifest(store):
    """Reads a store's manifest.

    Args:
        store (StoreConfig): The store.

    Returns:
        dict | None: The manifest, or ``None`` if the store has none, either
        because nothing has been ingested or because its vectors were built
        before manifests were recorded.

    Raises:
        RuntimeError: If the manifest isn't valid JSON.
    """
    path = manifest_path(store)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{path} isn't valid JSON: {error}. Delete it and ingest again to recreate it."
        ) from None


def write_manifest(store, chunk_count):
    """Records a store's fixed settings and chunk count in its manifest.

    Args:
        store (StoreConfig): The store whose vectors were just built.
        chunk_count (int): Number of chunks now in the vector store.

    Returns:
        dict: The manifest that was written.
    """
    manifest = {name: getattr(store.ingest, name) for name in FIXED_SETTINGS}
    manifest["chunk_count"] = chunk_count
    manifest["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path = manifest_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def find_mismatches(store, manifest, settings=FIXED_SETTINGS):
    """Lists the settings that differ between a store and its manifest.

    Embedding model names are compared with Ollama's implicit ``:latest``
    tag made explicit, so ``nomic-embed-text`` matches
    ``nomic-embed-text:latest``.

    Args:
        store (StoreConfig): The store, with its current settings.
        manifest (dict): The store's manifest, from ``read_manifest``.
        settings (Iterable[str]): ``[ingest]`` settings to compare. Defaults
            to all of ``FIXED_SETTINGS``.

    Returns:
        list[str]: One description per differing setting, naming both values
        and where the current one is set. Empty if they all match.
    """
    mismatches = []
    for name in settings:
        built = manifest.get(name)
        current = getattr(store.ingest, name)
        if name == "embedding_model" and built is not None:
            same = model_with_tag(built) == model_with_tag(current)
        else:
            same = built == current
        if not same:
            source = store.sources[_ENV_NAMES[("ingest", name)]]
            mismatches.append(
                f"ingest.{name}: vectors were built with {built!r}, but it's now "
                f"{current!r} (set in {source})"
            )
    return mismatches
