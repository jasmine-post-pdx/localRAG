"""Settings for the local RAG system.

Settings come in two kinds:

* Machine-wide settings, shared by every data store, are read from
  environment variables. Each is taken from the first of these that defines
  it: variables already set in the shell, ``.env`` in the project root
  (personal overrides, never committed), and ``.env.default`` (the committed
  defaults).
* Per-store settings are read for the active store, named by ``RAG_STORE``.
  Each is taken from the first of these that defines it: a shell variable
  with the setting's environment name (e.g. ``LLM_MODEL``),
  ``stores/<name>/store.toml``, and ``stores/defaults.toml``.

``.env`` can't set per-store settings, so one store's models and chunking
don't leak into another; a warning lists any it contains. An empty value
counts as unset.

Attributes:
    OLLAMA_HOST (str): Address of the Ollama server.
    STORE (str): Name of the active data store, a folder under ``stores/``.
    STORE_DIR (Path): Folder of the active store.
    STORE_CONFIG_PATH (Path): The active store's ``store.toml``.
    SETTING_SOURCES (dict[str, str]): Where each per-store setting's value
        came from, keyed by environment name: ``SHELL`` or a file path.
    EMBEDDING_MODEL (str): Ollama model used to embed chunks and questions.
    LLM_MODEL (str): Ollama chat model used to generate answers.
    LLM_TEMPERATURE (float): Sampling temperature for the chat model.
    LLM_CONTEXT_WINDOW (int): Chat model context window in tokens.
    DOCUMENTS_PATH (str): Directory searched for source documents.
    VECTORSTORE_PATH (str): Directory where Chroma saves its data.
    CHUNK_SIZE (int): Maximum characters per chunk.
    CHUNK_OVERLAP (int): Characters shared between consecutive chunks.
    EMBEDDING_BATCH_SIZE (int): Chunks sent to Ollama per embedding request.
    RETRIEVAL_K (int): Number of chunks retrieved for each question.
"""
import os
import tomllib
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from src.console import console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env.default"
STORES_DIR = PROJECT_ROOT / "stores"
STORE_DEFAULTS_PATH = STORES_DIR / "defaults.toml"

# Environment variable name -> (store.toml table, key, type). An empty table
# name means the key sits at the top level of the file.
STORE_SETTINGS = {
    "DOCUMENTS_PATH": ("", "documents_path", str),
    "VECTORSTORE_PATH": ("", "vectorstore_path", str),
    "EMBEDDING_MODEL": ("ingest", "embedding_model", str),
    "CHUNK_SIZE": ("ingest", "chunk_size", int),
    "CHUNK_OVERLAP": ("ingest", "chunk_overlap", int),
    "EMBEDDING_BATCH_SIZE": ("ingest", "embedding_batch_size", int),
    "LLM_MODEL": ("query", "llm_model", str),
    "LLM_TEMPERATURE": ("query", "temperature", float),
    "LLM_CONTEXT_WINDOW": ("query", "context_window", int),
    "RETRIEVAL_K": ("query", "retrieval_k", int),
}

# Keys allowed in store.toml that describe the store rather than configure it.
STORE_INFO_KEYS = {("", "description")}

_KIND_NAMES = {str: "text", int: "a whole number", float: "a number"}

SHELL = "the shell"

# Taken before the .env files are loaded, because only the shell may
# override per-store settings.
_shell_env = dict(os.environ)

# load_dotenv never replaces a variable that's already set, so loading .env
# first lets it override .env.default, and the shell overrides both.
load_dotenv(ENV_PATH)
load_dotenv(DEFAULT_ENV_PATH)


def _get_str(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Setting {name} is not set. It should be defined in .env.default; "
            "override it in .env if needed."
        )
    return value


def _dotted(table, key):
    return f"{table}.{key}" if table else key


def _read_store_file(path):
    """Reads a store TOML file as a dict keyed by ``(table, key)``.

    Raises:
        RuntimeError: If the file is missing, isn't valid TOML, or contains a
            key that isn't a known setting.
    """
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"{path} not found.") from None
    except tomllib.TOMLDecodeError as error:
        raise RuntimeError(f"{path} isn't valid TOML: {error}") from None

    values = {}
    for key, value in data.items():
        if isinstance(value, dict):
            values.update({(key, sub_key): v for sub_key, v in value.items()})
        else:
            values[("", key)] = value

    known = {(table, key) for table, key, _ in STORE_SETTINGS.values()} | STORE_INFO_KEYS
    unknown = sorted(_dotted(*k) for k in values.keys() - known)
    if unknown:
        raise RuntimeError(
            f"Unknown setting(s) in {path}: {', '.join(unknown)}. "
            f"Check the spelling against {STORE_DEFAULTS_PATH.name}."
        )
    return values


def _convert(name, value, kind, source):
    if isinstance(value, str):
        if kind is str:
            return value.strip()
        try:
            return kind(value.strip())
        except ValueError:
            pass
    elif not isinstance(value, bool):
        if isinstance(value, kind):
            return value
        if kind is float and isinstance(value, int):
            return float(value)
    raise RuntimeError(
        f"Setting {name} from {source} must be {_KIND_NAMES[kind]}, got {value!r}."
    )


def _load_store_settings(store, config_path):
    """Resolves every per-store setting for a store.

    Returns:
        tuple[dict[str, str | int | float], dict[str, str]]: The values, and
        where each came from (``SHELL``, or the path of the file that set it),
        both keyed by environment name.
    """
    store_values = _read_store_file(config_path)
    default_values = _read_store_file(STORE_DEFAULTS_PATH)
    settings = {}
    sources = {}
    for name, (table, key, kind) in STORE_SETTINGS.items():
        layers = (
            (SHELL, _shell_env.get(name)),
            (os.path.relpath(config_path), store_values.get((table, key))),
            (os.path.relpath(STORE_DEFAULTS_PATH), default_values.get((table, key))),
        )
        for source, value in layers:
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            settings[name] = _convert(name, value, kind, source)
            sources[name] = source
            break
        else:
            raise RuntimeError(
                f"Setting {_dotted(table, key)} isn't set for store {store!r}. "
                f"Add it to {config_path} or {STORE_DEFAULTS_PATH}."
            )
    return settings, sources


def _warn_about_store_settings_in_env():
    ignored = [
        name for name, value in dotenv_values(ENV_PATH).items()
        if name in STORE_SETTINGS and value
    ]
    if ignored:
        console.print(
            f"Ignoring {', '.join(ignored)} in .env: per-store settings belong in "
            f"{os.path.relpath(STORE_CONFIG_PATH)}.",
            style="warning",
        )


OLLAMA_HOST = _get_str("OLLAMA_HOST")

STORE = _get_str("RAG_STORE")
STORE_CONFIG_PATH = STORES_DIR / STORE / "store.toml"
if not STORE_CONFIG_PATH.is_file():
    _available = sorted(path.parent.name for path in STORES_DIR.glob("*/store.toml"))
    raise RuntimeError(
        f"Store {STORE!r} doesn't exist: {STORE_CONFIG_PATH} not found. "
        f"Available stores: {', '.join(_available) or 'none'}. "
        "Set RAG_STORE in .env or the shell to one of them."
    )

# Relative to the working directory rather than absolute, because chunk IDs
# include each document's path and should stay short and portable.
STORE_DIR = Path(os.path.relpath(STORES_DIR / STORE))

_warn_about_store_settings_in_env()
_settings, SETTING_SOURCES = _load_store_settings(STORE, STORE_CONFIG_PATH)

EMBEDDING_MODEL = _settings["EMBEDDING_MODEL"]
LLM_MODEL = _settings["LLM_MODEL"]
LLM_TEMPERATURE = _settings["LLM_TEMPERATURE"]
LLM_CONTEXT_WINDOW = _settings["LLM_CONTEXT_WINDOW"]

# Relative store paths are resolved from the store's folder.
DOCUMENTS_PATH = str(STORE_DIR / Path(_settings["DOCUMENTS_PATH"]).expanduser())
VECTORSTORE_PATH = str(STORE_DIR / Path(_settings["VECTORSTORE_PATH"]).expanduser())

CHUNK_SIZE = _settings["CHUNK_SIZE"]
CHUNK_OVERLAP = _settings["CHUNK_OVERLAP"]
EMBEDDING_BATCH_SIZE = _settings["EMBEDDING_BATCH_SIZE"]

RETRIEVAL_K = _settings["RETRIEVAL_K"]
