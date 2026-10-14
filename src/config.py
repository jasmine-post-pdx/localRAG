"""Settings for the local RAG system.

Settings come in two kinds, each loaded by a function so callers choose
when, and for which data store, settings are read. Importing this module
reads nothing.

* Machine-wide settings, shared by every data store, are loaded into an
  ``AppSettings`` by ``load_app_settings``. Each is taken from the first of
  these that defines it: variables set in the shell, ``.env`` in the project
  root (personal overrides, never committed), and ``.env.default`` (the
  committed defaults).
* Per-store settings are loaded into a ``StoreConfig`` by ``load_store``.
  Each is taken from the first of these that defines it: a shell variable
  with the setting's environment name (e.g. ``LLM_MODEL``),
  ``stores/<name>/store.toml``, and ``stores/defaults.toml``. The store's
  prompt template is ``stores/<name>/prompt.txt`` if it exists, otherwise
  ``stores/prompt.txt``.

``.env`` can't set per-store settings, so one store's models and chunking
don't leak into another; ``load_app_settings`` warns about any it contains.
An empty value counts as unset.

Attributes:
    STORE_SETTINGS (dict[str, tuple[str, str, type]]): Every per-store
        setting, keyed by environment name, with its ``store.toml`` table,
        key, and type. The table and key are also the setting's attribute
        path on ``StoreConfig``, e.g. ``store.query.llm_model``.
    SHELL (str): The source ``StoreConfig.sources`` reports for a value set
        by a shell variable.
"""
import os
import string
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from src.console import console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env.default"
STORES_DIR = PROJECT_ROOT / "stores"
STORE_DEFAULTS_PATH = STORES_DIR / "defaults.toml"
PROMPT_FILENAME = "prompt.txt"
PROMPT_FIELDS = {"context", "question"}

# An empty table name means the key sits at the top level of the file.
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

# Keys allowed in  .toml that describe the store rather than configure it.
STORE_INFO_KEYS = {("", "description")}

SHELL = "the shell"

_KIND_NAMES = {str: "text", int: "a whole number", float: "a number"}


@dataclass(frozen=True)
class AppSettings:
    """Machine-wide settings, shared by every data store.

    Attributes:
        ollama_host (str): Address of the Ollama server.
        default_store (str): Store to use when none is named, from
            ``RAG_STORE``.
    """
    ollama_host: str
    default_store: str


@dataclass(frozen=True)
class IngestSettings:
    """A store's ``[ingest]`` settings.

    Changing ``embedding_model``, ``chunk_size``, or ``chunk_overlap`` after
    ingesting requires rebuilding the store's vector store.

    Attributes:
        embedding_model (str): Ollama model used to embed chunks and questions.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Characters shared between consecutive chunks.
        embedding_batch_size (int): Chunks sent to Ollama per embedding request.
    """
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int


@dataclass(frozen=True)
class QuerySettings:
    """A store's ``[query]`` settings, all safe to change without re-ingesting.

    Attributes:
        llm_model (str): Ollama chat model used to generate answers.
        temperature (float): Sampling temperature for the chat model.
        context_window (int): Chat model context window in tokens.
        retrieval_k (int): Number of chunks retrieved for each question.
    """
    llm_model: str
    temperature: float
    context_window: int
    retrieval_k: int


@dataclass(frozen=True)
class StoreConfig:
    """Everything needed to ingest into or query one data store.

    Paths are relative to the working directory unless the store's settings
    give an absolute path.

    Attributes:
        name (str): The store's name, its folder under ``stores/``.
        description (str): Short note about what the store holds.
        root (Path): The store's folder.
        config_path (Path): The store's ``store.toml``.
        documents_path (Path): Folder searched for source documents.
        vectorstore_path (Path): Folder where Chroma saves the store's data.
        ingest (IngestSettings): Settings used when ingesting.
        query (QuerySettings): Settings used when answering questions.
        prompt_template (str): Prompt sent to the chat model, with
            ``{context}`` and ``{question}`` placeholders.
        prompt_path (Path): The file ``prompt_template`` was read from.
        sources (dict[str, str]): Where each setting's value came from, keyed
            by environment name: ``SHELL`` or a file path.
    """
    name: str
    description: str
    root: Path
    config_path: Path
    documents_path: Path
    vectorstore_path: Path
    ingest: IngestSettings
    query: QuerySettings
    prompt_template: str
    prompt_path: Path
    sources: dict[str, str]

    def get(self, name):
        """Returns a setting's value by environment name, e.g. ``"LLM_MODEL"``."""
        table, key, _ = STORE_SETTINGS[name]
        return getattr(getattr(self, table) if table else self, key)


def model_with_tag(name):
    """Returns a model name with Ollama's implicit ``:latest`` tag made explicit."""
    return name if ":" in name else f"{name}:latest"


def _dotted(table, key):
    return f"{table}.{key}" if table else key


def _read_prompt(path):
    """Reads a prompt template and checks its placeholders.

    Raises:
        RuntimeError: If the file is missing or its placeholders aren't
            exactly ``{context}`` and ``{question}``.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeError(f"{path} not found.") from None
    try:
        fields = {
            field for _, field, _, _ in string.Formatter().parse(text) if field is not None
        }
    except ValueError as error:
        raise RuntimeError(
            f"{path} isn't a valid prompt template: {error}. "
            "Write a literal brace as {{ or }}."
        ) from None
    if fields != PROMPT_FIELDS:
        found = ", ".join(sorted("{" + field + "}" for field in fields)) or "none"
        raise RuntimeError(
            f"{path} must use the placeholders {{context}} and {{question}} and no "
            f"others; found {found}. Write a literal brace as {{{{ or }}}}."
        )
    return text


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


def _store_dir(stores_dir, store):
    # Relative to the working directory rather than absolute, because chunk
    # IDs include each document's path and should stay short and portable.
    return Path(os.path.relpath(stores_dir / store))


def _warn_about_store_settings_in_env(env_values):
    ignored = [name for name, value in env_values.items() if name in STORE_SETTINGS and value]
    if ignored:
        console.print(
            f"Ignoring {', '.join(ignored)} in .env: per-store settings belong in "
            "stores/<name>/store.toml.",
            style="warning",
        )


def load_app_settings(shell_env=None, env_path=ENV_PATH, default_env_path=DEFAULT_ENV_PATH):
    """Loads the machine-wide settings.

    Each is taken from the first of ``shell_env``, ``env_path``, and
    ``default_env_path`` that sets it. Prints a warning if ``env_path``
    sets any per-store settings, which are ignored there.

    Args:
        shell_env (Mapping[str, str] | None): Shell variables. Defaults to
            ``os.environ``.
        env_path (Path): Personal overrides file. Defaults to ``.env``.
        default_env_path (Path): Committed defaults file. Defaults to
            ``.env.default``.

    Returns:
        AppSettings: The machine-wide settings.

    Raises:
        RuntimeError: If a setting isn't set anywhere.
    """
    if shell_env is None:
        shell_env = os.environ
    env_values = dotenv_values(env_path)
    layers = (shell_env, env_values, dotenv_values(default_env_path))

    def get(name):
        for layer in layers:
            value = (layer.get(name) or "").strip()
            if value:
                return value
        raise RuntimeError(
            f"Setting {name} is not set. It should be defined in .env.default; "
            "override it in .env if needed."
        )

    _warn_about_store_settings_in_env(env_values)
    return AppSettings(ollama_host=get("OLLAMA_HOST"), default_store=get("RAG_STORE"))


def list_stores(stores_dir=STORES_DIR):
    """Lists the names of the stores that have a ``store.toml``, sorted."""
    return sorted(path.parent.name for path in stores_dir.glob("*/store.toml"))


def load_store(name, stores_dir=STORES_DIR, shell_env=None):
    """Loads every per-store setting for a data store.

    Each setting is taken from the first of ``shell_env``,
    ``<stores_dir>/<name>/store.toml``, and ``<stores_dir>/defaults.toml``
    that sets it. Relative document and vector store paths are resolved from
    the store's folder.

    Args:
        name (str): Name of the store, a folder under ``stores_dir``.
        stores_dir (Path): Folder holding the store folders and
            ``defaults.toml``. Defaults to ``STORES_DIR``.
        shell_env (Mapping[str, str] | None): Variables that override the
            files, keyed by environment name. Defaults to ``os.environ``.

    Returns:
        StoreConfig: The store's settings.

    Raises:
        RuntimeError: If the store doesn't exist, a file is missing or
            invalid, a value has the wrong type, or a setting isn't set
            anywhere.
    """
    if shell_env is None:
        shell_env = os.environ
    config_path = stores_dir / name / "store.toml"
    defaults_path = stores_dir / STORE_DEFAULTS_PATH.name
    if not config_path.is_file():
        raise RuntimeError(
            f"Store {name!r} doesn't exist: {config_path} not found. "
            f"Available stores: {', '.join(list_stores(stores_dir)) or 'none'}."
        )

    store_values = _read_store_file(config_path)
    default_values = _read_store_file(defaults_path)
    tables = {"": {}, "ingest": {}, "query": {}}
    sources = {}
    for env_name, (table, key, kind) in STORE_SETTINGS.items():
        layers = (
            (SHELL, shell_env.get(env_name)),
            (os.path.relpath(config_path), store_values.get((table, key))),
            (os.path.relpath(defaults_path), default_values.get((table, key))),
        )
        for source, value in layers:
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            tables[table][key] = _convert(env_name, value, kind, source)
            sources[env_name] = source
            break
        else:
            raise RuntimeError(
                f"Setting {_dotted(table, key)} isn't set for store {name!r}. "
                f"Add it to {config_path} or {defaults_path}."
            )

    prompt_path = stores_dir / name / PROMPT_FILENAME
    if not prompt_path.is_file():
        prompt_path = stores_dir / PROMPT_FILENAME
    prompt_template = _read_prompt(prompt_path)

    root = _store_dir(stores_dir, name)
    return StoreConfig(
        name=name,
        description=str(store_values.get(("", "description"), "")),
        root=root,
        config_path=root / "store.toml",
        documents_path=root / Path(tables[""]["documents_path"]).expanduser(),
        vectorstore_path=root / Path(tables[""]["vectorstore_path"]).expanduser(),
        ingest=IngestSettings(**tables["ingest"]),
        query=QuerySettings(**tables["query"]),
        prompt_template=prompt_template,
        prompt_path=Path(os.path.relpath(prompt_path)),
        sources=sources,
    )
