"""Settings for the local RAG system, read from environment variables.

Each setting is taken from the first of these that defines it:

1. Variables already set in the shell.
2. ``.env`` in the project root: personal overrides, never committed.
3. ``.env.default`` in the project root: the committed defaults.

``.env.default`` documents every setting; ``.env.example`` shows sample
overrides. An empty value counts as unset.

Attributes:
    OLLAMA_HOST (str): Address of the Ollama server.
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
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# load_dotenv never replaces a variable that's already set, so loading .env
# first lets it override .env.default, and the shell overrides both.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.default")


def _get_str(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Setting {name} is not set. It should be defined in .env.default; "
            "override it in .env if needed."
        )
    return value


def _get_int(name):
    value = _get_str(name)
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(f"Setting {name} must be a whole number, got {value!r}.") from None


def _get_float(name):
    value = _get_str(name)
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"Setting {name} must be a number, got {value!r}.") from None


OLLAMA_HOST = _get_str("OLLAMA_HOST")

EMBEDDING_MODEL = _get_str("EMBEDDING_MODEL")
LLM_MODEL = _get_str("LLM_MODEL")
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE")
LLM_CONTEXT_WINDOW = _get_int("LLM_CONTEXT_WINDOW")

DOCUMENTS_PATH = _get_str("DOCUMENTS_PATH")
VECTORSTORE_PATH = _get_str("VECTORSTORE_PATH")

CHUNK_SIZE = _get_int("CHUNK_SIZE")
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP")
EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE")

RETRIEVAL_K = _get_int("RETRIEVAL_K")
