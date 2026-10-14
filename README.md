# RAG Sandbox — Learning & Experimentation Project

> Status: Active learning sandbox and a public demonstration of my transition
> from software test engineering into AI engineering.

## Why this exists

I'm a software development engineer in test (SDET) with more than 20 years of
experience, and I am now transitioning into AI engineering. This repository
gives me hands-on experience with retrieval-augmented generation fundamentals
as I work toward building a production-ready, evaluated AI application. It
also allows me to apply the testing discipline I have developed throughout my
career and is part of a series of public projects documenting the skills I am
building as I pursue an AI engineering role in early 2027.

The current prototype loads local documents (PDF, text, and Markdown),
splits them into overlapping chunks, embeds those chunks into a persistent
vector store, and answers questions about them interactively with a locally
hosted model.

## From tutorial to structured project

This project began with freeCodeCamp's
[How to Build Your Own Local AI: Create Free RAG and AI Agents with Qwen 3 and Ollama](https://www.freecodecamp.org/news/build-a-local-ai/).
The tutorial presents the RAG workflow in a single Python file, which is useful
for learning the end-to-end flow quickly. I reorganized that implementation
into a more conventional application structure:

- `src/ingest.py` owns document loading, chunking, embedding, and persistence.
- `src/retrieve.py` is reserved for standalone retrieval behavior as that layer
  develops.
- `src/generate.py` builds the retrieval and generation chain.
- `src/app.py` is the application entry point.
- `src/config.py` loads every tunable setting (models, paths, chunking,
  retrieval) into typed settings objects, from a data store's config and the
  environment files. The entry points pass those objects down, so the other
  modules hold no hard-coded values and can switch stores at runtime.
- `src/manifest.py` records the settings each data store's vectors were
  built with, so changed settings are caught before they cause errors.
- `src/stores.py` lists, inspects, creates, and selects data stores.
- `src/util.py` holds maintenance tasks, currently an interactive picker for
  a data store's embedding and chat models.
- `src/console.py` holds the shared Rich console that styles the terminal
  output (section rules, status lines, answer and error panels).
- `tests/` mirrors those responsibilities and holds the corresponding
  automated tests.

This was a deliberate engineering change, not just an organizational choice.
Separating the stages makes data ingestion, retrieval, prompt construction,
and generation easier to understand, test, replace, and experiment with
independently. It also gives the sandbox room to evolve beyond the tutorial
without turning one learning script into a growing monolith.

## What I'm learning here

- Chunking strategies and their effect on retrieval quality
- Embedding model tradeoffs
- Similarity search and the relationship between retrieval and generation
- Prompt construction and context-window constraints
- Evaluation of LLM output (I brought my testing discipline with me)
- The operational details of keeping a RAG pipeline local

## Stack

- **Python 3.14** — the project language, pinned through `.python-version`.
- **LangChain** — composes document loading, splitting, retrieval, prompting,
  and generation so each part of the pipeline can be changed independently.
- **Ollama** — keeps embedding and generation workloads local.
- **`nomic-embed-text`** — the current local embedding model.
- **`qwen3:8b`** — the current local chat model, configured for deterministic
  output with a temperature of `0`.
- **ChromaDB** — persists embeddings locally between ingestion and querying.
- **PyPDF** — extracts text and page metadata from the source PDF.

## Running it

### Prerequisites

- Python 3.14
- [Ollama](https://ollama.com/) installed and running

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the exact dependency versions captured during development, install
`requirements.lock` instead of `requirements.txt`.

### Configuration

The app can keep several independent **data stores**, each with its own
documents, vector store, and settings. Each store is a folder under
`stores/`:

```text
stores/
  defaults.toml     # default settings for every store, with explanations
  prompt.txt        # default prompt template for every store
  default/
    store.toml      # this store's settings
    prompt.txt      # optional: this store's own prompt template
    documents/
    vectorstore/    # includes manifest.json, written by ingestion
```

Settings come in two kinds:

- **Per-store settings** (models, chunking, retrieval, paths) are taken from
  the first of: a shell variable for one run, such as
  `LLM_MODEL=qwen3:4b python -m src.app`; the store's `store.toml`; and
  `stores/defaults.toml`.
- **Machine-wide settings** (the Ollama address and which store to use) are
  taken from the first of: your shell; `.env`, your personal overrides,
  ignored by Git; and `.env.default`, the committed defaults.

The app runs with no extra setup, using the `default` store. Manage stores
with `src.stores`:

```bash
python -m src.stores                      # list stores, their models, and what's ingested
python -m src.stores info research        # settings, where each is set, paths, status
python -m src.stores create research --description "ML papers" \
    --embedding-model mxbai-embed-large   # or --copy-from <store>
python -m src.stores use research         # make it the default (saved in .env)
```

`create` writes every ingest and query setting into the new store's
`store.toml`, so later changes to `stores/defaults.toml` can't silently
change it. Every other command takes `--store <name>` to choose a store for
one run; without it, they use the default store (`RAG_STORE`):

```bash
python -m src.ingest --store research
python -m src.app --store research
```

While asking questions in `src.app`, type `/use <name>` to switch to another
store, or `/use` on its own to list them.

Each store answers with the prompt in `stores/<name>/prompt.txt` if it has
one, otherwise `stores/prompt.txt`. A prompt must contain the placeholders
`{context}` (the retrieved chunks) and `{question}`, and no others; write a
literal brace as `{{` or `}}`.

Ingestion records a store's `embedding_model`, `chunk_size`, and
`chunk_overlap` in `manifest.json` inside its vector store. If you change
one of them afterwards, ingestion refuses to add to the store and the app
refuses to search it with a different embedding model, explaining what
changed. To rebuild with the new settings, delete the contents of the
store's vector store and ingest again.

To switch models without editing files, run the model picker. It shows a
store's embedding and chat models next to the models installed in Ollama,
and saves any change to that store's `store.toml`. Press Enter at a prompt
to keep the current model.

```bash
python -m src.util
python -m src.util --store research   # another store
```

### Models

Pull the two local models (or whichever ones you configured):

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Place your source files (`.pdf`, `.txt`, or `.md`) anywhere under the active
store's documents folder, which for the `default` store is:

```text
stores/default/documents/
```

With the Ollama server running, ingest the documents and then ask questions
about them:

```bash
python -m src.ingest
python -m src.app
```

`src.app` prompts for questions until you press Enter on an empty line.
Pressing Enter at the first prompt asks a sample question about the main
topics of the documents.

Each chunk is stored under a deterministic ID (`<source>:<page>:<chunk_index>`),
so re-running ingestion only embeds chunks from files that are new since the
last run. Chunks already in the store are not updated, so after editing or
removing a file, delete the contents of `stores/default/vectorstore/` and run `src.ingest`
again to rebuild.
Source documents and vector-store data are intentionally excluded from Git.

Run the automated tests from the project root with:

```bash
pytest
```

They cover loading settings and prompts, the manifest, creating and
selecting stores, saving model choices, the app's question loop and store
switching, and generation's prompt construction and chain wiring, plus the
parts of ingestion that don't need Ollama (finding files, splitting, and
chunk IDs). Fakes stand in for Ollama, Chroma, and the keyboard, and stores
load from temporary files, so your real stores aren't touched. Only the
retrieval test file is still a placeholder, matching `src/retrieve.py`.

## Notes & findings

- **2026-09-30:** Ingestion involves multiple steps and parameters. Re-ingesting
  the same PDF creates duplicate vectors, so stable chunk IDs or an explicit
  rebuild/update strategy are required.
- **2026-09-30:** "Runs locally" does not mean "has no dependencies." Both
  ingestion and querying require a running Ollama service and locally installed
  models; the application does not yet provide user-friendly checks for either.
- **2026-09-30:** Even when using only a short PDF as input, the LLM consumes
  significant CPU resources.

## What's next

The following are proposed next steps. They will be implemented to the extent
that they support this project's learning goals. Some may be deferred to future
projects.

- [ ] Explore the existing codebase to understand component interactions and
  assess how minor changes affect overall functionality.
- [x] Prevent duplicate ingestion with deterministic chunk IDs or a documented
  rebuild/update workflow.
- [x] Support multiple source documents instead of one fixed PDF.
- [x] Implement an interactive CLI rather than hard-coded sample questions.
- [x] Turn the placeholder tests into unit and integration tests and add a
  test runner.
- [ ] Add a small evaluation set with expected source passages and measure
  retrieval recall before tuning chunk size, overlap, or `k`.
- [x] Move paths, models, chunk settings, and retrieval settings into real
  configuration.
- [ ] Format retrieved context explicitly and return source/page citations.
- [ ] Add clear errors for missing documents, empty indexes, unavailable
  Ollama, missing models, and PDF extraction failures.
