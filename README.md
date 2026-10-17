# RAG Sandbox — Learning & Experimentation Project

> Status: While there are still opportunities for improvement, I consider this project complete. I went well beyond the original tutorial, expanding both the scope and complexity of the implementation beyond what I initially anticipated.
>
> Throughout development, I encountered areas where AI-generated code introduced duplication or lacked the polish I would expect in production software. I made a deliberate decision to prioritize the project's primary objective: developing hands-on experience with RAG systems and AI/LLM engineering, rather than spending additional time refining every implementation detail. In a production-focused project, code quality, maintainability, and eliminating unnecessary duplication would be a higher priority.
>
> The result is a significantly expanded learning project that deepened my understanding of RAG architecture, implementation trade-offs, and AI/LLM engineering practices. The experience has provided a strong foundation for my next project, where I intend to apply these lessons with a greater emphasis on production-level quality and engineering standards.

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
- `src/retrieve.py` owns standalone retrieval behavior for querying indexed
  chunks without invoking the LLM.
- `src/generate.py` builds the retrieval and generation chain.
- `src/app.py` is the application entry point, including store readiness checks
  and the interactive question loop.
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
    vectorstore/    # manifest.json is created after successful ingestion
```

Settings come in two kinds:

- **Per-store settings** (models, chunking, retrieval, paths) are taken from
  the first of: a shell variable for one run, such as
  `LLM_MODEL=qwen3:4b python -m src.app`; the store's `store.toml`; and
  `stores/defaults.toml`.
- **Machine-wide settings** (the Ollama address and which store to use) are
  taken from the first of: your shell; `.env`, your personal overrides,
  ignored by Git; and `.env.default`, the committed defaults.

To create personal machine-wide overrides, copy `.env.example` to `.env` and
uncomment the values you want to change. Do not put real secrets in `.env`;
the current app does not need any.

Commands use the `default` store unless you choose another one. Asking
questions still requires readable documents to be ingested first. Manage
stores with `src.stores`:

```bash
python -m src.stores                      # list stores, their models, and what's ingested
python -m src.stores info research        # settings, where each is set, paths, status
python -m src.stores create research --description "ML papers" \
    --embedding-model mxbai-embed-large   # or --copy-from <store>
python -m src.stores use research         # make it the default (saved in .env)
```

`create` writes every ingest and query setting into the new store's
`store.toml`, so later changes to `stores/defaults.toml` can't silently
change it. The runtime entry points take `--store <name>` to choose a store
for one run; without it, they use the default store (`RAG_STORE`):

```bash
python -m src.ingest --store research
python -m src.app --store research
python -m src.util --store research
```

While asking questions in `src.app`, type `/use <name>` to switch to another
store, or `/use` on its own to list them.

Each store answers with the prompt in `stores/<name>/prompt.txt` if it has
one, otherwise `stores/prompt.txt`. A prompt must contain the placeholders
`{context}` (the retrieved chunks) and `{question}`, and no others; write a
literal brace as `{{` or `}}`.

Ingestion records a store's `embedding_model`, `chunk_size`, `chunk_overlap`,
and indexed chunk count in `manifest.json` inside its vector store. If you
change one of the fixed settings afterwards, ingestion refuses to add to the
store and the app refuses to search it with a different embedding model,
explaining what changed. The app also refuses to answer from a store with no
indexed chunks. To rebuild with new settings or source content, delete the
contents of the store's vector store and ingest again.

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
removing a file, delete the contents of the active store's `vectorstore/`
folder and run `src.ingest` again to rebuild.
Source documents and vector-store data are intentionally excluded from Git.
Data stores other than `default` are also local-only by default, including
their `store.toml` and prompt files, unless the Git ignore rules are changed.

## Data And Privacy

Documents live under the active store's `documents/` folder. Chroma writes the
local index under that store's `vectorstore/` folder, including `manifest.json`
after successful ingestion. The app does not write question/answer logs; answers
and retrieved sources are printed to the terminal.

Embedding and chat requests are sent to the configured Ollama server
(`OLLAMA_HOST`, default `http://localhost:11434`). The app does not call hosted
LLM APIs by default. If you point `OLLAMA_HOST` at another machine, document
chunks and questions are sent to that Ollama server.

To clear or rebuild an index, delete the contents of the active store's
`vectorstore/` folder and run `python -m src.ingest` again. To remove local
source documents too, delete them from the store's `documents/` folder.

## Limitations And Troubleshooting

- Supported source files are `.pdf`, `.txt`, and `.md`. Other file types are
  ignored during ingestion.
- Scanned or image-only PDFs do not produce text chunks unless OCR has already
  been run outside this app.
- Ollama must be running, and the configured embedding and chat models must be
  pulled locally. The app reports startup errors with the relevant
  `ollama serve` or `ollama pull <model>` command.
- Empty stores and empty indexes cannot answer questions. Add readable files to
  the active store's `documents/` folder and run `python -m src.ingest`.
- Edited or deleted source files are not automatically reflected in Chroma.
  Rebuild the store's `vectorstore/` after changing source content.
- Changing `embedding_model`, `chunk_size`, or `chunk_overlap` after ingestion
  requires rebuilding the vector store. Query-only settings such as
  `retrieval_k`, `llm_model`, temperature, and context window can be changed
  without re-ingesting.
- Local models can use significant CPU, memory, and disk. If ingestion or
  generation is slow or unstable, try a smaller model, lower batch size, or
  smaller context window.

## Development

Run the automated tests from the project root with:

```bash
pytest
```

They cover loading settings and prompts, the manifest, creating and
selecting stores, saving model choices, the app's question loop and store
switching, and generation's prompt construction and chain wiring, plus the
parts of ingestion that don't need Ollama (finding files, splitting, empty
chunk handling, and chunk IDs). Fakes stand in for Ollama, Chroma, and the
keyboard, and stores load from temporary files, so your real stores aren't
touched. Retrieval is tested separately from generation so retrieval quality
can be evaluated without invoking the LLM.

No formatter or linter command is configured yet. If one is added later,
document the command here and include it in the development workflow.

To add another document type, add a loader to `LOADERS` in `src/ingest.py`, add
any needed package to `requirements.txt`, and cover file discovery/loading with
tests. To use another Ollama model, pull it with `ollama pull <model>` and set
it in a store's `store.toml`, with `python -m src.util`, or for one run through
a shell variable. Changing an embedding model requires rebuilding that store's
vector index.

## Notes & findings

- **2026-09-30:** Ingestion involves multiple steps and parameters. Re-ingesting
  the same PDF creates duplicate vectors, so stable chunk IDs or an explicit
  rebuild/update strategy are required.
- **2026-09-30:** "Runs locally" does not mean "has no dependencies." Both
  ingestion and querying require a running Ollama service and locally installed
  models. The original prototype did not provide user-friendly checks for
  either; this project now checks Ollama/model availability before indexing or
  answering.
- **2026-09-30:** Even when using only a short PDF as input, the LLM consumes
  significant CPU resources.
- **2026-10-02:** Deterministic chunk IDs prevent duplicate ingestion, but they
  also mean edited or deleted source files are not automatically reflected in
  Chroma. A rebuild workflow is still required when source content changes.
- **2026-10-02:** Supporting `.txt` and `.md` files exposed encoding concerns
  that PDFs did not, including the need to pin `chardet<6` for LangChain
  `TextLoader` compatibility.
- **2026-10-06:** Better terminal output is not just cosmetic. Rich panels,
  section headers, and clearer status messages make local AI failures easier to
  understand while experimenting.
- **2026-10-07:** Per-store configuration became necessary once multiple
  document collections and model choices existed. Machine-wide settings and
  store-specific settings need different override rules so one store does not
  accidentally affect another.
- **2026-10-07:** Vector stores are tightly coupled to their embedding model and
  chunking settings. Recording a manifest helps catch incompatible changes
  before Chroma or retrieval fails in less obvious ways.
- **2026-10-07:** Ollama model names need normalization because `model` and
  `model:latest` can refer to the same installed model. Without that, checks can
  report false setting mismatches.
- **2026-10-14:** PDF loading can fail in several ways: missing folders, no
  supported files, corrupt or password-protected PDFs, and scanned PDFs with no
  extractable text all need different user guidance.
- **2026-10-14:** Generated citations are not enough on their own. Returning and
  printing the retrieved source chunks gives a deterministic way to audit what
  context the answer used.
- **2026-10-15:** Separating retrieval from generation makes retrieval quality
  testable without invoking the LLM, which is a useful step toward recall
  evaluation and tuning.
- **2026-10-15:** Fakes and temporary stores make most of a local RAG app
  testable without touching Ollama, Chroma persistence, keyboard input, or real
  documents. Integration and evaluation tests are still a separate layer.

## What's next

The following are proposed next steps. They will be implemented to the extent
that they support this project's learning goals. Some may be deferred to future
projects.

- [x] Explore the existing codebase to understand component interactions and
  assess how minor changes affect overall functionality.
- [x] Prevent duplicate ingestion with deterministic chunk IDs or a documented
  rebuild/update workflow.
- [x] Support multiple source documents instead of one fixed PDF.
- [x] Implement an interactive CLI rather than hard-coded sample questions.
- [x] Turn the placeholder tests into unit and integration tests and add a
  test runner.
- [x] Move paths, models, chunk settings, and retrieval settings into real
  configuration.
- [x] Format retrieved context explicitly and return source/page citations.
- [x] Add clear errors for missing documents, empty indexes, unavailable
  Ollama, missing models, and PDF extraction failures.
