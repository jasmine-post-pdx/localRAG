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

The current prototype loads a local PDF, splits it into overlapping chunks,
embeds those chunks into a persistent vector store, and answers two sample
questions with a locally hosted model.

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
- `tests/` mirrors those responsibilities and will hold the corresponding
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

`sentence-transformers`, `unstructured[pdf]`, and `tiktoken` are installed as
experimental options but are not part of the current execution path.

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

Pull the two local models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Place your source files (`.pdf`, `.txt`, or `.md`) anywhere under:

```text
data/documents/
```

With the Ollama server running, ingest the documents and then run the sample
questions:

```bash
python -m src.ingest
python -m src.app
```

`src.app` currently asks two hard-coded questions. Change them in
`src/app.py` to match the contents of your document.

Each chunk is stored under a deterministic ID (`<source>:<page>:<chunk_index>`),
so re-running ingestion only embeds chunks from files that are new since the
last run. Chunks already in the store are not updated, so after editing or
removing a file, delete the contents of `data/vectorstore/` and run `src.ingest`
again to rebuild.
Source documents and vector-store data are intentionally excluded from Git.

Automated tests are not runnable yet: the files under `tests/` are placeholders
and no test runner is declared as a dependency.

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
- [ ] Implement an interactive CLI rather than hard-coded sample questions.
- [ ] Turn the placeholder tests into unit and integration tests and add a test
  runner.
- [ ] Add a small evaluation set with expected source passages and measure
  retrieval recall before tuning chunk size, overlap, or `k`.
- [ ] Move paths, models, chunk settings, and retrieval settings into real
  configuration.
- [ ] Format retrieved context explicitly and return source/page citations.
- [ ] Add clear errors for missing documents, empty indexes, unavailable
  Ollama, missing models, and PDF extraction failures.
- [ ] Decide whether the experimental dependencies should be integrated or
  removed.
