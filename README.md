# RAG Sandbox — Learning & Experimentation Project

> Status: Active learning sandbox (not a portfolio piece — that will live in a
> separate flagship repository when it exists).

## Why this exists

I'm a 20-year SDET transitioning into AI engineering. This repository is where
I'm getting hands-on with retrieval-augmented generation fundamentals before
building a production-ready, evaluated AI application, while applying the
testing discipline I have developed throughout my career.

The current prototype loads a local PDF, splits it into overlapping chunks,
embeds those chunks into a persistent vector store, and answers two sample
questions with a locally hosted model.

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
- **`qwen3:8b`** — the current local chat model, run deterministically with
  temperature `0`.
- **ChromaDB** — persists embeddings locally between ingestion and querying.
- **PyPDF** — extracts text and page metadata from the source PDF.

`sentence-transformers`, `unstructured[pdf]`, and `tiktoken` are installed as
experimentation options but are not part of the current execution path.

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

Place a PDF at:

```text
data/documents/document.pdf
```

With the Ollama server running, ingest the document and then run the sample
questions:

```bash
python -m src.ingest
python -m src.app
```

`src.app` currently asks two hard-coded questions. Change them in
`src/app.py` to match the contents of your document.

Re-running ingestion adds another copy of every chunk. To rebuild cleanly,
remove the contents of `data/vectorstore/` before running `src.ingest` again.
Source documents and vector-store data are intentionally excluded from Git.

Automated tests are not runnable yet: the files under `tests/` are placeholders
and no test runner is declared as a dependency.

## Notes & findings

- **2026-09-30:** Recursive character splitting at 1,000 characters with 200
  characters of overlap is a sensible baseline, but it is still only a
  hypothesis. I need retrieval evaluation before claiming that these values
  improve recall or answer quality.
- **2026-09-30:** Ingestion is additive. Chroma assigns new IDs on every run,
  so ingesting the same PDF twice creates duplicate vectors. Stable chunk IDs
  or an explicit rebuild/update strategy are required.
- **2026-09-30:** The planned `retrieve.py` module is still a stub. Retrieval
  currently happens inside the generation chain and returns the three most
  similar chunks.
- **2026-09-30:** The retrieved LangChain `Document` objects are inserted into
  the prompt without custom formatting. The answer does not expose page
  numbers, similarity scores, or citations, even though page metadata is
  retained during ingestion.
- **2026-09-30:** Configuration is only partially designed. `.env.example`
  describes configurable paths and models, but the code still hard-codes the
  PDF name, vector-store path, chunk settings, and model names.
- **2026-09-30:** "Runs locally" does not mean "has no dependencies." Both
  ingestion and querying require a running Ollama service and already-pulled
  models; the application does not yet provide friendly checks for either.
- **2026-09-30:** A test-shaped directory is not a test suite. The next useful
  step is to test deterministic boundaries—loading, chunking, retrieval, and
  prompt assembly—while mocking model calls, then add a smaller set of local
  integration and retrieval-quality evaluations.

## What's next

- [ ] Turn the placeholder tests into unit and integration tests and add a test
  runner.
- [ ] Add a small evaluation set with expected source passages and measure
  retrieval recall before tuning chunk size, overlap, or `k`.
- [ ] Prevent duplicate ingestion with deterministic chunk IDs or a documented
  rebuild/update workflow.
- [ ] Move paths, models, chunk settings, and retrieval settings into real
  configuration.
- [ ] Support multiple source documents instead of one fixed PDF.
- [ ] Implement an interactive CLI rather than hard-coded sample questions.
- [ ] Format retrieved context explicitly and return source/page citations.
- [ ] Add clear errors for missing documents, empty indexes, unavailable
  Ollama, missing models, and PDF extraction failures.
- [ ] Decide whether the experimental dependencies should be integrated or
  removed.
- [ ] Add formatting, linting, and CI once the first real tests exist.
