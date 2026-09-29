# localRAG

A local retrieval-augmented generation (RAG) app: index your own documents into
a local vector store and ask questions about them using a locally hosted LLM.

## Project structure

```
LOCALRAG/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
│
├── data/
│   ├── documents/          # Source documents
│   └── vectorstore/        # Local vector DB / embeddings
│
├── src/
│   ├── __init__.py
│   ├── ingest.py           # Load, chunk, embed, store documents
│   ├── retrieve.py         # Retrieve relevant chunks
│   ├── generate.py         # Send context + question to LLM
│   └── app.py              # Main application / CLI
│
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieve.py
│   └── test_generate.py
│
└── .venv/                  # Local virtual environment (not committed)
```
