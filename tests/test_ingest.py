# test_ingest.py
#
# Tests for src/ingest.py: finding, splitting, and identifying document
# chunks. Embedding and storage need a running Ollama server and aren't
# covered here.
import pytest
from langchain_core.documents import Document

from src.ingest import (
    assign_chunk_ids,
    find_document_files,
    require_usable_chunks,
    split_documents,
)


def test_find_document_files_returns_supported_files_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.MD").write_text("a")
    (tmp_path / "image.png").write_bytes(b"")

    files = find_document_files(tmp_path)

    assert files == [tmp_path / "b.txt", tmp_path / "nested" / "a.MD"]


def test_split_documents_respects_chunk_size_and_keeps_metadata():
    document = Document(page_content="word " * 100, metadata={"source": "doc.txt"})

    chunks = split_documents([document], chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 50 for chunk in chunks)
    assert all(chunk.metadata["source"] == "doc.txt" for chunk in chunks)


def test_require_usable_chunks_rejects_empty_text(tmp_path):
    chunks = [Document(page_content="   ", metadata={"source": "scan.pdf"})]

    with pytest.raises(RuntimeError, match="No readable text chunks"):
        require_usable_chunks(chunks, tmp_path)


def test_assign_chunk_ids_counts_per_source_and_page():
    chunks = [
        Document(page_content="", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="", metadata={"source": "b.txt"}),
    ]

    assign_chunk_ids(chunks)

    assert [c.metadata["id"] for c in chunks] == [
        "a.pdf:0:0", "a.pdf:0:1", "a.pdf:1:0", "b.txt:0:0",
    ]
