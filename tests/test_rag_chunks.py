import pytest

from phishing_ml.rag.chunks import (
    KnowledgeChunk,
    chunk_document,
    chunk_documents,
)
from phishing_ml.rag.documents import KnowledgeDocument


def _document(source: str, content: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        source=source,
        content=content,
        media_type="text/markdown",
    )


def test_chunk_document_creates_overlapping_chunks():
    document = _document(
        "docs/runbook.md",
        "line 1\nline 2\n line 3\nline 4\nline 5",
    )

    chunks = chunk_document(
        document,
        max_lines=3,
        overlap_lines=1,
    )

    assert chunks == [
        KnowledgeChunk(
            chunk_id="docs/runbook.md#L1-L3",
            source="docs/runbook.md",
            content="line 1\nline 2\n line 3",
            media_type="text/markdown",
            start_line=1,
            end_line=3,
        ),
        KnowledgeChunk(
            chunk_id="docs/runbook.md#L3-L5",
            source="docs/runbook.md",
            content=" line 3\nline 4\nline 5",
            media_type="text/markdown",
            start_line=3,
            end_line=5,
        ),
    ]


def test_chunk_documents_preserves_document_order():
    documents = [
        _document("README.md", "one\ntwo\nthree"),
        _document("docs/runbook.md", "four"),
    ]

    chunks = chunk_documents(
        documents,
        max_lines=2,
        overlap_lines=0,
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "README.md#L1-L2",
        "README.md#L3-L3",
        "docs/runbook.md#L1-L1",
    ]


def test_chunk_document_returns_empty_list_for_empty_content():
    assert chunk_document(_document("docs/empty.md", "")) == []


@pytest.mark.parametrize(
    ("max_lines", "overlap_lines", "message"),
    [
        (0, 0, "max_lines must be greater than zero"),
        (3, -1, "overlap_lines must not be negative"),
        (3, 3, "overlap_lines must be smaller than max_lines"),
    ],
)
def test_chunk_document_rejects_invalid_parameters(
    max_lines,
    overlap_lines,
    message,
):
    with pytest.raises(ValueError, match=message):
        chunk_document(
            _document("README.md", "content"),
            max_lines=max_lines,
            overlap_lines=overlap_lines,
        )
