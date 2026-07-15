import pytest

from phishing_ml.rag.chunks import KnowledgeChunk
from phishing_ml.rag.retriever import LexicalRetriever


def _chunk(chunk_id: str, content: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source=chunk_id.split("#", maxsplit=1)[0],
        content=content,
        media_type="text/markdown",
        start_line=1,
        end_line=1,
    )


def test_search_ranks_relevant_chunk_first():
    retriever = LexicalRetriever(
        [
            _chunk(
                "docs/quality.md#L1-L1",
                "Model quality gate checks recall metrics.",
            ),
            _chunk(
                "docs/docker.md#L1-L1",
                "Docker container deployment instructions.",
            ),
        ]
    )

    results = retriever.search("quality gate recall")

    assert results[0].chunk.chunk_id == "docs/quality.md#L1-L1"
    assert results[0].score > 0.0


def test_search_uses_chunk_id_to_break_score_ties():
    retriever = LexicalRetriever(
        [
            _chunk("b.md#L1-L1", "shared retrieval text"),
            _chunk("a.md#L1-L1", "shared retrieval text"),
        ]
    )

    results = retriever.search("shared retrieval", limit=1)

    assert [result.chunk.chunk_id for result in results] == [
        "a.md#L1-L1",
    ]


def test_search_returns_empty_list_for_unknown_query():
    retriever = LexicalRetriever(
        [_chunk("README.md#L1-L1", "phishing model metrics")]
    )

    assert retriever.search("quantum astronomy") == []


def test_retriever_requires_at_least_one_chunk():
    with pytest.raises(
        ValueError,
        match="At least one knowledge chunk is required",
    ):
        LexicalRetriever([])


@pytest.mark.parametrize(
    ("query", "limit", "minimum_score", "message"),
    [
        (" ", 5, 0.0, "Search query must not be empty"),
        ("model", 0, 0.0, "limit must be greater than zero"),
        (
            "model",
            5,
            -0.1,
            "minimum_score must be between 0.0 and 1.0",
        ),
        (
            "model",
            5,
            1.1,
            "minimum_score must be between 0.0 and 1.0",
        ),
    ],
)
def test_search_rejects_invalid_parameters(
    query,
    limit,
    minimum_score,
    message,
):
    retriever = LexicalRetriever(
        [_chunk("README.md#L1-L1", "model metrics")]
    )

    with pytest.raises(ValueError, match=message):
        retriever.search(
            query,
            limit=limit,
            minimum_score=minimum_score,
        )
