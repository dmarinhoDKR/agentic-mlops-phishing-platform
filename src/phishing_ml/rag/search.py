from pathlib import Path
from typing import Any

from phishing_ml.rag.chunks import chunk_documents
from phishing_ml.rag.documents import load_knowledge_documents
from phishing_ml.rag.retriever import (
    DEFAULT_RESULT_LIMIT,
    LexicalRetriever,
    RetrievalResult,
)


DEFAULT_PROJECT_ROOT = Path(".")


def search_project_knowledge(
    query: str,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    limit: int = DEFAULT_RESULT_LIMIT,
    minimum_score: float = 0.0,
) -> dict[str, Any]:
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("Search query must not be empty")

    resolved_project_root = Path(project_root).resolve()
    documents = load_knowledge_documents(
        project_root=resolved_project_root
    )
    chunks = chunk_documents(documents)

    retriever = LexicalRetriever(chunks)
    results = retriever.search(
        query=normalized_query,
        limit=limit,
        minimum_score=minimum_score,
    )

    return {
        "tool_name": "search_project_knowledge",
        "query": normalized_query,
        "project_root": str(resolved_project_root),
        "result_count": len(results),
        "results": [
            _serialize_result(result, rank)
            for rank, result in enumerate(results, start=1)
        ],
    }


def _serialize_result(
    result: RetrievalResult,
    rank: int,
) -> dict[str, Any]:
    chunk = result.chunk

    return {
        "rank": rank,
        "citation": chunk.chunk_id,
        "source": chunk.source,
        "content": chunk.content,
        "media_type": chunk.media_type,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "score": result.score,
    }
