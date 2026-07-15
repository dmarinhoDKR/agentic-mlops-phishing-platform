from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from phishing_ml.rag.chunks import KnowledgeChunk


DEFAULT_RESULT_LIMIT = 5


@dataclass(frozen=True)
class RetrievalResult:
    chunk: KnowledgeChunk
    score: float


class LexicalRetriever:
    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "At least one knowledge chunk is required"
            )

        self._chunks = tuple(chunks)
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
        )

        searchable_texts = [
            f"{chunk.source}\n{chunk.content}"
            for chunk in self._chunks
        ]
        self._matrix = self._vectorizer.fit_transform(
            searchable_texts
        )

    def search(
        self,
        query: str,
        limit: int = DEFAULT_RESULT_LIMIT,
        minimum_score: float = 0.0,
    ) -> list[RetrievalResult]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query must not be empty")

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0.0 and 1.0"
            )

        query_vector = self._vectorizer.transform(
            [normalized_query]
        )
        scores = cosine_similarity(
            query_vector,
            self._matrix,
        ).ravel()

        results = [
            RetrievalResult(
                chunk=self._chunks[index],
                score=float(score),
            )
            for index, score in enumerate(scores)
            if score > 0.0 and score >= minimum_score
        ]

        return sorted(
            results,
            key=lambda result: (
                -result.score,
                result.chunk.chunk_id,
            ),
        )[:limit]
