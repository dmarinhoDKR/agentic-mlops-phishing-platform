from collections.abc import Sequence
from dataclasses import dataclass

from phishing_ml.rag.documents import KnowledgeDocument


DEFAULT_MAX_LINES = 40
DEFAULT_OVERLAP_LINES = 5


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    content: str
    media_type: str
    start_line: int
    end_line: int


def chunk_document(
    document: KnowledgeDocument,
    max_lines: int = DEFAULT_MAX_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[KnowledgeChunk]:
    _validate_chunking_parameters(max_lines, overlap_lines)

    return _chunk_document(
        document=document,
        max_lines=max_lines,
        overlap_lines=overlap_lines,
    )


def chunk_documents(
    documents: Sequence[KnowledgeDocument],
    max_lines: int = DEFAULT_MAX_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[KnowledgeChunk]:
    _validate_chunking_parameters(max_lines, overlap_lines)

    chunks: list[KnowledgeChunk] = []

    for document in documents:
        chunks.extend(
            _chunk_document(
                document=document,
                max_lines=max_lines,
                overlap_lines=overlap_lines,
            )
        )

    return chunks


def _chunk_document(
    document: KnowledgeDocument,
    max_lines: int,
    overlap_lines: int,
) -> list[KnowledgeChunk]:
    lines = document.content.splitlines()

    if not lines:
        return []

    chunks: list[KnowledgeChunk] = []
    start_index = 0

    while start_index < len(lines):
        end_index = min(start_index + max_lines, len(lines))
        content = "\n".join(lines[start_index:end_index])

        if content.strip():
            start_line = start_index + 1
            end_line = end_index

            chunks.append(
                KnowledgeChunk(
                    chunk_id=(
                        f"{document.source}"
                        f"#L{start_line}-L{end_line}"
                    ),
                    source=document.source,
                    content=content,
                    media_type=document.media_type,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        if end_index == len(lines):
            break

        start_index = end_index - overlap_lines

    return chunks


def _validate_chunking_parameters(
    max_lines: int,
    overlap_lines: int,
) -> None:
    if max_lines <= 0:
        raise ValueError("max_lines must be greater than zero")

    if overlap_lines < 0:
        raise ValueError("overlap_lines must not be negative")

    if overlap_lines >= max_lines:
        raise ValueError(
            "overlap_lines must be smaller than max_lines"
        )
