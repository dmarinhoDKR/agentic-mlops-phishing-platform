from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


DEFAULT_KNOWLEDGE_SOURCES: tuple[Path, ...] = (
    Path("README.md"),
    Path("pyproject.toml"),
    Path("configs"),
    Path("reports"),
    Path("docs"),
    Path(".github/workflows"),
)

SUPPORTED_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".toml",
        ".yaml",
        ".yml",
    }
)

MEDIA_TYPES = {
    ".json": "application/json",
    ".md": "text/markdown",
    ".toml": "application/toml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


@dataclass(frozen=True)
class KnowledgeDocument:
    source: str
    content: str
    media_type: str


def discover_knowledge_files(
    project_root: str | Path = ".",
    source_paths: Sequence[str | Path] = DEFAULT_KNOWLEDGE_SOURCES,
) -> list[Path]:
    root = Path(project_root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")

    discovered_files: set[Path] = set()

    for source_path in source_paths:
        candidate = (root / source_path).resolve()

        if not candidate.is_relative_to(root):
            continue

        if candidate.is_file():
            _add_supported_file(
                discovered_files,
                candidate,
                root,
            )
            continue

        if candidate.is_dir():
            for child in candidate.rglob("*"):
                _add_supported_file(
                    discovered_files,
                    child,
                    root,
                )

    return sorted(
        discovered_files,
        key=lambda path: path.as_posix(),
    )


def load_knowledge_documents(
    project_root: str | Path = ".",
    source_paths: Sequence[str | Path] = DEFAULT_KNOWLEDGE_SOURCES,
) -> list[KnowledgeDocument]:
    root = Path(project_root).resolve()
    documents = []

    for file_path in discover_knowledge_files(
        project_root=root,
        source_paths=source_paths,
    ):
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            KnowledgeDocument(
                source=file_path.relative_to(root).as_posix(),
                content=content,
                media_type=MEDIA_TYPES[file_path.suffix.lower()],
            )
        )

    return documents


def _add_supported_file(
    discovered_files: set[Path],
    candidate: Path,
    project_root: Path,
) -> None:
    if not candidate.is_file():
        return

    resolved_candidate = candidate.resolve()

    if not resolved_candidate.is_relative_to(project_root):
        return

    if resolved_candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
        return

    discovered_files.add(resolved_candidate)
