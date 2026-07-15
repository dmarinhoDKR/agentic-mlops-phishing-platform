from pathlib import Path

import pytest

from phishing_ml.rag.documents import (
    KnowledgeDocument,
    discover_knowledge_files,
    load_knowledge_documents,
)


def test_discover_knowledge_files_filters_and_sorts(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True)

    (project_root / "README.md").write_text(
        "Project overview",
        encoding="utf-8",
    )
    (configs_dir / "baseline.yaml").write_text(
        "epochs: 10",
        encoding="utf-8",
    )
    (configs_dir / "ignored.txt").write_text(
        "Outside content",
        encoding="utf-8",
    )
    (tmp_path / "outside.md").write_text(
        "Outside content",
        encoding="utf-8",
    )

    discovered_files = discover_knowledge_files(
        project_root=project_root,
        source_paths=(
            Path("configs"),
            Path("README.md"),
            Path("../outside.md"),
            Path("README.md"),
        ),
    )

    relative_paths = [
        path.relative_to(project_root).as_posix()
        for path in discovered_files
    ]

    assert relative_paths == [
        "README.md",
        "configs/baseline.yaml",
    ]


def test_load_knowledge_documents_maps_content_and_media_types(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    configs_dir = project_root / "configs"
    reports_dir = project_root / "reports"
    docs_dir = project_root / "docs"

    configs_dir.mkdir(parents=True)
    reports_dir.mkdir()
    docs_dir.mkdir()

    (project_root / "README.md").write_text(
        " Project overview ",
        encoding="utf-8",
    )
    (configs_dir / "baseline.yml").write_text(
        "epochs: 10",
        encoding="utf-8",
    )
    (reports_dir / "metrics.json").write_text(
        '{"accuracy": 0.9}',
        encoding="utf-8",
    )
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "example"',
        encoding="utf-8",
    )
    (docs_dir / "empty.md").write_text(
        "   ",
        encoding="utf-8",
    )

    documents = load_knowledge_documents(
        project_root=project_root,
        source_paths=(
            Path("README.md"),
            Path("pyproject.toml"),
            Path("configs"),
            Path("reports"),
            Path("docs"),
        ),
    )
    documents_by_source = {
        document.source: document
        for document in documents
    }

    assert documents_by_source == {
        "README.md": KnowledgeDocument(
            source="README.md",
            content="Project overview",
            media_type="text/markdown",
        ),
        "configs/baseline.yml": KnowledgeDocument(
            source="configs/baseline.yml",
            content="epochs: 10",
            media_type="application/yaml",
        ),
        "pyproject.toml": KnowledgeDocument(
            source="pyproject.toml",
            content='[project]\nname = "example"',
            media_type="application/toml",
        ),
        "reports/metrics.json": KnowledgeDocument(
            source="reports/metrics.json",
            content='{"accuracy": 0.9}',
            media_type="application/json",
        ),
    }


def test_discover_knowledge_files_requires_existing_root(
    tmp_path: Path,
):
    missing_root = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="Project root not found",
    ):
        discover_knowledge_files(missing_root)


def test_discover_knowledge_files_requires_directory_root(
    tmp_path: Path,
):
    file_root = tmp_path / "project.txt"
    file_root.write_text("content", encoding="utf-8")

    with pytest.raises(
        NotADirectoryError,
        match="Project root is not a directory",
    ):
        discover_knowledge_files(file_root)
