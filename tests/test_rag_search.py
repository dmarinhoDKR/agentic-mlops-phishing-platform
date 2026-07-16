from pathlib import Path

import pytest

from phishing_ml.rag.search import search_project_knowledge


def _write_readme(project_root: Path) -> str:
    content = "Model quality gate validates recall metrics."
    (project_root / "README.md").write_text(
        content,
        encoding="utf-8",
    )

    return content


def test_search_project_knowledge_returns_structured_results(
    tmp_path: Path,
):
    content = _write_readme(tmp_path)

    result = search_project_knowledge(
        query=" quality gate recall ",
        project_root=tmp_path,
    )

    assert result["tool_name"] == "search_project_knowledge"
    assert result["query"] == "quality gate recall"
    assert result["project_root"] == str(tmp_path.resolve())
    assert result["result_count"] == 1

    first_result = result["results"][0]

    assert first_result["rank"] == 1
    assert first_result["citation"] == "README.md#L1-L1"
    assert first_result["source"] == "README.md"
    assert first_result["content"] == content
    assert first_result["media_type"] == "text/markdown"
    assert first_result["start_line"] == 1
    assert first_result["end_line"] == 1
    assert first_result["score"] > 0.0


def test_search_project_knowledge_returns_no_unknown_matches(
    tmp_path: Path,
):
    _write_readme(tmp_path)

    result = search_project_knowledge(
        query="quantum astronomy",
        project_root=tmp_path,
    )

    assert result["result_count"] == 0
    assert result["results"] == []


def test_search_project_knowledge_rejects_empty_query(
    tmp_path: Path,
):
    with pytest.raises(
        ValueError,
        match="Search query must not be empty",
    ):
        search_project_knowledge(
            query=" ",
            project_root=tmp_path,
        )


def test_search_project_knowledge_requires_documents(
    tmp_path: Path,
):
    with pytest.raises(
        ValueError,
        match="At least one knowledge chunk is required",
    ):
        search_project_knowledge(
            query="model metrics",
            project_root=tmp_path,
        )
