import json

from phishing_ml.agents import copilot


def test_main_runs_status_command(
    monkeypatch,
    capsys,
    tmp_path,
):
    calls = {}
    expected_status = {"status": "approved"}
    config_path = tmp_path / "quality_gate.yaml"

    def fake_build_model_status(config_path):
        calls["config_path"] = config_path
        return expected_status

    def fake_format_model_status(status):
        calls["status"] = status
        return "Model status: APPROVED"

    monkeypatch.setattr(
        copilot,
        "build_model_status",
        fake_build_model_status,
    )
    monkeypatch.setattr(
        copilot,
        "format_model_status",
        fake_format_model_status,
    )

    exit_code = copilot.main(
        ["status", "--config", str(config_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == {
        "config_path": config_path,
        "status": expected_status,
    }
    assert captured.out == "Model status: APPROVED\n"
    assert captured.err == ""


def test_main_runs_classify_command(
    monkeypatch,
    capsys,
    tmp_path,
):
    calls = {}
    expected_result = {
        "tool_name": "classify_message",
        "input_text": "Suspicious message",
        "label": 1,
        "class_name": "phishing",
        "phishing_probability": 0.92,
        "threshold": 0.7,
    }

    def fake_classify_message(text, threshold, artifacts_dir):
        calls["text"] = text
        calls["threshold"] = threshold
        calls["artifacts_dir"] = artifacts_dir

        return expected_result

    monkeypatch.setattr(
        copilot,
        "classify_message",
        fake_classify_message,
    )

    exit_code = copilot.main(
        [
            "classify",
            "Suspicious message",
            "--threshold",
            "0.7",
            "--artifacts-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == {
        "text": "Suspicious message",
        "threshold": 0.7,
        "artifacts_dir": tmp_path,
    }
    assert json.loads(captured.out) == expected_result
    assert captured.err == ""


def test_main_returns_one_for_tool_error(
    monkeypatch,
    capsys,
):
    def fake_classify_message(text, threshold, artifacts_dir):
        raise ValueError("Invalid test message")

    monkeypatch.setattr(
        copilot,
        "classify_message",
        fake_classify_message,
    )

    exit_code = copilot.main(
        ["classify", "Example message"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "error: Invalid test message\n"


def test_main_runs_search_command(
    monkeypatch,
    capsys,
    tmp_path,
):
    calls = {}
    expected_result = {
        "tool_name": "search_project_knowledge",
        "query": "model quality gate",
        "result_count": 0,
        "results": [],
    }

    def fake_search_project_knowledge(
        query,
        project_root,
        limit,
        minimum_score,
    ):
        calls["query"] = query
        calls["project_root"] = project_root
        calls["limit"] = limit
        calls["minimum_score"] = minimum_score

        return expected_result

    monkeypatch.setattr(
        copilot,
        "search_project_knowledge",
        fake_search_project_knowledge,
    )

    exit_code = copilot.main(
        [
            "search",
            "model quality gate",
            "--project-root",
            str(tmp_path),
            "--limit",
            "2",
            "--minimum-score",
            "0.1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == {
        "query": "model quality gate",
        "project_root": tmp_path,
        "limit": 2,
        "minimum_score": 0.1,
    }
    assert json.loads(captured.out) == expected_result
    assert captured.err == ""
