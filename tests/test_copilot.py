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


def test_main_runs_analyze_command(
    monkeypatch,
    capsys,
    tmp_path,
):
    calls = {}
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "quality_gate.yaml"
    expected_result = {
        "tool_name": "analyze_incident",
        "outcome": "phishing",
        "reason": "phishing_detected",
    }

    def fake_run_incident_workflow(
        message,
        threshold,
        artifacts_dir,
        quality_gate_config,
        project_root,
    ):
        calls["message"] = message
        calls["threshold"] = threshold
        calls["artifacts_dir"] = artifacts_dir
        calls["quality_gate_config"] = quality_gate_config
        calls["project_root"] = project_root

        return expected_result

    monkeypatch.setattr(
        copilot,
        "run_incident_workflow",
        fake_run_incident_workflow,
    )

    exit_code = copilot.main(
        [
            "analyze",
            "Suspicious message",
            "--threshold",
            "0.7",
            "--artifacts-dir",
            str(artifacts_dir),
            "--config",
            str(config_path),
            "--project-root",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == {
        "message": "Suspicious message",
        "threshold": 0.7,
        "artifacts_dir": artifacts_dir,
        "quality_gate_config": config_path,
        "project_root": tmp_path,
    }
    assert json.loads(captured.out) == expected_result
    assert captured.err == ""


def test_main_runs_analyze_summary_command(
    monkeypatch,
    capsys,
):
    expected_result = {
        "tool_name": "analyze_incident",
        "outcome": "phishing",
        "reason": "phishing_detected",
        "model_status": {
            "status": "approved",
            "failed_metrics": [],
            "metrics": {
                "accuracy": 0.91,
                "precision": 0.90,
                "recall": 0.89,
                "f1": 0.88,
            },
        },
        "classification": {
            "class_name": "phishing",
            "phishing_probability": 0.92,
        },
        "guidance": {
            "results": [
                {
                    "citation": (
                        "docs/phishing_incident_response.md#L1-L40"
                    )
                },
                {
                    "citation": (
                        "docs/phishing_incident_response.md#L36-L70"
                    )
                },
            ]
        },
    }

    def fake_run_incident_workflow(**kwargs):
        return expected_result

    monkeypatch.setattr(
        copilot,
        "run_incident_workflow",
        fake_run_incident_workflow,
    )

    exit_code = copilot.main(
        [
            "analyze",
            "Suspicious message",
            "--output",
            "summary",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == (
        "Workflow: LangGraph incident analysis\n"
        "Incident outcome: PHISHING\n"
        "Reason: phishing_detected\n"
        "Model status: APPROVED\n"
        "Quality metrics: accuracy=0.9100, precision=0.9000, "
        "recall=0.8900, f1=0.8800\n"
        "Classification: phishing\n"
        "Phishing probability: 0.9200\n"
        "Guidance sources (2):\n"
        "  1. docs/phishing_incident_response.md#L1-L40\n"
        "  2. docs/phishing_incident_response.md#L36-L70\n"
        "Automation boundary: privileged actions require explicit "
        "human approval.\n"
    )
    assert captured.err == ""


def test_format_incident_analysis_describes_blocked_inference():
    summary = copilot.format_incident_analysis(
        {
            "outcome": "blocked",
            "reason": "model_quality_gate_rejected",
            "model_status": {
                "status": "rejected",
                "failed_metrics": ["recall"],
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.80,
                    "recall": 0.70,
                    "f1": 0.75,
                },
            },
            "classification": None,
            "guidance": None,
        }
    )

    assert "Incident outcome: BLOCKED" in summary
    assert "Failed quality checks: recall" in summary
    assert summary.endswith(
        "Inference blocked by the model quality gate."
    )


def test_format_incident_analysis_describes_legitimate_message():
    summary = copilot.format_incident_analysis(
        {
            "outcome": "legitimate",
            "reason": "message_classified_as_legitimate",
            "model_status": {
                "status": "approved",
                "failed_metrics": [],
                "metrics": {
                    "accuracy": 0.91,
                    "precision": 0.90,
                    "recall": 0.89,
                    "f1": 0.88,
                },
            },
            "classification": {
                "class_name": "legitimate",
                "phishing_probability": 0.08,
            },
            "guidance": None,
        }
    )

    assert "Incident outcome: LEGITIMATE" in summary
    assert "Classification: legitimate" in summary
    assert summary.endswith(
        "No incident-response guidance was required."
    )
