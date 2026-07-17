import pytest

from phishing_ml.agents import workflow


def _fail_if_called(*args, **kwargs):
    raise AssertionError("Unexpected workflow tool call")


def test_workflow_blocks_when_model_is_rejected(
    monkeypatch,
    tmp_path,
):
    calls = {}
    quality_gate_config = tmp_path / "quality_gate.yaml"

    def fake_build_model_status(config_path):
        calls["config_path"] = config_path
        return {"status": "rejected"}

    monkeypatch.setattr(
        workflow,
        "build_model_status",
        fake_build_model_status,
    )
    monkeypatch.setattr(
        workflow,
        "classify_message",
        _fail_if_called,
    )
    monkeypatch.setattr(
        workflow,
        "search_project_knowledge",
        _fail_if_called,
    )

    result = workflow.run_incident_workflow(
        "Suspicious message",
        quality_gate_config=quality_gate_config,
    )

    assert calls == {
        "config_path": str(quality_gate_config),
    }
    assert result["outcome"] == "blocked"
    assert result["reason"] == "model_quality_gate_rejected"
    assert result["model_status"] == {"status": "rejected"}
    assert result["classification"] is None
    assert result["guidance"] is None


def test_workflow_finishes_without_guidance_for_legitimate_message(
    monkeypatch,
    tmp_path,
):
    calls = {}
    artifacts_dir = tmp_path / "artifacts"
    expected_classification = {
        "tool_name": "classify_message",
        "input_text": "Scheduled team meeting",
        "label": 0,
        "class_name": "legitimate",
        "phishing_probability": 0.08,
        "threshold": 0.5,
    }

    monkeypatch.setattr(
        workflow,
        "build_model_status",
        lambda config_path: {"status": "approved"},
    )

    def fake_classify_message(text, threshold, artifacts_dir):
        calls["text"] = text
        calls["threshold"] = threshold
        calls["artifacts_dir"] = artifacts_dir
        return expected_classification

    monkeypatch.setattr(
        workflow,
        "classify_message",
        fake_classify_message,
    )
    monkeypatch.setattr(
        workflow,
        "search_project_knowledge",
        _fail_if_called,
    )

    result = workflow.run_incident_workflow(
        " Scheduled team meeting ",
        artifacts_dir=artifacts_dir,
    )

    assert calls == {
        "text": "Scheduled team meeting",
        "threshold": 0.5,
        "artifacts_dir": str(artifacts_dir),
    }
    assert result["outcome"] == "legitimate"
    assert result["reason"] == "message_classified_as_legitimate"
    assert result["classification"] == expected_classification
    assert result["guidance"] is None


def test_workflow_retrieves_guidance_for_phishing_message(
    monkeypatch,
    tmp_path,
):
    calls = {}
    expected_classification = {
        "tool_name": "classify_message",
        "input_text": "Urgent password reset",
        "label": 1,
        "class_name": "phishing",
        "phishing_probability": 0.94,
        "threshold": 0.6,
    }
    expected_guidance = {
        "tool_name": "search_project_knowledge",
        "query": workflow.INCIDENT_GUIDANCE_QUERY,
        "result_count": 1,
        "results": [
            {
                "citation": (
                    "docs/phishing_incident_response.md#L1-L40"
                ),
            }
        ],
    }

    monkeypatch.setattr(
        workflow,
        "build_model_status",
        lambda config_path: {"status": "approved"},
    )
    monkeypatch.setattr(
        workflow,
        "classify_message",
        lambda text, threshold, artifacts_dir: (
            expected_classification
        ),
    )

    def fake_search_project_knowledge(
        query,
        project_root,
        limit,
    ):
        calls["query"] = query
        calls["project_root"] = project_root
        calls["limit"] = limit
        return expected_guidance

    monkeypatch.setattr(
        workflow,
        "search_project_knowledge",
        fake_search_project_knowledge,
    )

    result = workflow.run_incident_workflow(
        "Urgent password reset",
        threshold=0.6,
        project_root=tmp_path,
    )

    assert calls == {
        "query": workflow.INCIDENT_GUIDANCE_QUERY,
        "project_root": str(tmp_path),
        "limit": workflow.GUIDANCE_RESULT_LIMIT,
    }
    assert result["outcome"] == "phishing"
    assert result["reason"] == "phishing_detected"
    assert result["classification"] == expected_classification
    assert result["guidance"] == expected_guidance


@pytest.mark.parametrize(
    ("message", "threshold", "expected_error"),
    [
        (" ", 0.5, "Message text must not be empty"),
        (
            "Example message",
            -0.01,
            "Threshold must be between 0.0 and 1.0",
        ),
        (
            "Example message",
            1.01,
            "Threshold must be between 0.0 and 1.0",
        ),
    ],
)
def test_workflow_rejects_invalid_input(
    message,
    threshold,
    expected_error,
):
    with pytest.raises(ValueError, match=expected_error):
        workflow.run_incident_workflow(
            message,
            threshold=threshold,
        )
