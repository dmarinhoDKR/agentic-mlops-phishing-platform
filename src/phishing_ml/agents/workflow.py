from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from phishing_ml.agents.mlops_tools import (
    DEFAULT_ARTIFACTS_DIR,
    build_model_status,
    classify_message,
)
from phishing_ml.evaluation.quality_gate import DEFAULT_GATE_CONFIG_PATH
from phishing_ml.rag.search import (
    DEFAULT_PROJECT_ROOT,
    search_project_knowledge,
)


INCIDENT_GUIDANCE_QUERY = (
    "phishing incident response containment credential protection "
    "evidence preservation recovery escalation"
)
GUIDANCE_RESULT_LIMIT = 3


class IncidentWorkflowState(TypedDict):
    message: str
    threshold: float
    artifacts_dir: str
    quality_gate_config: str
    project_root: str
    model_status: NotRequired[dict[str, Any]]
    classification: NotRequired[dict[str, Any]]
    guidance: NotRequired[dict[str, Any]]
    outcome: NotRequired[
        Literal["blocked", "legitimate", "phishing"]
    ]
    response: NotRequired[dict[str, Any]]


def _check_model_status(
    state: IncidentWorkflowState,
) -> dict[str, Any]:
    return {
        "model_status": build_model_status(
            config_path=state["quality_gate_config"],
        )
    }


def _route_model_status(
    state: IncidentWorkflowState,
) -> Literal["classify", "blocked"]:
    model_status = state.get("model_status")

    if model_status is None:
        raise ValueError("Model status is missing")

    if model_status["status"] == "approved":
        return "classify"

    return "blocked"


def _classify_message(
    state: IncidentWorkflowState,
) -> dict[str, Any]:
    return {
        "classification": classify_message(
            text=state["message"],
            threshold=state["threshold"],
            artifacts_dir=state["artifacts_dir"],
        )
    }


def _route_classification(
    state: IncidentWorkflowState,
) -> Literal["retrieve", "finalize"]:
    classification = state.get("classification")

    if classification is None:
        raise ValueError("Classification result is missing")

    if int(classification["label"]) == 1:
        return "retrieve"

    return "finalize"


def _retrieve_guidance(
    state: IncidentWorkflowState,
) -> dict[str, Any]:
    return {
        "guidance": search_project_knowledge(
            query=INCIDENT_GUIDANCE_QUERY,
            project_root=state["project_root"],
            limit=GUIDANCE_RESULT_LIMIT,
        )
    }


def _finalize_incident(
    state: IncidentWorkflowState,
) -> dict[str, Any]:
    model_status = state.get("model_status")
    classification = state.get("classification")
    guidance = state.get("guidance")

    if model_status is None:
        raise ValueError("Model status is missing")

    if model_status["status"] != "approved":
        outcome = "blocked"
        reason = "model_quality_gate_rejected"
    elif classification is None:
        raise ValueError("Classification result is missing")
    elif int(classification["label"]) == 1:
        if guidance is None:
            raise ValueError("Incident guidance is missing")

        outcome = "phishing"
        reason = "phishing_detected"
    else:
        outcome = "legitimate"
        reason = "message_classified_as_legitimate"

    response = {
        "tool_name": "analyze_incident",
        "outcome": outcome,
        "reason": reason,
        "model_status": model_status,
        "classification": classification,
        "guidance": guidance,
    }

    return {
        "outcome": outcome,
        "response": response,
    }


def build_incident_workflow():
    graph = StateGraph(IncidentWorkflowState)

    graph.add_node("check_model_status", _check_model_status)
    graph.add_node("classify_message", _classify_message)
    graph.add_node("retrieve_guidance", _retrieve_guidance)
    graph.add_node("finalize_incident", _finalize_incident)

    graph.add_edge(START, "check_model_status")
    graph.add_conditional_edges(
        "check_model_status",
        _route_model_status,
        {
            "classify": "classify_message",
            "blocked": "finalize_incident",
        },
    )
    graph.add_conditional_edges(
        "classify_message",
        _route_classification,
        {
            "retrieve": "retrieve_guidance",
            "finalize": "finalize_incident",
        },
    )
    graph.add_edge("retrieve_guidance", "finalize_incident")
    graph.add_edge("finalize_incident", END)

    return graph.compile()


INCIDENT_WORKFLOW = build_incident_workflow()


def run_incident_workflow(
    message: str,
    threshold: float = 0.5,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    quality_gate_config: str | Path = DEFAULT_GATE_CONFIG_PATH,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    normalized_message = message.strip()

    if not normalized_message:
        raise ValueError("Message text must not be empty")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")

    final_state = INCIDENT_WORKFLOW.invoke(
        {
            "message": normalized_message,
            "threshold": threshold,
            "artifacts_dir": str(artifacts_dir),
            "quality_gate_config": str(quality_gate_config),
            "project_root": str(project_root),
        }
    )
    response = final_state.get("response")

    if not isinstance(response, dict):
        raise RuntimeError(
            "Incident workflow did not produce a response"
        )

    return response
