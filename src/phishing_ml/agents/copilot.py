import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from phishing_ml.agents.mlops_tools import (
    DEFAULT_ARTIFACTS_DIR,
    build_model_status,
    classify_message,
    format_model_status,
)
from phishing_ml.evaluation.quality_gate import DEFAULT_GATE_CONFIG_PATH
from phishing_ml.rag.retriever import DEFAULT_RESULT_LIMIT
from phishing_ml.rag.search import (
    DEFAULT_PROJECT_ROOT,
    search_project_knowledge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect model quality, classify suspicious messages, "
            "and search project knowledge."
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show the current model quality status.",
    )
    status_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_GATE_CONFIG_PATH,
        help="Path to the model quality gate configuration.",
    )
    status_parser.set_defaults(handler=_run_status)

    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify a message as phishing or legitimate.",
    )
    classify_parser.add_argument(
        "text",
        help="Message text to classify.",
    )
    classify_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for phishing classification.",
    )
    classify_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing the model and vectorizer artifacts.",
    )
    classify_parser.set_defaults(handler=_run_classify)

    search_parser = subparsers.add_parser(
        "search",
        help="Search the local project knowledge base.",
    )
    search_parser.add_argument(
        "query",
        help="Question or search terms for the knowledge base.",
    )
    search_parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory containing the project knowledge.",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RESULT_LIMIT,
        help="Maximum number of search results.",
    )
    search_parser.add_argument(
        "--minimum-score",
        type=float,
        default=0.0,
        help="Minimum cosine similarity score.",
    )
    search_parser.set_defaults(handler=_run_search)

    return parser


def _run_status(args: argparse.Namespace) -> int:
    status = build_model_status(config_path=args.config)
    print(format_model_status(status))

    return 0


def _run_classify(args: argparse.Namespace) -> int:
    result = classify_message(
        text=args.text,
        threshold=args.threshold,
        artifacts_dir=args.artifacts_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


def _run_search(args: argparse.Namespace) -> int:
    result = search_project_knowledge(
        query=args.query,
        project_root=args.project_root,
        limit=args.limit,
        minimum_score=args.minimum_score,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
