import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest

from rootwalk.benchmark import summarize_benchmark
from rootwalk.evaluate import evaluate_files


def _write(path: Path, value: object) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def _csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bundle(
    root: Path,
    *,
    run_id: str,
    status: str = "completed_partial",
    findings: list[object] | None = None,
    failure: object = None,
    abstain_reason: str | None = None,
    model_calls: int = 1,
    usage: dict[str, int] | None = None,
    task_id: str = "task-1",
    source_row_id: int = 1,
    manifest_metadata: dict[str, object] | None = None,
    wall_clock_seconds: float | None = None,
    latency_seconds: float | None = None,
    provider_models: list[str] | None = None,
) -> Path:
    root.mkdir()
    task = {"task_id": task_id, "source_row_id": source_row_id}
    report: dict[str, object] = {
        "run_id": run_id,
        "task_id": task_id,
        "status": status,
        "findings": findings or [],
        "failure": failure,
        "abstain_reason": abstain_reason,
        "model_calls": model_calls,
    }
    if usage is not None:
        report["model_usage"] = usage
    if wall_clock_seconds is not None:
        report["wall_clock_seconds"] = wall_clock_seconds
    if latency_seconds is not None:
        report["latency_seconds"] = latency_seconds
    files: dict[str, object] = {
        "task.json": task,
        "topology.json": {"snapshot_id": "topology-1"},
        "context_manifests.jsonl": [],
        "attempts.jsonl": [
            {
                "payload": {
                    "attempt_id": f"model:{index}",
                    "status": "success",
                    "provider_model": provider,
                }
            }
            for index, provider in enumerate(provider_models or ["provider-a"], 1)
        ],
        "decisions.jsonl": [],
        "observations.jsonl": [],
        "report.json": report,
        "investigation_events.jsonl": [
            {"payload": {"event_type": "run_sealed", "status": status}}
        ],
    }
    for name, value in files.items():
        if name.endswith(".jsonl"):
            path = root / name
            path.write_text(
                "".join(
                    json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n"
                    for item in cast(list[object], value)
                ),
                encoding="utf-8",
            )
        else:
            _write(root / name, value)
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
    }
    manifest = {"run_id": run_id, "file_hashes": hashes, **(manifest_metadata or {})}
    _write(root / "manifest.json", manifest)
    return root


def test_all_refusals_and_failures_remain_in_the_denominator(tmp_path: Path) -> None:
    first = _bundle(
        tmp_path / "first",
        run_id="run-1",
        abstain_reason="insufficient evidence",
        model_calls=1,
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    second = _bundle(
        tmp_path / "second",
        run_id="run-2",
        status="runtime_failure",
        failure="model timeout",
        model_calls=2,
    )

    result = summarize_benchmark(
        [second, first],
        run_metadata={
            "run-1": {"condition": "C0", "repeat": 1},
            "run-2": {"condition": "C0", "repeat": 2},
        },
    )

    assert result["run_count"] == 2
    runs = cast(list[dict[str, Any]], result["runs"])
    assert [run["run_id"] for run in runs] == ["run-1", "run-2"]
    assert all(run["refused"] for run in runs)
    assert runs[1]["failed"] is True
    assert runs[1]["failure"] == "model timeout"
    group = cast(list[dict[str, Any]], result["groups"])[0]
    assert group["denominator"] == 2
    assert group["diagnostic_output_rate"] == 0.0
    assert group["refused_count"] == 2
    assert group["refused_rate"] == 1.0
    assert group["failed_count"] == 1
    assert group["failed_rate"] == 0.5
    assert group["variation"]["prompt_tokens"]["missing_count"] == 1


@pytest.mark.parametrize(
    ("status", "failed"),
    [
        ("completed_grounded", False),
        ("completed_partial", False),
        ("no_progress", True),
        ("budget_exhausted", True),
        ("context_limit", True),
        ("runtime_failure", True),
        ("cancelled", True),
    ],
)
def test_terminal_status_drives_execution_failure(
    tmp_path: Path, status: str, failed: bool
) -> None:
    bundle = _bundle(tmp_path / "run", run_id="run-1", status=status)

    run = cast(list[dict[str, Any]], summarize_benchmark([bundle])["runs"])[0]

    assert run["failed"] is failed
    assert run["refused"] is True


def test_global_summary_keeps_failures_across_tasks_and_conditions(
    tmp_path: Path,
) -> None:
    success = _bundle(
        tmp_path / "success",
        run_id="run-1",
        task_id="task-1",
        findings=[{"root_cause": "database"}],
        model_calls=1,
        usage={"prompt_tokens": 10, "completion_tokens": 1},
    )
    failure = _bundle(
        tmp_path / "failure",
        run_id="run-2",
        task_id="task-2",
        status="runtime_failure",
        failure="model timeout",
        model_calls=2,
        usage={"prompt_tokens": 20, "completion_tokens": 2},
    )
    refusal = _bundle(
        tmp_path / "refusal",
        run_id="run-3",
        task_id="task-2",
        abstain_reason="insufficient evidence",
        model_calls=3,
    )

    result = summarize_benchmark(
        [failure, refusal, success],
        run_metadata={
            "run-1": {"condition": "C0", "repeat": 1},
            "run-2": {"condition": "C0", "repeat": 1},
            "run-3": {"condition": "C1", "repeat": 1},
        },
    )

    summary = cast(dict[str, Any], result["summary"])
    assert summary["task_count"] == 2
    assert summary["condition_count"] == 2
    assert summary["denominator"] == summary["run_count"] == 3
    assert summary["status_counts"] == {
        "completed_partial": 2,
        "runtime_failure": 1,
    }
    assert summary["diagnostic_output_count"] == 1
    assert summary["diagnostic_output_rate"] == 1 / 3
    assert summary["refused_count"] == 2
    assert summary["refused_rate"] == 2 / 3
    assert summary["failed_count"] == 1
    assert summary["failed_rate"] == 1 / 3
    assert summary["variation"]["model_calls"]["mean"] == 2.0
    assert summary["variation"]["prompt_tokens"]["missing_count"] == 1
    assert set(summary["variation"]) == {
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_tokens",
        "wall_clock_seconds",
        "latency_seconds",
        "cost",
        "evaluation_score",
    }


def test_provider_model_drift_is_reported_without_fabricating_failures(
    tmp_path: Path,
) -> None:
    first = _bundle(tmp_path / "first", run_id="run-1", provider_models=["provider-a"])
    second = _bundle(
        tmp_path / "second", run_id="run-2", provider_models=["provider-b"]
    )

    result = summarize_benchmark([first, second])

    assert result["provider_models"] == ["provider-a", "provider-b"]
    assert result["provider_model_controlled"] is False
    runs = cast(list[dict[str, Any]], result["runs"])
    assert [run["provider_models"] for run in runs] == [
        ["provider-a"],
        ["provider-b"],
    ]
    assert all(run["failed"] is False for run in runs)


def test_repeat_variation_and_explicit_pricing_are_deterministic(
    tmp_path: Path,
) -> None:
    first = _bundle(
        tmp_path / "first",
        run_id="run-1",
        findings=[{"root_cause": "database"}],
        model_calls=1,
        usage={"prompt_tokens": 1_000, "completion_tokens": 100},
        manifest_metadata={"benchmark": {"condition": "C1", "repeat": 1}},
        wall_clock_seconds=4.0,
        latency_seconds=1.0,
    )
    second = _bundle(
        tmp_path / "second",
        run_id="run-2",
        findings=[{"root_cause": "database"}],
        model_calls=3,
        usage={"prompt_tokens": 3_000, "completion_tokens": 300},
        manifest_metadata={"benchmark": {"condition": "C1", "repeat": 2}},
        wall_clock_seconds=8.0,
        latency_seconds=3.0,
    )
    pricing = {
        "prompt_per_million": 2.0,
        "completion_per_million": 4.0,
        "secret": 999.0,
    }

    result = summarize_benchmark([second, first], pricing=pricing)

    assert result == summarize_benchmark([first, second], pricing=pricing)
    runs = cast(list[dict[str, Any]], result["runs"])
    assert runs[0]["total_tokens"] == 1_100
    assert runs[0]["cache_read_tokens"] is None
    assert runs[0]["cost_basis"] == "cache_tokens_not_reported_assumed_zero"
    assert math.isclose(cast(float, runs[0]["cost"]), 0.0024)
    assert result["pricing"] == {
        "prompt_per_million": 2.0,
        "completion_per_million": 4.0,
        "cache_read_per_million": None,
    }
    assert "secret" not in json.dumps(result)
    group = cast(list[dict[str, Any]], result["groups"])[0]
    assert group["repeats"] == [1, 2]
    assert group["variation"]["model_calls"] == {
        "known_count": 2,
        "missing_count": 0,
        "min": 1.0,
        "max": 3.0,
        "mean": 2.0,
        "pstdev": 1.0,
    }
    assert group["variation"]["wall_clock_seconds"]["mean"] == 6.0
    assert group["variation"]["latency_seconds"]["pstdev"] == 1.0


def test_missing_usage_price_and_run_metadata_stay_unknown_or_null(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "run", run_id="run-1")

    result = summarize_benchmark([bundle])

    run = cast(list[dict[str, Any]], result["runs"])[0]
    assert run["repeat"] == "unknown"
    assert run["condition"] == "unknown"
    assert run["prompt_tokens"] is None
    assert run["completion_tokens"] is None
    assert run["total_tokens"] is None
    assert run["cache_read_tokens"] is None
    assert run["wall_clock_seconds"] is None
    assert run["latency_seconds"] is None
    assert run["cost"] is None
    assert run["cost_basis"] == "pricing_not_provided"
    assert result["pricing"] is None


@pytest.mark.parametrize("cache_field", ["cache_read_input_tokens", "cached_tokens"])
def test_cache_tokens_require_and_use_their_explicit_price(
    tmp_path: Path, cache_field: str
) -> None:
    bundle = _bundle(
        tmp_path / "run",
        run_id="run-1",
        usage={
            "prompt_tokens": 1_000,
            "completion_tokens": 100,
            cache_field: 200,
        },
    )
    ordinary_prices = {
        "prompt_per_million": 2.0,
        "completion_per_million": 4.0,
    }

    missing_cache_price = summarize_benchmark([bundle], pricing=ordinary_prices)
    priced = summarize_benchmark(
        [bundle], pricing={**ordinary_prices, "cache_read_per_million": 1.0}
    )

    missing_run = cast(list[dict[str, Any]], missing_cache_price["runs"])[0]
    assert missing_run["cache_read_tokens"] == 200
    assert missing_run["cost"] is None
    assert missing_run["cost_basis"] == "cache_price_not_provided"
    priced_run = cast(list[dict[str, Any]], priced["runs"])[0]
    assert priced_run["cost_basis"] == "reported_cache_tokens"
    assert math.isclose(cast(float, priced_run["cost"]), 0.0022)


def test_cache_tokens_cannot_exceed_prompt_tokens(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "run",
        run_id="run-1",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cached_tokens": 101,
        },
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        summarize_benchmark([bundle])


def test_independent_evaluator_summary_exposes_only_aggregate_results(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "run", run_id="run-1")
    evaluator = tmp_path / "evaluation.json"
    _write(
        evaluator,
        {
            "evaluator": "fixed-evaluator-v1",
            "score": 0.25,
            "failed_count": 1,
            "hidden_labels": ["must-not-be-copied"],
            "summary": {"strict_accuracy": 0.0, "total": 1},
        },
    )

    result = summarize_benchmark([bundle], evaluator_summaries={"run-1": evaluator})

    evaluation = cast(list[dict[str, Any]], result["runs"])[0]["evaluation"]
    assert evaluation == {
        "evaluator": "fixed-evaluator-v1",
        "score": 0.25,
        "failed_count": 1,
        "metrics": {"strict_accuracy": 0.0, "total": 1},
    }
    group = cast(list[dict[str, Any]], result["groups"])[0]
    assert group["variation"]["evaluation_score"]["mean"] == 0.25
    assert "hidden_labels" not in json.dumps(result)


def test_current_evaluate_files_output_supplies_the_per_run_score(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "run", run_id="run-1", source_row_id=1)
    predictions = tmp_path / "predictions.csv"
    labels = tmp_path / "labels.csv"
    _csv(predictions, ["row_id", "prediction"], [{"row_id": "1", "prediction": "{}"}])
    _csv(
        labels,
        ["scoring_points"],
        [{"scoring_points": "The only predicted root cause component is node-a"}],
    )
    evaluation = evaluate_files(predictions, labels)
    cast(dict[str, object], cast(list[object], evaluation["rows"])[0])["label"] = (
        "SECRET"
    )

    result = summarize_benchmark([bundle], evaluator_summaries={"run-1": evaluation})

    run = cast(list[dict[str, Any]], result["runs"])[0]
    assert run["evaluation_score"] == 0.0
    assert run["evaluation"] == {
        "score": 0.0,
        "evaluator_version": evaluation["evaluator_version"],
        "exporter_version": evaluation["exporter_version"],
        "rows": [{"row_id": "1", "score": 0.0}],
        "metrics": evaluation["summary"],
    }
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        (
            {
                "rows": [{"row_id": "1", "score": 0.0}],
                "summary": {"total": 2, "partial_accuracy": 0.0},
            },
            "summary.total",
        ),
        (
            {
                "rows": [
                    {"row_id": "1", "score": 0.0},
                    {"row_id": "2", "score": 0.0},
                ],
                "summary": {"total": 1, "partial_accuracy": 0.0},
            },
            "exactly one",
        ),
        (
            {
                "rows": [{"row_id": "2", "score": 0.0}],
                "summary": {"total": 1, "partial_accuracy": 0.0},
            },
            "source_row_id",
        ),
        ({"score": 0.0}, "ambiguous"),
        (
            {
                "rows": [{"row_id": "1", "score": 0.0}],
                "summary": {"total": 1, "partial_accuracy": 1.0},
            },
            "disagrees",
        ),
    ],
)
def test_per_run_evaluator_output_must_be_unambiguous(
    tmp_path: Path, evaluation: dict[str, object], message: str
) -> None:
    bundle = _bundle(tmp_path / "run", run_id="run-1", source_row_id=1)

    with pytest.raises(ValueError, match=message):
        summarize_benchmark([bundle], evaluator_summaries={"run-1": evaluation})


def test_bundle_integrity_and_seal_are_required(tmp_path: Path) -> None:
    corrupt = _bundle(tmp_path / "corrupt", run_id="corrupt")
    (corrupt / "report.json").write_text("tampered", encoding="utf-8")
    unsealed = _bundle(tmp_path / "unsealed", run_id="unsealed")
    (unsealed / "investigation_events.jsonl").write_text("", encoding="utf-8")
    manifest_path = unsealed / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    manifest["file_hashes"]["investigation_events.jsonl"] = hashlib.sha256(
        b""
    ).hexdigest()
    _write(manifest_path, manifest)

    with pytest.raises(ValueError, match="hash mismatch"):
        summarize_benchmark([corrupt])
    with pytest.raises(ValueError, match="run_sealed"):
        summarize_benchmark([unsealed])
