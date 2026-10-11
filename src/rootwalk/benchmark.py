"""Deterministic D9 summaries for sealed, label-free RunBundles."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .diff import diff_run_bundles

_UNKNOWN = "unknown"
_PRICE_KEYS = ("prompt_per_million", "completion_per_million")
_CACHE_KEYS = ("cache_read_input_tokens", "cached_tokens")
_VARIATION_FIELDS = (
    "model_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_tokens",
    "wall_clock_seconds",
    "latency_seconds",
    "cost",
    "evaluation_score",
)


def summarize_benchmark(
    run_bundles: Sequence[str | Path],
    *,
    run_metadata: Mapping[str, Mapping[str, object]] | None = None,
    evaluator_summaries: Mapping[str, str | Path | Mapping[str, object]] | None = None,
    pricing: Mapping[str, int | float] | None = None,
) -> dict[str, object]:
    """Summarize every run and repeat group without reading evaluation labels."""
    prices = _prices(pricing)
    runs = [
        _load_run(Path(bundle), run_metadata, evaluator_summaries, prices)
        for bundle in run_bundles
    ]
    if len({item["run_id"] for item in runs}) != len(runs):
        raise ValueError("benchmark run_id values must be unique")
    runs.sort(
        key=lambda item: (
            str(item["task_id"]),
            str(item["condition"]),
            str(item["repeat"]),
            str(item["run_id"]),
        )
    )
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        groups[(str(run["task_id"]), str(run["condition"]))].append(run)
    provider_models = sorted(
        {
            str(provider)
            for run in runs
            for provider in cast(list[object], run["provider_models"])
        }
    )
    return {
        "run_count": len(runs),
        "provider_models": provider_models,
        "provider_model_controlled": len(provider_models) == 1,
        "pricing": (
            None
            if prices is None
            else {
                "prompt_per_million": prices[0],
                "completion_per_million": prices[1],
                "cache_read_per_million": prices[2],
            }
        ),
        "summary": {
            "task_count": len({item["task_id"] for item in runs}),
            "condition_count": len({item["condition"] for item in runs}),
            **_aggregate(runs),
        },
        "runs": runs,
        "groups": [
            _group(task_id, condition, items)
            for (task_id, condition), items in sorted(groups.items())
        ],
    }


def _load_run(
    root: Path,
    explicit_metadata: Mapping[str, Mapping[str, object]] | None,
    evaluator_summaries: Mapping[str, str | Path | Mapping[str, object]] | None,
    prices: tuple[float, float, float | None] | None,
) -> dict[str, object]:
    diff_run_bundles(root, root)  # Reuse the RunBundle hash and file-set validator.
    manifest = _object(root / "manifest.json")
    task = _object(root / "task.json")
    report = _object(root / "report.json")
    run_id = _string(report.get("run_id") or manifest.get("run_id"), "run_id")
    task_id = _string(report.get("task_id") or task.get("task_id"), "task_id")
    if report.get("task_id") not in {None, task.get("task_id")}:
        raise ValueError(f"{run_id} report task_id does not match task.json")
    _require_sealed(root, run_id, report.get("status"))
    metadata = _metadata(explicit_metadata, root, run_id)
    condition = _run_meta("condition", metadata, manifest)
    repeat = _run_meta("repeat", metadata, manifest)
    status = _string(report.get("status"), "status")
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        raise ValueError(f"{run_id} report findings must be a list")
    failure = report.get("failure")
    failed = status not in {"completed_grounded", "completed_partial"}
    abstain_reason = report.get("abstain_reason")
    refused = not findings
    usage = report.get("model_usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise ValueError(f"{run_id} model_usage must be an object")
    usage = cast(dict[str, object], usage)
    prompt_tokens = _optional_number(usage.get("prompt_tokens"), "prompt_tokens")
    completion_tokens = _optional_number(
        usage.get("completion_tokens"), "completion_tokens"
    )
    total_tokens = _optional_number(usage.get("total_tokens"), "total_tokens")
    cache_read_tokens, cache_reported = _cache_usage(usage)
    if (
        cache_read_tokens is not None
        and prompt_tokens is not None
        and cache_read_tokens > prompt_tokens
    ):
        raise ValueError(f"{run_id} cache_read_tokens cannot exceed prompt_tokens")
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens
    model_calls = _number(report.get("model_calls"), "model_calls")
    wall_clock = _optional_number(
        report.get("wall_clock_seconds"), "wall_clock_seconds"
    )
    latency = _optional_number(report.get("latency_seconds"), "latency_seconds")
    source_row_id = task.get("source_row_id")
    evaluation = _evaluation(_lookup(evaluator_summaries, root, run_id), source_row_id)
    evaluation_score = (
        _optional_number(evaluation.get("score"), "evaluation score")
        if evaluation is not None
        else None
    )
    cost, cost_basis = _cost(
        prompt_tokens,
        completion_tokens,
        cache_read_tokens,
        cache_reported,
        prices,
    )
    provider_models = sorted(
        {
            _string(item.get("provider_model"), "provider_model")
            for item in (
                _jsonl(root / "attempts.jsonl")
                if (root / "attempts.jsonl").is_file()
                else []
            )
            if item.get("status") == "success"
            and isinstance(item.get("attempt_id"), str)
            and cast(str, item["attempt_id"]).startswith("model:")
            and item.get("provider_model") is not None
        }
    )
    return {
        "run_id": run_id,
        "bundle": str(root),
        "task_id": task_id,
        "source_row_id": source_row_id,
        "repeat": repeat,
        "condition": condition,
        "status": status,
        "refused": refused,
        "refusal_reason": (
            abstain_reason
            if abstain_reason is not None
            else failure
            if failure is not None
            else status
            if refused
            else None
        ),
        "failed": failed,
        "failure": failure,
        "provider_models": provider_models,
        "model_calls": model_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read_tokens,
        "wall_clock_seconds": wall_clock,
        "latency_seconds": latency,
        "cost": cost,
        "cost_basis": cost_basis,
        "evaluation_score": evaluation_score,
        "evaluation": evaluation,
    }


def _group(
    task_id: str, condition: str, runs: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "source_row_ids": sorted(
            {
                item["source_row_id"]
                for item in runs
                if isinstance(item["source_row_id"], int)
                and not isinstance(item["source_row_id"], bool)
            }
        ),
        "condition": condition,
        "repeats": [item["repeat"] for item in runs],
        **_aggregate(runs),
    }


def _aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    denominator = len(runs)
    refused = sum(item["refused"] is True for item in runs)
    failed = sum(item["failed"] is True for item in runs)
    variation = {
        name: _variation([item[name] for item in runs]) for name in _VARIATION_FIELDS
    }
    return {
        "denominator": denominator,
        "run_count": denominator,
        "status_counts": dict(
            sorted(Counter(str(item["status"]) for item in runs).items())
        ),
        "diagnostic_output_count": denominator - refused,
        "diagnostic_output_rate": (
            (denominator - refused) / denominator if denominator else None
        ),
        "refused_count": refused,
        "refused_rate": refused / denominator if denominator else None,
        "failed_count": failed,
        "failed_rate": failed / denominator if denominator else None,
        "variation": variation,
    }


def _variation(values: list[object]) -> dict[str, int | float | None]:
    known = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not known:
        return {
            "known_count": 0,
            "missing_count": len(values),
            "min": None,
            "max": None,
            "mean": None,
            "pstdev": None,
        }
    return {
        "known_count": len(known),
        "missing_count": len(values) - len(known),
        "min": min(known),
        "max": max(known),
        "mean": statistics.fmean(known),
        "pstdev": statistics.pstdev(known),
    }


def _run_meta(
    name: str, explicit: Mapping[str, object], manifest: Mapping[str, object]
) -> object:
    values: list[object] = [explicit.get(name), manifest.get(name)]
    for section_name in ("benchmark", "experiment"):
        section = manifest.get(section_name)
        if isinstance(section, dict):
            mapping = cast(dict[str, object], section)
            values.append(mapping.get(name))
            if name == "condition":
                values.append(mapping.get("condition_id"))
    value = next((item for item in values if item is not None), _UNKNOWN)
    if name == "condition":
        if value == _UNKNOWN:
            return value
        return _string(value, name)
    if value == _UNKNOWN:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("repeat must be an integer or non-empty string")
    if isinstance(value, str) and not value.strip():
        raise ValueError("repeat must be an integer or non-empty string")
    return value


def _metadata(
    values: Mapping[str, Mapping[str, object]] | None,
    root: Path,
    run_id: str,
) -> Mapping[str, object]:
    return _lookup(values, root, run_id) or {}


def _lookup(values: Mapping[str, Any] | None, root: Path, run_id: str) -> Any | None:
    if values is None:
        return None
    for key in (run_id, str(root), root.name):
        if key in values:
            return values[key]
    return None


def _evaluation(
    value: str | Path | Mapping[str, object] | None,
    source_row_id: object,
) -> dict[str, object] | None:
    if value is None:
        return None
    source = _object(Path(value)) if isinstance(value, (str, Path)) else dict(value)
    summary = source.get("summary")
    if summary is not None and not isinstance(summary, dict):
        raise ValueError("evaluator summary must be an object")
    metrics = {
        str(name): item
        for name, item in cast(dict[object, object], summary or {}).items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }
    if "total" in metrics and metrics["total"] != 1:
        raise ValueError("one RunBundle requires evaluator summary.total == 1")
    rows = source.get("rows")
    row_score: int | float | None = None
    safe_rows: list[dict[str, object]] | None = None
    if isinstance(rows, list):
        row_values = cast(list[object], rows)
        if len(row_values) != 1 or not isinstance(row_values[0], dict):
            raise ValueError("one RunBundle requires exactly one evaluator row")
        row = cast(dict[str, object], row_values[0])
        row_score = _score(row.get("score"), "evaluator row score")
        row_id = row.get("row_id")
        if row_id is not None and (
            isinstance(row_id, bool) or not isinstance(row_id, (int, str))
        ):
            raise ValueError("evaluator row_id must be a string or integer")
        if (
            source_row_id is not None
            and row_id is not None
            and str(row_id) != str(source_row_id)
        ):
            raise ValueError("evaluator row_id does not match task source_row_id")
        if metrics.get("total") != 1:
            raise ValueError("one RunBundle requires evaluator summary.total == 1")
        partial_accuracy = _score(
            metrics.get("partial_accuracy"), "evaluator partial_accuracy"
        )
        if partial_accuracy != row_score:
            raise ValueError("evaluator row score disagrees with partial_accuracy")
        safe_row: dict[str, object] = {"score": row_score}
        if row_id is not None:
            safe_row["row_id"] = row_id
        safe_rows = [safe_row]
    elif rows is not None and (
        isinstance(rows, bool) or not isinstance(rows, (int, float)) or rows != 1
    ):
        raise ValueError("legacy evaluator rows must equal one")

    legacy_score = (
        _score(source["score"], "legacy evaluator score") if "score" in source else None
    )
    if row_score is not None and legacy_score is not None and row_score != legacy_score:
        raise ValueError("legacy evaluator score disagrees with row score")
    score = row_score if row_score is not None else legacy_score
    if score is None:
        raise ValueError("evaluator output has no unambiguous per-run score")
    if row_score is None and rows != 1 and metrics.get("total") != 1:
        raise ValueError("legacy evaluator score is ambiguous")

    result: dict[str, object] = {"score": score}
    for name in (
        "evaluator",
        "evaluator_version",
        "exporter_version",
        "failed_count",
        "passed_count",
    ):
        item = source.get(name)
        if isinstance(item, (str, int, float)) and not isinstance(item, bool):
            result[name] = item
    if safe_rows is not None:
        result["rows"] = safe_rows
    elif rows == 1:
        result["rows"] = rows
    if metrics:
        result["metrics"] = metrics
    return result


def _score(value: object, name: str) -> int | float:
    score = _number(value, name)
    if score > 1:
        raise ValueError(f"{name} must be at most one")
    return score


def _require_sealed(root: Path, run_id: str, status: object) -> None:
    events = _jsonl(root / "investigation_events.jsonl")
    sealed = [
        item
        for item in events
        if item.get("event_type") == "run_sealed" and item.get("status") == status
    ]
    if not sealed:
        raise ValueError(f"{run_id} RunBundle has no matching run_sealed event")


def _prices(
    pricing: Mapping[str, int | float] | None,
) -> tuple[float, float, float | None] | None:
    if pricing is None:
        return None
    missing = [key for key in _PRICE_KEYS if key not in pricing]
    if missing:
        raise ValueError(f"pricing is missing: {', '.join(missing)}")
    prompt, completion = (float(_number(pricing[key], key)) for key in _PRICE_KEYS)
    cache = (
        float(_number(pricing["cache_read_per_million"], "cache_read_per_million"))
        if "cache_read_per_million" in pricing
        else None
    )
    return prompt, completion, cache


def _cost(
    prompt_tokens: int | float | None,
    completion_tokens: int | float | None,
    cache_read_tokens: int | float | None,
    cache_reported: bool,
    prices: tuple[float, float, float | None] | None,
) -> tuple[float | None, str]:
    if prices is None:
        return None, "pricing_not_provided"
    if prompt_tokens is None or completion_tokens is None:
        return None, "token_usage_incomplete"
    if cache_reported and cache_read_tokens is None:
        return None, "cache_usage_incomplete"
    if cache_reported and prices[2] is None:
        return None, "cache_price_not_provided"
    cache = cache_read_tokens or 0
    cache_price = prices[2] or 0
    cost = (
        (prompt_tokens - cache) * prices[0]
        + cache * cache_price
        + completion_tokens * prices[1]
    ) / 1_000_000
    basis = (
        "reported_cache_tokens"
        if cache_reported
        else "cache_tokens_not_reported_assumed_zero"
    )
    return cost, basis


def _cache_usage(usage: Mapping[str, object]) -> tuple[int | float | None, bool]:
    present = [key for key in _CACHE_KEYS if key in usage]
    if not present:
        return None, False
    values = [_optional_number(usage[key], key) for key in present]
    if any(value != values[0] for value in values[1:]):
        raise ValueError("cache token fields disagree")
    return values[0], True


def _number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


def _optional_number(value: object, name: str) -> int | float | None:
    return None if value is None else _number(value, name)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"benchmark input must be a JSON object: {path}")
    return cast(dict[str, object], value)


def _jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark input: {path}") from exc
    result: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"benchmark JSONL record must be an object: {path}")
        record = cast(dict[str, object], value)
        payload = record.get("payload", record)
        if not isinstance(payload, dict):
            raise ValueError(f"benchmark JSONL payload must be an object: {path}")
        result.append(cast(dict[str, object], payload))
    return result
