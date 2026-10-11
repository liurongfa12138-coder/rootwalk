"""Isolated, label-safe wrapper for the pinned OpenRCA evaluator."""

from __future__ import annotations

import csv
import itertools
import json
import re
from datetime import datetime
from pathlib import Path
from typing import cast

EXPORTER_VERSION = "rootwalk-openrca-csv-v5"
EVALUATOR_VERSION = (
    "microsoft/OpenRCA/main/evaluate.py@c1bd4af7f635171a1c31cdd567c07d698dff6abc"
)

_PREDICTION_PATTERN = re.compile(
    r"{\s*"
    r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
    r'(?:"root cause component":\s*"(.*?)")?,?\s*'
    r'(?:"root cause reason":\s*"(.*?)")?\s*}'
)
_COMPONENT_PATTERN = re.compile(
    r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
)
_REASON_PATTERN = re.compile(
    r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
)
_TIME_PATTERN = re.compile(
    r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes "
    r"\(i.e., <=1min\) of ([^\n]+)"
)


def evaluate_files(predictions: str | Path, labels: str | Path) -> dict[str, object]:
    """Score an exported CSV against an isolated OpenRCA ``query.csv``."""

    prediction_rows = _read_csv(Path(predictions), {"row_id", "prediction"})
    labels_path = Path(labels)
    alignment_path: Path | None = None
    if labels_path.is_dir():
        alignment_path = labels_path / "alignment.json"
        labels_path /= "query.csv"
    label_rows = _read_csv(labels_path, {"scoring_points"})

    if len(prediction_rows) != len(label_rows):
        raise ValueError("prediction and label row counts differ")

    row_ids = [row["row_id"] for row in prediction_rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("prediction row_id values must be unique")
    expected_ids = _expected_row_ids(label_rows, alignment_path)
    if row_ids != expected_ids:
        raise ValueError("prediction row_id values are not in label order")

    rows = [
        {
            "row_id": row_id,
            "score": _score(
                prediction.get("prediction") or "", label["scoring_points"]
            ),
        }
        for row_id, prediction, label in zip(
            row_ids, prediction_rows, label_rows, strict=True
        )
    ]
    total = len(rows)
    score_sum = sum(float(row["score"]) for row in rows)
    strict_correct = sum(row["score"] == 1.0 for row in rows)
    return {
        "exporter_version": EXPORTER_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "rows": rows,
        "summary": {
            "total": total,
            "strict_correct": strict_correct,
            "strict_accuracy": strict_correct / total if total else 0.0,
            "partial_score": score_sum,
            "partial_accuracy": score_sum / total if total else 0.0,
        },
    }


def _expected_row_ids(
    label_rows: list[dict[str, str]], alignment_path: Path | None
) -> list[str]:
    if alignment_path is not None and alignment_path.exists():
        if not alignment_path.is_file():
            raise ValueError(f"alignment is not a file: {alignment_path}")
        alignment = _read_alignment(alignment_path, len(label_rows))
        expected = [str(item["prediction_row_id"]) for item in alignment]
        if label_rows and "row_id" in label_rows[0]:
            label_ids = [
                label_rows[item["selected_query_row"]]["row_id"] for item in alignment
            ]
            if label_ids != expected:
                raise ValueError("label row_id values do not match alignment")
        return expected
    if label_rows and "row_id" in label_rows[0]:
        return [row["row_id"] for row in label_rows]
    return [str(index) for index in range(1, len(label_rows) + 1)]


def _read_alignment(path: Path, label_count: int) -> list[dict[str, int]]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read alignment: {path}") from exc
    if isinstance(raw, dict) and "rows" in raw:
        container = cast(dict[object, object], raw)
        if set(container) != {"rows"}:
            raise ValueError("alignment object must contain only rows")
        rows_value: object = container["rows"]
    else:
        rows_value = cast(object, [raw])
    if not isinstance(rows_value, list):
        raise ValueError("alignment row count must match label row count")
    rows = cast(list[object], rows_value)
    if len(rows) != label_count:
        raise ValueError("alignment row count must match label row count")

    required = {"source_row_id", "selected_query_row", "prediction_row_id"}
    result: list[dict[str, int]] = []
    for raw_item in rows:
        if not isinstance(raw_item, dict):
            raise ValueError("alignment rows must contain exactly the required fields")
        item = cast(dict[object, object], raw_item)
        if set(item) != required:
            raise ValueError("alignment rows must contain exactly the required fields")
        row: dict[str, int] = {}
        for key in required:
            value = item[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("alignment row values must be integers")
            row[key] = value
        if any(value < 0 for value in row.values()):
            raise ValueError("alignment row values must be non-negative")
        if row["source_row_id"] != row["prediction_row_id"]:
            raise ValueError("prediction_row_id must equal source_row_id")
        result.append(row)

    selected = [item["selected_query_row"] for item in result]
    if len(selected) != len(set(selected)):
        raise ValueError("selected_query_row values must be unique")
    if sorted(selected) != list(range(label_count)):
        raise ValueError("selected_query_row values must be contiguous from zero")
    result.sort(key=lambda item: item["selected_query_row"])
    prediction_ids = [item["prediction_row_id"] for item in result]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("alignment prediction_row_id values must be unique")
    return result


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError(f"CSV has no header: {path}")
        if len(fields) != len(set(fields)):
            raise ValueError(f"CSV has duplicate columns: {path}")
        missing = sorted(required_fields - set(fields))
        if missing:
            raise ValueError(f"CSV is missing required fields: {', '.join(missing)}")
        rows: list[dict[str, str]] = []
        for number, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"CSV row {number} has extra values: {path}")
            rows.append({key: value or "" for key, value in raw.items()})
        return rows


def _score(prediction: str, scoring_points: str) -> float:
    predicted = [
        {
            "root cause occurrence datetime": match[0],
            "root cause component": match[1],
            "root cause reason": match[2],
        }
        for match in _PREDICTION_PATTERN.findall(prediction)
    ]
    components = _COMPONENT_PATTERN.findall(scoring_points)
    reasons = _REASON_PATTERN.findall(scoring_points)
    times = _TIME_PATTERN.findall(scoring_points)
    fault_count = max(len(components), len(reasons), len(times))
    criteria_count = len(components) + len(reasons) + len(times)
    if criteria_count == 0:
        raise ValueError("label row has no supported scoring points")
    if len(predicted) != fault_count:
        return 0.0

    best = 0
    for permutation in itertools.permutations(predicted):
        score = 0
        for index in range(fault_count):
            if len(components) == fault_count:
                score += permutation[index]["root cause component"] == components[index]
            if len(reasons) == fault_count:
                score += permutation[index]["root cause reason"] == reasons[index]
            if len(times) == fault_count:
                score += _within_minute(
                    times[index],
                    permutation[index]["root cause occurrence datetime"],
                )
        best = max(best, score)
    return round(best / criteria_count, 2)


def _within_minute(expected: str, actual: str) -> bool:
    try:
        expected_time = datetime.strptime(expected, "%Y-%m-%d %H:%M:%S")
        actual_time = datetime.strptime(actual, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return abs(expected_time - actual_time).total_seconds() <= 60
