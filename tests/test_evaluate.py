import csv
import hashlib
import json
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from rootwalk.evaluate import EVALUATOR_VERSION, EXPORTER_VERSION, evaluate_files

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_bank_suite as suite  # noqa: E402
from scripts.prepare_openrca_eval import main as prepare_main  # noqa: E402
from scripts.prepare_openrca_eval import prepare_eval_package  # noqa: E402

SCORING_POINTS = (
    "The only predicted root cause component is SyntheticNode\n"
    "The only predicted root cause reason is SyntheticReason"
)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def labels(path: Path, count: int) -> Path:
    path.mkdir()
    query = path / "query.csv"
    write_csv(
        query,
        ["instruction", "task_index", "scoring_points"],
        [
            {
                "instruction": f"synthetic task {index}",
                "task_index": "task_6",
                "scoring_points": SCORING_POINTS,
            }
            for index in range(1, count + 1)
        ],
    )
    return path


def eval_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Any]]:
    row_ids = (2, 4, 9)
    instructions = {row_id: f"public instruction {row_id}" for row_id in row_ids}
    query = tmp_path / "source" / "query.csv"
    query.parent.mkdir()
    write_csv(
        query,
        ["task_index", "instruction", "scoring_points"],
        [
            {
                "task_index": "task_6" if index in row_ids else "task_1",
                "instruction": instructions.get(index, f"unused {index}"),
                "scoring_points": f"SECRET_LABEL_{index}",
            }
            for index in range(10)
        ],
    )
    spec = cast(
        dict[str, Any],
        json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "benchmarks/openrca-bank-d9-v1.json"
            ).read_text()
        ),
    )
    spec["cases"] = spec["cases"][:3]
    for case in spec["cases"]:
        row_id = cast(int, case["row_id"])
        case["instruction_sha256"] = hashlib.sha256(
            instructions[row_id].encode()
        ).hexdigest()
    projection = [
        {
            "instruction": instructions[row_id],
            "row_id": row_id,
            "task_index": "task_6",
        }
        for row_id in row_ids
    ]
    selection_hash = hashlib.sha256(
        (json.dumps(projection, separators=(",", ":"), sort_keys=True) + "\n").encode()
    ).hexdigest()
    spec["selection_sha256"] = selection_hash
    plans = [
        suite.CasePlan(
            row_id=cast(int, case["row_id"]),
            task_index="task_6",
            date=cast(str, case["date"]),
            reference_start=datetime.fromisoformat(
                cast(dict[str, str], case["reference_range"])["start"]
            ),
            incident_start=datetime.fromisoformat(
                cast(dict[str, str], case["incident_range"])["start"]
            ),
            incident_end=datetime.fromisoformat(
                cast(dict[str, str], case["incident_range"])["end"]
            ),
            allowed_start=datetime.fromisoformat(
                cast(dict[str, str], case["allowed_query_range"])["start"]
            ),
            allowed_end=datetime.fromisoformat(
                cast(dict[str, str], case["allowed_query_range"])["end"]
            ),
            source_dates=(cast(str, case["date"]),),
            instruction_sha256=cast(str, case["instruction_sha256"]),
        )
        for case in spec["cases"]
    ]
    case_plan_hash = suite.case_plan_sha256(plans)
    monkeypatch.setattr(suite, "DEFAULT_ROW_IDS", row_ids)
    monkeypatch.setattr(suite, "SELECTION_SHA256", selection_hash)
    monkeypatch.setattr(suite, "CASE_PLAN_SHA256", case_plan_hash)
    spec_path = tmp_path / "suite.json"
    spec_path.write_text(json.dumps(spec))
    return query, spec_path, spec


def test_prepare_eval_package_selects_non_contiguous_rows_privately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query, spec_path, _spec = eval_inputs(tmp_path, monkeypatch)
    output = tmp_path / "private-eval"

    result = prepare_eval_package(query, spec_path, output)

    with (output / "query.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == ["task_index", "instruction", "scoring_points"]
    alignment = json.loads((output / "alignment.json").read_text())
    assert [row["instruction"] for row in rows] == [
        "public instruction 2",
        "public instruction 4",
        "public instruction 9",
    ]
    assert [row["scoring_points"] for row in rows] == [
        "SECRET_LABEL_2",
        "SECRET_LABEL_4",
        "SECRET_LABEL_9",
    ]
    assert alignment == {
        "rows": [
            {
                "source_row_id": row_id,
                "selected_query_row": selected,
                "prediction_row_id": row_id,
            }
            for selected, row_id in enumerate((2, 4, 9))
        ]
    }
    assert result["rows"] == 3
    assert (
        result["query_sha256"]
        == hashlib.sha256((output / "query.csv").read_bytes()).hexdigest()
    )
    assert (
        result["alignment_sha256"]
        == hashlib.sha256((output / "alignment.json").read_bytes()).hexdigest()
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE((output / name).stat().st_mode) == 0o600
        for name in ("query.csv", "alignment.json")
    )
    assert prepare_eval_package(query, spec_path, output) == result

    (output / "query.csv").write_text("different")
    with pytest.raises(FileExistsError, match="different content"):
        prepare_eval_package(query, spec_path, output)


def test_prepare_eval_rejects_scoring_only_tamper_for_mirror_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query, spec_path, spec = eval_inputs(tmp_path, monkeypatch)
    original = query.read_bytes()
    spec["source_mirror"] = {}
    spec["source_files"] = [
        {
            "path": "Bank/query.csv",
            "bytes": len(original),
            "sha256": hashlib.sha256(original).hexdigest(),
        }
    ]
    spec_path.write_text(json.dumps(spec))

    with query.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[2]["scoring_points"] = "SECRET_EVILL_2"
    write_csv(query, ["task_index", "instruction", "scoring_points"], rows)

    with pytest.raises(ValueError, match="query.csv SHA-256"):
        prepare_eval_package(query, spec_path, tmp_path / "private-eval")


def test_prepare_eval_stdout_has_only_counts_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    query, spec_path, _spec = eval_inputs(tmp_path, monkeypatch)

    assert (
        prepare_main(
            [
                "--query",
                str(query),
                "--suite",
                str(spec_path),
                "--output",
                str(tmp_path / "private-eval"),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "SECRET_LABEL" not in output
    assert set(json.loads(output)) == {
        "rows",
        "selection_sha256",
        "case_plan_sha256",
        "query_sha256",
        "alignment_sha256",
    }


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("instruction", "instruction SHA-256"),
        ("selection_hash", "selection SHA-256"),
        ("case_order", "frozen order"),
        ("case_plan", "case plan SHA-256"),
    ],
)
def test_prepare_eval_rejects_hash_and_order_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    query, spec_path, spec = eval_inputs(tmp_path, monkeypatch)
    if tamper == "instruction":
        with query.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[2]["instruction"] += " changed"
        write_csv(query, ["task_index", "instruction", "scoring_points"], rows)
    elif tamper == "selection_hash":
        spec["selection_sha256"] = "0" * 64
        spec_path.write_text(json.dumps(spec))
    elif tamper == "case_order":
        cast(list[object], spec["cases"]).reverse()
        spec_path.write_text(json.dumps(spec))
    else:
        case = cast(list[dict[str, Any]], spec["cases"])[0]
        for range_name in (
            "reference_range",
            "incident_range",
            "allowed_query_range",
        ):
            interval = cast(dict[str, Any], case[range_name])
            for boundary in ("start", "end"):
                interval[boundary] = (
                    datetime.fromisoformat(cast(str, interval[boundary]))
                    + timedelta(minutes=1)
                ).isoformat()
        spec_path.write_text(json.dumps(spec))

    with pytest.raises(ValueError, match=message):
        prepare_eval_package(query, spec_path, tmp_path / "private-eval")


def test_prepare_eval_rejects_header_record_and_worktree_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query, spec_path, _spec = eval_inputs(tmp_path, monkeypatch)
    with query.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    write_csv(query, ["instruction", "task_index", "scoring_points"], rows)
    with pytest.raises(ValueError, match="header"):
        prepare_eval_package(query, spec_path, tmp_path / "private-eval")

    write_csv(query, ["task_index", "instruction", "scoring_points"], rows)
    record = query.with_name("record.csv")
    record.write_bytes(query.read_bytes())
    with pytest.raises(ValueError, match="named query.csv"):
        prepare_eval_package(record, spec_path, tmp_path / "private-eval")

    worktree_output = Path(__file__).resolve().parents[1] / ".private-eval-test"
    assert not worktree_output.exists()
    with pytest.raises(ValueError, match="outside the RootWalk Git worktree"):
        prepare_eval_package(query, spec_path, worktree_output)


def test_full_score_and_label_safe_summary(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions.csv"
    prediction = {
        "1": {
            "root cause component": "SyntheticNode",
            "root cause reason": "SyntheticReason",
        }
    }
    write_csv(
        prediction_file,
        ["row_id", "prediction", "run_id"],
        [{"row_id": "1", "prediction": json.dumps(prediction), "run_id": "run-1"}],
    )

    label_root = labels(tmp_path / "labels", 1)
    (label_root / "alignment.json").write_text(
        json.dumps(
            {
                "source_row_id": 1,
                "selected_query_row": 0,
                "prediction_row_id": 1,
            }
        )
    )

    result = evaluate_files(prediction_file, label_root)

    assert result == {
        "exporter_version": EXPORTER_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "rows": [{"row_id": "1", "score": 1.0}],
        "summary": {
            "total": 1,
            "strict_correct": 1,
            "strict_accuracy": 1.0,
            "partial_score": 1.0,
            "partial_accuracy": 1.0,
        },
    }
    assert "SyntheticNode" not in repr(result)
    assert "synthetic task" not in repr(result)


def test_official_regex_failures_stay_in_the_denominator(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions.csv"
    reversed_fields = {
        "1": {
            "root cause reason": "SyntheticReason",
            "root cause component": "SyntheticNode",
        }
    }
    extra_candidate = {
        "1": {
            "root cause component": "SyntheticNode",
            "root cause reason": "SyntheticReason",
        },
        "2": {
            "root cause component": "OtherNode",
            "root cause reason": "OtherReason",
        },
    }
    write_csv(
        prediction_file,
        ["row_id", "prediction"],
        [
            {"row_id": "1", "prediction": json.dumps(reversed_fields)},
            {"row_id": "2", "prediction": json.dumps(extra_candidate)},
            {"row_id": "3", "prediction": "{}"},
        ],
    )

    label_file = labels(tmp_path / "labels", 3) / "query.csv"
    result = evaluate_files(prediction_file, label_file)

    assert result["rows"] == [
        {"row_id": "1", "score": 0.0},
        {"row_id": "2", "score": 0.0},
        {"row_id": "3", "score": 0.0},
    ]
    assert result["summary"] == {
        "total": 3,
        "strict_correct": 0,
        "strict_accuracy": 0.0,
        "partial_score": 0.0,
        "partial_accuracy": 0.0,
    }


def test_non_contiguous_alignment_scores_in_selected_query_order(
    tmp_path: Path,
) -> None:
    prediction_file = tmp_path / "predictions.csv"
    label_root = labels(tmp_path / "labels", 3)
    write_csv(
        label_root / "query.csv",
        ["scoring_points"],
        [
            {
                "scoring_points": SCORING_POINTS.replace(
                    "SyntheticNode", f"SyntheticNode{row_id}"
                )
            }
            for row_id in (2, 4, 9)
        ],
    )
    (label_root / "alignment.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "source_row_id": source,
                        "selected_query_row": selected,
                        "prediction_row_id": source,
                    }
                    for selected, source in ((2, 9), (0, 2), (1, 4))
                ]
            }
        )
    )
    write_csv(
        prediction_file,
        ["row_id", "prediction"],
        [
            {
                "row_id": str(row_id),
                "prediction": json.dumps(
                    {
                        "1": {
                            "root cause component": f"SyntheticNode{row_id}",
                            "root cause reason": "SyntheticReason",
                        }
                    }
                ),
            }
            for row_id in (2, 4, 9)
        ],
    )

    result = evaluate_files(prediction_file, label_root)

    assert result["rows"] == [
        {"row_id": "2", "score": 1.0},
        {"row_id": "4", "score": 1.0},
        {"row_id": "9", "score": 1.0},
    ]


@pytest.mark.parametrize(
    "row_ids, message",
    [
        ((4, 2, 9), "label order"),
        ((2, 2, 9), "unique"),
    ],
)
def test_non_contiguous_prediction_rows_reject_wrong_order_and_duplicates(
    tmp_path: Path, row_ids: tuple[int, int, int], message: str
) -> None:
    prediction_file = tmp_path / "predictions.csv"
    label_root = labels(tmp_path / "labels", 3)
    (label_root / "alignment.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "source_row_id": source,
                        "selected_query_row": selected,
                        "prediction_row_id": source,
                    }
                    for selected, source in enumerate((2, 4, 9))
                ]
            }
        )
    )
    write_csv(
        prediction_file,
        ["row_id", "prediction"],
        [{"row_id": str(row_id), "prediction": "{}"} for row_id in row_ids],
    )

    with pytest.raises(ValueError, match=message):
        evaluate_files(prediction_file, label_root)


def test_alignment_rejects_wrong_count_and_types(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions.csv"
    write_csv(
        prediction_file,
        ["row_id", "prediction"],
        [{"row_id": str(row_id), "prediction": "{}"} for row_id in (2, 4, 9)],
    )
    label_root = labels(tmp_path / "labels", 3)
    alignment = label_root / "alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "source_row_id": 2,
                        "selected_query_row": 0,
                        "prediction_row_id": 2,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="row count"):
        evaluate_files(prediction_file, label_root)

    alignment.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "source_row_id": source,
                        "selected_query_row": str(selected),
                        "prediction_row_id": source,
                    }
                    for selected, source in enumerate((2, 4, 9))
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="must be integers"):
        evaluate_files(prediction_file, label_root)


def test_single_alignment_and_label_row_id_must_agree(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions.csv"
    write_csv(
        prediction_file,
        ["row_id", "prediction"],
        [{"row_id": "7", "prediction": "{}"}],
    )
    label_root = tmp_path / "labels"
    label_root.mkdir()
    write_csv(
        label_root / "query.csv",
        ["row_id", "scoring_points"],
        [{"row_id": "8", "scoring_points": SCORING_POINTS}],
    )
    (label_root / "alignment.json").write_text(
        json.dumps(
            {
                "source_row_id": 7,
                "selected_query_row": 0,
                "prediction_row_id": 7,
            }
        )
    )

    with pytest.raises(ValueError, match="label row_id"):
        evaluate_files(prediction_file, label_root)


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                {"row_id": "1", "prediction": "{}"},
                {"row_id": "1", "prediction": "{}"},
            ],
            "unique",
        ),
        (
            [
                {"row_id": "2", "prediction": "{}"},
                {"row_id": "1", "prediction": "{}"},
            ],
            "label order",
        ),
        ([{"row_id": "1", "prediction": "{}"}], "row counts differ"),
    ],
)
def test_row_mismatch_is_rejected(
    tmp_path: Path, rows: list[dict[str, str]], message: str
) -> None:
    prediction_file = tmp_path / "predictions.csv"
    write_csv(prediction_file, ["row_id", "prediction"], rows)

    with pytest.raises(ValueError, match=message):
        evaluate_files(prediction_file, labels(tmp_path / "labels", 2))
