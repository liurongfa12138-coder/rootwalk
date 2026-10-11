import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from rootwalk.cli import app
from rootwalk.inspection import inspect_bundle, inspect_many, render_inspection


def bundle_fixture(tmp_path: Path, *, legacy: bool = False) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    (root / "artifacts").mkdir()
    messages = [
        {"role": "system", "content": "fixed policy"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "convergence": {"required_next_operation": "update_hypothesis"},
                    "candidate_evidence_summary": {
                        "pairs": [{"candidate_id": "candidate"}]
                    },
                    "budget": {"model_calls_remaining": 3},
                    "evidence": [],
                }
            ),
        },
    ]
    body = json.dumps(messages).encode()
    digest = hashlib.sha256(body).hexdigest()
    (root / "artifacts" / digest).write_bytes(body)
    values: dict[str, Any] = {
        "task.json": {"task_id": "synthetic", "environment": "test"},
        "topology.json": {"nodes": [{"canonical_id": "db"}]},
        "report.json": {
            "status": "completed_partial",
            "findings": [],
            "model_calls": 1,
            "candidates": [],
            "model_usage": {"prompt_tokens": 100},
        },
        "artifact_index.json": [
            {
                "content_hash": digest,
                "storage": "embedded",
                "visibility": "model_readable",
                "bundle_path": "artifacts/" + digest,
            }
        ],
    }
    streams: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "context_manifests": [
            (1, {"call_number": 1, "model_input_ref": digest, "estimated_tokens": 300})
        ],
        "decisions": [
            (
                3,
                {
                    "decision_id": "decision:run:0001",
                    "operations": [{"type": "query", "tool": "query_candidate"}],
                },
            )
        ],
        "observations": [
            (
                6,
                {
                    "observation_id": "o1",
                    "attempt_id": "tool-a",
                    "tool": "query_metric",
                    "arguments": {"entity_id": "db", "metric_id": "cpu"},
                },
            )
        ],
        "evidence": [],
        "attempts": [
            (2, {"attempt_id": "model:1", "call_number": 1, "status": "started"}),
            (
                4,
                {
                    "attempt_id": "model:1",
                    "call_number": 1,
                    "status": "success",
                    "duration_ms": 20,
                    "usage": {"prompt_tokens": 100},
                },
            ),
            (
                5,
                {
                    "attempt_id": "tool-a",
                    "operation_id": ["run", "decision:run:0001", 0],
                    "status": "started",
                },
            ),
            (
                7,
                {
                    "attempt_id": "tool-a",
                    "operation_id": ["run", "decision:run:0001", 0],
                    "status": "success",
                    "duration_ms": 5,
                },
            ),
            (
                8,
                {
                    "attempt_id": "model:1:provider:1",
                    "call_number": 1,
                    "status": "success",
                },
            ),
        ],
        "model_requests": [
            (9, {"call_number": 1, "request_index": 1, "status": "invalid_response"}),
            (10, {"call_number": 1, "request_index": 2, "status": "success"}),
        ],
        "investigation_events": [
            (
                11,
                {
                    "event_type": "candidate_disposition_applied",
                    "decision_id": "decision:run:0001",
                    "original_operation": {"subject": "cache", "status": "open"},
                    "applied_operation": {"subject": "db", "status": "open"},
                },
            ),
            (12, {"event_type": "run_sealed", "status": "completed_partial"}),
        ],
    }
    if legacy:
        streams.pop("model_requests")
        for _, payload in streams["attempts"]:
            payload.pop("duration_ms", None)
    for filename, value in values.items():
        (root / filename).write_text(json.dumps(value))
    for name, rows in streams.items():
        (root / (name + ".jsonl")).write_text(
            "".join(
                json.dumps(
                    {
                        "revision": revision,
                        "payload": payload,
                        "stable_key": f"{name}:{revision}",
                    }
                )
                + "\n"
                for revision, payload in rows
            )
        )
    _manifest(root)
    return root


def _manifest(root: Path) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "code_revision": "frozen",
                "file_hashes": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob("*")
                    if p.is_file() and p.name != "manifest.json"
                },
            }
        )
    )


def test_inspection_distinguishes_model_decisions_effects_and_repair_requests(
    tmp_path: Path,
) -> None:
    root = bundle_fixture(tmp_path)
    result = inspect_bundle(root)
    summary = result["summary"]
    assert summary["tool_attempts"] == 1
    assert summary["provider_requests"] == 2
    assert summary["provider_status_counts"] == {"invalid_response": 1, "success": 1}
    assert summary["model_duration_ms"] == {
        "sum": 20,
        "known_count": 1,
        "missing_count": 0,
    }
    assert summary["tool_duration_ms"]["sum"] == 5
    row = result["rounds"][0]
    assert row["bindings"][0]["original"]["subject"] == "cache"
    assert row["bindings"][0]["applied"]["subject"] == "db"
    assert (
        row["context"]["view"]["convergence"]["required_next_operation"]
        == "update_hypothesis"
    )
    assert [item["code"] for item in result["diagnostic_signals"]] == [
        "provider_repairs_present"
    ]
    assert "Compiled input" in render_inspection(result)
    assert inspect_bundle(root, call_number=1) == result
    with pytest.raises(ValueError, match="absent"):
        inspect_bundle(root, call_number=2)


def test_inspection_preserves_missing_legacy_telemetry(tmp_path: Path) -> None:
    result = inspect_bundle(bundle_fixture(tmp_path, legacy=True))
    assert result["summary"]["provider_requests"] is None
    assert result["summary"]["model_duration_ms"] == {
        "sum": None,
        "known_count": 0,
        "missing_count": 1,
    }


def test_inspection_includes_orphan_started_provider_attempt(tmp_path: Path) -> None:
    root = bundle_fixture(tmp_path)
    path = root / "attempts.jsonl"
    records = [
        {
            "revision": 9,
            "stable_key": "provider:1:start",
            "payload": {
                "attempt_id": "attempt:provider:run:1:1",
                "call_number": 1,
                "provider_request_index": 1,
                "status": "started",
            },
        },
        {
            "revision": 10,
            "stable_key": "provider:3:start",
            "payload": {
                "attempt_id": "attempt:provider:run:1:3",
                "call_number": 1,
                "provider_request_index": 3,
                "status": "started",
            },
        },
    ]
    path.write_text(
        path.read_text() + "".join(json.dumps(row) + "\n" for row in records)
    )
    _manifest(root)

    result = inspect_bundle(root)

    assert result["summary"]["provider_requests"] == 3
    assert result["summary"]["provider_status_counts"] == {
        "invalid_response": 1,
        "success": 1,
        "started": 1,
    }
    requests = result["rounds"][0]["provider_requests"]
    assert [request["provider_request_index"] for request in requests] == [1, 2, 3]
    assert requests[0]["status"] == "invalid_response"
    assert requests[2]["status"] == "started"


def test_inspection_rejects_tampering_and_does_not_read_unmanifested_data(
    tmp_path: Path,
) -> None:
    root = bundle_fixture(tmp_path)
    (root / "labels.json").write_text("not json and not to be read")
    assert inspect_bundle(root)["integrity"]["unmanifested_files"] == ["labels.json"]
    path = next((root / "artifacts").iterdir())
    path.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_bundle(root)


def test_inspection_cli_protects_sealed_bundle(tmp_path: Path) -> None:
    root = bundle_fixture(tmp_path)
    runner = CliRunner()
    denied = runner.invoke(
        app, ["inspect", str(root), "--output", str(root / "audit.json")]
    )
    assert denied.exit_code == 1
    assert "outside the sealed bundle" in denied.output
    output = tmp_path / "audit.md"
    ok = runner.invoke(
        app,
        [
            "inspect",
            str(root),
            "--format",
            "markdown",
            "--call",
            "1",
            "--output",
            str(output),
        ],
    )
    assert ok.exit_code == 0, ok.output
    assert "## Call 1" in output.read_text()


def test_inspection_aggregate_preserves_missing_values(tmp_path: Path) -> None:
    current = bundle_fixture(tmp_path / "current")
    legacy = bundle_fixture(tmp_path / "legacy", legacy=True)

    result = inspect_many([current, legacy])

    assert result["run_count"] == 2
    assert result["provider_requests"] == {
        "sum": 2,
        "mean": 2.0,
        "min": 2,
        "max": 2,
        "known_count": 1,
        "missing_count": 1,
    }
    assert result["tool_counts"] == {"query_metric": 2}
    assert result["candidate_presence"] == {}
