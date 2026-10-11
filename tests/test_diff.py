import hashlib
import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from rootwalk.diff import diff_run_bundles


def _bundle(
    root: Path,
    *,
    decisions: list[object],
    contexts: list[object] | None = None,
    evidence: list[object] | None = None,
) -> Path:
    root.mkdir()
    files: dict[str, object] = {
        "task.json": {"task_id": "task-1", "environment": "test"},
        "topology.json": {"snapshot_id": "topology-1"},
        "context_manifests.jsonl": contexts
        if contexts is not None
        else [{"call_number": 1, "status": "ready"}],
        "decisions.jsonl": decisions,
        "observations.jsonl": [{"tool": "query_metric", "status": "success"}],
        "evidence.jsonl": evidence or [],
        "report.json": {"status": "completed"},
    }
    for name, value in files.items():
        path = root / name
        if name.endswith(".jsonl"):
            items = cast(list[object], value)
            path.write_text(
                "".join(json.dumps({"payload": item}) + "\n" for item in items),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": root.name,
                "rootwalk_version": "test",
                "file_hashes": hashes,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _replace_stream(root: Path, filename: str, records: list[object]) -> None:
    path = root / filename
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["file_hashes"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_same_bundle_has_no_divergence(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "run", decisions=[{"decision_summary": "same"}])

    result = diff_run_bundles(bundle, bundle)

    assert result["same"] is True
    assert result["first_divergence"] is None
    assert all(layer["same"] for layer in result["layers"].values())


def test_bundle_without_evidence_stream_gets_empty_compatible_layer(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "run", decisions=[])
    (bundle / "evidence.jsonl").unlink()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    del manifest["file_hashes"]["evidence.jsonl"]
    (bundle / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = diff_run_bundles(bundle, bundle)

    assert result["layers"]["evidence"] == {"same": True, "differences": []}


def test_first_divergence_is_the_first_changed_decision(tmp_path: Path) -> None:
    left = _bundle(
        tmp_path / "left",
        decisions=[{"decision_summary": "same"}, {"decision_summary": "inspect A"}],
    )
    right = _bundle(
        tmp_path / "right",
        decisions=[{"decision_summary": "same"}],
    )

    result = diff_run_bundles(left, right)

    assert result["same"] is False
    assert result["first_divergence"] == {
        "layer": "decision",
        "index": 1,
        "left": result["layers"]["decision"]["differences"][0]["left"],
        "right": None,
    }
    assert result["layers"]["environment_manifest"]["same"] is True


def test_first_divergence_can_be_evidence(tmp_path: Path) -> None:
    common = {
        "evidence_id": "evidence-1",
        "observation_refs": ["observation-1"],
        "subject_ids": ["service:a"],
        "predicate": "latency.high",
        "value": {"status": "observed", "p99_ms": 800},
        "artifact_refs": [{"content_hash": "hash-1", "visibility": "model_readable"}],
        "source_fingerprint": "source-1",
    }
    left = _bundle(tmp_path / "left", decisions=[], evidence=[common])
    right = _bundle(
        tmp_path / "right",
        decisions=[],
        evidence=[{**common, "value": {"status": "observed", "p99_ms": 1200}}],
    )

    result = diff_run_bundles(left, right)

    assert list(result["layers"]) == [
        "environment_manifest",
        "context_manifest",
        "decision",
        "observation",
        "evidence",
        "termination",
    ]
    assert result["first_divergence"]["layer"] == "evidence"
    assert result["first_divergence"]["index"] == 0
    assert result["first_divergence"]["left"]["value"]["p99_ms"] == 800
    assert result["first_divergence"]["right"]["value"]["p99_ms"] == 1200


def test_context_reference_differences_are_structured(tmp_path: Path) -> None:
    selected: dict[str, list[str]] = {
        "evidence_ids": ["evidence-1"],
        "observation_ids": ["observation-1"],
    }
    omitted: dict[str, list[str]] = {"evidence_ids": [], "observation_ids": []}
    left = _bundle(
        tmp_path / "left",
        decisions=[],
        contexts=[
            {
                "call_number": 1,
                "status": "ready",
                "selected_refs": selected,
                "omitted_refs": omitted,
            }
        ],
    )
    right = _bundle(
        tmp_path / "right",
        decisions=[],
        contexts=[
            {
                "call_number": 1,
                "status": "ready",
                "selected_refs": {**selected, "evidence_ids": []},
                "omitted_refs": {**omitted, "evidence_ids": ["evidence-1"]},
            }
        ],
    )

    first = diff_run_bundles(left, right)["first_divergence"]

    assert first["layer"] == "context_manifest"
    assert first["left"]["selected_refs"] == selected
    assert first["right"]["omitted_refs"]["evidence_ids"] == ["evidence-1"]


def test_context_reference_projection_ignores_non_reference_content(
    tmp_path: Path,
) -> None:
    selected: dict[str, object] = {
        "evidence_ids": ["evidence-1"],
        "body": "private model input",
        "path": "/private/context.json",
        "selector": {"line": 7},
    }
    omitted: dict[str, object] = {
        "observation_ids": ["observation-1"],
        "body": "private omitted input",
        "path": "/private/omitted.json",
    }
    left = _bundle(
        tmp_path / "left",
        decisions=[],
        contexts=[
            {
                "call_number": 1,
                "selected_refs": selected,
                "omitted_refs": omitted,
            }
        ],
    )
    right = _bundle(
        tmp_path / "right",
        decisions=[],
        contexts=[
            {
                "call_number": 1,
                "selected_refs": {**selected, "body": "different private input"},
                "omitted_refs": {**omitted, "path": "/different/private.json"},
            }
        ],
    )

    result = diff_run_bundles(left, right)

    assert result["same"] is True


def test_diff_rejects_non_monotonic_evidence_revisions(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "run", decisions=[])
    _replace_stream(
        bundle,
        "evidence.jsonl",
        [
            {
                "kind": "evidence",
                "revision": 4,
                "stable_key": "evidence-2",
                "payload": {"evidence_id": "evidence-2"},
            },
            {
                "kind": "evidence",
                "revision": 3,
                "stable_key": "evidence-1",
                "payload": {"evidence_id": "evidence-1"},
            },
        ],
    )

    with pytest.raises(ValueError, match="revision 3 is not strictly after 4"):
        diff_run_bundles(bundle, bundle)


def test_recorded_bundle_matches_itself() -> None:
    bundle = Path("examples/bank-recorded-run")

    assert diff_run_bundles(bundle, bundle)["same"] is True


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_bundle_integrity_is_checked_before_diff(tmp_path: Path, failure: str) -> None:
    left = _bundle(tmp_path / "left", decisions=[])
    right = tmp_path / "right"
    shutil.copytree(left, right)
    target = right / "decisions.jsonl"
    if failure == "missing":
        target.unlink()
    else:
        target.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing|hash mismatch"):
        diff_run_bundles(left, right)
