import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from rootwalk.store import (
    ArtifactBudgetExceeded,
    ArtifactStore,
    BundleExporter,
    RunSealedError,
    RunStore,
    StableKeyConflict,
    checkpoint_relation,
    export_self_contained_bundle,
)


def test_effect_append_is_idempotent_and_revisions_are_monotonic(
    tmp_path: Path,
) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        first = store.append("decision", "decision:1", {"operations": []})
        effect = {"decision_id": "1", "operation_index": 0, "status": "success"}
        effect_revision = store.append("tool_effect", "1:0:query_result", effect)

        assert (
            store.append("tool_effect", "1:0:query_result", effect) == effect_revision
        )
        assert store.revision == effect_revision
        assert first < effect_revision
        assert len(store.records()) == 2
        with pytest.raises(StableKeyConflict):
            store.append(
                "tool_effect", "1:0:query_result", {**effect, "status": "error"}
            )


def test_append_many_rolls_back_whole_batch_on_conflict(tmp_path: Path) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        existing = store.append("decision", "decision:1", {"summary": "first"})

        with pytest.raises(StableKeyConflict):
            store.append_many(
                [
                    ("evidence", "evidence:1", {"value": 1}),
                    ("decision", "decision:1", {"summary": "changed"}),
                ]
            )

        assert store.get("evidence:1") is None
        assert store.revision == existing
        revisions = store.append_many(
            [
                ("decision", "decision:1", {"summary": "first"}),
                ("evidence", "evidence:1", {"value": 1}),
            ]
        )
        assert revisions[0] == existing
        assert revisions[1] > existing


def test_append_many_and_seal_commits_records_run_and_outbox_atomically(
    tmp_path: Path,
) -> None:
    records = [
        ("evidence", "evidence:1", {"value": 1}),
        ("report", "report:run-1", {"status": "done"}),
    ]
    with RunStore(tmp_path / "run.sqlite3") as store:
        revisions, sealed = store.append_many_and_seal(
            records, "run-1", "done", "report:run-1"
        )

        assert revisions == [1, 2]
        assert sealed.report_revision == 2
        assert sealed.sealed_revision == 2
        assert store.seal_state("run-1") == sealed
        pending = store.outbox_jobs("pending")
        assert len(pending) == 1
        assert (pending[0].run_id, pending[0].report_ref) == (
            "run-1",
            "report:run-1",
        )


def test_append_many_and_seal_rolls_back_records_on_conflict_or_missing_report(
    tmp_path: Path,
) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        existing = store.append("decision", "decision:1", {"summary": "first"})

        with pytest.raises(StableKeyConflict):
            store.append_many_and_seal(
                [
                    ("evidence", "evidence:1", {"value": 1}),
                    ("decision", "decision:1", {"summary": "changed"}),
                    ("report", "report:run-1", {"status": "done"}),
                ],
                "run-1",
                "done",
                "report:run-1",
            )
        assert store.revision == existing
        assert store.get("evidence:1") is None
        assert store.get("report:run-1") is None
        assert store.seal_state("run-1") is None
        assert store.outbox_jobs() == []

        with pytest.raises(ValueError, match="committed report"):
            store.append_many_and_seal(
                [("evidence", "evidence:2", {"value": 2})],
                "run-1",
                "done",
                "report:missing",
            )
        assert store.revision == existing
        assert store.get("evidence:2") is None
        assert store.seal_state("run-1") is None
        assert store.outbox_jobs() == []


def test_append_many_and_seal_is_idempotent_without_reopening_sealed_writes(
    tmp_path: Path,
) -> None:
    records = [
        ("evidence", "evidence:1", {"value": 1}),
        ("report", "report:run-1", {"status": "done"}),
    ]
    with RunStore(tmp_path / "run.sqlite3") as store:
        first = store.append_many_and_seal(records, "run-1", "done", "report:run-1")
        second = store.append_many_and_seal(records, "run-1", "done", "report:run-1")

        assert second == first
        assert len(store.records()) == 2
        assert len(store.outbox_jobs()) == 1
        with pytest.raises(RunSealedError):
            store.append_many_and_seal(
                [("evidence", "evidence:new", {"value": 2})],
                "run-1",
                "done",
                "report:run-1",
            )
        assert store.get("evidence:new") is None


def test_seal_is_idempotent_and_only_late_audit_may_append(tmp_path: Path) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        report_revision = store.append("report", "report:run-1", {"status": "done"})

        sealed = store.seal("run-1", "done", "report:run-1")

        assert store.seal("run-1", "done", "report:run-1") == sealed
        assert sealed.report_revision == report_revision
        assert store.seal_state("run-1") == sealed
        with pytest.raises(RunSealedError):
            store.append("evidence", "evidence:late", {"value": 2})
        with pytest.raises(RunSealedError):
            store.seal("run-1", "failed", "report:run-1")

        audit_revision = store.append(
            "late_audit", "late_audit:1", {"outcome": "arrived after seal"}
        )
        assert audit_revision > sealed.sealed_revision
        assert store.domain_revision == sealed.sealed_revision
        assert store.get("report:run-1") is not None


def test_checkpoint_is_bounded_and_recovery_relation_is_explicit(
    tmp_path: Path,
) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        store.append_many(
            [
                ("investigation_event", "event:1", {"type": "start"}),
                ("evidence", "evidence:1", {"value": 1}),
                ("budget_ledger", "budget:1", {"tokens": 2}),
            ]
        )
        checkpoint = store.save_checkpoint(
            event_cursor=1,
            projection_revision=3,
            ledger_cursor=2,
            phase="execute",
            pending_refs=["operation:1"],
        )

        assert store.read_checkpoint() == checkpoint
        assert checkpoint_relation(store.domain_revision, 2) == "domain_ahead"
        assert checkpoint_relation(store.domain_revision, 3) == "consistent"
        assert checkpoint_relation(store.domain_revision, 4) == "checkpoint_ahead"
        with pytest.raises(ValueError, match="cannot lead"):
            store.save_checkpoint(
                event_cursor=4,
                projection_revision=3,
                ledger_cursor=2,
                phase="execute",
                pending_refs=[],
            )


def test_seal_enqueues_one_idempotent_outbox_job(tmp_path: Path) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        report_revision = store.append("report", "report:run-1", {"status": "done"})

        store.seal("run-1", "done", "report:run-1")
        store.seal("run-1", "done", "report:run-1")

        pending = store.outbox_jobs("pending")
        assert len(pending) == 1
        assert (pending[0].run_id, pending[0].report_revision) == (
            "run-1",
            report_revision,
        )
        assert store.mark_outbox_done("run-1", report_revision).status == "done"
        assert store.mark_outbox_done("run-1", report_revision).status == "done"
        assert store.outbox_jobs("pending") == []
        assert len(store.outbox_jobs("done")) == 1


def test_artifact_hash_is_stable_and_access_is_root_confined(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    content = b"stable model response\n"

    first = artifacts.put_bytes(content)
    second = artifacts.put_bytes(content)

    assert first == second == hashlib.sha256(content).hexdigest()
    assert artifacts.read_bytes(first) == content
    assert artifacts.path_for(first).parent == artifacts.root
    with pytest.raises(ValueError, match="SHA256"):
        artifacts.read_bytes("../outside")

    artifacts.path_for(first).write_bytes(b"tampered on disk")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        artifacts.committed_path(first)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        artifacts.read_bytes(first)


def test_artifact_store_rejects_symlink_and_replaced_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-artifacts"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        ArtifactStore(linked)

    root = tmp_path / "artifacts"
    artifacts = ArtifactStore(root)
    content_hash = artifacts.put_text("committed")
    moved = tmp_path / "moved-artifacts"
    root.rename(moved)
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="changed after initialization"):
        artifacts.put_text("must not escape")
    with pytest.raises(ValueError, match="changed after initialization"):
        artifacts.read_bytes(content_hash)
    assert list(outside.iterdir()) == []


def test_artifact_store_removes_crashed_staging_files(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    stale = root / ".artifact-crashed"
    stale.write_bytes(b"unfinished")
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    stale_link = root / ".artifact-link"
    stale_link.symlink_to(outside)

    artifacts = ArtifactStore(root, max_bytes=3)

    assert not stale.exists()
    assert not stale_link.exists()
    assert outside.read_bytes() == b"keep"
    assert artifacts.bytes_used == 0
    artifacts.put_bytes(b"abc")


def test_artifact_budget_is_recovered_deduplicated_and_atomic(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    artifacts = ArtifactStore(root, max_bytes=5)

    first = artifacts.put_bytes(b"abc")
    assert artifacts.put_bytes(b"abc") == first
    assert artifacts.bytes_used == 3
    assert [path.name for path in root.iterdir()] == [first]

    recovered = ArtifactStore(root, max_bytes=5)
    second = recovered.put_bytes(b"de")
    assert recovered.bytes_used == 5
    with pytest.raises(ArtifactBudgetExceeded) as error:
        recovered.put_bytes(b"f")

    assert (error.value.used, error.value.requested, error.value.limit) == (5, 1, 5)
    assert {path.name for path in root.iterdir()} == {first, second}
    assert not list(root.glob(".artifact-*"))


def test_model_response_and_decision_commit_before_tool_effect(tmp_path: Path) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        response_revision = store.append(
            "model_response", "model_response:1", {"content_hash": "a" * 64}
        )
        decision_revision = store.append(
            "decision",
            "decision:1",
            {"decision_summary": "inspect", "operations": [{"type": "query"}]},
        )
        effect_revision = store.append(
            "tool_effect", "1:0:query_result", {"status": "success"}
        )

        assert response_revision < effect_revision
        assert decision_revision < effect_revision


def test_bundle_exports_minimum_file_set_without_copying_external_input(
    tmp_path: Path,
) -> None:
    artifact_store = ArtifactStore(tmp_path / "artifact-store")
    embedded_hash = artifact_store.put_text("small generated observation")
    external_hash = "f" * 64
    with RunStore(tmp_path / "run.sqlite3") as store:
        store.append(
            "manifest",
            "manifest:1",
            {
                "rootwalk_version": "test",
                "model_version": "recorded",
                "prompt_version": "test",
                "tool_version": "test",
                "data_version": "bank-row-0001",
            },
        )
        store.append("task", "task:1", {"task_id": "task-1"})
        store.append("topology", "topology:1", {"snapshot_id": "topology-1"})
        for kind in (
            "decision",
            "attempt",
            "observation",
            "evidence",
            "investigation_event",
            "context_manifest",
            "budget_ledger",
        ):
            payload: dict[str, object] = {"id": f"{kind}-1"}
            if kind == "decision":
                payload["model_response_ref"] = embedded_hash
            elif kind == "context_manifest":
                payload["model_input_ref"] = embedded_hash
            elif kind == "observation":
                payload["artifact_refs"] = [{"content_hash": external_hash}]
            store.append(kind, f"{kind}:1", payload)
        store.append("report", "report:1", {"status": "partial"})
        store.append(
            "artifact",
            f"artifact:{embedded_hash}",
            {"content_hash": embedded_hash, "storage": "embedded"},
        )
        store.append(
            "artifact",
            "artifact:external-bank-input",
            {
                "content_hash": external_hash,
                "storage": "external",
                "uri": "/read-only/bank-input-92MiB",
            },
        )

        bundle = BundleExporter(store, artifact_store).export(tmp_path / "bundle")

    expected_files = {
        "artifact_index.json",
        "attempts.jsonl",
        "budget_ledger.jsonl",
        "context_manifests.jsonl",
        "decisions.jsonl",
        "evidence.jsonl",
        "investigation_events.jsonl",
        "manifest.json",
        "observations.jsonl",
        "report.json",
        "task.json",
        "topology.json",
    }
    assert expected_files <= {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
    }
    assert (bundle / "artifacts" / embedded_hash).read_text() == (
        "small generated observation"
    )
    assert not (bundle / "artifacts" / external_hash).exists()

    index = cast(
        list[dict[str, Any]],
        json.loads((bundle / "artifact_index.json").read_text()),
    )
    assert [entry["storage"] for entry in index] == ["embedded", "external"]
    manifest = cast(dict[str, Any], json.loads((bundle / "manifest.json").read_text()))
    assert manifest["bundle_schema_version"] == "1"
    assert manifest["domain_cursor"] == 13
    assert "artifacts/" + embedded_hash in manifest["file_hashes"]


def test_bundle_export_is_atomic_when_writing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    content_hash = artifacts.put_text("embedded")
    with RunStore(tmp_path / "run.sqlite3") as store:
        store.append("task", "task:1", {"task_id": "task-1"})
        store.append("topology", "topology:1", {"snapshot_id": "topology-1"})
        store.append("report", "report:1", {"status": "partial"})
        store.append(
            "artifact",
            f"artifact:{content_hash}",
            {"content_hash": content_hash, "storage": "embedded"},
        )
        exporter = BundleExporter(store, artifacts)
        original = exporter._write_jsonl  # pyright: ignore[reportPrivateUsage]

        def fail_after_write(path: Path, records: list[Any]) -> None:
            original(path, records)
            raise OSError("injected export crash")

        destination = tmp_path / "bundle"
        destination.mkdir()
        monkeypatch.setattr(exporter, "_write_jsonl", fail_after_write)
        with pytest.raises(OSError, match="injected export crash"):
            exporter.export(destination)

        assert destination.is_dir() and not any(destination.iterdir())
        assert not list(tmp_path.glob(".bundle.tmp-*"))
        monkeypatch.setattr(exporter, "_write_jsonl", original)
        assert exporter.export(destination) == destination


def _bundle_with_external(
    tmp_path: Path,
    content: bytes,
    visibility: str | tuple[str, ...] = "model_readable",
) -> tuple[Path, str]:
    content_hash = hashlib.sha256(content).hexdigest()
    visibilities = (visibility,) if isinstance(visibility, str) else visibility
    with RunStore(tmp_path / "portable-run.sqlite3") as store:
        store.append("task", "task:portable", {"task_id": "portable"})
        store.append("topology", "topology:portable", {"snapshot_id": "portable"})
        for index, item_visibility in enumerate(visibilities):
            store.append(
                "observation",
                f"observation:portable:{index}",
                {
                    "artifact_refs": [
                        {
                            "content_hash": content_hash,
                            "visibility": item_visibility,
                        }
                    ]
                },
            )
        store.append("report", "report:portable", {"status": "partial"})
        store.append(
            "artifact",
            f"artifact:{content_hash}",
            {
                "content_hash": content_hash,
                "storage": "external",
                "uri": f"bank://fixture/{content_hash}",
            },
        )
        bundle = BundleExporter(store, ArtifactStore(tmp_path / "cas")).export(
            tmp_path / "reference-bundle"
        )
    return bundle, content_hash


def _external_root(tmp_path: Path, content: bytes) -> tuple[Path, Path]:
    root = tmp_path / "public_input"
    artifact = root / "telemetry/metric/source.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(content)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": "telemetry/metric/source.csv",
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ]
            }
        )
    )
    return root, artifact


def test_self_contained_export_embeds_only_verified_external_artifacts(
    tmp_path: Path,
) -> None:
    content = b"verified external telemetry\n"
    bundle, content_hash = _bundle_with_external(tmp_path, content)
    external, _artifact = _external_root(tmp_path, content)

    exported = export_self_contained_bundle(
        bundle, tmp_path / "self-contained", external
    )

    assert (exported / "artifacts" / content_hash).read_bytes() == content
    source_index = json.loads((bundle / "artifact_index.json").read_text())
    target_index = json.loads((exported / "artifact_index.json").read_text())
    assert source_index[0]["storage"] == "external"
    assert target_index[0]["storage"] == "embedded"
    assert target_index[0]["bundle_path"] == f"artifacts/{content_hash}"
    assert "uri" not in target_index[0]
    manifest = json.loads((exported / "manifest.json").read_text())
    assert manifest["bundle_mode"] == "self_contained"
    assert manifest["external_artifacts_embedded"] == 1
    assert manifest["integrity"] == {
        "algorithm": "sha256",
        "scope": "all bundle files except manifest.json",
        "status": "verified",
        "verified_file_count": len(manifest["file_hashes"]),
    }
    assert manifest["bundle_size_bytes"] == sum(
        path.stat().st_size for path in exported.rglob("*") if path.is_file()
    )
    assert manifest["file_hashes"][f"artifacts/{content_hash}"] == content_hash


def test_self_contained_export_accepts_model_and_audit_visibility_mix(
    tmp_path: Path,
) -> None:
    content = b"shared model and audit artifact\n"
    bundle, content_hash = _bundle_with_external(
        tmp_path, content, visibility=("model_readable", "audit_only")
    )
    external, _artifact = _external_root(tmp_path, content)

    exported = export_self_contained_bundle(bundle, tmp_path / "portable", external)

    index = json.loads((exported / "artifact_index.json").read_text())
    assert index[0]["content_hash"] == content_hash
    assert index[0]["visibility"] == "model_readable"
    assert index[0]["storage"] == "embedded"


@pytest.mark.parametrize("failure", ["hash", "size", "traversal", "symlink"])
def test_self_contained_export_rejects_untrusted_external_artifact(
    tmp_path: Path, failure: str
) -> None:
    content = b"external telemetry\n"
    bundle, content_hash = _bundle_with_external(tmp_path, content)
    external, artifact = _external_root(tmp_path, content)
    manifest_path = external / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if failure == "hash":
        artifact.write_bytes(b"tampered telemetry\n")
    elif failure == "size":
        manifest["artifacts"][0]["bytes"] += 1
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "traversal":
        outside = tmp_path / "outside.csv"
        outside.write_bytes(content)
        manifest["artifacts"][0]["path"] = "../outside.csv"
        manifest_path.write_text(json.dumps(manifest))
    else:
        outside = tmp_path / "outside.csv"
        artifact.rename(outside)
        artifact.symlink_to(outside)

    with pytest.raises((ValueError, FileNotFoundError)):
        export_self_contained_bundle(bundle, tmp_path / "rejected", external)

    assert not (tmp_path / "rejected").exists()
    assert content_hash not in {
        path.name for path in tmp_path.rglob("*") if path.is_file()
    }


def test_self_contained_export_never_reads_evaluation_only_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"private evaluation label\n"
    bundle, _content_hash = _bundle_with_external(
        tmp_path, content, visibility="evaluation_only"
    )
    external, artifact = _external_root(tmp_path, content)
    original_open = Path.open

    def guarded_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == artifact:
            raise AssertionError("evaluation-only artifact was read")
        return cast(Any, original_open)(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="evaluation_only"):
        export_self_contained_bundle(bundle, tmp_path / "rejected", external)


def test_self_contained_export_rejects_embedded_evaluation_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = ArtifactStore(tmp_path / "cas")
    content_hash = artifacts.put_text("private evaluation label")
    with RunStore(tmp_path / "run.sqlite3") as store:
        store.append("task", "task:1", {"task_id": "task-1"})
        store.append("topology", "topology:1", {"snapshot_id": "topology-1"})
        store.append(
            "observation",
            "observation:1",
            {
                "artifact_refs": [
                    {
                        "content_hash": content_hash,
                        "visibility": "evaluation_only",
                    }
                ]
            },
        )
        store.append("report", "report:1", {"status": "partial"})
        store.append(
            "artifact",
            f"artifact:{content_hash}",
            {"content_hash": content_hash, "storage": "embedded"},
        )
        bundle = BundleExporter(store, artifacts).export(tmp_path / "bundle")
    original_artifact = bundle / "artifacts" / content_hash
    sensitive = bundle / "nonstandard/secret.bin"
    sensitive.parent.mkdir()
    original_artifact.rename(sensitive)
    index_path = bundle / "artifact_index.json"
    index = json.loads(index_path.read_text())
    index[0]["bundle_path"] = "nonstandard/secret.bin"
    index[0]["visibility"] = "model_readable"
    index_path.write_text(
        f"{json.dumps(index, separators=(',', ':'), sort_keys=True)}\n"
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_hashes"].pop(f"artifacts/{content_hash}")
    manifest["file_hashes"]["nonstandard/secret.bin"] = content_hash
    manifest["file_hashes"]["artifact_index.json"] = hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        f"{json.dumps(manifest, separators=(',', ':'), sort_keys=True)}\n"
    )
    original_open = Path.open

    def guarded_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == sensitive:
            raise AssertionError("embedded evaluation-only artifact was read")
        return cast(Any, original_open)(self, *args, **kwargs)

    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="evaluation_only"):
        export_self_contained_bundle(bundle, tmp_path / "rejected", external)


def test_self_contained_export_requires_canonical_embedded_path(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path / "cas")
    content_hash = artifacts.put_text("embedded")
    with RunStore(tmp_path / "run.sqlite3") as store:
        store.append("task", "task:1", {"task_id": "task-1"})
        store.append("topology", "topology:1", {"snapshot_id": "topology-1"})
        store.append("report", "report:1", {"status": "partial"})
        store.append(
            "artifact",
            f"artifact:{content_hash}",
            {"content_hash": content_hash, "storage": "embedded"},
        )
        bundle = BundleExporter(store, artifacts).export(tmp_path / "bundle")
    index_path = bundle / "artifact_index.json"
    index = json.loads(index_path.read_text())
    index[0]["bundle_path"] = "artifacts/not-the-content-hash"
    index_path.write_text(
        f"{json.dumps(index, separators=(',', ':'), sort_keys=True)}\n"
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_hashes"]["artifact_index.json"] = hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        f"{json.dumps(manifest, separators=(',', ':'), sort_keys=True)}\n"
    )
    external, _artifact = _external_root(tmp_path, b"unused")

    with pytest.raises(ValueError, match="embedded artifact must use physical path"):
        export_self_contained_bundle(bundle, tmp_path / "rejected", external)


@pytest.mark.parametrize(
    "reference_field", ["content_hash", "model_input_ref", "model_response_ref"]
)
def test_bundle_rejects_unregistered_artifact_reference(
    tmp_path: Path, reference_field: str
) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        store.append("task", "task:1", {"task_id": "task-1"})
        store.append("topology", "topology:1", {"snapshot_id": "topology-1"})
        store.append("decision", "decision:1", {reference_field: "d" * 64})
        store.append("report", "report:1", {"status": "partial"})

        with pytest.raises(ValueError, match="unregistered artifact reference"):
            BundleExporter(store, ArtifactStore(tmp_path / "artifacts")).export(
                tmp_path / "bundle"
            )

    assert not (tmp_path / "bundle").exists()
