# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import asyncio
import json
import multiprocessing
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest

from rootwalk.contracts import Decision
from rootwalk.model import Message, ModelCallResult, RecordedModelAdapter
from rootwalk.openrca import BankEnvironment
from rootwalk.runtime import RootWalkRuntime, RuntimeConfig
from rootwalk.store import ArtifactStore, RunSealedError, RunStore

PUBLIC_INPUT = (
    Path(__file__).resolve().parents[2]
    / "rootwalk-handoff"
    / "rootwalk-development-handoff"
    / "rootwalk-data-prep"
    / "slices"
    / "bank-row-0001"
    / "public_input"
)
CRASH_EXIT = 86
EnvironmentT = TypeVar("EnvironmentT", bound=BankEnvironment)


def _with_entry(environment: EnvironmentT) -> EnvironmentT:
    environment.task = environment.task.model_copy(
        update={"entry_entities": ["bank:entity:Tomcat01"]}
    )
    return environment


def _finish_decision(summary: str = "finish after recovery") -> dict[str, Any]:
    return {
        "decision_summary": summary,
        "operations": [{"type": "finalize", "abstain_reason": summary}],
    }


def _spawn_crash(target: Any, root: Path) -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=target, args=(root, PUBLIC_INPUT)
    )
    process.start()
    process.join(30)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("crash worker did not exit within 30 seconds")
    assert process.exitcode == CRASH_EXIT


class _FinalizeModel:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, messages: Sequence[Message]) -> object:
        del messages
        self.calls += 1
        return _finish_decision()


class _OrphanResponseModel:
    async def decide(self, messages: Sequence[Message]) -> ModelCallResult:
        del messages
        decision = Decision.model_validate(_finish_decision("orphan response"))
        return ModelCallResult(
            decision=decision,
            usage={"prompt_tokens": 7, "completion_tokens": 3},
            provider_model="crash-test",
            raw_response={
                "orphan_response_marker": True,
                "decision": decision.model_dump(mode="json"),
            },
        )


class _CrashAfterResponseArtifact(ArtifactStore):
    def put_json(self, content: object) -> str:
        content_hash = super().put_json(content)
        if isinstance(content, dict) and content.get("orphan_response_marker") is True:
            os._exit(CRASH_EXIT)
        return content_hash


def _crash_after_response_artifact(root: Path, public_input: Path) -> None:
    with RunStore(root / "run.sqlite3") as store:
        runtime = RootWalkRuntime(
            BankEnvironment(public_input),
            _OrphanResponseModel(),
            store,
            _CrashAfterResponseArtifact(root / "artifacts"),
            RuntimeConfig(max_model_calls=3),
        )
        asyncio.run(runtime.run("orphan-response", root / "unused-bundle"))


def test_response_artifact_without_decision_is_orphan_and_retried(
    tmp_path: Path,
) -> None:
    _spawn_crash(_crash_after_response_artifact, tmp_path)

    with RunStore(tmp_path / "run.sqlite3") as crashed:
        assert crashed.records("decision") == []
        assert crashed.records("model_response") == []
        registered = {
            cast(dict[str, Any], record.payload)["content_hash"]
            for record in crashed.records("artifact")
        }
        orphaned = {
            path.name for path in (tmp_path / "artifacts").iterdir() if path.is_file()
        } - registered
        assert len(orphaned) == 1
        orphan_payload = json.loads(
            (tmp_path / "artifacts" / orphaned.pop()).read_text(encoding="utf-8")
        )
        assert orphan_payload["orphan_response_marker"] is True

    model = _FinalizeModel()
    with RunStore(tmp_path / "run.sqlite3") as recovered:
        result = asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                model,
                recovered,
                ArtifactStore(tmp_path / "artifacts"),
                RuntimeConfig(max_model_calls=3),
            ).run("orphan-response", tmp_path / "bundle")
        )
        assert result.status == "completed_partial"
        assert model.calls == 1
        assert [record.stable_key for record in recovered.records("decision")] == [
            "decision:orphan-response:0002"
        ]
        assert len(recovered.records("model_response")) == 1
        starts = [
            cast(dict[str, Any], record.payload)["attempt_id"]
            for record in recovered.records("attempt")
            if record.stable_key.endswith(":start")
        ]
        assert starts == ["model:1", "model:2"]


class _CrashAfterFirstEffectStore(RunStore):
    def append_many(self, records: Iterable[tuple[str, str, object]]) -> list[int]:
        batch = list(records)
        revisions = super().append_many(batch)
        if any(kind == "tool_effect" for kind, _key, _payload in batch):
            os._exit(CRASH_EXIT)
        return revisions


def _crash_after_first_query_effect(root: Path, public_input: Path) -> None:
    decision = {
        "decision_summary": "run one successful and one failing query",
        "operations": [
            {"type": "query", "tool": "list_entities", "arguments": {}},
            {"type": "query", "tool": "list_metrics", "arguments": {}},
        ],
    }
    with _CrashAfterFirstEffectStore(root / "run.sqlite3") as store:
        runtime = RootWalkRuntime(
            _with_entry(BankEnvironment(public_input)),
            RecordedModelAdapter([decision]),
            store,
            ArtifactStore(root / "artifacts"),
            RuntimeConfig(max_model_calls=3),
        )
        asyncio.run(runtime.run("partial-query", root / "unused-bundle"))


class _RecoveryBank(BankEnvironment):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.list_entities_calls = 0
        self.list_metrics_calls = 0

    def list_entities(self) -> dict[str, Any]:
        self.list_entities_calls += 1
        raise AssertionError("committed query effect was replayed")

    def list_metrics(self, entity_id: str | None = None) -> dict[str, Any]:
        del entity_id
        self.list_metrics_calls += 1
        raise RuntimeError("injected second-query failure")


def test_domain_ahead_skips_effect_and_preserves_mixed_query_denominator(
    tmp_path: Path,
) -> None:
    _spawn_crash(_crash_after_first_query_effect, tmp_path)

    environment = _with_entry(_RecoveryBank(PUBLIC_INPUT))
    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            RootWalkRuntime(
                environment,
                _FinalizeModel(),
                store,
                ArtifactStore(tmp_path / "artifacts"),
                RuntimeConfig(max_model_calls=3),
            ).run("partial-query", tmp_path / "bundle")
        )

        assert environment.list_entities_calls == 0
        assert environment.list_metrics_calls == 1
        observations = [
            cast(dict[str, Any], item.payload) for item in store.records("observation")
        ]
        assert [(item["tool"], item["status"]) for item in observations] == [
            ("list_entities", "success"),
            ("list_metrics", "error"),
        ]
        effects = [
            cast(dict[str, Any], item.payload) for item in store.records("tool_effect")
        ]
        assert [item["status"] for item in effects] == ["success", "error"]
        tool_starts = [
            cast(dict[str, Any], item.payload)["tool"]
            for item in store.records("attempt")
            if isinstance(item.payload, dict) and item.payload.get("tool") is not None
        ]
        assert tool_starts == ["list_entities", "list_metrics"]
        assert result.report["budget"]["tool_attempts_used"] == 2

    bundled = [
        json.loads(line)
        for line in (tmp_path / "bundle" / "observations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [item["payload"]["tool"] for item in bundled] == [
        "list_entities",
        "list_metrics",
    ]


class _CrashAfterUnknownStore(RunStore):
    def append_many(self, records: Iterable[tuple[str, str, object]]) -> list[int]:
        batch = list(records)
        revisions = super().append_many(batch)
        if any(
            kind == "attempt"
            and isinstance(payload, dict)
            and payload.get("status") == "outcome_unknown"
            for kind, _key, payload in batch
        ):
            os._exit(CRASH_EXIT)
        return revisions


class _UnknownOutcomeModel:
    async def decide(self, messages: Sequence[Message]) -> object:
        del messages
        raise TimeoutError("provider outcome is unknown")


def _crash_after_outcome_unknown(root: Path, public_input: Path) -> None:
    with _CrashAfterUnknownStore(root / "run.sqlite3") as store:
        runtime = RootWalkRuntime(
            BankEnvironment(public_input),
            _UnknownOutcomeModel(),
            store,
            ArtifactStore(root / "artifacts"),
            RuntimeConfig(max_model_calls=3),
        )
        asyncio.run(runtime.run("unknown-outcome", root / "unused-bundle"))


def test_outcome_unknown_keeps_reservation_and_retries_with_new_attempt(
    tmp_path: Path,
) -> None:
    _spawn_crash(_crash_after_outcome_unknown, tmp_path)

    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                _FinalizeModel(),
                store,
                ArtifactStore(tmp_path / "artifacts"),
                RuntimeConfig(max_model_calls=3),
            ).run("unknown-outcome", tmp_path / "bundle")
        )
        attempts = [
            cast(dict[str, Any], item.payload) for item in store.records("attempt")
        ]
        assert [(item["attempt_id"], item["status"]) for item in attempts] == [
            ("model:1", "started"),
            ("model:1", "outcome_unknown"),
            ("model:2", "started"),
            ("model:2", "success"),
        ]
        ledger = [
            cast(dict[str, Any], item.payload)
            for item in store.records("budget_ledger")
        ]
        starts = [item for item in ledger if item["event"] == "model_call_started"]
        assert len(starts) == 2
        assert starts[0]["settlement"] == "reserved_unknown"
        assert (
            len([item for item in ledger if item["event"] == "model_call_completed"])
            == 1
        )
        assert result.report["budget"]["output_tokens_used"] == 4_000


def test_sealed_run_rejects_late_effect_but_accepts_audit(tmp_path: Path) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                RecordedModelAdapter([_finish_decision("sealed")]),
                store,
                ArtifactStore(tmp_path / "artifacts"),
                RuntimeConfig(max_model_calls=1),
            ).run("sealed-late-effect", tmp_path / "bundle")
        )
        report_before = store.get("report:sealed-late-effect")
        sealed = store.seal_state("sealed-late-effect")
        assert report_before is not None
        assert sealed is not None

        with pytest.raises(RunSealedError):
            store.append(
                "tool_effect",
                "tool_effect:sealed-late-effect:late:0",
                {"status": "success"},
            )
        audit_revision = store.append(
            "late_audit",
            "late_audit:sealed-late-effect:late:0",
            {"audit_type": "late_tool_effect", "status": "success"},
        )

        assert result.report == cast(dict[str, Any], report_before.payload)
        assert store.get("report:sealed-late-effect") == report_before
        assert store.records("tool_effect") == []
        assert store.domain_revision == sealed.sealed_revision
        assert audit_revision > sealed.sealed_revision
