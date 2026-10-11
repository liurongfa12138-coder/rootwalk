import asyncio
import json
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from rootwalk.cli import _close_tracer_provider  # pyright: ignore[reportPrivateUsage]
from rootwalk.model import ModelAdapter, RecordedModelAdapter
from rootwalk.observability import BoundedSpanExporter, ObservabilityConfig
from rootwalk.openrca import BankEnvironment
from rootwalk.query_worker import _capture_stderr  # pyright: ignore[reportPrivateUsage]
from rootwalk.runtime import RootWalkRuntime, RuntimeConfig
from rootwalk.store import ArtifactBudgetExceeded, ArtifactStore, RunStore


class Reply:
    def __init__(self, content: str, **extra: Any) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.model = "audit-model"
        self.extra = extra

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            "choices": [{"message": {"content": self.choices[0].message.content}}],
            **self.extra,
        }


def adapter(
    monkeypatch: pytest.MonkeyPatch,
    contents: Sequence[str | BaseException | Reply],
    *,
    max_output_tokens: int = 2_000,
) -> ModelAdapter:
    replies = iter(contents)

    async def create(**kwargs: Any) -> Reply:
        reply = next(replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, Reply) else Reply(reply)

    def client(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    def import_module(name: str) -> SimpleNamespace:
        return SimpleNamespace(AsyncArk=client)

    monkeypatch.setattr("rootwalk.model.import_module", import_module)
    monkeypatch.setenv("ARK_API_KEY", "synthetic-test-key")
    return ModelAdapter(
        model="audit-model",
        base_url="https://unused.invalid",
        max_output_tokens=max_output_tokens,
    )


@pytest.mark.parametrize("repaired", [False, True])
def test_provider_requests_and_failed_usage_are_auditable(
    tmp_path: Path,
    synthetic_public_input: Path,
    monkeypatch: pytest.MonkeyPatch,
    repaired: bool,
) -> None:
    valid = json.dumps(
        {
            "decision_summary": "Recorded decision",
            "operations": [{"type": "finalize", "abstain_reason": "test"}],
        }
    )
    contents = (
        ["invalid-private-response", valid]
        if repaired
        else ["invalid-private-response"] * 3
    )
    model = adapter(monkeypatch, contents)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(BoundedSpanExporter(exporter)))
    store = RunStore(tmp_path / "run.sqlite3")
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(synthetic_public_input),
            model,
            store,
            ArtifactStore(tmp_path / "artifacts"),
            tracer=provider.get_tracer("test.audit"),
            observability_config=ObservabilityConfig(enabled=True),
        ).run("audit", tmp_path / "bundle")
    )
    assert result.model_calls == 1
    assert result.report["model_usage"]["prompt_tokens"] == 3 * len(contents)
    assert result.report["model_usage"]["completion_tokens"] == 5 * len(contents)
    requests = [record.payload for record in store.records("model_request")]
    assert len(requests) == len(contents)
    assert all(item["duration_ms"] >= 0 for item in requests)  # type: ignore[index]
    artifacts = ArtifactStore(tmp_path / "artifacts")
    first = json.loads(artifacts.read_bytes(requests[0]["model_input_ref"]))  # type: ignore[index]
    second = json.loads(artifacts.read_bytes(requests[1]["model_input_ref"]))  # type: ignore[index]
    assert len(second["messages"]) == len(first["messages"]) + 2
    assert first["messages"][-1] != second["messages"][-1]
    assert len(
        (result.bundle_path / "model_requests.jsonl").read_text().splitlines()
    ) == len(contents)
    model_attempts = [record.payload for record in store.records("attempt")]
    assert model_attempts[-1]["provider_request_count"] == len(contents)  # type: ignore[index]
    assert model_attempts[-1]["duration_ms"] >= 0  # type: ignore[index]
    spans = exporter.get_finished_spans()
    assert "invalid-private-response" not in str([span.attributes for span in spans])
    for name in ("rootwalk.model", "rootwalk.run", "rootwalk.graph"):
        span = next(span for span in spans if span.name == name)
        assert span.status.status_code == (
            StatusCode.OK if repaired else StatusCode.ERROR
        )
    store.close()
    provider.shutdown()


@pytest.mark.parametrize(
    ("config", "model_output_tokens", "expected_attempts"),
    [
        (RuntimeConfig(max_model_calls=1), 2_000, 1),
        (
            RuntimeConfig(
                max_model_calls=3,
                max_output_tokens=9,
                max_output_tokens_per_call=5,
            ),
            5,
            1,
        ),
        (RuntimeConfig(max_model_calls=3, max_input_tokens=1), 2_000, 0),
    ],
)
def test_repairs_cannot_bypass_provider_attempt_budgets(
    tmp_path: Path,
    synthetic_public_input: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: RuntimeConfig,
    model_output_tokens: int,
    expected_attempts: int,
) -> None:
    valid = json.dumps(
        {
            "decision_summary": "unused repair",
            "operations": [{"type": "finalize", "abstain_reason": "test"}],
        }
    )
    model = adapter(
        monkeypatch,
        ["invalid-private-response", valid],
        max_output_tokens=model_output_tokens,
    )
    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            RootWalkRuntime(
                BankEnvironment(synthetic_public_input),
                model,
                store,
                ArtifactStore(tmp_path / "artifacts"),
                config,
            ).run("provider-budget", tmp_path / "bundle")
        )

        assert result.status == "budget_exhausted"
        assert result.report["budget"]["provider_attempts_used"] == expected_attempts
        assert len(store.records("model_request")) == expected_attempts
        provider_finishes = [
            record.payload
            for record in store.records("attempt")
            if ":provider:" in record.stable_key
            and record.stable_key.endswith(":finish")
        ]
        assert len(provider_finishes) == expected_attempts
        if provider_finishes:
            assert provider_finishes[0]["status"] == "invalid_response"  # type: ignore[index]


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (asyncio.CancelledError(), "cancelled"),
        (RuntimeError("Authorization: Bearer synthetic-test-key"), "runtime_failure"),
    ],
)
def test_cancelled_and_failed_provider_attempts_are_safe_and_accounted(
    tmp_path: Path,
    synthetic_public_input: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_status: str,
) -> None:
    model = adapter(monkeypatch, [failure], max_output_tokens=7)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            RootWalkRuntime(
                BankEnvironment(synthetic_public_input), model, store, artifacts
            ).run("provider-failure", tmp_path / "bundle")
        )

        assert result.status == expected_status
        assert result.report["budget"]["provider_attempts_used"] == 1
        assert result.report["budget"]["output_tokens_used"] == 7
        requests = store.records("model_request")
        assert len(requests) == 1
        assert requests[0].payload["status"] in {"cancelled", "error"}  # type: ignore[index]
        serialized = json.dumps(
            [record.payload for record in store.records()], sort_keys=True
        ) + "".join(
            path.read_text(errors="ignore")
            for path in result.bundle_path.rglob("*")
            if path.is_file()
        )
        assert "synthetic-test-key" not in serialized
        assert "Authorization: Bearer" not in serialized


@pytest.mark.parametrize(
    ("provider_result", "run_status", "attempt_status"),
    [
        (
            json.dumps(
                {
                    "decision_summary": "provider succeeded",
                    "operations": [{"type": "finalize", "abstain_reason": "test"}],
                }
            ),
            "completed_partial",
            "success",
        ),
        (RuntimeError("provider failed"), "runtime_failure", "error"),
        (asyncio.CancelledError(), "cancelled", "cancelled"),
    ],
)
def test_finished_hook_failure_preserves_provider_outcome_and_usage(
    tmp_path: Path,
    synthetic_public_input: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_result: str | BaseException,
    run_status: str,
    attempt_status: str,
) -> None:
    class FailFinishedAuditOnce(RootWalkRuntime):
        failed = False

        def _provider_attempt_event(
            self, call_number: int, event: str, attempt: Any
        ) -> None:
            if event == "finished" and not self.failed:
                self.failed = True
                raise OSError("injected finish persistence failure")
            super()._provider_attempt_event(call_number, event, attempt)

    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            FailFinishedAuditOnce(
                BankEnvironment(synthetic_public_input),
                adapter(monkeypatch, [provider_result], max_output_tokens=7),
                store,
                ArtifactStore(tmp_path / "artifacts"),
            ).run("finish-hook", tmp_path / "bundle")
        )

        assert result.status == run_status
        finishes = [
            record.payload
            for record in store.records("attempt")
            if ":provider:" in record.stable_key
            and record.stable_key.endswith(":finish")
        ]
        assert len(finishes) == 1
        assert finishes[0]["status"] == attempt_status  # type: ignore[index]
        if attempt_status == "success":
            assert result.report["model_usage"] == {
                "prompt_tokens": 3,
                "completion_tokens": 5,
            }


def test_started_hook_failure_prevents_provider_call_and_keeps_reservation(
    tmp_path: Path,
    synthetic_public_input: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailStartedAudit(RootWalkRuntime):
        def _provider_attempt_event(
            self, call_number: int, event: str, attempt: Any
        ) -> None:
            super()._provider_attempt_event(call_number, event, attempt)
            if event == "started":
                raise OSError("injected start persistence failure")

    model = adapter(monkeypatch, ["must-not-be-consumed"], max_output_tokens=7)
    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            FailStartedAudit(
                BankEnvironment(synthetic_public_input),
                model,
                store,
                ArtifactStore(tmp_path / "artifacts"),
            ).run("start-hook", tmp_path / "bundle")
        )

        assert result.status == "runtime_failure"
        assert model.last_requests == []
        assert result.report["budget"]["provider_attempts_used"] == 1
        assert result.report["budget"]["output_tokens_used"] == 7
        assert not [
            record
            for record in store.records("attempt")
            if record.stable_key.endswith(":finish")
        ]


def test_response_artifact_failure_keeps_provider_usage_and_attempt(
    tmp_path: Path,
    synthetic_public_input: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = json.dumps(
        {
            "decision_summary": "response reached provider",
            "operations": [{"type": "finalize", "abstain_reason": "test"}],
        }
    )
    model = adapter(monkeypatch, [valid])
    artifacts = ArtifactStore(tmp_path / "artifacts")
    original_put_json = artifacts.put_json

    def fail_response(value: object) -> str:
        if isinstance(value, dict) and "choices" in value:
            raise ArtifactBudgetExceeded(limit=1, used=1, requested=1)
        return original_put_json(cast(object, value))

    monkeypatch.setattr(artifacts, "put_json", fail_response)
    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            RootWalkRuntime(
                BankEnvironment(synthetic_public_input), model, store, artifacts
            ).run("artifact-limit", tmp_path / "bundle")
        )

        assert result.status == "budget_exhausted"
        assert result.report["model_usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 5,
        }
        assert result.report["budget"]["provider_attempts_used"] == 1
        assert any(
            record.stable_key.endswith(":finish")
            and record.payload["status"] == "success"  # type: ignore[index]
            for record in store.records("attempt")
            if ":provider:" in record.stable_key
        )


def test_provider_usage_survives_crash_before_decision_commit(
    tmp_path: Path,
    synthetic_public_input: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = json.dumps(
        {
            "decision_summary": "survives restart",
            "operations": [{"type": "finalize", "abstain_reason": "test"}],
        }
    )

    class CrashAfterProvider(RootWalkRuntime):
        def _unpack_response(self, response: object) -> Any:
            del response
            raise KeyboardInterrupt

    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifacts"
    first_store = RunStore(database)
    first_artifacts = ArtifactStore(artifact_root)
    first_runtime = CrashAfterProvider(
        BankEnvironment(synthetic_public_input),
        adapter(monkeypatch, [valid]),
        first_store,
        first_artifacts,
        RuntimeConfig(max_model_calls=2),
    )
    first_runtime._run_id = "provider-recovery"  # pyright: ignore[reportPrivateUsage]
    first_runtime._execution_attempt_id = "first"  # pyright: ignore[reportPrivateUsage]
    first_runtime._started_at = time.time()  # pyright: ignore[reportPrivateUsage]
    first_runtime._deadline_at = time.time() + 60  # pyright: ignore[reportPrivateUsage]
    first_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    input_ref = first_artifacts.put_json([{"role": "user", "content": "decide"}])
    first_runtime._register_embedded_artifact(  # pyright: ignore[reportPrivateUsage]
        input_ref
    )
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(
            first_runtime._call_model(  # pyright: ignore[reportPrivateUsage]
                {
                    "model_call_number": 1,
                    "model_input_ref": input_ref,
                    "estimated_input_tokens": 1,
                }
            )
        )
    first_store.close()

    second_store = RunStore(database)
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(synthetic_public_input),
            adapter(monkeypatch, [valid]),
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=2),
        ).run("provider-recovery", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert result.report["budget"]["provider_attempts_used"] == 2
    assert result.report["model_usage"] == {
        "prompt_tokens": 6,
        "completion_tokens": 10,
    }
    assert len(second_store.records("model_request")) == 2
    second_store.close()


def test_provider_artifacts_and_trace_recursively_redact_credentials(
    tmp_path: Path,
    synthetic_public_input: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "synthetic-test-key"
    other_secret = "other-secret"
    valid = json.dumps(
        {
            "decision_summary": "ordinary token content remains",
            "operations": [{"type": "finalize", "abstain_reason": "test"}],
        }
    )
    response = Reply(
        json.dumps(
            {
                "api_key": other_secret,
                "message": "ordinary token content remains",
            }
        ),
        authorization=f"Bearer {other_secret}",
        nested={
            "api_key": secret,
            "access_token": other_secret,
            "cookie": other_secret,
        },
    )
    model = adapter(monkeypatch, [response, valid])
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(BoundedSpanExporter(exporter, capture_content=True))
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with RunStore(tmp_path / "run.sqlite3") as store:
        result = asyncio.run(
            RootWalkRuntime(
                BankEnvironment(synthetic_public_input),
                model,
                store,
                artifacts,
                tracer=provider.get_tracer("test.secrets"),
                observability_config=ObservabilityConfig(
                    enabled=True, capture_content=True
                ),
                config=RuntimeConfig(
                    model_endpoint=(
                        "https://unused.invalid/v1?token=other-secret&mode=test"
                    )
                ),
            ).run("provider-secrets", tmp_path / "bundle")
        )
        records = json.dumps([record.payload for record in store.records()])
    provider.force_flush(1_000)
    artifact_text = "".join(
        path.read_text(errors="ignore") for path in artifacts.root.iterdir()
    )
    bundle_text = "".join(
        path.read_text(errors="ignore")
        for path in result.bundle_path.rglob("*")
        if path.is_file()
    )
    trace_text = str([span.attributes for span in exporter.get_finished_spans()])

    assert secret not in records + artifact_text + bundle_text + trace_text
    assert other_secret not in records + artifact_text + bundle_text + trace_text
    assert "ordinary token content remains" in artifact_text
    provider.shutdown()


def test_tools_have_elapsed_time_and_safe_trace_metadata(
    tmp_path: Path,
    synthetic_public_input: Path,
) -> None:
    model = RecordedModelAdapter(
        [
            {
                "decision_summary": "Inspect",
                "operations": [
                    {"type": "query", "tool": "summarize_anomalies", "arguments": {}}
                ],
            },
            {
                "decision_summary": "Stop",
                "operations": [{"type": "finalize", "abstain_reason": "test"}],
            },
        ]
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(BoundedSpanExporter(exporter)))
    with RunStore(tmp_path / "run.sqlite3") as store:
        asyncio.run(
            RootWalkRuntime(
                BankEnvironment(synthetic_public_input),
                model,
                store,
                ArtifactStore(tmp_path / "artifacts"),
                tracer=provider.get_tracer("test.audit"),
                observability_config=ObservabilityConfig(enabled=True),
            ).run("tools", tmp_path / "bundle")
        )
        finishes = [
            record.payload
            for record in store.records("attempt")
            if record.stable_key.endswith(":finish")
        ]
        assert finishes and finishes[0]["duration_ms"] >= 0  # type: ignore[index]
        assert finishes[0]["call_number"] == 1  # type: ignore[index]
    span = next(
        span for span in exporter.get_finished_spans() if span.name == "rootwalk.tool"
    )
    assert (span.attributes or {})["tool.name"] == "summarize_anomalies"
    assert (span.attributes or {})["outcome"] == "partial"
    assert span.status.status_code == StatusCode.OK
    provider.shutdown()


def test_cleanup_reports_delivery_failure() -> None:
    class Provider:
        telemetry_dropped_count = 7

        def force_flush(self, timeout: int) -> bool:
            return False

        def shutdown(self) -> None:
            pass

    assert _close_tracer_provider(Provider(), 1) == {
        "flush_completed": False,
        "cleanup_completed": True,
        "telemetry_dropped_count": 7,
    }


def test_stderr_capture_stops_without_waiting_for_pipe_eof() -> None:
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb")
    stop = threading.Event()
    captured: list[bytes] = []
    reader = threading.Thread(
        target=_capture_stderr, args=(stream, captured, stop), daemon=True
    )
    reader.start()
    try:
        os.write(write_fd, b"synthetic startup failure")
        stop.set()
        reader.join(0.5)
        assert not reader.is_alive()
        assert stream.closed
        assert captured == [b"synthetic startup failure"]
    finally:
        os.close(write_fd)
        stop.set()
        reader.join(0.5)
