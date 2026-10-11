import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from opentelemetry import trace
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    TraceState,
)

from rootwalk import query_worker
from rootwalk.query_worker import QueryWorker

PUBLIC_INPUT = (
    Path(__file__).resolve().parents[2]
    / "rootwalk-handoff"
    / "rootwalk-development-handoff"
    / "rootwalk-data-prep"
    / "slices"
    / "bank-row-0001"
    / "public_input"
)


@pytest.mark.openrca_bank
def test_real_metric_query_returns_19_samples() -> None:
    worker = QueryWorker(PUBLIC_INPUT)
    try:
        result = worker.execute(
            "query_metric",
            {
                "entity_id": "bank:entity:Tomcat01",
                "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                "period": "incident",
            },
            10.0,
        )
        assert result["status"] == "success"
        assert result["sample_count"] == 19
        assert "request_id" not in result
    finally:
        worker.close()


def test_consecutive_calls_reuse_one_serial_worker() -> None:
    worker = QueryWorker(PUBLIC_INPUT)
    try:
        first = worker.execute(
            "list_metrics", {"entity_id": "bank:entity:Tomcat01"}, 10
        )
        pid = worker.pid
        second = worker.execute("list_entities", {}, 10)

        assert first["status"] == second["status"] == "success"
        assert pid is not None
        assert worker.pid == pid
        assert second["count"] == 4
    finally:
        worker.close()
        worker.close()


def test_unknown_tool_is_rejected_without_reaching_the_child() -> None:
    worker = QueryWorker(PUBLIC_INPUT)
    try:
        result = worker.execute("open", {"path": "/etc/passwd"}, 1)

        assert result == {
            "status": "error",
            "error_type": "unknown_tool",
            "summary": "unknown query tool",
        }
        assert worker.pid is None
    finally:
        worker.close()


def test_optional_profile_tools_reach_the_worker() -> None:
    worker = QueryWorker(PUBLIC_INPUT)
    try:
        impact = worker.execute("query_impact_profile", {}, 10)
        trace = worker.execute(
            "query_trace_operation_profile",
            {"entity_id": "bank:entity:Tomcat01", "limit": 5},
            10,
        )

        assert impact.get("error_type") != "unknown_tool"
        assert trace.get("error_type") != "unknown_tool"
    finally:
        worker.close()


def test_timeout_terminates_worker_and_next_call_restarts_it() -> None:
    worker = QueryWorker(PUBLIC_INPUT)
    try:
        timed_out = worker.execute("list_entities", {}, 1e-9)

        assert timed_out["status"] == "cancelled"
        assert timed_out["error_type"] == "query_timeout"
        assert worker.pid is None

        restarted = worker.execute("list_entities", {}, 10)
        assert restarted["status"] == "success"
        assert restarted["count"] == 4
        assert worker.pid is not None
    finally:
        worker.close()


class _BlockingConnection:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.released = threading.Event()

    def send(self, message: object) -> None:
        del message
        self.started.set()

    def poll(self, timeout: float) -> bool:
        self.released.wait(timeout)
        return False

    def close(self) -> None:
        self.released.set()


class _BlockingProcess:
    pid = 123

    def __init__(self) -> None:
        self.alive = True

    def poll(self) -> int | None:
        return None if self.alive else 0

    def terminate(self) -> None:
        self.alive = False

    def wait(self, timeout: float) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        self.alive = False


def test_close_interrupts_active_execute_without_waiting_for_its_timeout() -> None:
    worker = QueryWorker(PUBLIC_INPUT, sandbox="off")
    connection = _BlockingConnection()
    process = _BlockingProcess()
    worker._connection = cast(Any, connection)  # pyright: ignore[reportPrivateUsage]
    worker._process = cast(Any, process)  # pyright: ignore[reportPrivateUsage]
    result: dict[str, Any] = {}

    thread = threading.Thread(
        target=lambda: result.update(worker.execute("list_entities", {}, 30))
    )
    thread.start()
    assert connection.started.wait(1)
    started = time.monotonic()
    worker.close()
    elapsed = time.monotonic() - started
    thread.join(1)

    assert elapsed < 1
    assert not thread.is_alive()
    assert process.alive is False
    assert result == {
        "status": "cancelled",
        "error_type": "worker_closed",
        "summary": "query worker is closed",
    }
    assert worker.execute("list_entities", {}, 1) == result


def test_child_argument_error_does_not_hang_or_kill_worker() -> None:
    worker = QueryWorker(PUBLIC_INPUT)
    try:
        invalid = worker.execute("list_entities", {"unexpected": True}, 10)
        pid = worker.pid
        valid = worker.execute("list_entities", {}, 10)

        assert invalid["error_type"] == "invalid_arguments"
        assert valid["status"] == "success"
        assert worker.pid == pid
    finally:
        worker.close()


def test_anomaly_entity_ids_cross_worker_as_a_json_list() -> None:
    worker = QueryWorker(PUBLIC_INPUT)
    try:
        result = worker.execute(
            "summarize_anomalies",
            {"entity_ids": ["bank:entity:Tomcat01"], "limit": 2},
            10,
        )

        assert result["status"] in {"success", "partial"}
        assert result["signals"]
        assert all(
            signal["entity_id"] == "bank:entity:Tomcat01"
            for signal in result["signals"]
        )
    finally:
        worker.close()


def test_trace_context_is_restored_around_child_tool_call() -> None:
    parent = SpanContext(
        trace_id=int("1" * 32, 16),
        span_id=int("2" * 16, 16),
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    observed: list[SpanContext] = []

    def tool() -> dict[str, object]:
        observed.append(trace.get_current_span().get_span_context())
        return {"status": "success"}

    with trace.use_span(NonRecordingSpan(parent), end_on_exit=False):
        headers = query_worker._current_trace_headers()  # pyright: ignore[reportPrivateUsage]
    result = query_worker._call_with_trace_context(  # pyright: ignore[reportPrivateUsage]
        tool, {}, headers
    )

    assert headers == {"traceparent": f"00-{'1' * 32}-{'2' * 16}-01"}
    assert result == {"status": "success"}
    assert len(observed) == 1
    assert observed[0].trace_id == parent.trace_id
    assert observed[0].span_id == parent.span_id
    assert observed[0].is_remote is True


def test_missing_trace_context_is_fail_open() -> None:
    observed: list[SpanContext] = []

    def tool() -> dict[str, object]:
        observed.append(trace.get_current_span().get_span_context())
        return {"status": "success"}

    assert query_worker._current_trace_headers() == {}  # pyright: ignore[reportPrivateUsage]
    result = query_worker._call_with_trace_context(  # pyright: ignore[reportPrivateUsage]
        tool, {}, None
    )

    assert result == {"status": "success"}
    assert len(observed) == 1
    assert observed[0].is_valid is False


def test_spawn_trace_headers_do_not_leak_into_result_or_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "rootwalk-private-trace-state"
    parent = SpanContext(
        trace_id=int("3" * 32, 16),
        span_id=int("4" * 16, 16),
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=TraceState([("vendor", secret)]),
    )
    worker = QueryWorker(PUBLIC_INPUT)
    try:
        with trace.use_span(NonRecordingSpan(parent), end_on_exit=False):
            headers = query_worker._current_trace_headers()  # pyright: ignore[reportPrivateUsage]
            result = worker.execute("list_entities", {}, 10)
    finally:
        worker.close()

    serialized = json.dumps(result, sort_keys=True)
    assert headers["tracestate"] == f"vendor={secret}"
    assert result["status"] == "success"
    assert "traceparent" not in serialized
    assert "tracestate" not in serialized
    assert secret not in serialized
    assert secret not in caplog.text


def test_sandbox_required_fails_closed_without_bwrap(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="bubblewrap is unavailable"):
        QueryWorker(
            PUBLIC_INPUT,
            sandbox="required",
            bwrap_path=tmp_path / "missing-bwrap",
        )


def test_auto_sandbox_warns_when_it_falls_back_to_spawn(tmp_path: Path) -> None:
    missing = tmp_path / "missing-bwrap"
    with pytest.warns(RuntimeWarning) as warnings:
        worker = QueryWorker(PUBLIC_INPUT, sandbox="auto", bwrap_path=missing)
    try:
        assert worker.sandbox_requested == "auto"
        assert worker.sandbox_effective == "spawn"
        assert worker.bwrap_version is None
        assert worker.bwrap_executable_sha256 is None
        message = str(warnings[0].message)
        assert message == "query sandbox unavailable; falling back to spawn"
        assert str(tmp_path) not in message
    finally:
        worker.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bwrap") is None,
    reason="mount-plan validation requires Linux and bwrap",
)
def test_required_sandbox_rejects_root_sized_runtime_mounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(query_worker.sys, "prefix", "/")
    monkeypatch.setattr(query_worker.sys, "_base_executable", "/bin/python3")

    with pytest.raises(RuntimeError, match="could not be configured"):
        QueryWorker(
            PUBLIC_INPUT,
            sandbox="required",
            bwrap_path=cast(str, shutil.which("bwrap")),
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bwrap") is None,
    reason="mount-plan validation requires Linux and bwrap",
)
@pytest.mark.parametrize(
    "base_executable",
    [
        Path.home() / "bin/python3",
        Path(query_worker.__file__).resolve().parents[4] / "bin/python3",
    ],
)
def test_required_sandbox_rejects_sensitive_ancestor_mounts(
    monkeypatch: pytest.MonkeyPatch,
    base_executable: Path,
) -> None:
    monkeypatch.setattr(query_worker.sys, "_base_executable", str(base_executable))

    with pytest.raises(RuntimeError, match="could not be configured"):
        QueryWorker(
            PUBLIC_INPUT,
            sandbox="required",
            bwrap_path=cast(str, shutil.which("bwrap")),
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bwrap") is None,
    reason="sandbox readability validation requires Linux and bwrap",
)
def test_required_sandbox_rejects_input_unreadable_by_sandbox_uid(
    tmp_path: Path,
) -> None:
    public_input = tmp_path / "public_input"
    public_input.mkdir()
    unreadable = public_input / "manifest.json"
    unreadable.write_text("{}", encoding="utf-8")
    unreadable.chmod(0)

    try:
        with pytest.raises(RuntimeError, match="could not be configured"):
            QueryWorker(
                public_input,
                sandbox="required",
                bwrap_path=cast(str, shutil.which("bwrap")),
            )
    finally:
        unreadable.chmod(0o600)


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bwrap") is None,
    reason="startup stderr capture requires Linux and bwrap",
)
def test_sandbox_startup_error_is_bounded_and_redacted(tmp_path: Path) -> None:
    real_bwrap = cast(str, shutil.which("bwrap"))
    wrapper = tmp_path / "bwrap"
    wrapper.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "bubblewrap test-wrapper"; exit 0; fi\n'
        'case "$*" in\n'
        "  *rootwalk.query_worker*) "
        'echo "ROOTWALK_SUPER_SECRET /data00/private: Permission denied" >&2; '
        "exit 1;;\n"
        f'  *) exec {shlex.quote(real_bwrap)} "$@";;\n'
        "esac\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    worker = QueryWorker(PUBLIC_INPUT, sandbox="required", bwrap_path=wrapper)
    try:
        result = worker.execute("list_entities", {}, 10)
    finally:
        worker.close()

    assert result["status"] == "error"
    assert result["error_type"] == "worker_startup_error"
    assert "sandbox runtime was not readable" in result["summary"]
    assert len(result["summary"]) < 160
    assert "ROOTWALK_SUPER_SECRET" not in result["summary"]
    assert "/data00/private" not in result["summary"]


def test_sandbox_off_keeps_portable_spawn_worker() -> None:
    worker = QueryWorker(PUBLIC_INPUT, sandbox="off")
    try:
        result = worker.execute("list_entities", {}, 10)
    finally:
        worker.close()

    assert result["status"] == "success"
    assert result["count"] == 4


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bwrap") is None,
    reason="real bubblewrap integration requires Linux and bwrap",
)
def test_real_bwrap_hides_hidden_eval_and_allows_public_queries(
    tmp_path: Path,
) -> None:
    hidden_eval = tmp_path / "hidden_eval"
    hidden_eval.mkdir()
    canary = hidden_eval / "answer.txt"
    canary.write_text("private-answer", encoding="utf-8")
    script = (
        "import json,os,sys; from pathlib import Path; "
        "from rootwalk.openrca import BankEnvironment; "
        "result=BankEnvironment('/public_input').list_entities(); "
        "print(json.dumps({'canary':Path(sys.argv[1]).exists(),"
        "'credential':os.getenv('ROOTWALK_TEST_CREDENTIAL'),"
        "'status':result['status'],'count':result['count']}))"
    )
    bwrap = cast(str, shutil.which("bwrap"))
    command = query_worker._sandbox_command(  # pyright: ignore[reportPrivateUsage]
        bwrap,
        PUBLIC_INPUT,
        [sys.executable, "-c", script, str(canary)],
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "ROOTWALK_TEST_CREDENTIAL": "must-not-cross"},
    )

    assert json.loads(completed.stdout) == {
        "canary": False,
        "credential": None,
        "status": "success",
        "count": 4,
    }
    worker = QueryWorker(PUBLIC_INPUT, sandbox="required", bwrap_path=bwrap)
    try:
        assert worker.sandbox_requested == "required"
        assert worker.sandbox_effective == "bwrap"
        assert worker.bwrap_version
        assert worker.bwrap_executable_sha256
        assert len(worker.bwrap_executable_sha256) == 64
        assert worker.execute("list_entities", {}, 10)["status"] == "success"
    finally:
        worker.close()
