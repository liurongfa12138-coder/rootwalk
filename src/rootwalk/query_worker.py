"""Killable process boundary for Bank read-only queries."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from io import BufferedReader
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Literal, cast

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .openrca import load_openrca_environment

QUERY_TOOLS = (
    "list_entities",
    "list_metrics",
    "query_metric",
    "query_metric_profile",
    "query_impact_profile",
    "query_trace_operation_profile",
    "query_service_profile",
    "query_log_samples",
    "summarize_log_patterns",
    "query_trace_neighbors",
    "query_trace_edges",
    "read_artifact_slice",
    "summarize_anomalies",
)
_TRACE_PROPAGATOR = TraceContextTextMapPropagator()
_TRACE_HEADER_NAMES = frozenset({"traceparent", "tracestate"})
SandboxMode = Literal["auto", "required", "off"]
EffectiveSandbox = Literal["bwrap", "spawn"]
_SANDBOX_INPUT = "/public_input"
_STDERR_LIMIT = 4096


def _error(error_type: str, summary: str, *, cancelled: bool = False) -> dict[str, Any]:
    return {
        "status": "cancelled" if cancelled else "error",
        "error_type": error_type,
        "summary": summary,
    }


def _send(connection: Connection, message: Mapping[str, Any]) -> None:
    connection.send(json.dumps(message, allow_nan=False, separators=(",", ":")))


def _load_object(raw: str) -> dict[str, Any]:
    value: object = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return cast(dict[str, Any], value)


def _current_trace_headers() -> dict[str, str]:
    carrier: dict[str, str] = {}
    try:
        _TRACE_PROPAGATOR.inject(carrier)
    except Exception:
        return {}
    return {name: carrier[name] for name in _TRACE_HEADER_NAMES if name in carrier}


def _call_with_trace_context(
    method: Callable[..., dict[str, Any]],
    arguments: dict[str, Any],
    raw_headers: object,
) -> dict[str, Any]:
    raw_mapping = (
        cast(dict[object, object], raw_headers) if isinstance(raw_headers, dict) else {}
    )
    headers = {
        name: value
        for name, value in raw_mapping.items()
        if isinstance(name, str)
        and name in _TRACE_HEADER_NAMES
        and isinstance(value, str)
    }
    try:
        child_context = _TRACE_PROPAGATOR.extract(headers, context=Context())
    except Exception:
        child_context = Context()
    token = otel_context.attach(child_context)
    try:
        return method(**arguments)
    finally:
        otel_context.detach(token)


def _select_bwrap(
    mode: SandboxMode, bwrap_path: str | Path | None = None
) -> str | None:
    if mode not in {"auto", "required", "off"}:
        raise ValueError("sandbox must be auto, required, or off")
    if mode == "off":
        return None
    candidate = str(bwrap_path) if bwrap_path is not None else shutil.which("bwrap")
    available = (
        sys.platform.startswith("linux")
        and candidate is not None
        and Path(candidate).is_file()
        and os.access(candidate, os.X_OK)
    )
    if available:
        return str(Path(cast(str, candidate)).resolve())
    if mode == "required":
        raise RuntimeError("query sandbox is required but bubblewrap is unavailable")
    return None


@lru_cache(maxsize=None)
def _bwrap_version(bwrap: str) -> str:
    try:
        process = subprocess.Popen(
            [bwrap, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise RuntimeError("bubblewrap version check failed") from exc
    try:
        stdout, _stderr = process.communicate(timeout=2)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise RuntimeError("bubblewrap version check failed") from exc
    raw = stdout[:256].decode("ascii", errors="ignore").strip()
    version = raw.splitlines()[0] if raw else ""
    if process.returncode != 0 or not re.fullmatch(
        r"[A-Za-z0-9 ._+()-]{1,128}", version
    ):
        raise RuntimeError("bubblewrap version check failed")
    return version


def required_bwrap_version() -> str:
    return required_bwrap_metadata()[0]


def required_bwrap_metadata() -> tuple[str, str]:
    bwrap = _select_bwrap("required")
    if bwrap is None:  # pragma: no cover - required mode raises first
        raise RuntimeError("query sandbox is required but bubblewrap is unavailable")
    return _bwrap_version(bwrap), _sha256_file(Path(bwrap))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sandbox_bindings(public_input: str | Path) -> list[tuple[Path, str]]:
    base_executable = Path(getattr(sys, "_base_executable", sys.executable))
    package_root = Path(__file__).resolve().parent
    workspace_root = package_root.parents[1]
    forbidden = {
        Path.home(),
        Path.home().resolve(),
        workspace_root,
        workspace_root.resolve(),
    }
    candidates = [
        (Path("/lib"), "/lib", True),
        (Path("/lib64"), "/lib64", True),
        (base_executable.parent.parent, str(base_executable.parent.parent), False),
        (Path(sys.prefix), str(Path(sys.prefix)), False),
        (package_root, "/app/rootwalk", False),
        (Path(public_input), _SANDBOX_INPUT, False),
    ]
    bindings: list[tuple[Path, str]] = []
    destinations: set[str] = set()
    for source, destination, optional in candidates:
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            if optional:
                continue
            raise RuntimeError("query sandbox mount plan is unavailable") from exc
        if resolved == Path("/") or any(
            resolved == sensitive or resolved in sensitive.parents
            for sensitive in forbidden
        ):
            raise RuntimeError("query sandbox mount plan is unsafe")
        if destination in destinations:
            continue
        destinations.add(destination)
        bindings.append((source, destination))
    return bindings


def _sandbox_command(
    bwrap: str, public_input: str | Path, command: Sequence[str]
) -> list[str]:
    mount_args = [
        argument
        for source, destination in _sandbox_bindings(public_input)
        for argument in ("--ro-bind", str(source), destination)
    ]
    return [
        bwrap,
        "--unshare-user",
        "--uid",
        "65534",
        "--gid",
        "65534",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--hostname",
        "rootwalk-query",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--setenv",
        "HOME",
        "/nonexistent",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--setenv",
        "PYTHONPATH",
        "/app",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        *mount_args,
        "--chdir",
        "/tmp",
        "--",
        *command,
    ]


def _probe_sandbox(bwrap: str, public_input: str | Path) -> None:
    probe = (
        "import os; from pathlib import Path; root=Path('/public_input'); "
        "paths=[root,*root.rglob('*')]; "
        "ok=all(os.access(p,os.R_OK|(os.X_OK if p.is_dir() else 0)) "
        "for p in paths); raise SystemExit(0 if ok else 1)"
    )
    try:
        process = subprocess.Popen(
            _sandbox_command(bwrap, public_input, [sys.executable, "-c", probe]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise RuntimeError("query sandbox preflight failed") from exc
    try:
        returncode = process.wait(timeout=10)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise RuntimeError("query sandbox preflight failed") from exc
    if returncode != 0:
        raise RuntimeError("query sandbox preflight failed")


def _capture_stderr(
    stream: BufferedReader, target: list[bytes], stop: threading.Event
) -> None:
    captured = bytearray()
    try:
        descriptor = stream.fileno()
        os.set_blocking(descriptor, False)
        while True:
            stopping = stop.is_set()
            ready, _, _ = select.select([descriptor], [], [], 0 if stopping else 0.05)
            if not ready:
                if stopping:
                    break
                continue
            try:
                chunk = os.read(descriptor, 4096)
            except BlockingIOError:
                continue
            if not chunk:
                break
            if len(captured) < _STDERR_LIMIT:
                captured.extend(chunk[: _STDERR_LIMIT - len(captured)])
            if stopping:
                break
    except OSError:
        pass
    finally:
        target.append(bytes(captured))
        # Only the reader closes its stream. BufferedReader.close() in another
        # thread can wait forever when a descendant keeps the pipe open.
        stream.close()


def _safe_startup_error(stderr: bytes) -> dict[str, Any]:
    lowered = stderr.decode("utf-8", errors="ignore").lower()
    if "operation not permitted" in lowered:
        reason = "sandbox namespace setup was not permitted"
    elif "permission denied" in lowered:
        reason = "sandbox runtime was not readable"
    elif "no such file" in lowered or "not found" in lowered:
        reason = "sandbox runtime binding was unavailable"
    else:
        reason = "sandbox process exited before initialization"
    digest = hashlib.sha256(stderr).hexdigest()[:12] if stderr else "none"
    return _error("worker_startup_error", f"{reason}; stderr_digest={digest}")


def _worker_main(public_input: str, connection: Connection) -> None:
    try:
        environment = load_openrca_environment(public_input)
        tools: dict[str, Callable[..., dict[str, Any]]] = {
            "list_entities": environment.list_entities,
            "list_metrics": environment.list_metrics,
            "query_metric": environment.query_metric,
            "query_metric_profile": environment.query_metric_profile,
            "query_impact_profile": environment.query_impact_profile,
            "query_trace_operation_profile": environment.query_trace_operation_profile,
            "query_service_profile": environment.query_service_profile,
            "query_log_samples": environment.query_log_samples,
            "summarize_log_patterns": environment.summarize_log_patterns,
            "query_trace_neighbors": environment.query_trace_neighbors,
            "query_trace_edges": environment.query_trace_edges,
            "read_artifact_slice": environment.read_artifact_slice,
            "summarize_anomalies": environment.summarize_anomalies,
        }
    except Exception:
        _send(
            connection,
            {
                "kind": "startup_error",
                "result": _error(
                    "worker_startup_error", "query worker could not initialize"
                ),
            },
        )
        connection.close()
        return

    try:
        while True:
            raw_request = connection.recv()
            if not isinstance(raw_request, str):
                continue
            request = _load_object(raw_request)
            if request.get("kind") == "close":
                return

            request_id = request.get("request_id")
            tool = request.get("tool")
            arguments = request.get("arguments")
            if not isinstance(request_id, int):
                continue
            if (
                not isinstance(tool, str)
                or tool not in tools
                or not isinstance(arguments, dict)
            ):
                _send(
                    connection,
                    {
                        "kind": "result",
                        "request_id": request_id,
                        "result": _error("invalid_request", "invalid query request"),
                    },
                )
                continue
            try:
                result = _call_with_trace_context(
                    tools[tool],
                    cast(dict[str, Any], arguments),
                    request.get("_trace_context"),
                )
                # Round-trip before crossing the boundary: only JSON data leaves.
                result = _load_object(json.dumps(result, allow_nan=False))
            except TypeError:
                result = _error("invalid_arguments", "invalid query arguments")
            except Exception:
                result = _error("query_execution_error", "query execution failed")
            _send(
                connection,
                {"kind": "result", "request_id": request_id, "result": result},
            )
    except (EOFError, BrokenPipeError, OSError, json.JSONDecodeError):
        return
    finally:
        connection.close()


class QueryWorker:
    """Run the seven Bank query tools serially in one killable child process."""

    def __init__(
        self,
        public_input: str | Path,
        *,
        sandbox: SandboxMode = "auto",
        bwrap_path: str | Path | None = None,
    ) -> None:
        self._public_input = str(Path(public_input).resolve(strict=True))
        self._sandbox_requested: SandboxMode = sandbox
        self._bwrap = _select_bwrap(sandbox, bwrap_path)
        self._bwrap_version: str | None = None
        self._bwrap_executable_sha256: str | None = None
        if self._bwrap is not None:
            try:
                self._bwrap_version = _bwrap_version(self._bwrap)
                self._bwrap_executable_sha256 = _sha256_file(Path(self._bwrap))
                _probe_sandbox(self._bwrap, self._public_input)
            except RuntimeError:
                if sandbox == "required":
                    raise RuntimeError(
                        "query sandbox is required but could not be configured"
                    ) from None
                self._bwrap = None
                self._bwrap_version = None
                self._bwrap_executable_sha256 = None
                warnings.warn(
                    "query sandbox unavailable; falling back to spawn",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif sandbox == "auto":
            warnings.warn(
                "query sandbox unavailable; falling back to spawn",
                RuntimeWarning,
                stacklevel=2,
            )
        self._context = mp.get_context("spawn")
        self._process: BaseProcess | subprocess.Popen[bytes] | None = None
        self._connection: Connection | None = None
        self._stderr_stream: BufferedReader | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop: threading.Event | None = None
        self._stderr_capture: list[bytes] = []
        self._worker_ready = False
        self._next_request_id = 1
        self._closed = False
        self._lock = threading.Lock()
        self._state_lock = threading.RLock()

    @property
    def sandbox_requested(self) -> SandboxMode:
        return self._sandbox_requested

    @property
    def sandbox_effective(self) -> EffectiveSandbox:
        return "bwrap" if self._bwrap is not None else "spawn"

    @property
    def bwrap_version(self) -> str | None:
        return self._bwrap_version

    @property
    def bwrap_executable_sha256(self) -> str | None:
        return self._bwrap_executable_sha256

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            process = self._process
            return (
                process.pid if process is not None and self._is_alive(process) else None
            )

    def execute(
        self, tool: str, args: Mapping[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        """Execute one JSON query, terminating the child when its deadline expires."""
        with self._lock:
            if closed := self._closed_result():
                return closed
            if tool not in QUERY_TOOLS:
                return _error("unknown_tool", "unknown query tool")
            if (
                isinstance(timeout_s, bool)
                or not math.isfinite(timeout_s)
                or timeout_s <= 0
            ):
                return _error(
                    "invalid_timeout", "timeout_s must be positive and finite"
                )
            try:
                normalized_args = _load_object(
                    json.dumps(args, allow_nan=False, separators=(",", ":"))
                )
            except (TypeError, ValueError):
                return _error("invalid_arguments", "arguments must be a JSON object")

            request_id = self._next_request_id
            self._next_request_id += 1
            with self._state_lock:
                if self._closed:
                    return _error("worker_closed", "query worker is closed")
                try:
                    self._ensure_worker()
                except (OSError, RuntimeError):
                    return _error("worker_start_error", "query worker could not start")
                connection = self._connection
            if connection is None:
                return _error("worker_unavailable", "query worker is unavailable")
            request: dict[str, Any] = {
                "kind": "execute",
                "request_id": request_id,
                "tool": tool,
                "arguments": normalized_args,
            }
            trace_headers = _current_trace_headers()
            if trace_headers:
                request["_trace_context"] = trace_headers
            try:
                _send(connection, request)
                if not connection.poll(float(timeout_s)):
                    closed = self._closed_result()
                    self._stop_worker()
                    if closed is not None:
                        return closed
                    return _error(
                        "query_timeout",
                        "query deadline exceeded and worker was terminated",
                        cancelled=True,
                    )
                raw_response = connection.recv()
                if not isinstance(raw_response, str):
                    raise ValueError("non-JSON worker response")
                response = _load_object(raw_response)
            except (
                BrokenPipeError,
                EOFError,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ):
                closed = self._closed_result()
                with self._state_lock:
                    startup = not self._worker_ready
                stderr = self._stop_worker()
                if closed is not None:
                    return closed
                if startup:
                    return _safe_startup_error(stderr)
                return _error("worker_failure", "query worker exited unexpectedly")

            if response.get("kind") == "startup_error":
                return _safe_startup_error(self._stop_worker())
            if (
                response.get("kind") != "result"
                or response.get("request_id") != request_id
                or not isinstance(response.get("result"), dict)
            ):
                self._stop_worker()
                return _error("worker_protocol_error", "invalid query worker response")
            with self._state_lock:
                self._worker_ready = True
            return cast(dict[str, Any], response["result"])

    def close(self) -> None:
        """Stop the child process; repeated calls are harmless."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._stop_worker()

    def _closed_result(self) -> dict[str, Any] | None:
        with self._state_lock:
            if not self._closed:
                return None
        return _error("worker_closed", "query worker is closed", cancelled=True)

    def _ensure_worker(self) -> None:
        with self._state_lock:
            if self._process is not None and self._is_alive(self._process):
                return
            self._stop_worker()
            parent, child = self._context.Pipe(duplex=True)
            try:
                if self._bwrap is None:
                    stderr_stream: BufferedReader | None = None
                    stderr_thread: threading.Thread | None = None
                    stderr_capture: list[bytes] = []
                    stderr_stop: threading.Event | None = None
                    process: BaseProcess | subprocess.Popen[bytes] = (
                        self._context.Process(
                            target=_worker_main,
                            args=(self._public_input, child),
                            name="rootwalk-query-worker",
                            daemon=True,
                        )
                    )
                    process.start()
                else:
                    stderr_capture = []
                    child_command = [
                        sys.executable,
                        "-m",
                        "rootwalk.query_worker",
                        "--worker",
                        _SANDBOX_INPUT,
                        str(child.fileno()),
                    ]
                    process = subprocess.Popen(
                        _sandbox_command(
                            self._bwrap, self._public_input, child_command
                        ),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        close_fds=True,
                        pass_fds=(child.fileno(),),
                    )
                    stderr_stream = cast(BufferedReader, process.stderr)
                    stderr_stop = threading.Event()
                    stderr_thread = threading.Thread(
                        target=_capture_stderr,
                        args=(stderr_stream, stderr_capture, stderr_stop),
                        name="rootwalk-query-worker-stderr",
                        daemon=True,
                    )
                    stderr_thread.start()
            except (OSError, RuntimeError):
                parent.close()
                child.close()
                raise
            child.close()
            self._connection = parent
            self._process = process
            self._stderr_stream = stderr_stream
            self._stderr_thread = stderr_thread
            self._stderr_stop = stderr_stop
            self._stderr_capture = stderr_capture
            self._worker_ready = False

    def _stop_worker(self) -> bytes:
        with self._state_lock:
            connection, process = self._connection, self._process
            stderr_stream = self._stderr_stream
            stderr_thread = self._stderr_thread
            stderr_stop = self._stderr_stop
            stderr_capture = self._stderr_capture
            self._connection = None
            self._process = None
            self._stderr_stream = None
            self._stderr_thread = None
            self._stderr_stop = None
            self._stderr_capture = []
            self._worker_ready = False
        if process is not None:
            if self._is_alive(process):
                process.terminate()
            try:
                self._wait(process, 1.0)
            except subprocess.TimeoutExpired:
                pass
            if self._is_alive(process):
                process.kill()
                try:
                    self._wait(process, 1.0)
                except subprocess.TimeoutExpired:
                    pass
            if isinstance(process, BaseProcess) and not self._is_alive(process):
                process.close()
        if connection is not None:
            connection.close()
        if stderr_thread is not None:
            if stderr_stop is not None:
                stderr_stop.set()
            stderr_thread.join(0.2)
        elif stderr_stream is not None:
            stderr_stream.close()
        return b"".join(stderr_capture)[:_STDERR_LIMIT]

    @staticmethod
    def _is_alive(process: BaseProcess | subprocess.Popen[bytes]) -> bool:
        return (
            process.is_alive()
            if isinstance(process, BaseProcess)
            else process.poll() is None
        )

    @staticmethod
    def _wait(process: BaseProcess | subprocess.Popen[bytes], timeout: float) -> None:
        if isinstance(process, BaseProcess):
            process.join(timeout)
        else:
            process.wait(timeout)


def _child_entrypoint() -> None:
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit(2)
    try:
        descriptor = int(sys.argv[3])
    except ValueError as exc:
        raise SystemExit(2) from exc
    _worker_main(sys.argv[2], Connection(descriptor))


if __name__ == "__main__":
    _child_entrypoint()
