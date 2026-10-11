import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rootwalk.model import RecordedModelAdapter
from rootwalk.observability import ObservabilityConfig, create_tracer_provider
from rootwalk.openrca import BankEnvironment
from rootwalk.runtime import RootWalkRuntime
from rootwalk.store import ArtifactStore, RunStore


@pytest.mark.parametrize("capture_content", [False, True])
def test_runtime_exports_configured_content_to_local_otlp(
    tmp_path: Path, synthetic_public_input: Path, capture_content: bool
) -> None:
    received: list[tuple[str, str, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            received.append(
                (
                    self.path,
                    self.headers.get("Content-Type", ""),
                    self.rfile.read(length),
                )
            )
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    private = b"ROOTWALK_PRIVATE_OTLP_TEST_BODY"
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        config = ObservabilityConfig(
            enabled=True,
            capture_content=capture_content,
            endpoint=f"http://127.0.0.1:{port}/api/public/otel/v1/traces",
        )
        provider = create_tracer_provider(config)
        store = RunStore(tmp_path / "run.sqlite3")
        try:
            result = asyncio.run(
                RootWalkRuntime(
                    BankEnvironment(synthetic_public_input),
                    RecordedModelAdapter(
                        [
                            {
                                "decision_summary": private.decode(),
                                "operations": [
                                    {
                                        "type": "finalize",
                                        "abstain_reason": private.decode(),
                                    }
                                ],
                            }
                        ]
                    ),
                    store,
                    ArtifactStore(tmp_path / "artifacts"),
                    tracer=provider.get_tracer("test.otlp"),
                    observability_config=config,
                ).run("otlp-loopback", tmp_path / "bundle")
            )
            started = time.monotonic()
            flushed = provider.force_flush(2_000)
            flush_seconds = time.monotonic() - started
        finally:
            store.close()
            provider.shutdown()
            server.shutdown()
            thread.join(timeout=1)

    assert result.status == "completed_partial"
    assert flushed and flush_seconds < 2.5
    assert received
    assert {path for path, _, _ in received} == {"/api/public/otel/v1/traces"}
    assert all(
        content_type == "application/x-protobuf" for _, content_type, _ in received
    )
    assert all(body for _, _, body in received)
    payload = b"".join(body for _, _, body in received)
    assert (private in payload) is capture_content
