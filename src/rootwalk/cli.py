import asyncio
import csv
import fcntl
import hashlib
import io
import json
import os
import platform
import re
import threading
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import typer

from .context import (
    ContextStrategy,
    ValidatorMode,
    candidate_disposition_queue,
    has_strict_directional_metric_shift,
)
from .contracts import Evidence, Hypothesis, Observation, Task, TopologySnapshot
from .diff import diff_run_bundles
from .environment import RuntimeEnvironment
from .evaluate import evaluate_files
from .inspection import inspect_bundle, inspect_many, render_inspection
from .model import (
    ModelAdapter,
    ModelAdapterError,
    RecordedJSON,
    RecordedModelAdapter,
)
from .observability import ObservabilityConfig, create_tracer_provider
from .occurrence import (
    OccurrenceAtom,
    build_occurrence_groups,
    matching_occurrence_group,
)
from .openrca import (
    dominant_market_fault_type,
    load_openrca_environment,
    market_output_component,
    mechanism_signature_rank,
)
from .query_worker import QueryWorker, SandboxMode
from .runtime import (
    RootWalkRuntime,
    RuntimeConfig,
    select_family_compatible_terminal_candidate,
    select_terminal_candidate,
    supported_candidate_binding,
)
from .store import ArtifactStore, RunStore, export_self_contained_bundle
from .version import TERMINAL_POLICY_VERSION, installed_source_revision

app = typer.Typer(help="Rootwalk command-line tools.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Rootwalk command-line tools."""


def _config_exists() -> bool:
    configured_path = os.environ.get("ROOTWALK_CONFIG")
    path = (
        Path(configured_path).expanduser()
        if configured_path
        else Path.home() / ".config" / "rootwalk" / "config.toml"
    )
    return path.is_file()


@contextmanager
def _work_directory(path: Path) -> Generator[tuple[Path, int], None, None]:
    if path.is_symlink():
        raise ValueError("work directory must not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("work directory must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("work directory could not be pinned") from exc
    try:
        path_stat = os.stat(path, follow_symlinks=False)
        descriptor_stat = os.fstat(descriptor)
        if (path_stat.st_dev, path_stat.st_ino) != (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ):
            raise ValueError("work directory changed during initialization")
        pinned = Path("/proc/self/fd") / str(descriptor)
        if not pinned.is_dir():
            raise ValueError("work directory pin is unavailable")
        yield pinned, descriptor
    finally:
        os.close(descriptor)


def _close_tracer_provider(
    provider: Any, cleanup_grace_seconds: int
) -> dict[str, object]:
    result: dict[str, object] = {"flush_completed": False, "cleanup_completed": False}
    if cleanup_grace_seconds <= 0:
        return {**result, "telemetry_dropped_count": provider.telemetry_dropped_count}

    def close() -> None:
        try:
            result["flush_completed"] = provider.force_flush(
                cleanup_grace_seconds * 1_000
            )
        except Exception:
            pass
        try:
            provider.shutdown()
        except Exception:
            pass

    thread = threading.Thread(target=close, daemon=True)
    thread.start()
    thread.join(cleanup_grace_seconds)
    return {
        **result,
        "cleanup_completed": not thread.is_alive(),
        "telemetry_dropped_count": getattr(provider, "telemetry_dropped_count", 0),
    }


@app.command()
def doctor() -> None:
    """Report the Python version and whether configuration exists."""
    typer.echo(f"Python: {platform.python_version()}")
    typer.echo(f"Config: {'present' if _config_exists() else 'missing'}")


def _live_model_configuration(
    model_provider: str,
    model: str | None,
    base_url: str | None,
) -> tuple[str, str, str]:
    provider = model_provider.strip().lower()
    if provider != "ark":
        raise ValueError("--model-provider must be ark")
    model_env = "ARK_MODEL"
    base_url_env = "ARK_BASE_URL"

    resolved_model = model or os.environ.get(model_env)
    resolved_base_url = base_url or os.environ.get(base_url_env)
    if not resolved_model or not resolved_model.strip():
        raise ValueError(f"provide --model or set {model_env}")
    if not resolved_base_url or not resolved_base_url.strip():
        raise ValueError(f"provide --base-url or set {base_url_env}")
    return provider, resolved_model, resolved_base_url


def _live_model_adapter(
    *,
    model_provider: str,
    model: str,
    base_url: str,
    timeout_s: float,
    max_output_tokens: int,
    reasoning_effort: Any,
) -> ModelAdapter:
    if model_provider != "ark":
        raise ValueError("--model-provider must be ark")
    return ModelAdapter(
        model=model,
        base_url=base_url,
        timeout_s=timeout_s,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )


@app.command("probe-model")
def probe_model(
    model: str | None = typer.Option(
        None, "--model", help="Provider model ID; provider-specific env is supported."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Ark provider base URL; ARK_BASE_URL is also supported.",
    ),
    model_provider: str = typer.Option(
        "ark",
        "--model-provider",
        envvar="ROOTWALK_MODEL_PROVIDER",
        help="Live model provider: ark.",
    ),
    timeout_s: float = typer.Option(12.0, min=0.1),
    max_output_tokens: int = typer.Option(2_000, min=1),
    reasoning_effort: str | None = typer.Option(None),
) -> None:
    """Send one synthetic, schema-validated request to a live model provider."""
    messages = [
        {
            "role": "user",
            "content": (
                "Return one RootWalk query operation for list_entities. "
                "Use base_revision 0 and do not include prose outside JSON."
            ),
        }
    ]
    try:
        provider, resolved_model, resolved_base_url = _live_model_configuration(
            model_provider, model, base_url
        )
        result = asyncio.run(
            _live_model_adapter(
                model_provider=provider,
                model=resolved_model,
                base_url=resolved_base_url,
                timeout_s=timeout_s,
                max_output_tokens=max_output_tokens,
                reasoning_effort=_reasoning_effort(reasoning_effort),
            ).decide(messages)
        )
    except (ModelAdapterError, ValueError) as exc:
        typer.echo(f"Model probe failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        json.dumps(
            {
                "status": "success",
                "model_provider": provider,
                "provider_model": result.provider_model,
                "usage": result.usage,
                "operation_types": [op.type for op in result.decision.operations],
            },
            sort_keys=True,
        )
    )


@app.command()
def run(
    data_root: Path = typer.Option(..., file_okay=False),
    output: Path = typer.Option(...),
    run_id: str = typer.Option(...),
    recorded_decisions: Path | None = typer.Option(None, dir_okay=False),
    model: str | None = typer.Option(
        None, "--model", help="Provider model ID; provider-specific env is supported."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Ark provider base URL; ARK_BASE_URL is also supported.",
    ),
    model_provider: str = typer.Option(
        "ark",
        "--model-provider",
        envvar="ROOTWALK_MODEL_PROVIDER",
        help="Live model provider: ark.",
    ),
    model_timeout_s: float = typer.Option(60.0, min=0.1),
    wall_clock_seconds: int | None = typer.Option(None, min=1),
    max_output_tokens: int = typer.Option(2_000, min=1),
    max_model_calls: int = typer.Option(12, min=1),
    max_input_tokens: int = typer.Option(144_000, min=1),
    max_tool_attempts: int = typer.Option(40, min=0),
    max_validator_retries: int = typer.Option(2, min=0),
    max_no_progress_rounds: int = typer.Option(3, min=1),
    reasoning_effort: str | None = typer.Option(None),
    context_strategy: str = typer.Option("evidence"),
    validator_mode: str = typer.Option("enforce"),
    query_sandbox: str = typer.Option(
        "auto",
        "--query-sandbox",
        help="Query worker sandbox: auto, required, or off.",
    ),
    task: Path | None = typer.Option(None, dir_okay=False),
    otel: bool = typer.Option(
        False, "--otel/--no-otel", envvar="ROOTWALK_OTEL_ENABLED"
    ),
    otel_endpoint: str | None = typer.Option(
        None, "--otel-endpoint", envvar="OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ),
    otel_capture_content: bool = typer.Option(
        False,
        "--otel-capture-content/--no-otel-capture-content",
        envvar="ROOTWALK_OTEL_CAPTURE_CONTENT",
    ),
    otel_sample_ratio: float = typer.Option(
        1.0,
        "--otel-sample-ratio",
        min=0.0,
        max=1.0,
        envvar="ROOTWALK_OTEL_SAMPLE_RATIO",
    ),
) -> None:
    """Run one Bank investigation and write a portable bundle."""
    try:
        if context_strategy not in {"recent", "evidence", "frontier_reflection"}:
            raise ValueError(
                "--context-strategy must be recent, evidence, or frontier_reflection"
            )
        if validator_mode not in {"observe_only", "enforce"}:
            raise ValueError("--validator-mode must be observe_only or enforce")
        if query_sandbox not in {"auto", "required", "off"}:
            raise ValueError("--query-sandbox must be auto, required, or off")
        validated_context_strategy = cast(ContextStrategy, context_strategy)
        validated_validator_mode = cast(ValidatorMode, validator_mode)
        validated_query_sandbox = cast(SandboxMode, query_sandbox)
        frontier_mode = validated_context_strategy == "frontier_reflection"
        effective_wall_clock = wall_clock_seconds or (900 if frontier_mode else 300)
        effective_output_tokens = (
            max(max_output_tokens, 4_000) if frontier_mode else max_output_tokens
        )
        effective_context_tokens = 24_000 if frontier_mode else 12_000
        effective_tool_attempts = (
            max(max_tool_attempts, 200) if frontier_mode else max_tool_attempts
        )
        effective_validator_retries = (
            max(max_validator_retries, 5) if frontier_mode else max_validator_retries
        )
        effective_reasoning_effort = (
            reasoning_effort or "max" if frontier_mode else reasoning_effort
        )
        root = data_root.resolve(strict=True)
        output = output.resolve()
        expected_task = (root / "task.json").resolve(strict=True)
        if task is not None and task.resolve(strict=True) != expected_task:
            raise ValueError("--task must be data-root/task.json")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            try:
                _read_bundle(output)
            except Exception as exc:
                raise ValueError(
                    "--output exists but is not an empty directory or complete bundle"
                ) from exc
            raise ValueError(
                "--output already contains a complete bundle; use rootwalk replay"
            )

        if recorded_decisions is not None:
            if (
                model is not None
                or base_url is not None
                or model_provider.strip().lower() != "ark"
            ):
                raise ValueError(
                    "use --recorded-decisions or live model provider options, not both"
                )
            records = json.loads(recorded_decisions.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError("--recorded-decisions must contain a JSON array")
            adapter = RecordedModelAdapter(cast(list[RecordedJSON], records))
            runtime_config = RuntimeConfig(
                context_strategy=validated_context_strategy,
                validator_mode=validated_validator_mode,
                model_provider="recorded",
                model_id="recorded",
                model_timeout_seconds=None,
                wall_clock_seconds=effective_wall_clock,
                max_context_tokens=effective_context_tokens,
                max_output_tokens_per_call=effective_output_tokens,
                max_model_calls=max_model_calls,
                max_input_tokens=max_input_tokens,
                max_tool_attempts=effective_tool_attempts,
                max_validator_retries=effective_validator_retries,
                max_no_progress_rounds=max_no_progress_rounds,
            )
        else:
            provider, resolved_model, resolved_base_url = _live_model_configuration(
                model_provider, model, base_url
            )
            adapter = _live_model_adapter(
                model_provider=provider,
                model=resolved_model,
                base_url=resolved_base_url,
                timeout_s=model_timeout_s,
                max_output_tokens=effective_output_tokens,
                reasoning_effort=_reasoning_effort(effective_reasoning_effort),
            )
            runtime_config = RuntimeConfig(
                context_strategy=validated_context_strategy,
                validator_mode=validated_validator_mode,
                model_provider=provider,
                model_id=resolved_model,
                model_endpoint=resolved_base_url,
                model_timeout_seconds=model_timeout_s,
                wall_clock_seconds=effective_wall_clock,
                max_context_tokens=effective_context_tokens,
                max_output_tokens_per_call=effective_output_tokens,
                max_model_calls=max_model_calls,
                max_input_tokens=max_input_tokens,
                max_tool_attempts=effective_tool_attempts,
                max_validator_retries=effective_validator_retries,
                max_no_progress_rounds=max_no_progress_rounds,
                reasoning_effort=effective_reasoning_effort,
            )

        environment: RuntimeEnvironment = load_openrca_environment(root)
        observability_config = ObservabilityConfig(
            enabled=otel,
            endpoint=otel_endpoint if otel else None,
            capture_content=otel_capture_content if otel else False,
            sample_ratio=otel_sample_ratio if otel else 1.0,
        )
        work = output.with_name(f".{output.name}.work")
        with _work_directory(work) as (work_root, work_fd):
            lock_descriptor = os.open(
                "run.lock",
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=work_fd,
            )
            with os.fdopen(lock_descriptor, "a+") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(
                        f"work directory is locked by another run: {work}"
                    ) from exc
                with RunStore(work_root / "run.sqlite3") as store:
                    for manifest in store.records("manifest"):
                        payload = manifest.payload
                        owner = (
                            cast(dict[object, object], payload).get("run_id")
                            if isinstance(payload, dict)
                            else None
                        )
                        if owner != run_id:
                            raise ValueError(
                                "work directory belongs to a different run_id; "
                                f"refusing to reuse {work}"
                            )
                    worker = QueryWorker(root, sandbox=validated_query_sandbox)
                    runtime_config = replace(
                        runtime_config,
                        query_sandbox_requested=worker.sandbox_requested,
                        query_sandbox_effective=worker.sandbox_effective,
                        query_sandbox_bwrap_version=worker.bwrap_version,
                        query_sandbox_bwrap_executable_sha256=(
                            worker.bwrap_executable_sha256
                        ),
                    )
                    provider = (
                        create_tracer_provider(observability_config) if otel else None
                    )
                    telemetry: dict[str, object] | None = None
                    try:
                        runtime = RootWalkRuntime(
                            environment,
                            adapter,
                            store,
                            ArtifactStore(work_root / "artifacts"),
                            runtime_config,
                            query_executor=worker,
                            tracer=(
                                provider.get_tracer("rootwalk.runtime")
                                if provider is not None
                                else None
                            ),
                            observability_config=observability_config,
                        )
                        result = asyncio.run(runtime.run(run_id, output))
                    finally:
                        worker.close()
                        if provider is not None:
                            telemetry = _close_tracer_provider(
                                provider, runtime_config.cleanup_grace_seconds
                            )
                            # This late audit is outside the sealed report and
                            # remains available even if the exporter is down.
                            try:
                                store.append(
                                    "late_audit",
                                    f"telemetry_delivery:{uuid.uuid4().hex}",
                                    {
                                        "audit_type": "telemetry_delivery",
                                        "run_id": run_id,
                                        **telemetry,
                                    },
                                )
                            except Exception:
                                pass
    except Exception as exc:
        typer.echo(f"Run failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        json.dumps(
            {
                "bundle": str(result.bundle_path),
                "model_calls": result.model_calls,
                "run_id": result.run_id,
                "status": result.status,
                "telemetry": telemetry,
            },
            sort_keys=True,
        )
    )


@app.command()
def replay(bundle: Path = typer.Argument(...)) -> None:
    """Validate a bundle and print its saved report summary without a model."""
    try:
        _, report, _, _ = _read_bundle(bundle)
        raw_findings = report.get("findings", [])
        raw_paths = report.get("paths", [])
        if not isinstance(raw_findings, list) or not isinstance(raw_paths, list):
            raise ValueError("report findings and paths must be arrays")
        findings = cast(list[object], raw_findings)
        paths = cast(list[object], raw_paths)
        summary = {
            "finding_count": len(findings),
            "model_calls": report.get("model_calls"),
            "path_count": len(paths),
            "run_id": report.get("run_id"),
            "status": report.get("status"),
        }
    except Exception as exc:
        typer.echo(f"Replay failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, sort_keys=True))


@app.command("inspect")
def inspect_investigation(
    bundle: Path = typer.Argument(..., file_okay=False),
    call: int | None = typer.Option(None, min=1, help="Show one saved model round."),
    format: str = typer.Option("json", help="json or markdown"),
    output: Path | None = typer.Option(None),
) -> None:
    """Inspect choices, effects, context allocation, and budgets without a model."""
    try:
        if format not in {"json", "markdown"}:
            raise ValueError("--format must be json or markdown")
        if output is not None and output.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("inspection output must be outside the sealed bundle")
        result = inspect_bundle(bundle, call_number=call)
        content = (
            render_inspection(result)
            if format == "markdown"
            else json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        if output is None:
            typer.echo(content, nl=False)
        else:
            output.write_text(content, encoding="utf-8")
    except Exception as exc:
        typer.echo(f"Inspect failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("inspect-many")
def inspect_investigations(
    bundles: list[Path] = typer.Argument(..., file_okay=False),
    output: Path | None = typer.Option(None),
) -> None:
    """Aggregate saved inspection facts across comparable bundles."""
    try:
        content = (
            json.dumps(
                inspect_many(bundles), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n"
        )
        if output is None:
            typer.echo(content, nl=False)
        else:
            resolved = output.resolve()
            if any(resolved.is_relative_to(bundle.resolve()) for bundle in bundles):
                raise ValueError(
                    "inspection output must be outside every sealed bundle"
                )
            output.write_text(content, encoding="utf-8")
    except Exception as exc:
        typer.echo(f"Inspect failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("export")
def export_bundle(
    bundle: Path = typer.Argument(...),
    format: str | None = typer.Option(None, "--format"),
    output: Path | None = typer.Option(None),
    self_contained: bool = typer.Option(False, "--self-contained"),
    external_root: Path | None = typer.Option(None, "--external-root"),
) -> None:
    """Export predictions, or explicitly build a self-contained RunBundle."""
    try:
        if self_contained:
            if format is not None:
                raise ValueError("--format cannot be combined with --self-contained")
            if output is None or external_root is None:
                raise ValueError(
                    "--self-contained requires --output and --external-root"
                )
            exported = export_self_contained_bundle(bundle, output, external_root)
            manifest = json.loads((exported / "manifest.json").read_text())
            typer.echo(
                json.dumps(
                    {
                        "bundle": str(exported),
                        "bundle_size_bytes": manifest["bundle_size_bytes"],
                        "external_artifacts_embedded": manifest[
                            "external_artifacts_embedded"
                        ],
                    },
                    sort_keys=True,
                )
            )
            return
        if external_root is not None:
            raise ValueError("--external-root requires --self-contained")
        if format != "openrca":
            raise ValueError("--format must be openrca")
        bundle_root, report, task_payload, topology_payload = _read_bundle(bundle)
        task = Task.model_validate_json(json.dumps(task_payload))
        topology = TopologySnapshot.model_validate_json(json.dumps(topology_payload))
        if task.source_row_id is None:
            raise ValueError("bundle task has no source_row_id")
        raw_findings = report.get("findings")
        if not isinstance(raw_findings, list):
            raise ValueError("report findings must be an array")
        findings = cast(list[object], raw_findings)
        original_ids = {
            node.canonical_id: (
                market_output_component(node.original_id)
                if task.environment == "openrca_market"
                else node.original_id
            )
            for node in topology.nodes
        }
        prediction: dict[str, dict[str, str]] = {}
        for index, raw_finding in enumerate(findings, 1):
            finding = Hypothesis.model_validate(raw_finding)
            if finding.subject not in original_ids:
                raise ValueError(
                    f"finding subject is absent from topology: {finding.subject}"
                )
            item: dict[str, str] = {}
            if "root cause occurrence datetime" in task.requested_outputs:
                item["root cause occurrence datetime"] = _finding_occurrence_datetime(
                    bundle_root, finding, task
                )
            if "root cause component" in task.requested_outputs:
                item["root cause component"] = original_ids[finding.subject]
            if "root cause reason" in task.requested_outputs:
                item["root cause reason"] = (
                    _market_finding_reason(bundle_root, finding)
                    if task.environment
                    in {"openrca_bank", "openrca_market", "openrca_telecom"}
                    else finding.statement
                )
            prediction[str(index)] = item
        if not findings:
            fallback: dict[str, Any] | None = None
            occurrence = _validated_occurrence_recommendation(bundle_root, report, task)
            if occurrence is not None:
                prediction["1"] = {
                    "root cause occurrence datetime": occurrence["occurrence_datetime"]
                }
                typer.echo(
                    json.dumps(
                        {
                            "event": (
                                "low_confidence_occurrence_recommendation_exported"
                            ),
                            "run_id": report.get("run_id"),
                            "policy": occurrence["policy"],
                            "confidence": "low",
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    err=True,
                )
            else:
                fallback = _validated_benchmark_recommendation(
                    bundle_root, report, task, original_ids
                )
            if occurrence is None and fallback is not None:
                fallback_item: dict[str, str] = {}
                if "root cause occurrence datetime" in task.requested_outputs:
                    fallback_item["root cause occurrence datetime"] = cast(
                        str, fallback.get("occurrence_datetime", "")
                    )
                if "root cause component" in task.requested_outputs:
                    fallback_item["root cause component"] = original_ids[
                        fallback["subject"]
                    ]
                if "root cause reason" in task.requested_outputs:
                    fallback_item["root cause reason"] = fallback["statement"]
                prediction["1"] = fallback_item
                marker = {
                    "event": "low_confidence_benchmark_recommendation_exported",
                    "run_id": report.get("run_id"),
                    "policy": fallback["policy"],
                    "candidate_id": fallback["candidate_id"],
                    "confidence": "low",
                }
                for name in ("hypothesis_id", "rejected_candidate_id"):
                    if name in fallback:
                        marker[name] = fallback[name]
                typer.echo(
                    json.dumps(marker, separators=(",", ":"), sort_keys=True),
                    err=True,
                )

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream, fieldnames=["row_id", "prediction"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(
            {
                "row_id": task.source_row_id,
                "prediction": json.dumps(
                    prediction, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
        content = stream.getvalue()
        if output is None:
            typer.echo(content, nl=False)
        else:
            output.write_text(content, encoding="utf-8")
    except Exception as exc:
        typer.echo(f"Export failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("diff")
def diff_bundles(
    left: Path = typer.Argument(...),
    right: Path = typer.Argument(...),
    output: Path | None = typer.Option(None),
) -> None:
    """Compare two verified RunBundles and locate their first divergence."""
    try:
        content = json.dumps(
            diff_run_bundles(left, right),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if output is None:
            typer.echo(content)
        else:
            output.write_text(f"{content}\n", encoding="utf-8")
    except Exception as exc:
        typer.echo(f"Diff failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("evaluate")
def evaluate_predictions(
    predictions: Path = typer.Option(..., dir_okay=False),
    labels: Path = typer.Option(...),
    output: Path | None = typer.Option(None),
) -> None:
    """Score predictions in an isolated process that can see hidden labels."""
    try:
        content = json.dumps(
            evaluate_files(predictions, labels),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if output is None:
            typer.echo(content)
        else:
            output.write_text(f"{content}\n", encoding="utf-8")
    except Exception as exc:
        typer.echo(f"Evaluate failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _read_bundle(
    bundle: Path,
) -> tuple[Path, dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    root = bundle.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("bundle must be a directory")
    manifest = cast(
        object,
        json.loads((root / "manifest.json").read_text(encoding="utf-8")),
    )
    if not isinstance(manifest, dict):
        raise ValueError("bundle manifest has no file_hashes object")
    manifest_object = cast(dict[object, object], manifest)
    raw_file_hashes = manifest_object.get("file_hashes")
    if not isinstance(raw_file_hashes, dict):
        raise ValueError("bundle manifest has no file_hashes object")
    file_hashes = cast(dict[object, object], raw_file_hashes)
    for relative, expected_hash in file_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("bundle manifest contains an invalid file hash entry")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("bundle manifest contains an invalid file hash entry")
        path = (root / candidate).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("bundle manifest file escapes its root") from exc
        if not path.is_file():
            raise ValueError(f"bundle file is not regular: {relative}")
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"bundle file hash mismatch: {relative}")
    expected_size = manifest_object.get("bundle_size_bytes")
    if expected_size is not None:
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ValueError("bundle manifest contains an invalid bundle_size_bytes")
        total_size = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("bundle with bundle_size_bytes contains a symlink")
            if path.is_file():
                total_size += path.stat().st_size
        if total_size != expected_size:
            raise ValueError(
                f"bundle size mismatch: expected={expected_size}, actual={total_size}"
            )

    report = cast(
        object, json.loads((root / "report.json").read_text(encoding="utf-8"))
    )
    task = cast(object, json.loads((root / "task.json").read_text(encoding="utf-8")))
    topology = cast(
        object, json.loads((root / "topology.json").read_text(encoding="utf-8"))
    )
    if not all(isinstance(item, dict) for item in (report, task, topology)):
        raise ValueError("bundle report, task, and topology must be JSON objects")
    return (
        root,
        cast(dict[str, Any], report),
        cast(dict[str, Any], task),
        cast(dict[str, Any], topology),
    )


def _validated_occurrence_recommendation(
    bundle: Path,
    report: Mapping[str, Any],
    task: Task,
) -> dict[str, str] | None:
    raw_recommendation = report.get("occurrence_recommendation")
    if raw_recommendation is None:
        return None
    if not isinstance(raw_recommendation, Mapping):
        raise ValueError("occurrence recommendation must be an object")
    recommendation = cast(Mapping[str, object], raw_recommendation)
    expected_keys = {
        "schema_version",
        "policy",
        "confidence",
        "is_finding",
        "subject_id",
        "evidence_refs",
        "onset_interval",
        "occurrence_datetime",
        "audit_ref",
    }
    raw_refs = recommendation.get("evidence_refs")
    refs = (
        cast(list[str], raw_refs)
        if isinstance(raw_refs, list)
        and raw_refs
        and all(isinstance(item, str) and item for item in cast(list[object], raw_refs))
        else []
    )
    if (
        set(recommendation) != expected_keys
        or recommendation.get("schema_version") != "1"
        or recommendation.get("policy") != "source-backed-occurrence-v1"
        or recommendation.get("confidence") != "low"
        or recommendation.get("is_finding") is not False
        or task.requested_outputs != ["root cause occurrence datetime"]
        or report.get("status") != "completed_partial"
        or report.get("outcome_type") != "occurrence_recommendation"
        or report.get("findings") != []
        or report.get("hypothesis_ids") not in (None, [])
        or report.get("path_ids") not in (None, [])
        or report.get("occurrence_evidence_refs") != refs
        or report.get("abstain_reason") is not None
        or len(refs) > 8
        or len(refs) != len(set(refs))
    ):
        raise ValueError("occurrence recommendation has invalid fields")
    evidence_by_id = {
        str(item["evidence_id"]): item
        for item in _bundle_record_payloads(bundle, "evidence.jsonl")
        if isinstance(item.get("evidence_id"), str)
    }
    if set(refs) - evidence_by_id.keys():
        raise ValueError("occurrence recommendation Evidence is absent")
    impact_probe_attempted = any(
        item.get("tool") == "query_impact_profile"
        for item in _bundle_record_payloads(bundle, "observations.jsonl")
    )
    if not impact_probe_attempted:
        raise ValueError("occurrence recommendation impact probe is absent")
    occurrence_atoms: list[OccurrenceAtom] = []
    for item in evidence_by_id.values():
        raw_subjects = item.get("subject_ids")
        raw_onset = item.get("onset_interval")
        if (
            item.get("semantic_quality") != "typed"
            or not isinstance(raw_subjects, list)
            or not raw_subjects
            or not isinstance(raw_subjects[0], str)
            or not isinstance(raw_onset, Mapping)
        ):
            continue
        onset_values = cast(Mapping[str, object], raw_onset)
        if not isinstance(onset_values.get("start"), str) or not isinstance(
            onset_values.get("end"), str
        ):
            continue
        try:
            group_start = datetime.fromisoformat(cast(str, onset_values["start"]))
            group_end = datetime.fromisoformat(cast(str, onset_values["end"]))
        except ValueError:
            continue
        raw_value = item.get("value")
        summary = (
            cast(Mapping[str, object], raw_value).get("finding_gate_summary")
            if isinstance(raw_value, Mapping)
            else None
        )
        values: Mapping[str, object] = (
            cast(Mapping[str, object], summary) if isinstance(summary, Mapping) else {}
        )
        occurrence_atoms.append(
            OccurrenceAtom(
                evidence_id=cast(str, item["evidence_id"]),
                subject_id=raw_subjects[0],
                start=group_start,
                end=group_end,
                fault_type_ready=values.get("fault_type_ready") is True,
                material_event_ready=(
                    values.get("material_event_ready") is True
                    and isinstance(values.get("mechanism_signature"), str)
                ),
            )
        )
    selected_group = matching_occurrence_group(
        build_occurrence_groups(occurrence_atoms), refs
    )
    if selected_group is None:
        raise ValueError("occurrence recommendation Evidence is not one eligible group")
    selected = [evidence_by_id[evidence_id] for evidence_id in refs]
    subject = recommendation.get("subject_id")
    if not isinstance(subject, str) or not subject:
        raise ValueError("occurrence recommendation subject is invalid")
    intervals: list[tuple[datetime, datetime]] = []
    for item in selected:
        raw_subjects = item.get("subject_ids")
        raw_onset = item.get("onset_interval")
        if (
            item.get("semantic_quality") != "typed"
            or not isinstance(raw_subjects, list)
            or subject not in cast(list[object], raw_subjects)
            or not isinstance(raw_onset, Mapping)
        ):
            raise ValueError("occurrence recommendation Evidence is ineligible")
        onset_values = cast(Mapping[str, object], raw_onset)
        if not isinstance(onset_values.get("start"), str) or not isinstance(
            onset_values.get("end"), str
        ):
            raise ValueError("occurrence recommendation Evidence is ineligible")
        try:
            start = datetime.fromisoformat(cast(str, onset_values["start"]))
            end = datetime.fromisoformat(cast(str, onset_values["end"]))
        except ValueError as exc:
            raise ValueError("occurrence Evidence onset is invalid") from exc
        if not (
            start < end
            and task.incident_range.start <= start
            and end <= task.incident_range.end
        ):
            raise ValueError("occurrence Evidence onset is outside the incident")
        intervals.append((start, end))
    overlap_start = max(start for start, _end in intervals)
    overlap_end = min(end for _start, end in intervals)
    if overlap_start >= overlap_end:
        raise ValueError("occurrence Evidence onset intervals do not overlap")
    if len(selected) == 1:
        raw_value = selected[0].get("value")
        summary = (
            cast(Mapping[str, object], raw_value).get("finding_gate_summary")
            if isinstance(raw_value, Mapping)
            else None
        )
        if not isinstance(summary, Mapping) or (
            cast(Mapping[str, object], summary).get("fault_type_ready") is not True
        ):
            raise ValueError(
                "single occurrence Evidence must be a typed fault-ready event"
            )
    midpoint = overlap_start + (overlap_end - overlap_start) / 2
    expected_interval = {
        "start": overlap_start.isoformat(),
        "end": overlap_end.isoformat(),
        "end_exclusive": True,
    }
    if (
        recommendation.get("onset_interval") != expected_interval
        or recommendation.get("occurrence_datetime") != midpoint.isoformat()
    ):
        raise ValueError("occurrence recommendation onset audit is invalid")
    audit_ref = recommendation.get("audit_ref")
    events = [
        event
        for stable_key, event in _investigation_event_records(bundle)
        if stable_key == audit_ref
        and event.get("event_type") == "occurrence_recommendation_selected"
        and event.get("recommendation") == recommendation
    ]
    if len(events) != 1:
        raise ValueError("occurrence recommendation audit event is absent")
    if _bundle_frontier_policy_version(bundle) < 44:
        raise ValueError("occurrence recommendation requires frontier policy v44")
    return {
        "policy": "source-backed-occurrence-v1",
        "occurrence_datetime": midpoint.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _validated_benchmark_recommendation(
    bundle: Path,
    report: Mapping[str, Any],
    task: Task,
    original_ids: Mapping[str, str],
) -> dict[str, Any] | None:
    if report.get("status") != "completed_partial":
        return None
    recommendation = report.get("benchmark_recommendation")
    if recommendation is None:
        return None
    if (
        task.expected_fault_count != 1
        or not isinstance(report.get("abstain_reason"), str)
        or not cast(str, report["abstain_reason"]).strip()
        or report.get("hypothesis_ids") not in (None, [])
        or report.get("findings") != []
        or report.get("path_ids") not in (None, [])
        or report.get("paths") not in (None, [])
        or not isinstance(recommendation, Mapping)
    ):
        raise ValueError("benchmark recommendation is ineligible for export")
    candidate = cast(Mapping[str, object], recommendation)
    if candidate.get("policy") == "frontier-supported-v1":
        return _validated_frontier_recommendation(
            bundle, report, task, candidate, original_ids
        )
    _context_policy_version, terminal_policy_version = _bundle_policy_provenance(bundle)
    current_policy = bool(
        terminal_policy_version is not None and terminal_policy_version >= 6
    )
    if candidate.get("policy") == "initial-top1-v1":
        if current_policy:
            raise ValueError("v81 bundle cannot use initial-top1-v1")
        return _validated_initial_top1_recommendation(
            bundle, report, candidate, original_ids
        )
    if candidate.get("policy") == "rejected-supported-filtered-log-v1":
        return _validated_rejected_supported_recommendation(
            bundle, report, candidate, original_ids
        )
    expected_keys = {
        "schema_version",
        "policy",
        "confidence",
        "usage",
        "is_finding",
        "candidate_id",
        "hypothesis",
        "selection",
        "eligible_terminal_count",
        "audit_ref",
    }
    raw_hypothesis = candidate.get("hypothesis")
    raw_selection = candidate.get("selection")
    hypothesis = (
        Hypothesis.model_validate(raw_hypothesis)
        if isinstance(raw_hypothesis, Mapping)
        else None
    )
    selection: Mapping[str, object] = (
        cast(Mapping[str, object], raw_selection)
        if isinstance(raw_selection, Mapping)
        else cast(Mapping[str, object], {})
    )
    hypothesis_id = hypothesis.hypothesis_id if hypothesis is not None else None
    candidate_id = candidate.get("candidate_id")
    subject = hypothesis.subject if hypothesis is not None else None
    statement = hypothesis.statement if hypothesis is not None else None
    policy = candidate.get("policy")
    terminal_with_log_state = policy in {
        "terminal-candidate-v2",
        "terminal-candidate-v3",
        "terminal-candidate-v4",
        "terminal-candidate-v5",
        "terminal-candidate-v6",
    }
    rank_key = (
        "terminal_queue_rank"
        if policy
        in {
            "terminal-candidate-v3",
            "terminal-candidate-v4",
            "terminal-candidate-v5",
            "terminal-candidate-v6",
        }
        else "initial_queue_rank"
    )
    expected_selection_keys = {
        "inconclusive",
        "source_strict",
        "ever_open",
        "initial_queue_rank",
    }
    if policy in {
        "terminal-candidate-v3",
        "terminal-candidate-v4",
        "terminal-candidate-v5",
        "terminal-candidate-v6",
    }:
        expected_selection_keys.remove("initial_queue_rank")
        expected_selection_keys.add("terminal_queue_rank")
    if terminal_with_log_state:
        expected_selection_keys.add("has_log_source")
    if policy in {
        "terminal-candidate-v4",
        "terminal-candidate-v5",
        "terminal-candidate-v6",
    }:
        expected_selection_keys.add("direct_strict")
    if policy == "terminal-candidate-v5":
        expected_selection_keys.update(
            {
                "supported_subject_match",
                "unique_supported_terminal_subject",
                "base_candidate_id",
                "base_hypothesis_id",
                "overlap_supported_hypothesis_ids",
            }
        )
    if policy == "terminal-candidate-v6":
        expected_selection_keys.update(
            {
                "known_mismatch_filter_applied",
                "pre_family_policy",
                "base_candidate_id",
                "base_hypothesis_id",
                "base_statement",
                "base_resource_family",
                "excluded_candidate_ids",
            }
        )
    queue_rank = selection.get(rank_key)
    eligible_terminal_count = candidate.get("eligible_terminal_count")
    audit_ref = candidate.get("audit_ref")
    if (
        set(candidate) != expected_keys
        or candidate.get("schema_version") != "1"
        or policy
        not in {
            "terminal-candidate-v1",
            "terminal-candidate-v2",
            "terminal-candidate-v3",
            "terminal-candidate-v4",
            "terminal-candidate-v5",
            "terminal-candidate-v6",
        }
        or candidate.get("confidence") != "low"
        or candidate.get("usage") != "openrca_export_fallback_only"
        or candidate.get("is_finding") is not False
        or hypothesis is None
        or hypothesis.status != "inconclusive"
        or not isinstance(hypothesis_id, str)
        or not hypothesis_id
        or not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(subject, str)
        or subject not in original_ids
        or not isinstance(statement, str)
        or not statement
        or set(selection) != expected_selection_keys
        or selection.get("inconclusive") is not True
        or not isinstance(selection.get("source_strict"), bool)
        or terminal_with_log_state
        and not isinstance(selection.get("has_log_source"), bool)
        or policy
        in {"terminal-candidate-v4", "terminal-candidate-v5", "terminal-candidate-v6"}
        and not isinstance(selection.get("direct_strict"), bool)
        or policy == "terminal-candidate-v5"
        and selection.get("supported_subject_match") is not True
        or policy == "terminal-candidate-v5"
        and not isinstance(selection.get("unique_supported_terminal_subject"), str)
        or policy == "terminal-candidate-v5"
        and not isinstance(selection.get("base_candidate_id"), str)
        or policy == "terminal-candidate-v5"
        and not isinstance(selection.get("base_hypothesis_id"), str)
        or policy == "terminal-candidate-v5"
        and (
            not isinstance(selection.get("overlap_supported_hypothesis_ids"), list)
            or not selection.get("overlap_supported_hypothesis_ids")
            or not all(
                isinstance(item, str) and item
                for item in cast(
                    list[object], selection["overlap_supported_hypothesis_ids"]
                )
            )
        )
        or policy == "terminal-candidate-v6"
        and selection.get("known_mismatch_filter_applied") is not True
        or policy == "terminal-candidate-v6"
        and selection.get("pre_family_policy")
        not in {"terminal-candidate-v4", "terminal-candidate-v5"}
        or policy == "terminal-candidate-v6"
        and not isinstance(selection.get("base_candidate_id"), str)
        or policy == "terminal-candidate-v6"
        and not isinstance(selection.get("base_hypothesis_id"), str)
        or policy == "terminal-candidate-v6"
        and not isinstance(selection.get("base_statement"), str)
        or policy == "terminal-candidate-v6"
        and not isinstance(selection.get("base_resource_family"), str)
        or policy == "terminal-candidate-v6"
        and (
            not isinstance(selection.get("excluded_candidate_ids"), list)
            or not selection.get("excluded_candidate_ids")
            or not all(
                isinstance(item, str) and item
                for item in cast(list[object], selection["excluded_candidate_ids"])
            )
        )
        or not isinstance(selection.get("ever_open"), bool)
        or not isinstance(queue_rank, int)
        or isinstance(queue_rank, bool)
        or queue_rank < 0
        or not isinstance(eligible_terminal_count, int)
        or isinstance(eligible_terminal_count, bool)
        or eligible_terminal_count < 1
        or not isinstance(audit_ref, str)
        or not audit_ref
        or policy == "terminal-candidate-v6"
        and (terminal_policy_version is None or terminal_policy_version < 6)
        or current_policy
        and policy
        not in {
            "terminal-candidate-v4",
            "terminal-candidate-v5",
            "terminal-candidate-v6",
        }
    ):
        raise ValueError("benchmark recommendation has invalid fields")

    event_records = _investigation_event_records(bundle)
    recommendation_events = [
        event
        for stable_key, event in event_records
        if stable_key == audit_ref
        if event.get("event_type") == "benchmark_recommendation_selected"
        and event.get("recommendation") == recommendation
    ]
    binding_events = [
        event
        for _stable_key, event in event_records
        if event.get("event_type") == "candidate_disposition_applied"
        and event.get("hypothesis_id") == hypothesis_id
        and event.get("candidate_id") == candidate_id
        and event.get("subject") == subject
        and event.get("selected_fault_type") == statement
        and event.get("status") == hypothesis.status
    ]
    v4_binding_events = [
        event
        for _stable_key, event in event_records
        if event.get("event_type") == "candidate_disposition_applied"
        and event.get("hypothesis_id") == hypothesis_id
    ]
    binding_valid = (
        bool(v4_binding_events)
        and all(
            event.get("candidate_id") == candidate_id
            and event.get("subject") == subject
            and event.get("selected_fault_type") == statement
            and event.get("status") in {"open", "inconclusive"}
            for event in v4_binding_events
        )
        and v4_binding_events[-1].get("status") == hypothesis.status
        if policy
        in {"terminal-candidate-v4", "terminal-candidate-v5", "terminal-candidate-v6"}
        else len(binding_events) == 1
    )
    if len(recommendation_events) != 1 or not binding_valid:
        raise ValueError(
            "benchmark recommendation does not match its audit and disposition records"
        )
    report_candidates = report.get("candidates")
    matching_candidates: list[Mapping[str, object]] = (
        [
            cast(Mapping[str, object], item)
            for item in cast(list[object], report_candidates)
            if isinstance(item, Mapping)
            and cast(Mapping[str, object], item).get("hypothesis")
            == hypothesis.model_dump(mode="json")
        ]
        if isinstance(report_candidates, list)
        else []
    )
    if policy in {"terminal-candidate-v5", "terminal-candidate-v6"} or (
        current_policy and policy == "terminal-candidate-v4"
    ):
        _validate_terminal_v5_or_v6_recommendation(
            bundle,
            report,
            event_records,
            candidate,
            hypothesis,
            enforce_expected_policy=current_policy,
        )
    if len(matching_candidates) != 1:
        raise ValueError("benchmark recommendation does not match report candidates")
    return {
        "subject": subject,
        "statement": statement,
        "policy": policy,
        "hypothesis_id": hypothesis_id,
        "candidate_id": candidate_id,
    }


def _validated_frontier_recommendation(
    bundle: Path,
    report: Mapping[str, Any],
    task: Task,
    recommendation: Mapping[str, object],
    original_ids: Mapping[str, str],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "policy",
        "confidence",
        "usage",
        "is_finding",
        "candidate_id",
        "hypothesis",
        "selection",
        "audit_ref",
    }
    if (
        set(recommendation) != expected_keys
        or recommendation.get("schema_version") != "1"
        or recommendation.get("policy") != "frontier-supported-v1"
        or recommendation.get("confidence") != "low"
        or recommendation.get("usage") != "openrca_export_fallback_only"
        or recommendation.get("is_finding") is not False
    ):
        raise ValueError("frontier benchmark recommendation has invalid fields")
    frontier_policy = _bundle_frontier_policy_version(bundle)
    raw_manifest: object = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    terminal_policy = (
        cast(Mapping[str, object], raw_manifest).get("terminal_policy_version")
        if isinstance(raw_manifest, Mapping)
        else None
    )
    if (
        frontier_policy < 16
        or not isinstance(terminal_policy, int)
        or isinstance(terminal_policy, bool)
        or terminal_policy < 26
        or terminal_policy > TERMINAL_POLICY_VERSION
    ):
        raise ValueError(
            "frontier benchmark recommendation requires terminal p26 or newer"
        )
    raw_hypothesis = recommendation.get("hypothesis")
    raw_selection = recommendation.get("selection")
    if not isinstance(raw_hypothesis, Mapping) or not isinstance(
        raw_selection, Mapping
    ):
        raise ValueError("frontier benchmark recommendation is malformed")
    hypothesis = Hypothesis.model_validate(raw_hypothesis)
    selection = cast(Mapping[str, object], raw_selection)
    hypothesis_id = hypothesis.hypothesis_id
    if (
        hypothesis.status not in {"open", "supported"}
        or hypothesis.subject not in original_ids
        or report.get("current_best_hypothesis_id") != hypothesis_id
        or recommendation.get("candidate_id") != f"frontier:{hypothesis_id}"
    ):
        raise ValueError("frontier benchmark recommendation hypothesis is invalid")
    expected_selection_keys = {
        "current_best_hypothesis_id",
        "supporting_evidence_id",
        "public_fault_type",
        "mechanism_signature",
        "mechanism_rank",
        "root_cause_score",
        "unresolved_anomaly_ids",
    }
    if (
        set(selection) != expected_selection_keys
        or selection.get("current_best_hypothesis_id") != hypothesis_id
    ):
        raise ValueError("frontier benchmark recommendation selection is invalid")
    evidence_id = selection.get("supporting_evidence_id")
    fault_type = selection.get("public_fault_type")
    signature = selection.get("mechanism_signature")
    rank = selection.get("mechanism_rank")
    raw_score = selection.get("root_cause_score")
    if (
        not isinstance(evidence_id, str)
        or not isinstance(fault_type, str)
        or not isinstance(signature, str)
        or not isinstance(rank, int)
        or isinstance(rank, bool)
        or not isinstance(raw_score, (int, float))
        or isinstance(raw_score, bool)
    ):
        raise ValueError("frontier benchmark recommendation selection is malformed")
    evidence_payloads = _bundle_record_payloads(bundle, "evidence.jsonl")
    evidence_by_id = {
        str(item["evidence_id"]): item
        for item in evidence_payloads
        if isinstance(item.get("evidence_id"), str)
    }
    selected_evidence = evidence_by_id.get(evidence_id)
    if selected_evidence is None:
        raise ValueError("frontier recommendation Evidence is absent")
    raw_value = selected_evidence.get("value")
    summary = (
        cast(Mapping[str, object], raw_value).get("finding_gate_summary")
        if isinstance(raw_value, Mapping)
        else None
    )
    subjects = selected_evidence.get("subject_ids")
    if (
        selected_evidence.get("evidence_kind")
        not in {"metric_profile", "service_profile"}
        or selected_evidence.get("semantic_quality") != "typed"
        or not isinstance(summary, Mapping)
        or hypothesis.subject
        not in (cast(list[object], subjects) if isinstance(subjects, list) else [])
    ):
        raise ValueError("frontier recommendation Evidence is not eligible")
    summary_values = cast(Mapping[str, object], summary)
    expected_score = summary_values.get("root_cause_score")
    if (
        (
            summary_values.get("fault_type_ready") is not True
            and summary_values.get("material_event_ready") is not True
        )
        or summary_values.get("boundary_aligned") is True
        or summary_values.get("requested_fault_type") != fault_type
        or summary_values.get("mechanism_signature") != signature
        or mechanism_signature_rank(summary_values) != rank
        or not isinstance(expected_score, (int, float))
        or isinstance(expected_score, bool)
        or float(expected_score) != float(raw_score)
    ):
        raise ValueError("frontier recommendation typed gate does not match audit")
    latest_assessments = {
        (str(item.get("evidence_id")), str(item.get("hypothesis_id"))): item
        for item in _bundle_record_payloads(bundle, "evidence_assessments.jsonl")
        if isinstance(item.get("evidence_id"), str)
        and isinstance(item.get("hypothesis_id"), str)
    }
    selected_assessment = latest_assessments.get((evidence_id, hypothesis_id))
    if selected_assessment is None or selected_assessment.get("impact") != "supports":
        raise ValueError(
            "frontier recommendation lacks an explicit supports assessment"
        )
    if any(
        item.get("hypothesis_id") == hypothesis_id and item.get("impact") == "refutes"
        for item in latest_assessments.values()
    ):
        raise ValueError("frontier recommendation hypothesis is explicitly refuted")
    for item in evidence_payloads:
        value = item.get("value")
        other_summary = (
            cast(Mapping[str, object], value).get("finding_gate_summary")
            if isinstance(value, Mapping)
            else None
        )
        if not isinstance(other_summary, Mapping):
            continue
        other_values = cast(Mapping[str, object], other_summary)
        other_score = other_values.get("root_cause_score")
        raw_subjects = item.get("subject_ids")
        other_subjects = (
            cast(list[object], raw_subjects) if isinstance(raw_subjects, list) else []
        )
        if (
            (
                other_values.get("fault_type_ready") is True
                or other_values.get("material_event_ready") is True
            )
            and other_values.get("boundary_aligned") is not True
            and isinstance(other_score, (int, float))
            and not isinstance(other_score, bool)
            and any(
                isinstance(subject, str) and subject != hypothesis.subject
                for subject in other_subjects
            )
            and (
                mechanism_signature_rank(other_values),
                float(other_score),
            )
            >= (rank, float(raw_score))
        ):
            raise ValueError("frontier recommendation has an equal or stronger peer")
    raw_anomalies = report.get("material_anomalies")
    raw_dispositions = report.get("anomaly_dispositions")
    if not isinstance(raw_anomalies, list) or not isinstance(raw_dispositions, list):
        raise ValueError("frontier recommendation lacks anomaly audit state")
    anomaly_ids: set[str] = set()
    for raw_item in cast(list[object], raw_anomalies):
        if not isinstance(raw_item, Mapping):
            continue
        item = cast(Mapping[str, object], raw_item)
        anomaly_id = item.get("anomaly_id")
        if isinstance(anomaly_id, str):
            anomaly_ids.add(anomaly_id)
    latest_dispositions: dict[str, Mapping[str, object]] = {}
    for raw_item in cast(list[object], raw_dispositions):
        if not isinstance(raw_item, Mapping):
            continue
        item = cast(Mapping[str, object], raw_item)
        anomaly_id = item.get("anomaly_id")
        if isinstance(anomaly_id, str):
            latest_dispositions[anomaly_id] = item
    unresolved = sorted(
        anomaly_id
        for anomaly_id in anomaly_ids
        if anomaly_id not in latest_dispositions
        or latest_dispositions[anomaly_id].get("disposition") == "unresolved"
    )
    if selection.get("unresolved_anomaly_ids") != unresolved or not unresolved:
        raise ValueError("frontier recommendation unresolved audit is invalid")
    candidates = report.get("candidates")
    matching: list[Mapping[str, object]] = []
    for raw_item in (
        cast(list[object], candidates) if isinstance(candidates, list) else []
    ):
        if not isinstance(raw_item, Mapping):
            continue
        item = cast(Mapping[str, object], raw_item)
        if item.get("hypothesis") == hypothesis.model_dump(mode="json"):
            matching.append(item)
    audit_ref = recommendation.get("audit_ref")
    recommendation_events = [
        event
        for stable_key, event in _investigation_event_records(bundle)
        if stable_key == audit_ref
        and event.get("event_type") == "benchmark_recommendation_selected"
        and event.get("recommendation") == recommendation
    ]
    if len(matching) != 1 or len(recommendation_events) != 1:
        raise ValueError("frontier recommendation does not match its report audit")
    occurrence = ""
    if "root cause occurrence datetime" in task.requested_outputs:
        occurrence_hypothesis = hypothesis.model_copy(
            update={
                "supporting_evidence_refs": sorted(
                    {*hypothesis.supporting_evidence_refs, evidence_id}
                )
            }
        )
        occurrence = _finding_occurrence_datetime(bundle, occurrence_hypothesis, task)
    return {
        "subject": hypothesis.subject,
        "statement": fault_type,
        "policy": "frontier-supported-v1",
        "hypothesis_id": hypothesis_id,
        "candidate_id": recommendation.get("candidate_id"),
        "occurrence_datetime": occurrence,
    }


def _bundle_context_policy_version(bundle: Path) -> int | None:
    payloads = _bundle_record_payloads(bundle, "context_manifests.jsonl")
    if not payloads:
        return None
    versions = [payload.get("policy_version") for payload in payloads]
    if any(not isinstance(value, str) or not value for value in versions):
        raise ValueError("context manifest policy_version must be non-empty")
    unique = set(cast(list[str], versions))
    if len(unique) != 1:
        raise ValueError("context manifest policy_version values must be consistent")
    match = re.fullmatch(r"m2-context-v(\d+)", next(iter(unique)))
    if match is None:
        raise ValueError("context manifest policy_version is invalid")
    return int(match.group(1))


def _bundle_frontier_policy_version(bundle: Path) -> int:
    payloads = _bundle_record_payloads(bundle, "context_manifests.jsonl")
    if not payloads:
        raise ValueError("frontier recommendation requires context manifests")
    versions = [payload.get("policy_version") for payload in payloads]
    if any(not isinstance(value, str) or not value for value in versions):
        raise ValueError("frontier context policy_version must be non-empty")
    unique = set(cast(list[str], versions))
    if len(unique) != 1:
        raise ValueError("frontier context policy_version values must be consistent")
    match = re.fullmatch(r"frontier-reflection-v(\d+)", next(iter(unique)))
    if match is None:
        raise ValueError("frontier context policy_version is invalid")
    return int(match.group(1))


def _bundle_policy_provenance(bundle: Path) -> tuple[int | None, int | None]:
    context_policy_version = _bundle_context_policy_version(bundle)
    raw_manifest: object = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("bundle manifest must be an object")
    manifest = cast(Mapping[str, object], raw_manifest)
    raw_terminal_policy_version = manifest.get("terminal_policy_version")
    if raw_terminal_policy_version is None:
        terminal_policy_version = None
    elif isinstance(raw_terminal_policy_version, int) and not isinstance(
        raw_terminal_policy_version, bool
    ):
        terminal_policy_version = raw_terminal_policy_version
    else:
        raise ValueError("terminal_policy_version must be an integer")
    if (
        terminal_policy_version is not None
        and terminal_policy_version > TERMINAL_POLICY_VERSION
    ):
        raise ValueError("terminal_policy_version is newer than this exporter")
    if (
        terminal_policy_version is not None
        and terminal_policy_version >= 6
        and (context_policy_version is None or context_policy_version < 81)
    ):
        raise ValueError("terminal policy epoch 6 requires context policy v81 or newer")
    if (
        context_policy_version is not None
        and context_policy_version >= 81
        and (terminal_policy_version is None or terminal_policy_version < 6)
    ):
        raise ValueError("context policy v81 requires terminal policy epoch 6")
    if (
        manifest.get("code_revision") == installed_source_revision()
        and terminal_policy_version != TERMINAL_POLICY_VERSION
    ):
        raise ValueError(
            "current code revision requires the current terminal policy epoch"
        )
    return context_policy_version, terminal_policy_version


def _validate_terminal_v5_or_v6_recommendation(
    bundle: Path,
    report: Mapping[str, Any],
    events: Sequence[tuple[str, Mapping[str, Any]]],
    recommendation: Mapping[str, object],
    selected_hypothesis: Hypothesis,
    *,
    enforce_expected_policy: bool,
) -> None:
    evidence_items = [
        Evidence.model_validate_json(json.dumps(payload))
        for payload in _bundle_record_payloads(bundle, "evidence.jsonl")
    ]
    observation_items = [
        Observation.model_validate_json(json.dumps(payload))
        for payload in _bundle_record_payloads(bundle, "observations.jsonl")
    ]
    if len({item.evidence_id for item in evidence_items}) != len(evidence_items):
        raise ValueError("bundle contains duplicate Evidence IDs")
    if len({item.observation_id for item in observation_items}) != len(
        observation_items
    ):
        raise ValueError("bundle contains duplicate Observation IDs")
    evidence_by_id = {item.evidence_id: item for item in evidence_items}
    observations = {item.observation_id: item for item in observation_items}
    queue = candidate_disposition_queue(observation_items, evidence_items)
    queue_by_id = {
        str(item["candidate_id"]): (rank, item)
        for rank, item in enumerate(queue)
        if isinstance(item.get("candidate_id"), str)
    }
    raw_report_candidates = report.get("candidates")
    if not isinstance(raw_report_candidates, list):
        raise ValueError("benchmark recommendation has no candidate history")
    report_hypotheses = [
        Hypothesis.model_validate(item["hypothesis"])
        for raw_item in cast(list[object], raw_report_candidates)
        if isinstance(raw_item, Mapping)
        for item in [cast(Mapping[str, Any], raw_item)]
        if isinstance(item.get("hypothesis"), Mapping)
    ]
    if len({item.hypothesis_id for item in report_hypotheses}) != len(
        report_hypotheses
    ) or any(
        not isinstance(raw_item, Mapping)
        or cast(Mapping[str, object], raw_item).get("selected") is not False
        for raw_item in cast(list[object], raw_report_candidates)
    ):
        raise ValueError("terminal-candidate-v5 report candidate is invalid")
    hypotheses = report_hypotheses
    supported_hypotheses_by_subject: dict[str, list[str]] = {}
    for hypothesis in hypotheses:
        bindings = [
            event
            for _stable_key, event in events
            if event.get("event_type") == "candidate_disposition_applied"
            and event.get("hypothesis_id") == hypothesis.hypothesis_id
        ]
        if (
            supported_candidate_binding(
                hypothesis,
                bindings,
                queue_by_id,
                evidence_by_id,
                observations,
            )
            is not None
        ):
            supported_hypotheses_by_subject.setdefault(hypothesis.subject, []).append(
                hypothesis.hypothesis_id
            )
    eligible: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        if hypothesis.status != "inconclusive":
            continue
        bindings = [
            event
            for _stable_key, event in events
            if event.get("event_type") == "candidate_disposition_applied"
            and event.get("hypothesis_id") == hypothesis.hypothesis_id
        ]
        candidate_ids = {
            event.get("candidate_id")
            for event in bindings
            if isinstance(event.get("candidate_id"), str) and event.get("candidate_id")
        }
        if (
            not bindings
            or len(candidate_ids) != 1
            or not all(
                event.get("candidate_id") in candidate_ids
                and event.get("subject") == hypothesis.subject
                and event.get("selected_fault_type") == hypothesis.statement
                and event.get("status") in {"open", "inconclusive"}
                for event in bindings
            )
            or bindings[-1].get("status") != "inconclusive"
        ):
            continue
        candidate_id = next(iter(candidate_ids))
        if not isinstance(candidate_id, str) or candidate_id not in queue_by_id:
            continue
        terminal_rank, candidate = queue_by_id[candidate_id]
        selection = {
            "inconclusive": True,
            "source_strict": has_strict_directional_metric_shift(
                candidate.get("source_paired_metric_summary"),
                candidate.get("source_pressure_direction"),
            ),
            "has_log_source": candidate.get("has_log_source") is True,
            "direct_strict": bool(
                candidate.get("metric_role") == "direct"
                and has_strict_directional_metric_shift(
                    candidate.get("paired_metric_summary"),
                    candidate.get("candidate_pressure_direction"),
                )
            ),
            "ever_open": any(event.get("status") == "open" for event in bindings),
            "terminal_queue_rank": terminal_rank,
        }
        eligible.append(
            {
                "candidate_id": candidate_id,
                "hypothesis": hypothesis.model_dump(mode="json"),
                "selection": selection,
            }
        )
    expected, switch = select_terminal_candidate(
        eligible, supported_hypotheses_by_subject
    )
    if expected is None:
        raise ValueError("terminal candidate selection is empty")
    policy = recommendation.get("policy")
    pre_family_policy = (
        "terminal-candidate-v5" if switch is not None else "terminal-candidate-v4"
    )
    pre_family = (
        {
            **expected,
            "selection": {
                **cast(dict[str, Any], expected["selection"]),
                **switch,
            },
        }
        if switch is not None
        else expected
    )
    if not enforce_expected_policy and policy == "terminal-candidate-v5":
        expected_policy = "terminal-candidate-v5"
        expected = pre_family
    else:
        family_selected, family_audit = select_family_compatible_terminal_candidate(
            eligible,
            pre_family,
            {
                candidate_id: cast(str, candidate.get("resource_family"))
                for candidate_id, (_rank, candidate) in queue_by_id.items()
                if isinstance(candidate.get("resource_family"), str)
            },
            pre_family_policy,
        )
        if family_audit is not None:
            expected_policy = "terminal-candidate-v6"
            expected = {
                **family_selected,
                "selection": {
                    **cast(dict[str, Any], family_selected["selection"]),
                    **family_audit,
                },
            }
        elif switch is not None:
            expected_policy = "terminal-candidate-v5"
            expected = pre_family
        else:
            expected_policy = "terminal-candidate-v4"
    if (
        enforce_expected_policy
        or policy in {"terminal-candidate-v5", "terminal-candidate-v6"}
    ) and policy != expected_policy:
        raise ValueError(
            f"terminal policy downgrade: expected {expected_policy}, got {policy}"
        )
    if (
        recommendation.get("candidate_id") != expected["candidate_id"]
        or recommendation.get("hypothesis") != expected["hypothesis"]
        or recommendation.get("selection") != expected["selection"]
        or recommendation.get("eligible_terminal_count") != len(eligible)
        or selected_hypothesis.model_dump(mode="json") != expected["hypothesis"]
    ):
        raise ValueError(f"{policy} selection is not reproducible")


def _validated_rejected_supported_recommendation(
    bundle: Path,
    report: Mapping[str, Any],
    recommendation: Mapping[str, object],
    original_ids: Mapping[str, str],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "policy",
        "confidence",
        "usage",
        "is_finding",
        "candidate_id",
        "hypothesis",
        "filtered_log_evidence_ids",
        "weak_finding_rejected_ref",
        "disposition_ref",
        "audit_ref",
    }
    raw_hypothesis = recommendation.get("hypothesis")
    hypothesis = (
        Hypothesis.model_validate(raw_hypothesis)
        if isinstance(raw_hypothesis, Mapping)
        else None
    )
    candidate_id = recommendation.get("candidate_id")
    raw_filtered_ids = recommendation.get("filtered_log_evidence_ids")
    weak_ref = recommendation.get("weak_finding_rejected_ref")
    disposition_ref = recommendation.get("disposition_ref")
    audit_ref = recommendation.get("audit_ref")
    if (
        set(recommendation) != expected_keys
        or recommendation.get("schema_version") != "1"
        or recommendation.get("policy") != "rejected-supported-filtered-log-v1"
        or recommendation.get("confidence") != "low"
        or recommendation.get("usage") != "openrca_export_fallback_only"
        or recommendation.get("is_finding") is not False
        or hypothesis is None
        or hypothesis.status != "supported"
        or hypothesis.subject not in original_ids
        or not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(raw_filtered_ids, list)
        or not raw_filtered_ids
        or not all(
            isinstance(item, str) and item
            for item in cast(list[object], raw_filtered_ids)
        )
        or not isinstance(weak_ref, str)
        or not weak_ref
        or not isinstance(disposition_ref, str)
        or not disposition_ref
        or not isinstance(audit_ref, str)
        or not audit_ref
    ):
        raise ValueError("benchmark recommendation has invalid fields")
    filtered_ids = cast(list[str], raw_filtered_ids)
    if filtered_ids != sorted(set(filtered_ids)):
        raise ValueError("benchmark recommendation has invalid fields")
    events = _investigation_event_records(bundle)
    evidence_items = [
        Evidence.model_validate_json(json.dumps(payload))
        for payload in _bundle_record_payloads(bundle, "evidence.jsonl")
    ]
    observation_items = [
        Observation.model_validate_json(json.dumps(payload))
        for payload in _bundle_record_payloads(bundle, "observations.jsonl")
    ]
    if len({item.evidence_id for item in evidence_items}) != len(evidence_items):
        raise ValueError("bundle contains duplicate Evidence IDs")
    if len({item.observation_id for item in observation_items}) != len(
        observation_items
    ):
        raise ValueError("bundle contains duplicate Observation IDs")
    evidence_by_id = {item.evidence_id: item for item in evidence_items}
    observations = {item.observation_id: item for item in observation_items}
    queue = candidate_disposition_queue(observation_items, evidence_items)
    if _has_v4_terminal_candidate(report, events, queue):
        raise ValueError("v4 terminal candidate must take precedence over P0 fallback")
    matching_queue_candidates = [
        item for item in queue if item.get("candidate_id") == candidate_id
    ]
    if len(matching_queue_candidates) != 1:
        raise ValueError("benchmark recommendation candidate is absent from queue")
    queue_candidate = matching_queue_candidates[0]
    queue_fault_types = [queue_candidate.get("fault_type")]
    raw_queue_aliases = queue_candidate.get("fault_type_aliases", [])
    if isinstance(raw_queue_aliases, list):
        queue_fault_types.extend(cast(list[object], raw_queue_aliases))
    queue_fault_types = [
        item for item in queue_fault_types if isinstance(item, str) and item
    ]
    if (
        queue_candidate.get("entity_id") != hypothesis.subject
        or hypothesis.statement not in queue_fault_types
    ):
        raise ValueError("benchmark recommendation does not match candidate queue")
    selected_events = [
        event
        for stable_key, event in events
        if stable_key == audit_ref
        and event.get("event_type") == "benchmark_recommendation_selected"
        and event.get("recommendation") == recommendation
    ]
    weak_events = [
        event
        for stable_key, event in events
        if stable_key == weak_ref
        and event.get("event_type") == "weak_finding_rejected"
        and event.get("hypothesis_id") == hypothesis.hypothesis_id
        and event.get("candidate_id") == candidate_id
    ]
    disposition_events = [
        event
        for stable_key, event in events
        if stable_key == disposition_ref
        and event.get("event_type") == "candidate_disposition_applied"
        and event.get("hypothesis_id") == hypothesis.hypothesis_id
        and event.get("candidate_id") == candidate_id
        and event.get("subject") == hypothesis.subject
        and event.get("selected_fault_type") == hypothesis.statement
        and event.get("status") == "supported"
    ]
    if (
        len(selected_events) != 1
        or len(weak_events) != 1
        or len(disposition_events) != 1
    ):
        raise ValueError(
            "benchmark recommendation does not match its weak and disposition audits"
        )
    weak_event = weak_events[0]
    original_finalize = weak_event.get("original_finalize_operation")
    if (
        weak_event.get("visibility") != "audit_only"
        or weak_event.get("policy") != "strong-source-no-higher-peer-v1"
        or weak_event.get("abstain_reason") != report.get("abstain_reason")
        or not isinstance(original_finalize, Mapping)
    ):
        raise ValueError("benchmark recommendation weak audit is invalid")
    original_finalize_values = cast(Mapping[str, Any], original_finalize)
    if (
        original_finalize_values.get("type") != "finalize"
        or original_finalize_values.get("abstain_reason") is not None
        or original_finalize_values.get("hypothesis_ids") != [hypothesis.hypothesis_id]
    ):
        raise ValueError("benchmark recommendation weak audit is invalid")
    disposition = disposition_events[0]
    original = disposition.get("original_operation")
    applied = disposition.get("applied_operation")
    allowed_fault_types = disposition.get("allowed_fault_types")
    log_hint = disposition.get("log_selector_hint")
    expected_log_hint = queue_candidate.get("log_selector_hint")
    if (
        not isinstance(original, Mapping)
        or not isinstance(applied, Mapping)
        or not isinstance(allowed_fault_types, list)
        or not isinstance(log_hint, Mapping)
        or disposition.get("allowed_fault_types") != queue_fault_types
        or disposition.get("resource_family") != queue_candidate.get("resource_family")
        or log_hint != expected_log_hint
    ):
        raise ValueError("benchmark recommendation disposition is invalid")
    original_values = cast(Mapping[str, Any], original)
    applied_values = cast(Mapping[str, Any], applied)
    allowed_values = cast(list[object], allowed_fault_types)
    log_hint_values = cast(Mapping[str, Any], log_hint)
    if (
        hypothesis.statement not in allowed_fault_types
        or original_values.get("status") != "supported"
        or original_values.get("subject") != hypothesis.subject
        or not isinstance(original_values.get("statement"), str)
        or _resolve_fault_type(cast(str, original_values["statement"]), allowed_values)
        != hypothesis.statement
        or applied_values.get("status") != "supported"
        or applied_values.get("subject") != hypothesis.subject
        or applied_values.get("statement") != hypothesis.statement
        or not isinstance(log_hint_values.get("exact_log_names"), list)
    ):
        raise ValueError("benchmark recommendation disposition is invalid")
    binding_history = [
        (stable_key, event)
        for stable_key, event in events
        if event.get("event_type") == "candidate_disposition_applied"
        and event.get("hypothesis_id") == hypothesis.hypothesis_id
    ]
    if (
        not binding_history
        or binding_history[-1][0] != disposition_ref
        or not all(
            event.get("candidate_id") == candidate_id
            and event.get("subject") == hypothesis.subject
            and event.get("selected_fault_type") == hypothesis.statement
            and event.get("status") in {"open", "supported"}
            and event.get("allowed_fault_types") == queue_fault_types
            and event.get("resource_family") == queue_candidate.get("resource_family")
            and event.get("log_selector_hint") == expected_log_hint
            for _stable_key, event in binding_history
        )
    ):
        raise ValueError("benchmark recommendation binding history is invalid")
    common_refs = (
        set(cast(list[str], original_values.get("supporting_evidence_refs", [])))
        & set(cast(list[str], applied_values.get("supporting_evidence_refs", [])))
        & set(hypothesis.supporting_evidence_refs)
    )
    pair_refs = {
        disposition.get("reference_evidence_id"),
        disposition.get("incident_evidence_id"),
    }
    if (
        not pair_refs <= common_refs
        or not set(filtered_ids) <= common_refs
        or not set(filtered_ids)
        <= set(cast(list[str], disposition.get("log_evidence_ids", [])))
    ):
        raise ValueError("benchmark recommendation evidence references are invalid")
    reference_evidence_id = disposition.get("reference_evidence_id")
    incident_evidence_id = disposition.get("incident_evidence_id")
    if (
        not isinstance(reference_evidence_id, str)
        or not isinstance(incident_evidence_id, str)
        or not _valid_direct_metric_pair(
            reference_evidence_id,
            incident_evidence_id,
            evidence_by_id,
            observations,
            hypothesis.subject,
            queue_candidate.get("metric_id"),
        )
    ):
        raise ValueError("benchmark recommendation direct metric pair is invalid")
    if not all(
        _valid_filtered_log_evidence(
            evidence_by_id.get(evidence_id),
            observations,
            hypothesis.subject,
        )
        for evidence_id in filtered_ids
    ):
        raise ValueError("benchmark recommendation filtered log evidence is invalid")
    report_candidates = report.get("candidates")
    matching_candidates = (
        [
            item
            for raw_item in cast(list[object], report_candidates)
            if isinstance(raw_item, Mapping)
            for item in [cast(Mapping[str, object], raw_item)]
            if item.get("hypothesis") == hypothesis.model_dump(mode="json")
            and item.get("selected") is False
        ]
        if isinstance(report_candidates, list)
        else []
    )
    if len(matching_candidates) != 1:
        raise ValueError("benchmark recommendation does not match report candidates")
    return {
        "subject": hypothesis.subject,
        "statement": hypothesis.statement,
        "policy": "rejected-supported-filtered-log-v1",
        "hypothesis_id": hypothesis.hypothesis_id,
        "candidate_id": candidate_id,
    }


def _valid_filtered_log_evidence(
    evidence: Evidence | None,
    observations: Mapping[str, Observation],
    subject: str,
) -> bool:
    if (
        evidence is None
        or evidence.predicate != "query_log_samples.result"
        or subject not in evidence.subject_ids
        or not isinstance(evidence.value, Mapping)
    ):
        return False
    evidence_value = cast(Mapping[str, Any], evidence.value)
    summary = evidence_value.get("paired_log_summary")
    if not isinstance(summary, Mapping):
        return False
    summary_values = cast(Mapping[str, Any], summary)
    if summary_values.get("counts_complete") is not True or not isinstance(
        summary_values.get("samples_truncated"), bool
    ):
        return False
    period_counts: dict[str, Mapping[str, Any]] = {}
    for period in ("reference", "incident"):
        raw_counts = summary_values.get(period)
        if not isinstance(raw_counts, Mapping):
            return False
        counts = cast(Mapping[str, Any], raw_counts)
        matched_count = counts.get("matched_count")
        log_name_counts = counts.get("log_name_counts")
        if (
            not isinstance(matched_count, int)
            or isinstance(matched_count, bool)
            or matched_count < 0
            or not isinstance(log_name_counts, Mapping)
        ):
            return False
        typed_counts = cast(Mapping[object, object], log_name_counts)
        if (
            any(
                not isinstance(name, str)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for name, count in typed_counts.items()
            )
            or sum(cast(int, count) for count in typed_counts.values()) != matched_count
        ):
            return False
        period_counts[period] = counts
    incident_values = period_counts["incident"]
    matched_count = incident_values.get("matched_count")
    counts = incident_values.get("log_name_counts")
    if cast(int, matched_count) <= 0 or not isinstance(counts, Mapping):
        return False
    for observation_id in evidence.observation_refs:
        observation = observations.get(observation_id)
        log_name = observation.arguments.get("log_name") if observation else None
        if (
            observation is not None
            and observation.tool == "query_log_samples"
            and observation.status in {"success", "partial"}
            and observation.arguments.get("entity_id") == subject
            and isinstance(log_name, str)
            and bool(log_name.strip())
            and isinstance(cast(Mapping[str, object], counts).get(log_name), int)
            and cast(int, cast(Mapping[str, object], counts)[log_name]) > 0
        ):
            return True
    return False


def _valid_direct_metric_pair(
    reference_evidence_id: str,
    incident_evidence_id: str,
    evidence_by_id: Mapping[str, Evidence],
    observations: Mapping[str, Observation],
    subject: str,
    metric_id: object,
) -> bool:
    if reference_evidence_id == incident_evidence_id or not isinstance(metric_id, str):
        return False
    for evidence_id, period in (
        (reference_evidence_id, "reference"),
        (incident_evidence_id, "incident"),
    ):
        evidence = evidence_by_id.get(evidence_id)
        if (
            evidence is None
            or evidence.predicate != "query_metric.result"
            or subject not in evidence.subject_ids
        ):
            return False
        matching = [
            observation
            for observation_id in evidence.observation_refs
            for observation in [observations.get(observation_id)]
            if observation is not None
            and observation.tool == "query_metric"
            and observation.status in {"success", "partial"}
            and observation.arguments.get("entity_id") == subject
            and observation.arguments.get("metric_id") == metric_id
            and observation.arguments.get("period", "incident") == period
        ]
        if len(matching) != 1:
            return False
    return True


def _has_v4_terminal_candidate(
    report: Mapping[str, Any],
    events: Sequence[tuple[str, Mapping[str, Any]]],
    queue: Sequence[Mapping[str, Any]],
) -> bool:
    raw_candidates = report.get("candidates")
    if not isinstance(raw_candidates, list):
        return False
    queue_ids = {
        item.get("candidate_id")
        for item in queue
        if isinstance(item.get("candidate_id"), str)
    }
    for raw_candidate in cast(list[object], raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            continue
        raw_hypothesis = cast(Mapping[str, object], raw_candidate).get("hypothesis")
        if not isinstance(raw_hypothesis, Mapping):
            continue
        hypothesis = Hypothesis.model_validate(raw_hypothesis)
        if hypothesis.status != "inconclusive":
            continue
        bindings = [
            event
            for _stable_key, event in events
            if event.get("event_type") == "candidate_disposition_applied"
            and event.get("hypothesis_id") == hypothesis.hypothesis_id
        ]
        candidate_ids = {
            event.get("candidate_id")
            for event in bindings
            if isinstance(event.get("candidate_id"), str) and event.get("candidate_id")
        }
        if (
            bindings
            and len(candidate_ids) == 1
            and next(iter(candidate_ids)) in queue_ids
            and all(
                event.get("candidate_id") in candidate_ids
                and event.get("subject") == hypothesis.subject
                and event.get("selected_fault_type") == hypothesis.statement
                and event.get("status") in {"open", "inconclusive"}
                for event in bindings
            )
            and bindings[-1].get("status") == "inconclusive"
        ):
            return True
    return False


def _resolve_fault_type(statement: str, values: list[object]) -> str | None:
    normalized = " ".join(statement.split()).casefold()
    matches = [
        value
        for value in values
        if isinstance(value, str)
        and (
            statement == value
            or bool(
                re.search(
                    rf"(?<!\w){re.escape(' '.join(value.split()).casefold())}(?!\w)",
                    normalized,
                )
            )
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _bundle_record_payloads(bundle: Path, filename: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in (bundle / filename).read_text(encoding="utf-8").splitlines():
        raw_record: object = json.loads(line)
        if not isinstance(raw_record, dict):
            raise ValueError(f"{filename} record must be an object")
        payload = cast(dict[str, object], raw_record).get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"{filename} payload must be an object")
        payloads.append(cast(dict[str, Any], payload))
    return payloads


def _market_finding_reason(bundle: Path, finding: Hypothesis) -> str:
    """Export the proof-bound public fault type, not the internal explanation."""
    supporting_summaries: list[Mapping[str, object]] = []
    same_subject_fault_types: list[object] = []
    supporting = set(finding.supporting_evidence_refs)
    for payload in _bundle_record_payloads(bundle, "evidence.jsonl"):
        raw_value = payload.get("value")
        if not isinstance(raw_value, Mapping):
            continue
        summary = cast(Mapping[str, object], raw_value).get("finding_gate_summary")
        if not isinstance(summary, Mapping):
            continue
        summary_values = cast(Mapping[str, object], summary)
        proof_bound = summary_values.get("fault_type_ready") is True or (
            summary_values.get("material_event_ready") is True
            and isinstance(summary_values.get("mechanism_signature"), str)
        )
        if not proof_bound:
            continue
        fault_type = summary_values.get("requested_fault_type")
        raw_subjects = payload.get("subject_ids")
        subjects = (
            cast(list[object], raw_subjects) if isinstance(raw_subjects, list) else []
        )
        if finding.subject in subjects and isinstance(fault_type, str):
            same_subject_fault_types.append(fault_type)
        if payload.get("evidence_id") in supporting:
            supporting_summaries.append(summary_values)
    declared = _resolve_fault_type(finding.statement, same_subject_fault_types)
    if declared is not None:
        return declared
    dominant = dominant_market_fault_type(supporting_summaries)
    if dominant is not None:
        return dominant
    return finding.statement


def _finding_occurrence_datetime(bundle: Path, finding: Hypothesis, task: Task) -> str:
    """Export the earliest source-backed onset supporting one Finding."""
    supporting = set(finding.supporting_evidence_refs)
    candidates: list[tuple[int, datetime]] = []
    for payload in _bundle_record_payloads(bundle, "evidence.jsonl"):
        if payload.get("evidence_id") not in supporting:
            continue
        raw_onset = payload.get("onset_interval")
        if not isinstance(raw_onset, Mapping):
            continue
        onset_values = cast(Mapping[str, object], raw_onset)
        raw_start = onset_values.get("start")
        raw_end = onset_values.get("end")
        if not isinstance(raw_start, str) or not isinstance(raw_end, str):
            continue
        try:
            onset_start = datetime.fromisoformat(raw_start)
            onset_end = datetime.fromisoformat(raw_end)
        except ValueError:
            continue
        onset = onset_start + (onset_end - onset_start) / 2
        if (
            onset.tzinfo is not None
            and task.incident_range.start <= onset < task.incident_range.end
        ):
            rank = 0
            raw_value = payload.get("value")
            if isinstance(raw_value, Mapping):
                summary = cast(Mapping[str, object], raw_value).get(
                    "finding_gate_summary"
                )
                if isinstance(summary, Mapping):
                    summary_values = cast(Mapping[str, object], summary)
                    if summary_values.get("fault_type_ready") is True or (
                        summary_values.get("material_event_ready") is True
                        and isinstance(summary_values.get("mechanism_signature"), str)
                    ):
                        rank = mechanism_signature_rank(summary_values)
            if payload.get("evidence_kind") == "trace_edge":
                rank = max(rank, 1)
            if payload.get("evidence_kind") == "trace_operation_profile" and isinstance(
                raw_value, Mapping
            ):
                raw_operations = cast(Mapping[str, object], raw_value).get("operations")
                operations = (
                    cast(list[object], raw_operations)
                    if isinstance(raw_operations, list)
                    else []
                )
                if any(
                    isinstance(operation, Mapping)
                    and cast(Mapping[str, object], operation).get("identifiability")
                    == "identifiable"
                    for operation in operations
                ):
                    rank = max(rank, 3)
            candidates.append((rank, onset))
    if not candidates:
        raise ValueError(
            "finding has no source-backed onset for requested occurrence datetime"
        )
    best_rank = max(rank for rank, _onset in candidates)
    return min(onset for rank, onset in candidates if rank == best_rank).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _validated_initial_top1_recommendation(
    bundle: Path,
    report: Mapping[str, Any],
    recommendation: Mapping[str, object],
    original_ids: Mapping[str, str],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "policy",
        "confidence",
        "usage",
        "is_finding",
        "candidate_id",
        "subject",
        "statement",
        "selection",
        "rejected_candidate_id",
        "weak_finding_rejected_ref",
        "audit_ref",
    }
    raw_selection = recommendation.get("selection")
    selection = (
        cast(Mapping[str, object], raw_selection)
        if isinstance(raw_selection, Mapping)
        else cast(Mapping[str, object], {})
    )
    candidate_id = recommendation.get("candidate_id")
    subject = recommendation.get("subject")
    statement = recommendation.get("statement")
    rejected_candidate_id = recommendation.get("rejected_candidate_id")
    weak_ref = recommendation.get("weak_finding_rejected_ref")
    audit_ref = recommendation.get("audit_ref")
    if (
        set(recommendation) != expected_keys
        or recommendation.get("schema_version") != "1"
        or recommendation.get("policy") != "initial-top1-v1"
        or recommendation.get("confidence") != "low"
        or recommendation.get("usage") != "openrca_export_fallback_only"
        or recommendation.get("is_finding") is not False
        or not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(subject, str)
        or subject not in original_ids
        or not isinstance(statement, str)
        or not statement
        or set(selection) != {"initial_queue_rank", "source_strict", "trigger"}
        or selection.get("initial_queue_rank") != 0
        or not isinstance(selection.get("source_strict"), bool)
        or selection.get("trigger") != "weak_finding_rejected"
        or not (
            rejected_candidate_id is None
            or isinstance(rejected_candidate_id, str)
            and rejected_candidate_id
        )
        or not isinstance(weak_ref, str)
        or not weak_ref
        or not isinstance(audit_ref, str)
        or not audit_ref
        or report.get("abstain_reason")
        != "selected Finding did not satisfy the pre-registered strong-Finding policy"
    ):
        raise ValueError("benchmark recommendation has invalid fields")

    event_records = _investigation_event_records(bundle)
    recommendation_events = [
        event
        for stable_key, event in event_records
        if stable_key == audit_ref
        and event.get("event_type") == "benchmark_recommendation_selected"
        and event.get("recommendation") == recommendation
    ]
    weak_events = [
        event
        for stable_key, event in event_records
        if stable_key == weak_ref
        and event.get("event_type") == "weak_finding_rejected"
        and event.get("visibility") == "audit_only"
        and event.get("candidate_id") == rejected_candidate_id
        and event.get("initial_top1_candidate_id") == candidate_id
    ]
    if len(recommendation_events) != 1 or len(weak_events) != 1:
        raise ValueError(
            "benchmark recommendation does not match its weak-finding audit records"
        )
    weak_event = weak_events[0]
    if (
        set(recommendation_events[0]) != {"event_type", "visibility", "recommendation"}
        or recommendation_events[0].get("visibility") != "audit_only"
        or set(weak_event)
        != {
            "event_type",
            "visibility",
            "policy",
            "hypothesis_id",
            "candidate_id",
            "source_strict",
            "candidate_rank",
            "higher_rank_same_fault_candidate_ids",
            "initial_top1_candidate_id",
            "initial_top1_subject",
            "initial_top1_fault_type",
            "initial_top1_source_strict",
            "original_finalize_operation",
        }
        or weak_event.get("policy") != "strong-source-no-higher-peer-v1"
        or weak_event.get("initial_top1_subject") != subject
        or weak_event.get("initial_top1_fault_type") != statement
        or weak_event.get("initial_top1_source_strict")
        != selection.get("source_strict")
        or not (
            weak_event.get("hypothesis_id") is None
            or isinstance(weak_event.get("hypothesis_id"), str)
            and weak_event.get("hypothesis_id")
        )
        or not isinstance(weak_event.get("source_strict"), bool)
        or not (
            weak_event.get("candidate_rank") is None
            or isinstance(weak_event.get("candidate_rank"), int)
            and not isinstance(weak_event.get("candidate_rank"), bool)
            and cast(int, weak_event["candidate_rank"]) >= 0
        )
        or not isinstance(weak_event.get("higher_rank_same_fault_candidate_ids"), list)
        or not all(
            isinstance(item, str) and item
            for item in cast(
                list[object], weak_event["higher_rank_same_fault_candidate_ids"]
            )
        )
        or not isinstance(weak_event.get("original_finalize_operation"), Mapping)
    ):
        raise ValueError("weak-finding audit event has invalid fields")
    return {
        "subject": subject,
        "statement": statement,
        "policy": "initial-top1-v1",
        "candidate_id": candidate_id,
        "rejected_candidate_id": rejected_candidate_id,
    }


def _investigation_event_records(bundle: Path) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    stable_keys: set[str] = set()
    events_path = bundle / "investigation_events.jsonl"
    for line in events_path.read_text(encoding="utf-8").splitlines():
        raw_record: object = json.loads(line)
        if not isinstance(raw_record, dict):
            raise ValueError("investigation event record must be an object")
        raw_payload = cast(dict[str, object], raw_record).get("payload")
        stable_key = cast(dict[str, object], raw_record).get("stable_key")
        if not isinstance(raw_payload, dict):
            raise ValueError("investigation event payload must be an object")
        if not isinstance(stable_key, str):
            raise ValueError("investigation event stable_key must be a string")
        if stable_key in stable_keys:
            raise ValueError(
                "bundle contains duplicate investigation event stable keys"
            )
        stable_keys.add(stable_key)
        records.append((stable_key, cast(dict[str, Any], raw_payload)))
    return records


def _reasoning_effort(value: str | None) -> Any:
    if value not in {None, "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError(
            "--reasoning-effort must be minimal, low, medium, high, xhigh, or max"
        )
    return value
