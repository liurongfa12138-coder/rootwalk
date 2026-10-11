import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from rootwalk.context import ContextCompilation, compile_context
from rootwalk.contracts import AllowedScope, Task, TimeRange
from rootwalk.experiments import (
    build_experiment_matrix,
    compare_actual_model_inputs,
)
from rootwalk.runtime import RuntimeConfig


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _task(version: str = "v1") -> Task:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    window = TimeRange(start=start, end=start + timedelta(minutes=10))
    return Task(
        task_id=f"task-{version}",
        environment="test",
        question="Why is checkout unhealthy?",
        incident_range=window,
        timezone="UTC",
        allowed_query_range=window,
        entry_entities=["checkout"],
        allowed_scope=AllowedScope(
            entities=["checkout", "database"], tools=["query_metric"]
        ),
        requested_outputs=["root_cause"],
        dataset_manifest_ref=f"dataset-{version}",
    )


def _compile(
    strategy: str,
    validator_mode: str,
    *,
    task: Task | None = None,
    tools: dict[str, str] | None = None,
    topology: dict[str, object] | None = None,
) -> ContextCompilation:
    return compile_context(
        task or _task(),
        tools or {"query_metric": "query_metric(name: str)"},
        {"max_model_calls": 7, "model_calls_used": 0},
        revision=9,
        observations=[
            {
                "observation_id": "observation-1",
                "attempt_id": "attempt-1",
                "tool": "query_metric",
                "status": "success",
                "summary": "database latency increased",
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence-1",
                "subject_ids": ["database"],
                "predicate": "latency_increased",
                "value": True,
                "source_fingerprint": "source-1",
            }
        ],
        hypotheses=[
            {
                "hypothesis_id": "hypothesis-1",
                "subject": "database",
                "status": "open",
                "supporting_evidence_refs": ["evidence-1"],
            }
        ],
        paths=[
            {
                "path_id": "path-1",
                "hypothesis_id": "hypothesis-1",
                "root_cause_entity_id": "database",
                "symptom_entity_ids": ["checkout"],
                "grounding": {"quality_errors": ["missing support"]},
            }
        ],
        feedback=[
            {
                "feedback_id": "feedback-path",
                "type": "path_quality",
                "error": "missing support",
            },
            {
                "feedback_id": "feedback-exit",
                "type": "exit_gate",
                "error": "unsupported final",
            },
        ],
        strategy=cast(Any, strategy),
        validator_mode=cast(Any, validator_mode),
        max_tokens=20_000,
        topology_summary=topology,
    )


def _save_context(
    root: Path,
    compiled: ContextCompilation,
    config: RuntimeConfig,
    *,
    task: Task | None = None,
    topology_prompt: bool | None = None,
) -> Path:
    root.mkdir()
    (root / "artifacts").mkdir()
    task_payload = (task or _task()).model_dump(mode="json")
    task_bytes = _canonical(task_payload)
    context_bytes = _canonical(compiled.messages)
    assert (
        hashlib.sha256(context_bytes).hexdigest() == compiled.manifest["message_hash"]
    )
    context_manifest = {**compiled.manifest, "call_number": 1}
    context_manifest_bytes = _canonical({"payload": context_manifest}) + b"\n"
    (root / "task.json").write_bytes(task_bytes)
    (root / "context_manifests.jsonl").write_bytes(context_manifest_bytes)
    artifact_name = f"artifacts/{compiled.manifest['message_hash']}"
    (root / artifact_name).write_bytes(context_bytes)
    config_values = asdict(config)
    environment = {
        "rootwalk_version": "test",
        "model_provider": config.model_provider,
        "model_id": config.model_id,
        "model_endpoint": config.model_endpoint,
        "model_timeout_seconds": config.model_timeout_seconds,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": "test-prompt",
        "tool_version": "test-tools",
        "code_revision": "a" * 64,
        "uv_lock_sha256": "b" * 64,
        "direct_dependency_versions": {"dependency": "1.0"},
        "direct_dependency_fingerprint": "c" * 64,
        "observability": {
            "enabled": False,
            "capture_content": False,
            "sample_ratio": 1.0,
            "endpoint_sha256": None,
        },
        "data_version": "dataset-v1",
        **{
            key: config_values[key]
            for key in (
                "wall_clock_seconds",
                "cleanup_grace_seconds",
                "max_model_calls",
                "max_input_tokens",
                "max_output_tokens",
                "max_output_tokens_per_call",
                "max_tool_attempts",
                "max_artifact_bytes",
                "tool_timeout_seconds",
                "max_validator_retries",
                "max_no_progress_rounds",
            )
        },
    }
    if topology_prompt is not None:
        environment["experiment"] = {"topology_prompt": topology_prompt}
    environment["file_hashes"] = {
        "task.json": hashlib.sha256(task_bytes).hexdigest(),
        "context_manifests.jsonl": hashlib.sha256(context_manifest_bytes).hexdigest(),
        artifact_name: hashlib.sha256(context_bytes).hexdigest(),
    }
    (root / "manifest.json").write_bytes(_canonical(environment))
    return root


def test_fixed_matrix_reuses_c1_as_g0_and_preserves_other_config() -> None:
    base = RuntimeConfig(
        max_model_calls=7,
        max_artifact_bytes=12_345,
        model_provider="recorded",
        model_id="fixed-model",
    )

    matrix = build_experiment_matrix(base, include_topology=True)

    assert list(matrix) == ["C0", "C1", "G0", "G1", "T0", "T1"]
    assert (
        matrix["C0"].runtime_config.context_strategy,
        matrix["C0"].runtime_config.validator_mode,
    ) == ("recent", "observe_only")
    assert (
        matrix["G1"].runtime_config.context_strategy,
        matrix["G1"].runtime_config.validator_mode,
    ) == ("evidence", "enforce")
    assert matrix["C1"].runtime_config is matrix["G0"].runtime_config
    assert matrix["G0"].reuse_of == "C1"
    assert matrix["T0"].metadata == {"topology_prompt": False}
    assert matrix["T1"].metadata == {"topology_prompt": True}
    assert all(arm.runtime_config.model_id == "fixed-model" for arm in matrix.values())
    assert all(arm.runtime_config.max_model_calls == 7 for arm in matrix.values())
    assert all(
        arm.runtime_config.max_artifact_bytes == 12_345 for arm in matrix.values()
    )


def test_recorded_context_comparison_reports_only_actual_mechanism_differences(
    tmp_path: Path,
) -> None:
    base = RuntimeConfig(
        max_model_calls=7,
        model_provider="recorded",
        model_id="fixed-model",
    )
    matrix = build_experiment_matrix(base)
    c0 = _save_context(
        tmp_path / "c0",
        _compile("recent", "observe_only"),
        matrix["C0"].runtime_config,
    )
    c1 = _save_context(
        tmp_path / "c1",
        _compile("evidence", "observe_only"),
        matrix["C1"].runtime_config,
    )
    g1 = _save_context(
        tmp_path / "g1",
        _compile("evidence", "enforce"),
        matrix["G1"].runtime_config,
    )

    context_result = compare_actual_model_inputs(
        c0, c1, target_variable="context_strategy"
    )
    gate_result = compare_actual_model_inputs(c1, g1, target_variable="validator_mode")

    assert context_result["all_message_hashes_equal"] is False
    assert context_result["all_message_content_equal"] is False
    assert context_result["calls"][0]["changed_message_indices"] == [1]
    assert set(context_result["calls"][0]["changed_context_keys"]) == {
        "context_policy",
        "evidence",
        "paths",
    }
    assert context_result == compare_actual_model_inputs(
        c0, c1, target_variable="context_strategy"
    )
    assert gate_result["conditions"]["control"] == {
        "condition_id": "G0",
        "recorded_as": "C1",
        "reused": True,
    }
    assert gate_result["recorded_conditions"] == ["C1", "G1"]
    assert gate_result["observe_only_feedback_check"]["control"] == {
        "applies": True,
        "passed": True,
        "checked_call_numbers": [1],
        "path_quality": 0,
        "exit_gate": 0,
    }
    assert gate_result["observe_only_feedback_check"]["treatment"]["path_quality"] > 0
    assert gate_result["observe_only_feedback_check"]["treatment"]["exit_gate"] > 0
    assert "benefit" not in json.dumps(gate_result)
    assert "improvement" not in json.dumps(gate_result)


def test_optional_topology_metadata_is_compared_against_actual_inputs(
    tmp_path: Path,
) -> None:
    base = RuntimeConfig(
        max_model_calls=7,
        model_provider="recorded",
        model_id="fixed-model",
    )
    matrix = build_experiment_matrix(base, include_topology=True)
    t0 = _save_context(
        tmp_path / "t0",
        _compile("evidence", "observe_only"),
        matrix["T0"].runtime_config,
        topology_prompt=False,
    )
    t1 = _save_context(
        tmp_path / "t1",
        _compile(
            "evidence",
            "observe_only",
            topology={"nodes": [{"id": "checkout"}]},
        ),
        matrix["T1"].runtime_config,
        topology_prompt=True,
    )

    result = compare_actual_model_inputs(t0, t1, target_variable="topology_prompt")

    assert result["control_value"] is False
    assert result["treatment_value"] is True
    assert result["calls"][0]["changed_context_keys"] == ["topology"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("model_id", "other-model", "model differs"),
        ("model_timeout_seconds", 99.0, "model differs"),
        ("max_model_calls", 8, "budget"),
        ("max_artifact_bytes", 1, "budget differs"),
        ("data_version", "dataset-v2", "data_version differs"),
        ("uv_lock_sha256", "d" * 64, "implementation_versions differs"),
        (
            "direct_dependency_fingerprint",
            "e" * 64,
            "implementation_versions differs",
        ),
        (
            "observability",
            {"enabled": True},
            "observability differs",
        ),
    ],
)
def test_comparison_rejects_changed_controls(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    base = RuntimeConfig(
        max_model_calls=7,
        model_provider="recorded",
        model_id="fixed-model",
    )
    matrix = build_experiment_matrix(base)
    c0 = _save_context(
        tmp_path / "c0",
        _compile("recent", "observe_only"),
        matrix["C0"].runtime_config,
    )
    c1 = _save_context(
        tmp_path / "c1",
        _compile("evidence", "observe_only"),
        matrix["C1"].runtime_config,
    )
    manifest_path = c1 / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    manifest[field] = value
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(ValueError, match=error):
        compare_actual_model_inputs(c0, c1, target_variable="context_strategy")


def test_comparison_rejects_changed_tool_catalog_and_task(tmp_path: Path) -> None:
    base = RuntimeConfig(
        max_model_calls=7,
        model_provider="recorded",
        model_id="fixed-model",
    )
    matrix = build_experiment_matrix(base)
    c0 = _save_context(
        tmp_path / "c0",
        _compile("recent", "observe_only"),
        matrix["C0"].runtime_config,
    )
    changed_tools = _save_context(
        tmp_path / "tools",
        _compile(
            "evidence",
            "observe_only",
            tools={"query_metric": "query_metric(other: str)"},
        ),
        matrix["C1"].runtime_config,
    )
    changed_task_value = _task("v2")
    changed_task = _save_context(
        tmp_path / "task",
        _compile("evidence", "observe_only", task=changed_task_value),
        matrix["C1"].runtime_config,
        task=changed_task_value,
    )

    with pytest.raises(ValueError, match="tools differs"):
        compare_actual_model_inputs(
            c0, changed_tools, target_variable="context_strategy"
        )
    with pytest.raises(ValueError, match="task differs"):
        compare_actual_model_inputs(
            c0, changed_task, target_variable="context_strategy"
        )


def test_comparison_rejects_tampered_context_artifact(tmp_path: Path) -> None:
    base = RuntimeConfig(
        max_model_calls=7,
        model_provider="recorded",
        model_id="fixed-model",
    )
    matrix = build_experiment_matrix(base)
    c0 = _save_context(
        tmp_path / "c0",
        _compile("recent", "observe_only"),
        matrix["C0"].runtime_config,
    )
    c1 = _save_context(
        tmp_path / "c1",
        _compile("evidence", "observe_only"),
        matrix["C1"].runtime_config,
    )
    artifact = next((c1 / "artifacts").iterdir())
    artifact.write_text("tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        compare_actual_model_inputs(c0, c1, target_variable="context_strategy")
