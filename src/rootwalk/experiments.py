"""Minimal, mechanism-only experiment definitions and context comparison."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from .runtime import RuntimeConfig

ExperimentVariable = Literal["context_strategy", "validator_mode", "topology_prompt"]
ExperimentValue = str | bool

_MODEL_FIELDS = (
    "model_provider",
    "model_id",
    "model_endpoint",
    "model_timeout_seconds",
    "reasoning_effort",
)
_BUDGET_FIELDS = (
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
_FEEDBACK_MARKERS = ("path_quality", "exit_gate")
_MISSING = object()


def _empty_metadata() -> dict[str, bool]:
    return {}


@dataclass(frozen=True, slots=True)
class ExperimentArm:
    """One fixed mechanism condition; ``reuse_of`` means no separate run."""

    condition_id: str
    target_variable: ExperimentVariable
    target_value: ExperimentValue
    runtime_config: RuntimeConfig
    metadata: Mapping[str, bool] = field(default_factory=_empty_metadata)
    reuse_of: str | None = None


def build_experiment_matrix(
    base_config: RuntimeConfig | None = None,
    *,
    include_topology: bool = False,
) -> dict[str, ExperimentArm]:
    """Build the fixed M3 mechanism matrix while preserving other settings."""
    base = base_config or RuntimeConfig()
    recent_observe = replace(
        base, context_strategy="recent", validator_mode="observe_only"
    )
    evidence_observe = replace(
        base, context_strategy="evidence", validator_mode="observe_only"
    )
    evidence_enforce = replace(
        base, context_strategy="evidence", validator_mode="enforce"
    )
    arms = {
        "C0": ExperimentArm("C0", "context_strategy", "recent", recent_observe),
        "C1": ExperimentArm("C1", "context_strategy", "evidence", evidence_observe),
        "G0": ExperimentArm(
            "G0",
            "validator_mode",
            "observe_only",
            evidence_observe,
            reuse_of="C1",
        ),
        "G1": ExperimentArm("G1", "validator_mode", "enforce", evidence_enforce),
    }
    if include_topology:
        arms.update(
            {
                "T0": ExperimentArm(
                    "T0",
                    "topology_prompt",
                    False,
                    evidence_observe,
                    {"topology_prompt": False},
                ),
                "T1": ExperimentArm(
                    "T1",
                    "topology_prompt",
                    True,
                    evidence_observe,
                    {"topology_prompt": True},
                ),
            }
        )
    return arms


@dataclass(frozen=True, slots=True)
class _ContextCall:
    call_number: int
    manifest: dict[str, Any]
    messages: list[dict[str, str]]
    context: dict[str, Any]
    raw: bytes


@dataclass(frozen=True, slots=True)
class _SavedContexts:
    environment: dict[str, Any]
    task: dict[str, Any]
    calls: tuple[_ContextCall, ...]


def compare_actual_model_inputs(
    control_bundle: str | Path,
    treatment_bundle: str | Path,
    *,
    target_variable: ExperimentVariable,
) -> dict[str, Any]:
    """Compare saved model inputs after validating the experiment controls."""
    if target_variable not in {
        "context_strategy",
        "validator_mode",
        "topology_prompt",
    }:
        raise ValueError("unsupported experiment target variable")
    control = _load_saved_contexts(Path(control_bundle), "control")
    treatment = _load_saved_contexts(Path(treatment_bundle), "treatment")
    controls = _controlled_values(control, "control")
    treatment_controls = _controlled_values(treatment, "treatment")
    for name, value in controls.items():
        if value != treatment_controls[name]:
            raise ValueError(f"{name} differs outside target variable")

    control_value = _target_value(control, target_variable, "control")
    treatment_value = _target_value(treatment, target_variable, "treatment")
    expected = {
        "context_strategy": ("recent", "evidence"),
        "validator_mode": ("observe_only", "enforce"),
        "topology_prompt": (False, True),
    }[target_variable]
    if (control_value, treatment_value) != expected:
        raise ValueError(
            f"{target_variable} must compare fixed values {expected[0]!r} and "
            f"{expected[1]!r}"
        )
    _validate_non_target_policy(control, treatment, target_variable)

    condition_ids = {
        "context_strategy": ("C0", "C1"),
        "validator_mode": ("G0", "G1"),
        "topology_prompt": ("T0", "T1"),
    }[target_variable]
    matrix = build_experiment_matrix(
        include_topology=target_variable == "topology_prompt"
    )
    control_arm, treatment_arm = (matrix[item] for item in condition_ids)
    call_numbers = sorted(
        {item.call_number for item in control.calls}
        | {item.call_number for item in treatment.calls}
    )
    control_calls = {item.call_number: item for item in control.calls}
    treatment_calls = {item.call_number: item for item in treatment.calls}
    calls = [
        _compare_call(
            call_number,
            control_calls.get(call_number),
            treatment_calls.get(call_number),
        )
        for call_number in call_numbers
    ]
    feedback_checks = {
        "control": _feedback_check(control),
        "treatment": _feedback_check(treatment),
    }
    for side, check in feedback_checks.items():
        if check["applies"] and not check["passed"]:
            raise ValueError(
                f"{side} observe_only input contains path_quality/exit_gate feedback"
            )

    changed_calls = [
        cast(int, item["call_number"])
        for item in calls
        if not item["message_content_equal"]
    ]
    return {
        "target_variable": target_variable,
        "control_value": control_value,
        "treatment_value": treatment_value,
        "conditions": {
            "control": _condition_record(control_arm),
            "treatment": _condition_record(treatment_arm),
        },
        "recorded_conditions": sorted(
            {
                control_arm.reuse_of or control_arm.condition_id,
                treatment_arm.reuse_of or treatment_arm.condition_id,
            }
        ),
        "controlled_variables": {name: True for name in controls},
        "all_message_hashes_equal": all(item["message_hash_equal"] for item in calls),
        "all_message_content_equal": not changed_calls,
        "changed_call_numbers": changed_calls,
        "calls": calls,
        "observe_only_feedback_check": feedback_checks,
    }


def _condition_record(arm: ExperimentArm) -> dict[str, object]:
    recorded_as = arm.reuse_of or arm.condition_id
    return {
        "condition_id": arm.condition_id,
        "recorded_as": recorded_as,
        "reused": arm.reuse_of is not None,
    }


def _load_saved_contexts(root: Path, side: str) -> _SavedContexts:
    if not root.is_dir():
        raise ValueError(f"{side} bundle is not a directory: {root}")
    environment = _read_json_object(root, root / "manifest.json", side)
    task = _read_json_object(root, root / "task.json", side)
    _verify_declared_file(root, environment, "task.json", side)
    _verify_declared_file(root, environment, "context_manifests.jsonl", side)
    manifests = _read_jsonl_objects(root, root / "context_manifests.jsonl", side)
    calls = tuple(
        sorted(
            (_load_call(root, environment, manifest, side) for manifest in manifests),
            key=lambda item: item.call_number,
        )
    )
    if not calls:
        raise ValueError(f"{side} has no saved context manifests")
    if len({item.call_number for item in calls}) != len(calls):
        raise ValueError(f"{side} has duplicate context call numbers")
    for call in calls:
        if call.context.get("task") != task:
            raise ValueError(f"{side} context task does not match task.json")
    return _SavedContexts(environment, task, calls)


def _load_call(
    root: Path,
    environment: dict[str, Any],
    manifest: dict[str, Any],
    side: str,
) -> _ContextCall:
    call_number = manifest.get("call_number")
    if not isinstance(call_number, int) or isinstance(call_number, bool):
        raise ValueError(f"{side} context manifest has invalid call_number")
    message_hash = manifest.get("message_hash")
    model_input_ref = manifest.get("model_input_ref")
    if not _is_hash(message_hash) or model_input_ref != message_hash:
        raise ValueError(f"{side} context manifest has invalid message hash")
    artifact_name = f"artifacts/{message_hash}"
    _verify_declared_file(root, environment, artifact_name, side)
    raw = _read_bytes(root, root / artifact_name, side)
    if hashlib.sha256(raw).hexdigest() != message_hash:
        raise ValueError(f"{side} context artifact hash mismatch")
    try:
        decoded: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{side} context artifact is not valid JSON") from exc
    messages = _messages(decoded, side)
    users = [item for item in messages if item["role"] == "user"]
    if len(users) != 1:
        raise ValueError(f"{side} context must contain exactly one user message")
    try:
        context: object = json.loads(users[0]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{side} user context is not valid JSON") from exc
    if not isinstance(context, dict):
        raise ValueError(f"{side} user context must be a JSON object")
    context = cast(dict[str, Any], context)
    policy = context.get("context_policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{side} user context has no context_policy")
    policy = cast(dict[str, Any], policy)
    for key in ("strategy", "validator_mode"):
        if policy.get(key) != manifest.get(key):
            raise ValueError(f"{side} context {key} does not match its manifest")
    return _ContextCall(call_number, manifest, messages, context, raw)


def _controlled_values(run: _SavedContexts, side: str) -> dict[str, object]:
    environment = run.environment
    model = _required_fields(environment, _MODEL_FIELDS, side)
    budget = _required_fields(environment, _BUDGET_FIELDS, side)
    wall_clock_limit = environment.get("wall_clock_limit_seconds", _MISSING)
    if wall_clock_limit is _MISSING:
        wall_clock_limit = _required(environment, "wall_clock_seconds", side)
    budget["wall_clock_limit_seconds"] = wall_clock_limit
    context_limits = {item.manifest.get("max_context_tokens") for item in run.calls}
    if None in context_limits or len(context_limits) != 1:
        raise ValueError(f"{side} has inconsistent context token budgets")
    budget["max_context_tokens"] = context_limits.pop()
    tool_version = _required(environment, "tool_version", side)
    prompt_version = _required(environment, "prompt_version", side)
    rootwalk_version = _required(environment, "rootwalk_version", side)
    code_revision = _required(environment, "code_revision", side)
    uv_lock_sha256 = _required(environment, "uv_lock_sha256", side)
    dependency_versions = _required(environment, "direct_dependency_versions", side)
    dependency_fingerprint = _required(
        environment, "direct_dependency_fingerprint", side
    )
    observability = _required(environment, "observability", side)
    data_version = _required(environment, "data_version", side)
    catalogs = [item.context.get("tools") for item in run.calls]
    if any(not isinstance(item, dict) for item in catalogs):
        raise ValueError(f"{side} context has an invalid tool catalog")
    if any(item != catalogs[0] for item in catalogs[1:]):
        raise ValueError(f"{side} tool catalog changes between calls")
    policy_versions = {
        (
            item.manifest.get("policy_version"),
            item.manifest.get("token_estimator_version"),
        )
        for item in run.calls
    }
    if any(None in item for item in policy_versions) or len(policy_versions) != 1:
        raise ValueError(f"{side} has inconsistent context implementation versions")
    max_model_calls = environment["max_model_calls"]
    for item in run.calls:
        context_budget = item.context.get("budget")
        if not isinstance(context_budget, dict):
            raise ValueError(f"{side} context has an invalid budget")
        context_budget = cast(dict[str, Any], context_budget)
        if context_budget.get("max_model_calls") != max_model_calls:
            raise ValueError(f"{side} context budget does not match manifest")
    return {
        "model": model,
        "tools": {"version": tool_version, "catalog": catalogs[0]},
        "budget": budget,
        "task": run.task,
        "data_version": data_version,
        "observability": observability,
        "implementation_versions": {
            "rootwalk": rootwalk_version,
            "prompt": prompt_version,
            "tool": tool_version,
            "code_revision": code_revision,
            "uv_lock_sha256": uv_lock_sha256,
            "direct_dependency_versions": dependency_versions,
            "direct_dependency_fingerprint": dependency_fingerprint,
            "context": next(iter(policy_versions)),
        },
    }


def _target_value(
    run: _SavedContexts, target: ExperimentVariable, side: str
) -> ExperimentValue:
    if target == "topology_prompt":
        experiment = run.environment.get("experiment", {})
        if not isinstance(experiment, dict):
            raise ValueError(f"{side} experiment metadata must be an object")
        experiment = cast(dict[str, Any], experiment)
        value = experiment.get(
            "topology_prompt", run.environment.get("topology_prompt")
        )
        if not isinstance(value, bool):
            raise ValueError(f"{side} has no boolean topology_prompt metadata")
        return value
    manifest_field = {
        "context_strategy": "strategy",
        "validator_mode": "validator_mode",
    }[target]
    values = {item.manifest.get(manifest_field) for item in run.calls}
    if None in values or len(values) != 1:
        raise ValueError(f"{side} has inconsistent {target}")
    return cast(str, values.pop())


def _validate_non_target_policy(
    control: _SavedContexts,
    treatment: _SavedContexts,
    target: ExperimentVariable,
) -> None:
    for variable in ("context_strategy", "validator_mode"):
        if variable == target:
            continue
        left = _target_value(control, cast(ExperimentVariable, variable), "control")
        right = _target_value(
            treatment, cast(ExperimentVariable, variable), "treatment"
        )
        if left != right:
            raise ValueError(f"{variable} differs outside target variable")


def _compare_call(
    call_number: int,
    control: _ContextCall | None,
    treatment: _ContextCall | None,
) -> dict[str, object]:
    if control is None or treatment is None:
        return {
            "call_number": call_number,
            "control_message_hash": (
                control.manifest["message_hash"] if control else None
            ),
            "treatment_message_hash": (
                treatment.manifest["message_hash"] if treatment else None
            ),
            "message_hash_equal": False,
            "message_content_equal": False,
            "changed_message_indices": None,
            "changed_context_keys": None,
        }
    changed_indices = [
        index
        for index in range(max(len(control.messages), len(treatment.messages)))
        if _at(control.messages, index) != _at(treatment.messages, index)
    ]
    changed_keys = sorted(
        key
        for key in control.context.keys() | treatment.context.keys()
        if control.context.get(key, _MISSING) != treatment.context.get(key, _MISSING)
    )
    return {
        "call_number": call_number,
        "control_message_hash": control.manifest["message_hash"],
        "treatment_message_hash": treatment.manifest["message_hash"],
        "message_hash_equal": (
            control.manifest["message_hash"] == treatment.manifest["message_hash"]
        ),
        "message_content_equal": control.messages == treatment.messages,
        "artifact_bytes_equal": control.raw == treatment.raw,
        "changed_message_indices": changed_indices,
        "changed_context_keys": changed_keys,
    }


def _feedback_check(run: _SavedContexts) -> dict[str, object]:
    mode = _target_value(run, "validator_mode", "saved contexts")
    counts = {marker: 0 for marker in _FEEDBACK_MARKERS}
    for call in run.calls:
        _count_feedback_markers(call.context, counts)
    applies = mode == "observe_only"
    return {
        "applies": applies,
        "passed": not applies or not any(counts.values()),
        "checked_call_numbers": [item.call_number for item in run.calls],
        **counts,
    }


def _count_feedback_markers(value: object, counts: dict[str, int]) -> None:
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        for key, item in mapping.items():
            marker = str(key).strip().lower()
            if marker in counts:
                counts[marker] += 1
            if marker in {"type", "kind", "category", "check", "validator"}:
                item_marker = str(item).strip().lower()
                if item_marker in counts:
                    counts[item_marker] += 1
            _count_feedback_markers(item, counts)
    elif isinstance(value, list):
        for item in cast(list[object], value):
            _count_feedback_markers(item, counts)


def _messages(value: object, side: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{side} context artifact must contain a message list")
    messages: list[dict[str, str]] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            raise ValueError(f"{side} context artifact has an invalid message")
        mapping = cast(dict[object, object], item)
        role, content = mapping.get("role"), mapping.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"{side} context artifact has an invalid message")
        messages.append({"role": role, "content": content})
    return messages


def _read_json_object(root: Path, path: Path, side: str) -> dict[str, Any]:
    try:
        value: object = json.loads(_read_bytes(root, path, side))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {side} {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{side} {path.name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _read_jsonl_objects(root: Path, path: Path, side: str) -> list[dict[str, Any]]:
    try:
        lines = _read_bytes(root, path, side).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"cannot read {side} {path.name}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in {side} {path.name} line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{side} context manifest must be a JSON object")
        record = cast(dict[str, Any], value)
        payload = record.get("payload", record)
        if not isinstance(payload, dict):
            raise ValueError(f"{side} context manifest payload must be an object")
        records.append(cast(dict[str, Any], payload))
    return records


def _read_bytes(root: Path, path: Path, side: str) -> bytes:
    root = root.resolve()
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{side} file is missing or unsafe: {path.name}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{side} file is missing or unsafe: {path.name}")
    return resolved.read_bytes()


def _verify_declared_file(
    root: Path, environment: Mapping[str, Any], name: str, side: str
) -> None:
    hashes = environment.get("file_hashes")
    if hashes is None:
        return
    if not isinstance(hashes, dict):
        raise ValueError(f"{side} manifest has no valid hash for {name}")
    hashes = cast(dict[str, Any], hashes)
    if not _is_hash(hashes.get(name)):
        raise ValueError(f"{side} manifest has no valid hash for {name}")
    actual = hashlib.sha256(_read_bytes(root, root / name, side)).hexdigest()
    if actual != hashes[name]:
        raise ValueError(f"{side} bundle hash mismatch: {name}")


def _required_fields(
    source: Mapping[str, Any], fields: tuple[str, ...], side: str
) -> dict[str, object]:
    return {field: _required(source, field, side) for field in fields}


def _required(source: Mapping[str, Any], field: str, side: str) -> object:
    if field not in source:
        raise ValueError(f"{side} manifest has no {field}")
    return source[field]


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(item in "0123456789abcdef" for item in value)
    )


def _at(items: list[dict[str, str]], index: int) -> object:
    return items[index] if index < len(items) else None
