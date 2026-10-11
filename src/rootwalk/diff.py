"""Deterministic, observation-only comparison of two RunBundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

_REQUIRED_FILES = (
    "task.json",
    "topology.json",
    "context_manifests.jsonl",
    "decisions.jsonl",
    "observations.jsonl",
    "report.json",
)
_VOLATILE_MANIFEST_FIELDS = {
    "artifact_index",
    "deadline_at",
    "domain_cursor",
    "file_hashes",
    "run_id",
}
_LAYER_ORDER = (
    "environment_manifest",
    "context_manifest",
    "decision",
    "observation",
    "evidence",
    "termination",
)
_CONTEXT_REF_FIELDS = (
    "observation_ids",
    "evidence_ids",
    "hypothesis_ids",
    "path_ids",
    "gap_refs",
    "feedback_refs",
)


def diff_run_bundles(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Compare two valid RunBundles without making a causal claim."""
    left_layers = _load_bundle(Path(left), "left")
    right_layers = _load_bundle(Path(right), "right")
    layers: dict[str, object] = {}
    first: dict[str, object] | None = None

    for name in _LAYER_ORDER:
        differences: list[dict[str, object]] = []
        left_items = left_layers[name]
        right_items = right_layers[name]
        for index in range(max(len(left_items), len(right_items))):
            left_item = left_items[index] if index < len(left_items) else None
            right_item = right_items[index] if index < len(right_items) else None
            left_value = _comparison_value(name, left_item)
            right_value = _comparison_value(name, right_item)
            if left_value == right_value:
                continue
            difference = {
                "index": index,
                "left": _summary(name, left_value),
                "right": _summary(name, right_value),
            }
            differences.append(difference)
            if first is None:
                first = {"layer": name, **difference}
        layers[name] = {"same": not differences, "differences": differences}

    return {"same": first is None, "first_divergence": first, "layers": layers}


def _load_bundle(root: Path, side: str) -> dict[str, list[object]]:
    if not root.is_dir():
        raise ValueError(f"{side} RunBundle is not a directory: {root}")
    manifest = _read_json(root / "manifest.json", side)
    if not isinstance(manifest, dict):
        raise ValueError(f"{side} manifest.json must contain a JSON object")
    manifest = cast(dict[str, Any], manifest)
    _validate_hashes(root, manifest, side)
    for filename in _REQUIRED_FILES:
        if filename not in manifest["file_hashes"]:
            raise ValueError(f"{side} manifest has no hash for {filename}")

    environment = {
        "manifest": {
            key: value
            for key, value in manifest.items()
            if key not in _VOLATILE_MANIFEST_FIELDS
        },
        "task": _read_json(root / "task.json", side),
        "topology": _read_json(root / "topology.json", side),
    }
    return {
        "environment_manifest": [environment],
        "context_manifest": _read_jsonl(root / "context_manifests.jsonl", side),
        "decision": _read_jsonl(root / "decisions.jsonl", side),
        "observation": _read_jsonl(root / "observations.jsonl", side),
        "evidence": _read_optional_jsonl(root, manifest, "evidence.jsonl", side),
        "termination": [_read_json(root / "report.json", side)],
    }


def _validate_hashes(root: Path, manifest: dict[str, Any], side: str) -> None:
    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, dict):
        raise ValueError(f"{side} manifest.file_hashes must be a JSON object")
    root = root.resolve()
    for raw_name, expected in cast(dict[object, object], file_hashes).items():
        if not isinstance(raw_name, str) or not isinstance(expected, str):
            raise ValueError(f"{side} manifest.file_hashes entries must be strings")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"{side} manifest has invalid SHA-256 for {raw_name}")
        path = root / raw_name
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"{side} bundle file is missing or unsafe: {raw_name}"
            ) from exc
        if path.is_symlink() or not resolved.is_file():
            raise ValueError(f"{side} bundle file is missing or unsafe: {raw_name}")
        if _sha256(resolved) != expected:
            raise ValueError(f"{side} bundle hash mismatch: {raw_name}")


def _read_json(path: Path, side: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {side} bundle file: {path.name}") from exc


def _read_jsonl(path: Path, side: str) -> list[object]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {side} bundle file: {path.name}") from exc
    records: list[object] = []
    revisioned: bool | None = None
    previous_revision: int | None = None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in {side} {path.name} line {line_number}"
            ) from exc
        if isinstance(record, dict):
            mapping = cast(dict[str, object], record)
            if "payload" in mapping:
                has_revision = "revision" in mapping
                if revisioned is not None and revisioned != has_revision:
                    raise ValueError(
                        f"{side} {path.name} mixes revisioned and legacy records"
                    )
                revisioned = has_revision
                if has_revision:
                    revision = mapping["revision"]
                    if not isinstance(revision, int) or isinstance(revision, bool):
                        raise ValueError(
                            f"{side} {path.name} line {line_number} "
                            "has invalid revision"
                        )
                    if previous_revision is not None and revision <= previous_revision:
                        raise ValueError(
                            f"{side} {path.name} line {line_number} revision "
                            f"{revision} is not strictly after {previous_revision}"
                        )
                    previous_revision = revision
                record = mapping["payload"]
        records.append(cast(object, record))
    return records


def _read_optional_jsonl(
    root: Path, manifest: dict[str, Any], filename: str, side: str
) -> list[object]:
    file_hashes = cast(dict[str, object], manifest["file_hashes"])
    if filename not in file_hashes:
        return []
    return _read_jsonl(root / filename, side)


def _context_refs(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    refs = cast(dict[str, object], value)
    projected: dict[str, list[str]] = {}
    for field in _CONTEXT_REF_FIELDS:
        items = refs.get(field)
        if isinstance(items, list):
            projected[field] = [
                item for item in cast(list[object], items) if isinstance(item, str)
            ]
    return projected


def _comparison_value(layer: str, value: object) -> object:
    if layer != "context_manifest" or not isinstance(value, dict):
        return value
    normalized = dict(cast(dict[str, object], value))
    for field in ("selected_refs", "omitted_refs"):
        if field in normalized:
            normalized[field] = _context_refs(normalized[field])
    return normalized


def _summary(layer: str, value: object) -> object:
    if value is None or not isinstance(value, dict):
        return value
    item = cast(dict[str, Any], value)
    fields = {
        "environment_manifest": (),
        "context_manifest": (
            "call_number",
            "status",
            "strategy",
            "message_hash",
            "estimated_tokens",
            "selected_refs",
            "omitted_refs",
        ),
        "decision": (
            "decision_id",
            "decision_summary",
            "intent",
            "operations",
        ),
        "observation": (
            "observation_id",
            "tool",
            "arguments",
            "status",
            "summary",
            "artifact_refs",
        ),
        "evidence": (
            "evidence_id",
            "observation_refs",
            "subject_ids",
            "predicate",
            "value",
            "unit",
            "observed_interval",
            "onset_interval",
            "time_uncertainty",
            "artifact_refs",
            "extractor",
            "source_fingerprint",
            "dataset_snapshot_id",
        ),
        "termination": (
            "status",
            "failure",
            "abstain_reason",
            "hypothesis_ids",
            "path_ids",
            "unresolved_questions",
        ),
    }[layer]
    summary: dict[str, Any] = {field: item[field] for field in fields if field in item}
    if layer == "environment_manifest":
        manifest = cast(dict[str, Any], item.get("manifest", {}))
        task = cast(dict[str, Any], item.get("task", {}))
        topology = cast(dict[str, Any], item.get("topology", {}))
        summary = {
            "environment": task.get("environment"),
            "task_id": task.get("task_id"),
            "data_version": manifest.get("data_version"),
            "model_provider": manifest.get("model_provider"),
            "model_id": manifest.get("model_id"),
            "topology_id": topology.get("snapshot_id"),
        }
    summary["digest"] = hashlib.sha256(_canonical(item)).hexdigest()
    return summary


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
