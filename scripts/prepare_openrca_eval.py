#!/usr/bin/env python3
"""Prepare the private OpenRCA label package for the frozen D9 suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import cast

try:
    from scripts import prepare_bank_suite as suite
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import prepare_bank_suite as suite  # pyright: ignore[reportMissingImports]

_ROOT = Path(__file__).resolve().parents[1]
_QUERY_FIELDS = ["task_index", "instruction", "scoring_points"]


def prepare_eval_package(
    query: str | Path,
    suite_spec: str | Path,
    output: str | Path,
) -> dict[str, int | str]:
    """Select frozen labels and atomically publish an evaluator-only package."""

    query_path = _input_file(Path(query), "query.csv")
    spec_path = _input_file(Path(suite_spec))
    output_path = _output_path(Path(output), query_path, spec_path)
    spec, plans = suite.plan_cases(spec_path)
    identity = suite.suite_identity(spec)
    if tuple(plan.row_id for plan in plans) != tuple(
        cast(tuple[int, ...], identity["row_ids"])
    ):
        raise ValueError("suite must contain the complete frozen case set")
    _verify_canonical_query_identity(query_path, spec)
    source_rows = _query_rows(query_path)

    selected: list[dict[str, str]] = []
    projection: list[dict[str, int | str]] = []
    for plan in plans:
        if plan.row_id >= len(source_rows):
            raise ValueError(f"canonical query.csv has no row {plan.row_id}")
        row = source_rows[plan.row_id]
        if row["task_index"] != plan.task_index:
            raise ValueError(f"row {plan.row_id} task_index does not match suite")
        if not row["scoring_points"]:
            raise ValueError(f"row {plan.row_id} has empty scoring_points")
        if hashlib.sha256(row["instruction"].encode()).hexdigest() != (
            plan.instruction_sha256
        ):
            raise ValueError(f"row {plan.row_id} instruction SHA-256 mismatch")
        selected.append(row)
        projection.append(
            {
                "instruction": row["instruction"],
                "row_id": plan.row_id,
                "task_index": row["task_index"],
            }
        )

    selection_hash = _canonical_sha256(projection)
    if selection_hash != spec["selection_sha256"]:
        raise ValueError("selected query rows do not match the frozen selection hash")
    case_plan_hash = suite.case_plan_sha256(plans)
    if case_plan_hash != identity["case_plan_sha256"]:
        raise ValueError("suite case plan SHA-256 does not match the frozen D9 plan")

    query_bytes = _csv_bytes(selected)
    alignment_bytes = _json_bytes(
        {
            "rows": [
                {
                    "source_row_id": plan.row_id,
                    "selected_query_row": index,
                    "prediction_row_id": plan.row_id,
                }
                for index, plan in enumerate(plans)
            ]
        }
    )
    files = {"query.csv": query_bytes, "alignment.json": alignment_bytes}
    _publish(output_path, files)
    return {
        "rows": len(selected),
        "selection_sha256": selection_hash,
        "case_plan_sha256": case_plan_hash,
        "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "alignment_sha256": hashlib.sha256(alignment_bytes).hexdigest(),
    }


def _input_file(path: Path, required_name: str | None = None) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("input files must not be symlinks")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("input path must be a regular file")
    if required_name is not None and resolved.name != required_name:
        raise ValueError(f"input must be named {required_name}")
    return resolved


def _verify_canonical_query_identity(query_path: Path, spec: dict[str, object]) -> None:
    if "source_mirror" not in spec:
        return
    sources = suite.mirror_source_index(spec)
    expected = sources.get("Bank/query.csv")
    if expected is None:
        raise ValueError("mirror suite does not declare Bank/query.csv")
    expected_size = expected.get("bytes")
    expected_hash = expected.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or not isinstance(expected_hash, str)
        or query_path.stat().st_size != expected_size
    ):
        raise ValueError("canonical query.csv size does not match suite")
    digest = hashlib.sha256()
    with query_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("canonical query.csv SHA-256 does not match suite")


def _output_path(output: Path, *inputs: Path) -> Path:
    if output.is_symlink():
        raise ValueError("output must not be a symlink")
    resolved = output.expanduser().resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError("output must not be a filesystem root")
    if resolved == _ROOT or _ROOT in resolved.parents or resolved in _ROOT.parents:
        raise ValueError("output must be outside the RootWalk Git worktree")
    if any(
        resolved == source or resolved in source.parents or source in resolved.parents
        for source in inputs
    ):
        raise ValueError("output must not overlap an input path")
    return resolved


def _query_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != _QUERY_FIELDS:
            raise ValueError(
                "canonical query.csv header must be " + ",".join(_QUERY_FIELDS)
            )
        rows: list[dict[str, str]] = []
        for number, raw in enumerate(reader, 2):
            if None in raw or any(raw[field] is None for field in _QUERY_FIELDS):
                raise ValueError(f"canonical query.csv row {number} is malformed")
            rows.append({field: raw[field] or "" for field in _QUERY_FIELDS})
        return rows


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_QUERY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _canonical_sha256(value: object) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _publish(output: Path, files: dict[str, bytes]) -> None:
    if output.exists():
        entries = list(output.iterdir()) if output.is_dir() else []
        actual = (
            {path.name: path.read_bytes() for path in entries}
            if not output.is_symlink()
            and all(path.is_file() and not path.is_symlink() for path in entries)
            else {}
        )
        modes = (
            {path.name: path.stat().st_mode & 0o777 for path in entries}
            if output.is_dir() and not output.is_symlink()
            else {}
        )
        if (
            actual == files
            and output.stat().st_mode & 0o777 == 0o700
            and all(mode == 0o600 for mode in modes.values())
        ):
            return
        raise FileExistsError("output already exists with different content")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        staging.chmod(0o700)
        for name, content in files.items():
            target = staging / name
            target.write_bytes(content)
            target.chmod(0o600)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_eval_package(args.query, args.suite, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
