#!/usr/bin/env python3
"""Run the prepared, label-free Bank D9 matrix through the existing CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from rootwalk.benchmark import summarize_benchmark
from rootwalk.query_worker import QueryWorker, required_bwrap_metadata
from rootwalk.runtime import RuntimeConfig
from rootwalk.version import runtime_version_manifest, source_tree_revision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_bank_suite as bank_suite  # noqa: E402

# Backward-compatible names for callers that import the original v1 constants.
CASE_PLAN_SHA256 = bank_suite.CASE_PLAN_SHA256
DEFAULT_ROW_IDS = bank_suite.DEFAULT_ROW_IDS
SELECTION_SHA256 = bank_suite.SELECTION_SHA256
case_plan_sha256 = bank_suite.case_plan_sha256
plan_cases = bank_suite.plan_cases
suite_identity = bank_suite.suite_identity

CONDITIONS = (
    ("C0", "c0", "recent", "observe_only"),
    ("C1/G0", "c1-g0", "evidence", "observe_only"),
    ("G1", "g1", "evidence", "enforce"),
)
_BUDGET_FIELDS = (
    "cleanup_grace_seconds",
    "max_model_calls",
    "max_context_tokens",
    "max_input_tokens",
    "max_output_tokens",
    "max_output_tokens_per_call",
    "max_tool_attempts",
    "max_artifact_bytes",
    "tool_timeout_seconds",
    "max_validator_retries",
    "max_no_progress_rounds",
)
_DISABLED_OBSERVABILITY: dict[str, object] = {
    "enabled": False,
    "capture_content": False,
    "sample_ratio": 1.0,
    "endpoint_sha256": None,
}


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    public_input: Path
    row_id: int
    task_id: str
    manifest_sha256: str
    suite_id: str
    suite_spec_sha256: str


@dataclass(frozen=True, slots=True)
class Plan:
    case: Case
    condition: str
    slug: str
    strategy: str
    validator_mode: str
    repeat: int
    run_id: str
    output_root: Path
    bundle: Path
    prediction: Path


@dataclass(frozen=True, slots=True)
class Settings:
    model: str
    base_url: str
    reasoning_effort: str | None
    model_timeout_seconds: float
    max_output_tokens: int
    run_timeout_seconds: float
    export_timeout_seconds: float
    query_sandbox_bwrap_version: str | None
    query_sandbox_bwrap_executable_sha256: str | None
    max_model_calls: int = 12
    max_input_tokens: int = 144_000
    max_tool_attempts: int = 40
    max_validator_retries: int = 2
    max_no_progress_rounds: int = 3
    model_provider: str = "ark"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument(
        "--suite-spec",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks/openrca-bank-d9-v1.json",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-provider", choices=("ark",), default="ark")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pricing-json", type=Path)
    parser.add_argument("--rootwalk-bin", default="rootwalk")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--model-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, default=2_000)
    parser.add_argument("--max-model-calls", type=int, default=12)
    parser.add_argument("--max-input-tokens", type=int, default=144_000)
    parser.add_argument("--max-tool-attempts", type=int, default=40)
    parser.add_argument("--max-validator-retries", type=int, default=2)
    parser.add_argument("--max-no-progress-rounds", type=int, default=3)
    parser.add_argument("--run-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--export-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--confirm-paid-run", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument(
        "--condition",
        action="append",
        choices=[condition for condition, _slug, _strategy, _mode in CONDITIONS],
    )
    parser.add_argument("--exclude-row-id", action="append", default=[], type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.repeats < 1:
            raise ValueError("--repeats must be at least one")
        if args.max_new_runs is not None and args.max_new_runs < 1:
            raise ValueError("--max-new-runs must be a positive integer")
        if args.max_model_calls < 1 or args.max_input_tokens < 1:
            raise ValueError("model call and input token budgets must be positive")
        if args.max_tool_attempts < 0:
            raise ValueError("--max-tool-attempts must be non-negative")
        if args.max_validator_retries < 0:
            raise ValueError("--max-validator-retries must be non-negative")
        if args.max_no_progress_rounds < 1:
            raise ValueError("--max-no-progress-rounds must be positive")
        if not args.dry_run and not args.confirm_paid_run:
            raise ValueError("non-dry-run execution requires --confirm-paid-run")
        if (
            min(
                args.model_timeout_seconds,
                args.max_output_tokens,
                args.run_timeout_seconds,
                args.export_timeout_seconds,
            )
            <= 0
        ):
            raise ValueError("timeouts and max output tokens must be positive")
        suite, cases = _suite(args.suite_root, args.suite_spec)
        excluded_row_ids = sorted(cast(list[int], args.exclude_row_id))
        if len(excluded_row_ids) != len(set(excluded_row_ids)):
            raise ValueError("--exclude-row-id values must be unique")
        frozen_row_ids = {case.row_id for case in cases}
        unknown_row_ids = sorted(set(excluded_row_ids) - frozen_row_ids)
        if unknown_row_ids:
            raise ValueError(
                "--exclude-row-id values are not in the frozen suite: "
                f"{unknown_row_ids}"
            )
        excluded = set(excluded_row_ids)
        cases = [case for case in cases if case.row_id not in excluded]
        if not cases:
            raise ValueError("--exclude-row-id must leave at least one case")
        selected_row_ids = sorted(case.row_id for case in cases)
        requested_conditions = set(cast(list[str] | None, args.condition) or ())
        conditions = tuple(
            item
            for item in CONDITIONS
            if not requested_conditions or item[0] in requested_conditions
        )
        output = _output_root(args.output_root, args.suite_root)
        plans = _plans(cases, output, args.repeats, conditions)
        pricing = _pricing(args.pricing_json)
        run_plan = {
            "schema_version": "rootwalk-bank-run-plan-v1",
            "suite_id": suite["suite_id"],
            "suite_spec_sha256": suite["suite_spec_sha256"],
            "suite_manifest_sha256": _sha256(
                args.suite_root.expanduser().resolve(strict=True)
                / "suite_manifest.json"
            ),
            "case_manifest_sha256s": {
                case.case_id: case.manifest_sha256 for case in cases
            },
            "selected_row_ids": selected_row_ids,
            "excluded_row_ids": excluded_row_ids,
            "repeats": args.repeats,
            "run_ids": [plan.run_id for plan in plans],
        }
        if args.condition is not None:
            run_plan["conditions"] = [item[0] for item in conditions]
        budget_settings = {
            "max_model_calls": args.max_model_calls,
            "max_input_tokens": args.max_input_tokens,
            "max_tool_attempts": args.max_tool_attempts,
            "max_validator_retries": args.max_validator_retries,
            "max_no_progress_rounds": args.max_no_progress_rounds,
        }
        if budget_settings != {
            "max_model_calls": 12,
            "max_input_tokens": 144_000,
            "max_tool_attempts": 40,
            "max_validator_retries": 2,
            "max_no_progress_rounds": 3,
        }:
            run_plan["runtime_budget_settings"] = budget_settings
        sandbox_version: str | None = None
        sandbox_sha256: str | None = None
        if not args.dry_run:
            _freeze_run_plan(output, run_plan)
            probe = QueryWorker(cases[0].public_input, sandbox="required")
            try:
                sandbox_version = probe.bwrap_version
                sandbox_sha256 = probe.bwrap_executable_sha256
            finally:
                probe.close()
            if (sandbox_version, sandbox_sha256) != required_bwrap_metadata():
                raise RuntimeError("query sandbox preflight provenance changed")
        settings = Settings(
            args.model,
            args.base_url,
            args.reasoning_effort,
            args.model_timeout_seconds,
            args.max_output_tokens,
            args.run_timeout_seconds,
            args.export_timeout_seconds,
            sandbox_version,
            sandbox_sha256,
            args.max_model_calls,
            args.max_input_tokens,
            args.max_tool_attempts,
            args.max_validator_retries,
            args.max_no_progress_rounds,
            args.model_provider,
        )
        commands = [_run_command(args.rootwalk_bin, plan, settings) for plan in plans]
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "commands": commands,
                        "dry_run": True,
                        "excluded_row_ids": excluded_row_ids,
                        "max_new_runs": args.max_new_runs,
                        "planned_runs": len(plans),
                        "sandbox_preflight": {
                            "requested": "required",
                            "status": "not_run",
                        },
                        "selected_row_ids": selected_row_ids,
                        "suite_id": suite.get("suite_id"),
                    },
                    sort_keys=True,
                )
            )
            return 0
        return _execute(
            plans,
            output,
            args.rootwalk_bin,
            settings,
            pricing,
            args.max_new_runs,
            excluded_row_ids,
            conditions,
        )
    except Exception as exc:
        print(json.dumps({"error": _safe_error(exc), "status": "invalid"}))
        return 2


def _suite(
    root: Path,
    spec_path: Path = Path(__file__).resolve().parents[1]
    / "benchmarks/openrca-bank-d9-v1.json",
) -> tuple[dict[str, object], list[Case]]:
    suite = root.expanduser().resolve(strict=True)
    manifest_path = _safe_file(suite, suite / "suite_manifest.json")
    manifest = _object(manifest_path)
    spec, frozen_plans = plan_cases(spec_path)
    identity = suite_identity(spec)
    expected_row_ids = tuple(plan.row_id for plan in frozen_plans)
    expected_fields = {
        "schema_version": "rootwalk-openrca-prepared-suite-v1",
        "suite_id": spec["suite_id"],
        "selection_sha256": identity["selection_sha256"],
        "case_plan_sha256": case_plan_sha256(frozen_plans),
        "upstream_commit": spec["upstream_commit"],
        "suite_spec_sha256": _sha256(spec_path.resolve(strict=True)),
    }
    if "source_files_sha256" in identity:
        expected_fields.update(
            {
                "source_files_sha256": identity["source_files_sha256"],
                "source_total_bytes": spec["source_total_bytes"],
            }
        )
    for name, expected in expected_fields.items():
        if manifest.get(name) != expected:
            raise ValueError(f"suite_manifest {name} does not match the frozen suite")
    _suite_outputs(suite, manifest.get("outputs"))
    raw_cases = manifest.get("cases")
    row_ids = manifest.get("row_ids")
    if not isinstance(raw_cases, list) or not isinstance(row_ids, list):
        raise ValueError("suite_manifest requires cases and row_ids arrays")
    row_values = cast(list[object], row_ids)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in row_values
        )
        or tuple(row_values) != expected_row_ids
        or len(row_values) != len(set(row_values))
    ):
        raise ValueError("suite row_ids do not match the frozen ordered selection")
    expected_cases = {plan.row_id: plan.case_id for plan in frozen_plans}
    cases: list[Case] = []
    for raw in cast(list[object], raw_cases):
        if not isinstance(raw, dict):
            raise ValueError("suite case must be an object")
        item = cast(dict[str, object], raw)
        case_id = _string(item.get("case_id"), "case_id")
        row_id = _integer(item.get("row_id"), "row_id")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", case_id):
            raise ValueError(f"unsafe case_id slug: {case_id!r}")
        if expected_cases.get(row_id) != case_id:
            raise ValueError(f"case_id does not match frozen row {row_id}")
        relative = _string(item.get("public_input"), "public_input")
        if relative != f"{case_id}/public_input":
            raise ValueError(f"{case_id} public_input path is not canonical")
        public = _safe_directory(suite, suite / relative)
        case_manifest = _safe_file(suite, public / "manifest.json")
        expected = _string(item.get("manifest_sha256"), "manifest_sha256")
        if not _is_hash(expected) or _sha256(case_manifest) != expected:
            raise ValueError(f"{case_id} manifest SHA-256 mismatch")
        task = _object(_safe_file(suite, public / "task.json"))
        if task.get("source_row_id") != row_id:
            raise ValueError(f"{case_id} task source_row_id mismatch")
        cases.append(
            Case(
                case_id,
                public,
                row_id,
                _string(task.get("task_id"), "task_id"),
                expected,
                _string(manifest.get("suite_id"), "suite_id"),
                _string(manifest.get("suite_spec_sha256"), "suite_spec_sha256"),
            )
        )
    if [case.row_id for case in cases] != row_values:
        raise ValueError("suite cases do not match ordered row_ids")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("suite case_id values must be unique")
    return manifest, cases


def _suite_outputs(root: Path, raw_outputs: object) -> None:
    if not isinstance(raw_outputs, list):
        raise ValueError("suite_manifest outputs must be an array")
    declared: dict[str, tuple[int, str]] = {}
    for raw in cast(list[object], raw_outputs):
        if not isinstance(raw, dict):
            raise ValueError("suite output must be an object")
        item = cast(dict[str, object], raw)
        relative = _string(item.get("path"), "output path")
        size = _integer(item.get("bytes"), "output bytes")
        digest = _string(item.get("sha256"), "output sha256")
        if relative in declared or not _is_hash(digest):
            raise ValueError(f"invalid or duplicate suite output: {relative}")
        path = _safe_file(root, root / relative)
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError(f"suite output hash/bytes mismatch: {relative}")
        declared[relative] = (size, digest)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "suite_manifest.json"
    }
    if set(declared) != actual:
        raise ValueError("suite outputs do not bind the complete prepared tree")


def _plans(
    cases: list[Case],
    output: Path,
    repeats: int,
    conditions: Sequence[tuple[str, str, str, str]] = CONDITIONS,
) -> list[Plan]:
    plans: list[Plan] = []
    for case in cases:
        for repeat in range(1, repeats + 1):
            for condition, slug, strategy, validator_mode in conditions:
                suite_slug = re.sub(r"[^a-z0-9]+", "-", case.suite_id.lower()).strip(
                    "-"
                )
                case_identity = hashlib.sha256(
                    f"{case.suite_spec_sha256}:{case.manifest_sha256}".encode("ascii")
                ).hexdigest()
                run_id = f"{suite_slug}-{case.case_id}-{case_identity}-{slug}-r{repeat}"
                plans.append(
                    Plan(
                        case,
                        condition,
                        slug,
                        strategy,
                        validator_mode,
                        repeat,
                        run_id,
                        output,
                        output / "runs" / run_id,
                        output / "predictions" / "by-run" / f"{run_id}.csv",
                    )
                )
    return plans


def _output_root(value: Path, suite_value: Path) -> Path:
    if value.is_symlink():
        raise ValueError("output root must not be a symlink")
    output = value.expanduser().resolve(strict=False)
    suite = suite_value.expanduser().resolve(strict=True)
    if output == Path(output.anchor):
        raise ValueError("output root must not be a filesystem root")
    if output == suite or output in suite.parents or suite in output.parents:
        raise ValueError("output root must not overlap the prepared suite")
    if output.exists() and not output.is_dir():
        raise ValueError("output root exists and is not a directory")
    return output


def _run_command(executable: str, plan: Plan, settings: Settings) -> list[str]:
    command = [
        executable,
        "run",
        "--data-root",
        str(plan.case.public_input),
        "--output",
        str(plan.bundle),
        "--run-id",
        plan.run_id,
        "--model-provider",
        settings.model_provider,
        "--model",
        settings.model,
        "--base-url",
        settings.base_url,
        "--model-timeout-s",
        str(settings.model_timeout_seconds),
        "--max-output-tokens",
        str(settings.max_output_tokens),
        "--max-model-calls",
        str(settings.max_model_calls),
        "--max-input-tokens",
        str(settings.max_input_tokens),
        "--max-tool-attempts",
        str(settings.max_tool_attempts),
        "--max-validator-retries",
        str(settings.max_validator_retries),
        "--max-no-progress-rounds",
        str(settings.max_no_progress_rounds),
        "--context-strategy",
        plan.strategy,
        "--validator-mode",
        plan.validator_mode,
        "--query-sandbox",
        "required",
        "--no-otel",
    ]
    if settings.reasoning_effort is not None:
        command.extend(["--reasoning-effort", settings.reasoning_effort])
    return command


def _execute(
    plans: list[Plan],
    output: Path,
    executable: str,
    settings: Settings,
    pricing: Mapping[str, int | float] | None,
    max_new_runs: int | None,
    excluded_row_ids: list[int],
    conditions: Sequence[tuple[str, str, str, str]] = CONDITIONS,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    _safe_output_path(output, output)
    _mkdir_output(output, output / "runs")
    _mkdir_output(output, output / "predictions" / "by-run")
    bundles: list[Path] = []
    metadata: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    missing_runs: list[dict[str, object]] = []
    deferred_runs: list[dict[str, object]] = []
    new_runs_started = 0
    expected_code_revision = source_tree_revision()
    source_revision_changed = False

    for plan in plans:
        _safe_output_path(output, plan.bundle)
        _safe_output_path(output, plan.prediction)
        metadata[plan.run_id] = {
            "condition": plan.condition,
            "repeat": plan.repeat,
        }
        if source_revision_changed:
            deferred_runs.append(_deferred_run(plan, "source_revision_changed"))
            _placeholder(plan.prediction, plan.case.row_id, output)
            continue
        bundle_error = _bundle_error(
            plan, settings, code_revision=expected_code_revision
        )
        if bundle_error is not None:
            command = _run_command(executable, plan, settings)
            if plan.bundle.exists() and (
                not plan.bundle.is_dir() or any(plan.bundle.iterdir())
            ):
                failure = _failure(
                    plan,
                    "preflight",
                    None,
                    bundle_error,
                    "ExistingBundleMismatch",
                    command,
                    counted_as_run=False,
                )
                failures.append(failure)
                missing_runs.append(_missing_run(plan, failure))
                _placeholder(plan.prediction, plan.case.row_id, output)
                continue
            if max_new_runs is not None and new_runs_started >= max_new_runs:
                deferred_runs.append(_deferred_run(plan))
                _placeholder(plan.prediction, plan.case.row_id, output)
                continue
            actual_code_revision = source_tree_revision()
            if actual_code_revision != expected_code_revision:
                source_revision_changed = True
                deferred_runs.append(_deferred_run(plan, "source_revision_changed"))
                _placeholder(plan.prediction, plan.case.row_id, output)
                continue
            new_runs_started += 1
            returncode, error_type, command_error = _invoke(
                command, settings.run_timeout_seconds
            )
            actual_code_revision = source_tree_revision()
            if actual_code_revision != expected_code_revision:
                source_revision_changed = True
                error = (
                    "source revision changed during run: "
                    f"expected={expected_code_revision}, actual={actual_code_revision}"
                )
                if command_error is not None:
                    error = f"{error}; subprocess: {command_error}"
                failure = _failure(
                    plan,
                    "run",
                    returncode,
                    error,
                    "SourceRevisionChanged",
                    command,
                    counted_as_run=True,
                )
                failures.append(failure)
                missing_runs.append(_missing_run(plan, failure))
                _placeholder(plan.prediction, plan.case.row_id, output)
                continue
            bundle_error = _bundle_error(
                plan, settings, code_revision=expected_code_revision
            )
            if bundle_error is not None:
                failure = _failure(
                    plan,
                    "run",
                    returncode,
                    (
                        f"{command_error}; bundle validation: {bundle_error}"
                        if command_error is not None
                        else bundle_error
                    ),
                    error_type or "BundleValidationError",
                    command,
                    counted_as_run=True,
                )
                failures.append(failure)
                missing_runs.append(_missing_run(plan, failure))
        if bundle_error is None:
            bundles.append(plan.bundle)
            export_error = _export_prediction(executable, plan, settings)
            if export_error is not None:
                failures.append(export_error)
                _placeholder(plan.prediction, plan.case.row_id, output)
        else:
            _placeholder(plan.prediction, plan.case.row_id, output)

    _merge_predictions(plans, output)
    result = summarize_benchmark(bundles, run_metadata=metadata, pricing=pricing)
    _initialize_counts(result, len(bundles))
    _include_external_runs(result, [*missing_runs, *deferred_runs])
    summary = cast(dict[str, object], result["summary"])
    result["runner"] = {
        "bundle_run_count": len(bundles),
        "complete": not failures and not deferred_runs,
        "deferred_count": summary["deferred_count"],
        "excluded_row_ids": excluded_row_ids,
        "attempted_count": summary["attempted_count"],
        "max_new_runs": max_new_runs,
        "new_runs_started": new_runs_started,
        "paid_run_confirmed": True,
        "planned_count": len(plans),
        "runner_failure_count": len(failures),
        "selected_row_ids": sorted({plan.case.row_id for plan in plans}),
        "settings": {
            **asdict(settings),
            "conditions": [
                {
                    "condition": condition,
                    "context_strategy": strategy,
                    "validator_mode": validator_mode,
                }
                for condition, _slug, strategy, validator_mode in conditions
            ],
            "runtime_budget_defaults": _runtime_defaults(settings),
            "runtime_versions": {
                **runtime_version_manifest(),
                "code_revision": expected_code_revision,
            },
            "observability": _DISABLED_OBSERVABILITY,
        },
    }
    result["runner_failures"] = failures
    _write_json(output / "runner_failures.json", failures, output)
    _write_json(output / "benchmark_summary.json", result, output)
    if deferred_runs and not failures:
        status = "deferred"
    elif failures:
        status = "completed_with_failures"
    else:
        status = "completed"
    print(
        json.dumps(
            {
                "failures": len(failures),
                "output_root": str(output),
                "planned_runs": len(plans),
                "status": status,
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


def _bundle_error(
    plan: Plan,
    settings: Settings,
    *,
    code_revision: str | None = None,
) -> str | None:
    _safe_output_path(plan.output_root, plan.bundle)
    if not plan.bundle.is_dir():
        return "bundle is missing"
    try:
        if (
            settings.query_sandbox_bwrap_version is None
            or settings.query_sandbox_bwrap_executable_sha256 is None
        ):
            raise ValueError("required query sandbox preflight is missing")
        summary = summarize_benchmark(
            [plan.bundle],
            run_metadata={
                plan.run_id: {"condition": plan.condition, "repeat": plan.repeat}
            },
        )
        run = cast(list[dict[str, object]], summary["runs"])[0]
        if run["run_id"] != plan.run_id or run["source_row_id"] != plan.case.row_id:
            raise ValueError("bundle identity does not match the planned run")
        manifest = _object(plan.bundle / "manifest.json")
        budget_defaults = _runtime_defaults(settings)
        max_context_tokens = budget_defaults.pop("max_context_tokens")
        versions = runtime_version_manifest()
        if code_revision is not None:
            versions = {**versions, "code_revision": code_revision}
        expected = {
            **versions,
            "model_provider": settings.model_provider,
            "model_id": settings.model,
            "model_endpoint": settings.base_url,
            "model_timeout_seconds": settings.model_timeout_seconds,
            "context_strategy": plan.strategy,
            "validator_mode": plan.validator_mode,
            "reasoning_effort": settings.reasoning_effort,
            "query_sandbox_requested": "required",
            "query_sandbox_effective": "bwrap",
            "query_sandbox_bwrap_version": settings.query_sandbox_bwrap_version,
            "query_sandbox_bwrap_executable_sha256": (
                settings.query_sandbox_bwrap_executable_sha256
            ),
            "observability": _DISABLED_OBSERVABILITY,
            **budget_defaults,
        }
        mismatched = [
            name for name, value in expected.items() if manifest.get(name) != value
        ]
        if mismatched:
            raise ValueError(f"bundle config mismatch: {', '.join(mismatched)}")
        report = _object(plan.bundle / "report.json")
        report_mismatched = [
            name
            for name in (
                "query_sandbox_requested",
                "query_sandbox_effective",
                "query_sandbox_bwrap_version",
                "query_sandbox_bwrap_executable_sha256",
                "observability",
            )
            if report.get(name) != expected[name]
        ]
        if report_mismatched:
            raise ValueError(
                f"bundle report config mismatch: {', '.join(report_mismatched)}"
            )
        context_limits = {
            item.get("max_context_tokens")
            for item in _jsonl(plan.bundle / "context_manifests.jsonl")
        }
        if context_limits != {max_context_tokens}:
            raise ValueError(
                "bundle max_context_tokens does not match current defaults"
            )
    except Exception as exc:
        return _safe_error(exc)
    return None


def _runtime_defaults(settings: Settings) -> dict[str, object]:
    values = asdict(
        RuntimeConfig(
            max_output_tokens_per_call=settings.max_output_tokens,
            max_model_calls=settings.max_model_calls,
            max_input_tokens=settings.max_input_tokens,
            max_tool_attempts=settings.max_tool_attempts,
            max_validator_retries=settings.max_validator_retries,
            max_no_progress_rounds=settings.max_no_progress_rounds,
        )
    )
    return {
        "wall_clock_limit_seconds": values["wall_clock_seconds"],
        **{name: values[name] for name in _BUDGET_FIELDS},
    }


def _export_prediction(
    executable: str, plan: Plan, settings: Settings
) -> dict[str, object] | None:
    _safe_output_path(plan.output_root, plan.prediction)
    command = [
        executable,
        "export",
        str(plan.bundle),
        "--format",
        "openrca",
        "--output",
        str(plan.prediction),
    ]
    returncode, error_type, error = _invoke(command, settings.export_timeout_seconds)
    try:
        _prediction_row(plan.prediction, plan.case.row_id, plan.output_root)
    except Exception as exc:
        error = error or _safe_error(exc)
    if returncode == 0 and error is None:
        return None
    return _failure(
        plan,
        "export",
        returncode,
        error or "export command failed",
        error_type or "PredictionValidationError",
        command,
        counted_as_run=False,
    )


def _invoke(
    command: list[str], timeout_seconds: float
) -> tuple[int | None, str | None, str | None]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return (
            completed.returncode,
            None if completed.returncode == 0 else "NonZeroExit",
            (
                None
                if completed.returncode == 0
                else _redact(completed.stderr.strip()) or "command exited non-zero"
            ),
        )
    except subprocess.TimeoutExpired:
        return None, "TimeoutExpired", f"command exceeded {timeout_seconds:g}s"
    except OSError as exc:
        return None, type(exc).__name__, _safe_error(exc)


def _failure(
    plan: Plan,
    stage: str,
    returncode: int | None,
    error: str,
    error_type: str,
    command: list[str],
    *,
    counted_as_run: bool,
) -> dict[str, object]:
    return {
        "command": command,
        "condition": plan.condition,
        "counted_as_run": counted_as_run,
        "error": _redact(error),
        "error_type": error_type,
        "repeat": plan.repeat,
        "returncode": returncode,
        "row_id": plan.case.row_id,
        "run_id": plan.run_id,
        "stage": stage,
        "task_id": plan.case.task_id,
    }


def _missing_run(plan: Plan, failure: Mapping[str, object]) -> dict[str, object]:
    return {
        "bundle": None,
        "cache_read_tokens": None,
        "completion_tokens": None,
        "condition": plan.condition,
        "cost": None,
        "cost_basis": "run_bundle_missing",
        "evaluation": None,
        "evaluation_score": None,
        "failed": True,
        "failure": failure["error"],
        "latency_seconds": None,
        "model_calls": None,
        "prompt_tokens": None,
        "refusal_reason": failure["error"],
        "refused": True,
        "repeat": plan.repeat,
        "run_id": plan.run_id,
        "source_row_id": plan.case.row_id,
        "status": "runner_failure",
        "task_id": plan.case.task_id,
        "total_tokens": None,
        "wall_clock_seconds": None,
    }


def _deferred_run(
    plan: Plan, reason: str = "max_new_runs_reached"
) -> dict[str, object]:
    return {
        "bundle": None,
        "cache_read_tokens": None,
        "completion_tokens": None,
        "condition": plan.condition,
        "cost": None,
        "cost_basis": "run_deferred",
        "deferred_reason": reason,
        "diagnostic_output": False,
        "evaluation": None,
        "evaluation_score": None,
        "failed": False,
        "failure": None,
        "latency_seconds": None,
        "model_calls": None,
        "prompt_tokens": None,
        "refusal_reason": None,
        "refused": False,
        "repeat": plan.repeat,
        "run_id": plan.run_id,
        "source_row_id": plan.case.row_id,
        "status": "deferred",
        "task_id": plan.case.task_id,
        "total_tokens": None,
        "wall_clock_seconds": None,
    }


def _include_external_runs(
    result: dict[str, object], external: list[dict[str, object]]
) -> None:
    if not external:
        return
    runs = cast(list[dict[str, object]], result["runs"])
    runs.extend(external)
    runs.sort(
        key=lambda item: (
            str(item["task_id"]),
            str(item["condition"]),
            str(item["repeat"]),
            str(item["run_id"]),
        )
    )
    result["planned_count"] = len(runs)
    result["attempted_count"] = sum(item["status"] != "deferred" for item in runs)
    result["deferred_count"] = sum(item["status"] == "deferred" for item in runs)
    summary = cast(dict[str, object], result["summary"])
    _add_external(summary, external)
    summary["task_count"] = len({item["task_id"] for item in runs})
    summary["condition_count"] = len({item["condition"] for item in runs})

    groups = cast(list[dict[str, object]], result["groups"])
    by_key = {(item["task_id"], item["condition"]): item for item in groups}
    grouped: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for item in external:
        grouped[(item["task_id"], item["condition"])].append(item)
    variation_names = cast(dict[str, object], summary["variation"]).keys()
    for key, items in grouped.items():
        group = by_key.get(key)
        if group is None:
            group = cast(
                dict[str, object],
                {
                    "condition": key[1],
                    "repeats": [],
                    "source_row_ids": [],
                    "task_id": key[0],
                    "planned_count": 0,
                    "attempted_count": 0,
                    "bundle_run_count": 0,
                    "status_counts": {},
                    "diagnostic_output_count": 0,
                    "diagnostic_output_rate_planned": None,
                    "refused_count": 0,
                    "refused_rate": None,
                    "failed_count": 0,
                    "failed_rate": None,
                    "deferred_count": 0,
                    "variation": {
                        name: {
                            "known_count": 0,
                            "missing_count": 0,
                            "min": None,
                            "max": None,
                            "mean": None,
                            "pstdev": None,
                        }
                        for name in variation_names
                    },
                },
            )
            groups.append(group)
        _add_external(group, items)
        repeats = cast(list[object], group["repeats"])
        repeats.extend(item["repeat"] for item in items)
        repeats.sort(key=str)
        rows = cast(list[int], group["source_row_ids"])
        rows[:] = sorted({*rows, *(cast(int, item["source_row_id"]) for item in items)})
    groups.sort(key=lambda item: (str(item["task_id"]), str(item["condition"])))


def _add_external(target: dict[str, object], external: list[dict[str, object]]) -> None:
    count = len(external)
    attempted_add = sum(item["status"] != "deferred" for item in external)
    planned = cast(int, target["planned_count"]) + count
    attempted = cast(int, target["attempted_count"]) + attempted_add
    target["planned_count"] = planned
    target["attempted_count"] = attempted
    statuses = cast(dict[str, int], target["status_counts"])
    for status, status_count in Counter(
        str(item["status"]) for item in external
    ).items():
        statuses[status] = statuses.get(status, 0) + status_count
    refused = cast(int, target["refused_count"]) + sum(
        item["refused"] is True for item in external
    )
    failed = cast(int, target["failed_count"]) + sum(
        item["failed"] is True for item in external
    )
    diagnostic = cast(int, target["diagnostic_output_count"]) + sum(
        item.get("diagnostic_output") is True for item in external
    )
    target["refused_count"] = refused
    target["failed_count"] = failed
    target["diagnostic_output_count"] = diagnostic
    target["diagnostic_output_rate_planned"] = diagnostic / planned
    target["refused_rate"] = refused / attempted if attempted else None
    target["failed_rate"] = failed / attempted if attempted else None
    target["deferred_count"] = cast(int, target.get("deferred_count", 0)) + sum(
        item["status"] == "deferred" for item in external
    )
    for variation in cast(dict[str, dict[str, object]], target["variation"]).values():
        variation["missing_count"] = cast(int, variation["missing_count"]) + count


def _initialize_counts(result: dict[str, object], bundle_count: int) -> None:
    result.pop("run_count", None)
    result.update(
        {
            "planned_count": bundle_count,
            "attempted_count": bundle_count,
            "bundle_run_count": bundle_count,
            "deferred_count": 0,
        }
    )
    targets = [
        cast(dict[str, object], result["summary"]),
        *cast(list[dict[str, object]], result["groups"]),
    ]
    for target in targets:
        count = cast(int, target["run_count"])
        target.pop("denominator", None)
        target.pop("run_count", None)
        diagnostic_rate = target.pop("diagnostic_output_rate")
        target.update(
            {
                "planned_count": count,
                "attempted_count": count,
                "bundle_run_count": count,
                "deferred_count": 0,
                "diagnostic_output_rate_planned": diagnostic_rate,
            }
        )


def _merge_predictions(plans: list[Plan], output_root: Path) -> None:
    grouped: dict[tuple[str, int], list[Plan]] = defaultdict(list)
    for plan in plans:
        grouped[(plan.slug, plan.repeat)].append(plan)
    for (slug, repeat), items in grouped.items():
        rows = [
            _prediction_row(item.prediction, item.case.row_id, output_root)
            for item in items
        ]
        target = output_root / "predictions" / f"{slug}-r{repeat}.csv"
        _safe_output_path(output_root, target)
        _mkdir_output(output_root, target.parent)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["row_id", "prediction"])
            writer.writeheader()
            writer.writerows(rows)


def _prediction_row(path: Path, expected_row: int, output_root: Path) -> dict[str, str]:
    _safe_output_path(output_root, path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["row_id", "prediction"]:
            raise ValueError("prediction CSV has an invalid header")
        rows = list(reader)
    if len(rows) != 1 or rows[0]["row_id"] != str(expected_row):
        raise ValueError("prediction CSV row_id does not match the planned case")
    return {"row_id": rows[0]["row_id"], "prediction": rows[0]["prediction"]}


def _placeholder(path: Path, row_id: int, output_root: Path) -> None:
    _safe_output_path(output_root, path)
    _mkdir_output(output_root, path.parent)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "prediction"])
        writer.writeheader()
        writer.writerow({"row_id": row_id, "prediction": "{}"})


def _safe_output_path(root: Path, path: Path) -> None:
    if root.is_symlink():
        raise ValueError("output root must not be a symlink")
    resolved_root = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output path escapes output root") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"output path component must not be a symlink: {current}")
        if current.exists():
            try:
                current.resolve(strict=True).relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError("output path escapes output root") from exc


def _mkdir_output(root: Path, path: Path) -> None:
    _safe_output_path(root, path)
    path.mkdir(parents=True, exist_ok=True)
    _safe_output_path(root, path)


def _pricing(path: Path | None) -> Mapping[str, int | float] | None:
    if path is None:
        return None
    value = _object(path.expanduser().resolve(strict=True))
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value.values()
    ):
        raise ValueError("pricing JSON values must be numbers")
    return cast(dict[str, int | float], value)


def _safe_file(root: Path, path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"input must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"input escapes suite root: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"input is not a file: {path}")
    return resolved


def _safe_directory(root: Path, path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"input must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"input escapes suite root: {path}") from exc
    if not resolved.is_dir():
        raise ValueError(f"input is not a directory: {path}")
    return resolved


def _object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return cast(dict[str, object], value)


def _write_json(path: Path, value: object, output_root: Path) -> None:
    _safe_output_path(output_root, path)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _freeze_run_plan(output_root: Path, run_plan: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _safe_output_path(output_root, output_root)
    path = output_root / "run_plan.json"
    _safe_output_path(output_root, path)
    if path.exists():
        if _object(path) != run_plan:
            raise ValueError("run_plan.json does not match the requested run plan")
        return
    if any(output_root.iterdir()):
        raise ValueError("non-empty output root requires an existing run_plan.json")
    _write_json(path, run_plan, output_root)


def _jsonl(path: Path) -> list[dict[str, object]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL: {path}") from exc
    result: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object: {path}")
        record = cast(dict[str, object], value)
        payload = record.get("payload", record)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL payload must be an object: {path}")
        result.append(cast(dict[str, object], payload))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hash(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _redact(value: str) -> str:
    redacted = value
    for name in ("ARK_API_KEY",):
        secret = os.environ.get(name)
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _safe_error(exc: Exception) -> str:
    return _redact(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
