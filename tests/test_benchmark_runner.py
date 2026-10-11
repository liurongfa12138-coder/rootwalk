# pyright: reportPrivateUsage=false

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_bank_benchmark as runner  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _suite(root: Path, spec_path: Path = runner.bank_suite.SPEC_PATH) -> Path:
    spec, case_plans = runner.plan_cases(spec_path)
    identity = runner.suite_identity(spec)
    row_ids = tuple(plan.row_id for plan in case_plans)
    cases: list[dict[str, object]] = []
    for row_id in row_ids:
        case_id = f"bank-row-{row_id:04d}"
        public = root / case_id / "public_input"
        task = {"task_id": f"openrca-{case_id}", "source_row_id": row_id}
        _write(public / "task.json", task)
        _write(public / "topology.json", {"snapshot_id": f"topology-{row_id}"})
        _write(public / "manifest.json", {"dataset_id": case_id})
        cases.append(
            {
                "case_id": case_id,
                "manifest_sha256": hashlib.sha256(
                    (public / "manifest.json").read_bytes()
                ).hexdigest(),
                "public_input": f"{case_id}/public_input",
                "row_id": row_id,
            }
        )
    outputs = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    suite_manifest: dict[str, object] = {
        "case_plan_sha256": runner.case_plan_sha256(case_plans),
        "cases": cases,
        "outputs": outputs,
        "row_ids": list(row_ids),
        "schema_version": "rootwalk-openrca-prepared-suite-v1",
        "selection_sha256": identity["selection_sha256"],
        "suite_id": spec["suite_id"],
        "suite_spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "upstream_commit": spec["upstream_commit"],
    }
    if "source_files_sha256" in identity:
        suite_manifest.update(
            {
                "source_files_sha256": identity["source_files_sha256"],
                "source_total_bytes": spec["source_total_bytes"],
            }
        )
    _write(root / "suite_manifest.json", suite_manifest)
    return root


def _bundle(
    root: Path,
    data_root: Path,
    run_id: str,
    *,
    model: str = "fixed-model",
    model_provider: str = "ark",
    base_url: str = "https://example.invalid/v3",
    strategy: str = "recent",
    validator_mode: str = "observe_only",
    max_output_tokens: int = 2_000,
    model_timeout_seconds: float = 60.0,
    reasoning_effort: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sandbox_version, sandbox_sha256 = runner.required_bwrap_metadata()
    task = cast(dict[str, object], json.loads((data_root / "task.json").read_text()))
    topology = cast(
        dict[str, object], json.loads((data_root / "topology.json").read_text())
    )
    report: dict[str, object] = {
        "abstain_reason": "synthetic refusal",
        "failure": None,
        "findings": [],
        "model_calls": 1,
        "model_usage": {"completion_tokens": 2, "prompt_tokens": 10},
        "run_id": run_id,
        "query_sandbox_requested": "required",
        "query_sandbox_effective": "bwrap",
        "query_sandbox_bwrap_version": sandbox_version,
        "query_sandbox_bwrap_executable_sha256": sandbox_sha256,
        "observability": runner._DISABLED_OBSERVABILITY,
        "status": "completed_partial",
        "task_id": task["task_id"],
    }
    files: dict[str, object] = {
        "task.json": task,
        "topology.json": topology,
        "context_manifests.jsonl": [{"payload": {"max_context_tokens": 12_000}}],
        "attempts.jsonl": [
            {
                "payload": {
                    "attempt_id": "model:1",
                    "status": "success",
                    "provider_model": model,
                }
            }
        ],
        "decisions.jsonl": [],
        "observations.jsonl": [],
        "report.json": report,
        "investigation_events.jsonl": [
            {"payload": {"event_type": "run_sealed", "status": "completed_partial"}}
        ],
    }
    for name, value in files.items():
        path = root / name
        if name.endswith(".jsonl"):
            path.write_text(
                "".join(
                    json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n"
                    for item in cast(list[object], value)
                ),
                encoding="utf-8",
            )
        else:
            _write(path, value)
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
    }
    _write(
        root / "manifest.json",
        {
            **runner.runtime_version_manifest(),
            "cleanup_grace_seconds": 10,
            "context_strategy": strategy,
            "file_hashes": hashes,
            "max_input_tokens": 144_000,
            "max_artifact_bytes": 64 * 1024 * 1024,
            "max_model_calls": 12,
            "max_no_progress_rounds": 3,
            "max_output_tokens": 24_000,
            "max_output_tokens_per_call": max_output_tokens,
            "max_tool_attempts": 40,
            "max_validator_retries": 2,
            "model_endpoint": base_url,
            "model_id": model,
            "model_provider": model_provider,
            "model_timeout_seconds": model_timeout_seconds,
            "observability": runner._DISABLED_OBSERVABILITY,
            "query_sandbox_requested": "required",
            "query_sandbox_effective": "bwrap",
            "query_sandbox_bwrap_version": sandbox_version,
            "query_sandbox_bwrap_executable_sha256": sandbox_sha256,
            "reasoning_effort": reasoning_effort,
            "run_id": run_id,
            "tool_timeout_seconds": 20.0,
            "validator_mode": validator_mode,
            "wall_clock_limit_seconds": 300,
        },
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_dry_run_validates_suite_and_never_invokes_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = _suite(tmp_path / "suite")
    output = tmp_path / "output"
    monkeypatch.setenv("ARK_API_KEY", "SECRET_ARK_KEY")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run invoked subprocess")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner, "QueryWorker", forbidden)
    monkeypatch.setattr(runner, "required_bwrap_metadata", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(output),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--dry-run",
        ]
    )

    content = capsys.readouterr().out
    result = json.loads(content)
    assert code == 0
    assert result["excluded_row_ids"] == []
    assert result["planned_runs"] == 162
    assert result["selected_row_ids"] == list(runner.DEFAULT_ROW_IDS)
    assert [_option(command, "--run-id") for command in result["commands"][:3]] == [
        "openrca-bank-d9-v1-bank-row-0002-c6656813c1f4fcfa9b576ec8200416020d291c058c8779dec30cd58c9dcb3c76-c0-r1",
        "openrca-bank-d9-v1-bank-row-0002-c6656813c1f4fcfa9b576ec8200416020d291c058c8779dec30cd58c9dcb3c76-c1-g0-r1",
        "openrca-bank-d9-v1-bank-row-0002-c6656813c1f4fcfa9b576ec8200416020d291c058c8779dec30cd58c9dcb3c76-g1-r1",
    ]
    assert all(
        _option(command, "--query-sandbox") == "required"
        for command in result["commands"]
    )
    assert all("--no-otel" in command for command in result["commands"])
    assert "ARK_API_KEY" not in content and "SECRET_ARK_KEY" not in content
    assert result["sandbox_preflight"] == {
        "requested": "required",
        "status": "not_run",
    }
    assert not output.exists()


def test_condition_filter_plans_only_g1_and_is_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = _suite(tmp_path / "suite")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("filtered dry-run invoked runtime dependency")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--condition",
            "G1",
            "--max-model-calls",
            "18",
            "--max-input-tokens",
            "216000",
            "--max-tool-attempts",
            "40",
            "--max-validator-retries",
            "5",
            "--max-no-progress-rounds",
            "6",
            "--condition",
            "G1",
            "--dry-run",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["planned_runs"] == 54
    assert all(
        "-g1-r" in _option(command, "--run-id") for command in result["commands"]
    )
    assert all(
        _option(command, "--context-strategy") == "evidence"
        and _option(command, "--validator-mode") == "enforce"
        and _option(command, "--max-model-calls") == "18"
        and _option(command, "--max-input-tokens") == "216000"
        and _option(command, "--max-tool-attempts") == "40"
        and _option(command, "--max-validator-retries") == "5"
        and _option(command, "--max-no-progress-rounds") == "6"
        for command in result["commands"]
    )


def test_invalid_condition_is_rejected_by_argparse(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        runner.main(
            [
                "--suite-root",
                str(tmp_path / "suite"),
                "--output-root",
                str(tmp_path / "output"),
                "--model",
                "fixed-model",
                "--base-url",
                "https://example.invalid/v3",
                "--condition",
                "invalid",
                "--dry-run",
            ]
        )


def test_filtered_run_plan_and_execute_settings_are_bound_to_g1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite(tmp_path / "suite")
    output = tmp_path / "output"
    captured: dict[str, object] = {}

    class Probe:
        bwrap_version = "test-bwrap"
        bwrap_executable_sha256 = "a" * 64

        def close(self) -> None:
            pass

    def fake_execute(
        plans: list[runner.Plan],
        _output: Path,
        _executable: str,
        settings: runner.Settings,
        _pricing: object,
        _max_new_runs: int | None,
        _excluded: list[int],
        conditions: object,
    ) -> int:
        captured["plans"] = plans
        captured["conditions"] = conditions
        captured["settings"] = settings
        return 0

    def fake_probe(*_args: object, **_kwargs: object) -> Probe:
        return Probe()

    monkeypatch.setattr(runner, "QueryWorker", fake_probe)
    monkeypatch.setattr(
        runner, "required_bwrap_metadata", lambda: ("test-bwrap", "a" * 64)
    )
    monkeypatch.setattr(runner, "_execute", fake_execute)

    assert (
        runner.main(
            [
                "--suite-root",
                str(suite),
                "--output-root",
                str(output),
                "--model",
                "fixed-model",
                "--base-url",
                "https://example.invalid/v3",
                "--condition",
                "G1",
                "--max-model-calls",
                "18",
                "--max-input-tokens",
                "216000",
                "--max-tool-attempts",
                "40",
                "--max-validator-retries",
                "5",
                "--max-no-progress-rounds",
                "6",
                "--confirm-paid-run",
            ]
        )
        == 0
    )

    plans = cast(list[runner.Plan], captured["plans"])
    assert len(plans) == 54
    assert {plan.condition for plan in plans} == {"G1"}
    assert captured["conditions"] == (runner.CONDITIONS[2],)
    settings = cast(runner.Settings, captured["settings"])
    assert (
        settings.max_model_calls,
        settings.max_input_tokens,
        settings.max_tool_attempts,
        settings.max_validator_retries,
        settings.max_no_progress_rounds,
    ) == (18, 216_000, 40, 5, 6)
    run_plan = cast(dict[str, Any], json.loads((output / "run_plan.json").read_text()))
    assert run_plan["conditions"] == ["G1"]
    assert run_plan["runtime_budget_settings"] == {
        "max_model_calls": 18,
        "max_input_tokens": 216_000,
        "max_tool_attempts": 40,
        "max_validator_retries": 5,
        "max_no_progress_rounds": 6,
    }
    assert len(run_plan["run_ids"]) == 54


def test_v2_dry_run_validates_source_identity_and_unique_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = runner.bank_suite.V2_SPEC_PATH
    suite = _suite(tmp_path / "suite", spec_path)
    spec, case_plans = runner.plan_cases(spec_path)
    identity = runner.suite_identity(spec)
    manifest, cases = runner._suite(suite, spec_path)
    assert manifest["source_files_sha256"] == identity["source_files_sha256"]
    assert manifest["source_total_bytes"] == spec["source_total_bytes"]
    assert (
        manifest["suite_spec_sha256"]
        == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    )
    assert len(cases) == len(case_plans) == 17

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run invoked runtime dependency")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner, "QueryWorker", forbidden)
    monkeypatch.setattr(runner, "required_bwrap_metadata", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--suite-spec",
            str(spec_path),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--dry-run",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    run_ids = [_option(command, "--run-id") for command in result["commands"]]

    assert code == 0
    assert result["suite_id"] == "openrca-bank-d9-v2"
    assert result["planned_runs"] == 17 * 3 * 3 == 153
    assert len(set(run_ids)) == 153
    assert all(run_id.startswith("openrca-bank-d9-v2-") for run_id in run_ids)


def test_v2_dry_run_excludes_holdout_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = runner.bank_suite.V2_SPEC_PATH
    suite = _suite(tmp_path / "suite", spec_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run invoked runtime dependency")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner, "QueryWorker", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--suite-spec",
            str(spec_path),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--exclude-row-id",
            "2",
            "--exclude-row-id",
            "128",
            "--repeats",
            "3",
            "--dry-run",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    selected_row_ids = sorted(set(runner.bank_suite.V2_ROW_IDS) - {2, 128})
    planned_case_ids = {
        Path(_option(command, "--data-root")).parent.name
        for command in result["commands"]
    }

    assert code == 0
    assert result["excluded_row_ids"] == [2, 128]
    assert result["selected_row_ids"] == selected_row_ids
    assert result["planned_runs"] == 135
    assert planned_case_ids == {f"bank-row-{row_id:04d}" for row_id in selected_row_ids}
    assert not (tmp_path / "output").exists()


def test_run_plan_freezes_holdout_before_resume_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_spec = runner.bank_suite.V2_SPEC_PATH
    v2_suite = _suite(tmp_path / "v2-suite", v2_spec)
    v1_suite = _suite(tmp_path / "v1-suite")
    output = tmp_path / "output"
    calls = {"probe": 0, "execute": 0}

    class Probe:
        bwrap_version = "test-bwrap"
        bwrap_executable_sha256 = "a" * 64

        def close(self) -> None:
            pass

    def fake_probe(*_args: object, **_kwargs: object) -> Probe:
        calls["probe"] += 1
        return Probe()

    def fake_execute(*_args: object, **_kwargs: object) -> int:
        calls["execute"] += 1
        return 0

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("run plan mismatch invoked subprocess")

    def arguments(
        suite: Path,
        spec: Path,
        *,
        excluded: tuple[int, ...] = (2, 128),
        repeats: int = 3,
        conditions: tuple[str, ...] = (),
        max_model_calls: int = 12,
        max_input_tokens: int = 144_000,
        max_tool_attempts: int = 40,
        max_validator_retries: int = 2,
        max_no_progress_rounds: int = 3,
    ) -> list[str]:
        values = [
            "--suite-root",
            str(suite),
            "--suite-spec",
            str(spec),
            "--output-root",
            str(output),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--repeats",
            str(repeats),
            "--confirm-paid-run",
            "--max-model-calls",
            str(max_model_calls),
            "--max-input-tokens",
            str(max_input_tokens),
            "--max-tool-attempts",
            str(max_tool_attempts),
            "--max-validator-retries",
            str(max_validator_retries),
            "--max-no-progress-rounds",
            str(max_no_progress_rounds),
        ]
        for row_id in excluded:
            values.extend(["--exclude-row-id", str(row_id)])
        for condition in conditions:
            values.extend(["--condition", condition])
        return values

    monkeypatch.setattr(runner, "QueryWorker", fake_probe)
    monkeypatch.setattr(
        runner, "required_bwrap_metadata", lambda: ("test-bwrap", "a" * 64)
    )
    monkeypatch.setattr(runner, "_execute", fake_execute)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    base_args = arguments(v2_suite, v2_spec)

    assert runner.main(base_args) == 0
    plan_path = output / "run_plan.json"
    frozen_bytes = plan_path.read_bytes()
    run_plan = cast(dict[str, Any], json.loads(frozen_bytes))
    selected_row_ids = sorted(set(runner.bank_suite.V2_ROW_IDS) - {2, 128})
    assert run_plan["suite_id"] == "openrca-bank-d9-v2"
    assert (
        run_plan["suite_spec_sha256"]
        == hashlib.sha256(v2_spec.read_bytes()).hexdigest()
    )
    assert (
        run_plan["suite_manifest_sha256"]
        == hashlib.sha256((v2_suite / "suite_manifest.json").read_bytes()).hexdigest()
    )
    assert run_plan["selected_row_ids"] == selected_row_ids
    assert run_plan["excluded_row_ids"] == [2, 128]
    assert run_plan["repeats"] == 3
    assert "conditions" not in run_plan
    assert "runtime_budget_settings" not in run_plan
    assert len(cast(list[str], run_plan["run_ids"])) == 135
    assert len(cast(dict[str, str], run_plan["case_manifest_sha256s"])) == 15

    assert runner.main(base_args) == 0
    assert calls == {"probe": 2, "execute": 2}

    changed_plans = [
        arguments(v2_suite, v2_spec, excluded=()),
        arguments(v2_suite, v2_spec, excluded=(2, 124)),
        arguments(v2_suite, v2_spec, repeats=2),
        arguments(v2_suite, v2_spec, conditions=("G1",)),
        arguments(v2_suite, v2_spec, max_model_calls=18),
        arguments(v2_suite, v2_spec, max_input_tokens=216_000),
        arguments(v2_suite, v2_spec, max_tool_attempts=41),
        arguments(v2_suite, v2_spec, max_validator_retries=5),
        arguments(v2_suite, v2_spec, max_no_progress_rounds=6),
        arguments(v1_suite, runner.bank_suite.SPEC_PATH),
    ]
    for changed_args in changed_plans:
        assert runner.main(changed_args) == 2
        assert calls == {"probe": 2, "execute": 2}
        assert plan_path.read_bytes() == frozen_bytes

    for invalid_plan in ("{}\n", "{"):
        plan_path.write_text(invalid_plan, encoding="utf-8")
        assert runner.main(base_args) == 2
        assert calls == {"probe": 2, "execute": 2}


def test_nonempty_output_without_run_plan_fails_before_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = _suite(tmp_path / "suite")
    output = tmp_path / "output"
    output.mkdir()
    (output / "orphan").write_text("partial", encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("non-empty output invoked runtime dependency")

    monkeypatch.setattr(runner, "QueryWorker", forbidden)
    monkeypatch.setattr(runner, "_execute", forbidden)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(output),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--confirm-paid-run",
        ]
    )

    assert code == 2
    assert "non-empty output root" in json.loads(capsys.readouterr().out)["error"]
    assert not (output / "run_plan.json").exists()


def test_run_id_binds_suite_spec_and_case_manifest(tmp_path: Path) -> None:
    def run_id(suite_spec_sha256: str, manifest_sha256: str) -> str:
        case = runner.Case(
            "bank-row-0002",
            tmp_path,
            2,
            "openrca-bank-row-0002",
            manifest_sha256,
            "openrca-bank-d9-v2",
            suite_spec_sha256,
        )
        return runner._plans([case], tmp_path / "output", 1)[0].run_id

    suite_spec_sha256 = "1" * 64
    manifest_sha256 = "2" * 64
    case_identity = hashlib.sha256(
        f"{suite_spec_sha256}:{manifest_sha256}".encode("ascii")
    ).hexdigest()
    original = run_id(suite_spec_sha256, manifest_sha256)

    assert original == (f"openrca-bank-d9-v2-bank-row-0002-{case_identity}-c0-r1")
    assert (
        len(
            {
                original,
                run_id("3" * 64, manifest_sha256),
                run_id(suite_spec_sha256, "4" * 64),
            }
        )
        == 3
    )


def test_paid_run_requires_explicit_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _suite(tmp_path / "suite")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unconfirmed run invoked subprocess")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
        ]
    )

    assert code == 2
    assert not (tmp_path / "output").exists()


def test_paid_run_requires_real_sandbox_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _suite(tmp_path / "suite")
    calls: list[str] = []

    def unavailable(_public_input: Path, *, sandbox: str) -> None:
        calls.append(sandbox)
        raise RuntimeError("sandbox unavailable")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("failed preflight invoked a paid subprocess")

    monkeypatch.setattr(runner, "QueryWorker", unavailable)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--confirm-paid-run",
        ]
    )

    assert code == 2
    assert calls == ["required"]


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_max_new_runs_must_be_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
) -> None:
    suite = _suite(tmp_path / "suite")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid limit invoked subprocess")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--max-new-runs",
            limit,
            "--dry-run",
        ]
    )

    assert code == 2


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max-model-calls", "0"),
        ("--max-input-tokens", "0"),
        ("--max-tool-attempts", "-1"),
        ("--max-validator-retries", "-1"),
        ("--max-no-progress-rounds", "0"),
    ],
)
def test_runtime_budgets_are_validated_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    suite = _suite(tmp_path / "suite")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid budget invoked subprocess")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            option,
            value,
            "--dry-run",
        ]
    )

    assert code == 2


@pytest.mark.parametrize(
    ("excluded_row_ids", "error"),
    [
        ([999], "not in the frozen suite"),
        ([2, 2], "must be unique"),
        (list(runner.DEFAULT_ROW_IDS), "leave at least one case"),
    ],
)
def test_excluded_row_ids_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    excluded_row_ids: list[int],
    error: str,
) -> None:
    suite = _suite(tmp_path / "suite")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid exclusion invoked subprocess")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    args = [
        "--suite-root",
        str(suite),
        "--output-root",
        str(tmp_path / "output"),
        "--model",
        "fixed-model",
        "--base-url",
        "https://example.invalid/v3",
        "--dry-run",
    ]
    for row_id in excluded_row_ids:
        args.extend(["--exclude-row-id", str(row_id)])

    assert runner.main(args) == 2
    assert error in json.loads(capsys.readouterr().out)["error"]


def test_max_new_runs_counts_only_started_runs_and_resumes_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _suite(tmp_path / "suite")
    output = tmp_path / "output"
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1] == "run":
            _bundle(
                Path(_option(command, "--output")),
                Path(_option(command, "--data-root")),
                _option(command, "--run-id"),
                strategy=_option(command, "--context-strategy"),
                validator_mode=_option(command, "--validator-mode"),
                max_output_tokens=int(_option(command, "--max-output-tokens")),
                model_timeout_seconds=float(_option(command, "--model-timeout-s")),
            )
        else:
            bundle = Path(command[2])
            task = json.loads((bundle / "task.json").read_text())
            target = Path(_option(command, "--output"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["row_id", "prediction"])
                writer.writeheader()
                writer.writerow({"row_id": task["source_row_id"], "prediction": "{}"})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = [
        "--suite-root",
        str(suite),
        "--output-root",
        str(output),
        "--model",
        "fixed-model",
        "--base-url",
        "https://example.invalid/v3",
        "--repeats",
        "1",
        "--max-new-runs",
        "2",
        "--exclude-row-id",
        "128",
        "--confirm-paid-run",
    ]

    initial_args = args.copy()
    initial_args[initial_args.index("--max-new-runs") + 1] = "1"
    assert runner.main(initial_args) == 0
    commands.clear()

    assert runner.main(args) == 0
    first_started = [
        _option(command, "--run-id") for command in commands if command[1] == "run"
    ]
    assert first_started == [
        "openrca-bank-d9-v1-bank-row-0002-c6656813c1f4fcfa9b576ec8200416020d291c058c8779dec30cd58c9dcb3c76-c1-g0-r1",
        "openrca-bank-d9-v1-bank-row-0002-c6656813c1f4fcfa9b576ec8200416020d291c058c8779dec30cd58c9dcb3c76-g1-r1",
    ]
    first = json.loads((output / "benchmark_summary.json").read_text())
    assert {
        key: first["summary"][key]
        for key in (
            "planned_count",
            "attempted_count",
            "bundle_run_count",
            "deferred_count",
        )
    } == {
        "planned_count": 51,
        "attempted_count": 3,
        "bundle_run_count": 3,
        "deferred_count": 48,
    }
    assert first["summary"]["deferred_count"] == 48
    assert first["summary"]["failed_count"] == 0
    assert first["summary"]["refused_count"] == 3
    assert first["summary"]["refused_rate"] == 1.0
    assert first["summary"]["diagnostic_output_rate_planned"] == 0.0
    assert "run_count" not in first["summary"]
    assert "denominator" not in first["summary"]
    deferred_group = next(
        group
        for group in first["groups"]
        if group["task_id"] == "openrca-bank-row-0004" and group["condition"] == "C0"
    )
    assert deferred_group["planned_count"] == 1
    assert deferred_group["attempted_count"] == 0
    assert deferred_group["bundle_run_count"] == 0
    assert deferred_group["deferred_count"] == 1
    assert deferred_group["failed_rate"] is None
    assert deferred_group["refused_rate"] is None
    assert deferred_group["diagnostic_output_rate_planned"] == 0.0
    assert first["runner"]["complete"] is False
    assert first["runner"]["excluded_row_ids"] == [128]
    assert first["runner"]["new_runs_started"] == 2
    assert first["runner"]["selected_row_ids"] == sorted(
        set(runner.DEFAULT_ROW_IDS) - {128}
    )
    assert first["runner"]["deferred_count"] == 48

    commands.clear()
    assert runner.main(args) == 0
    second_started = [
        _option(command, "--run-id") for command in commands if command[1] == "run"
    ]
    assert second_started == [
        "openrca-bank-d9-v1-bank-row-0004-b76fbfaa71d874cca19034b8c9f7bfe0a50091db22c817e9b5616486b6647a46-c0-r1",
        "openrca-bank-d9-v1-bank-row-0004-b76fbfaa71d874cca19034b8c9f7bfe0a50091db22c817e9b5616486b6647a46-c1-g0-r1",
    ]
    second = json.loads((output / "benchmark_summary.json").read_text())
    assert second["summary"]["planned_count"] == 51
    assert second["summary"]["attempted_count"] == 5
    assert second["summary"]["bundle_run_count"] == 5
    assert second["summary"]["deferred_count"] == 46
    assert second["runner"]["bundle_run_count"] == 5
    assert second["runner"]["new_runs_started"] == 2


def test_nonzero_sealed_bundle_recovers_but_existing_mismatch_is_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite(tmp_path / "suite")
    _manifest, cases = runner._suite(suite)
    output = tmp_path / "output"
    condition = (runner.CONDITIONS[2],)
    plan = runner._plans(cases[:1], output, 1, condition)[0]
    settings = runner.Settings(
        "fixed-model",
        "https://example.invalid/v3",
        None,
        60.0,
        2_000,
        900.0,
        60.0,
        *runner.required_bwrap_metadata(),
    )
    commands: list[list[str]] = []
    plan.bundle.mkdir(parents=True)

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1] == "run":
            _bundle(
                Path(_option(command, "--output")),
                Path(_option(command, "--data-root")),
                _option(command, "--run-id"),
                strategy=_option(command, "--context-strategy"),
                validator_mode=_option(command, "--validator-mode"),
                max_output_tokens=int(_option(command, "--max-output-tokens")),
                model_timeout_seconds=float(_option(command, "--model-timeout-s")),
            )
            return subprocess.CompletedProcess(command, 1, "", "post-seal failure")
        bundle = Path(command[2])
        task = json.loads((bundle / "task.json").read_text())
        target = Path(_option(command, "--output"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["row_id", "prediction"])
            writer.writeheader()
            writer.writerow({"row_id": task["source_row_id"], "prediction": "{}"})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert (
        runner._execute([plan], output, "rootwalk", settings, None, None, [], condition)
        == 0
    )
    assert [command[1] for command in commands] == ["run", "export"]
    assert json.loads((output / "runner_failures.json").read_text()) == []
    assert runner._bundle_error(plan, settings) is None

    monkeypatch.setenv("ARK_API_KEY", "SECRET_ARK_KEY")

    def nonzero_with_secret(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 1, "", "post-seal failure: SECRET_ARK_KEY"
        )

    monkeypatch.setattr(runner.subprocess, "run", nonzero_with_secret)
    assert runner._invoke(["rootwalk", "run"], 1.0) == (
        1,
        "NonZeroExit",
        "post-seal failure: [REDACTED]",
    )

    mismatched_output = tmp_path / "mismatched-output"
    mismatched_plan = runner._plans(cases[:1], mismatched_output, 1, condition)[0]
    _bundle(
        mismatched_plan.bundle,
        mismatched_plan.case.public_input,
        mismatched_plan.run_id,
        base_url="https://stale.example.invalid/v3",
        strategy=mismatched_plan.strategy,
        validator_mode=mismatched_plan.validator_mode,
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("existing mismatched bundle invoked subprocess")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)

    assert (
        runner._execute(
            [mismatched_plan],
            mismatched_output,
            "rootwalk",
            settings,
            None,
            None,
            [],
            condition,
        )
        == 1
    )
    failures = json.loads((mismatched_output / "runner_failures.json").read_text())
    assert len(failures) == 1
    assert failures[0]["stage"] == "preflight"
    assert failures[0]["error_type"] == "ExistingBundleMismatch"
    assert failures[0]["counted_as_run"] is False
    assert failures[0]["returncode"] is None
    assert failures[0]["error"] == (
        "ValueError: bundle config mismatch: model_endpoint"
    )


def test_source_revision_change_stops_before_next_paid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite(tmp_path / "suite")
    _manifest, cases = runner._suite(suite)
    output = tmp_path / "output"
    condition = (runner.CONDITIONS[2],)
    plans = runner._plans(cases[:2], output, 1, condition)
    settings = runner.Settings(
        "fixed-model",
        "https://example.invalid/v3",
        None,
        60.0,
        2_000,
        900.0,
        60.0,
        *runner.required_bwrap_metadata(),
    )
    expected_revision = "a" * 64
    changed_revision = "b" * 64
    revisions = iter([expected_revision, expected_revision, changed_revision])
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command[1] == "run"
        _bundle(
            Path(_option(command, "--output")),
            Path(_option(command, "--data-root")),
            _option(command, "--run-id"),
            strategy=_option(command, "--context-strategy"),
            validator_mode=_option(command, "--validator-mode"),
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "source_tree_revision", lambda: next(revisions))
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert (
        runner._execute(plans, output, "rootwalk", settings, None, None, [], condition)
        == 1
    )
    assert len(commands) == 1
    failures = json.loads((output / "runner_failures.json").read_text())
    assert len(failures) == 1
    assert failures[0]["run_id"] == plans[0].run_id
    assert failures[0]["stage"] == "run"
    assert failures[0]["error_type"] == "SourceRevisionChanged"
    assert failures[0]["counted_as_run"] is True
    summary = json.loads((output / "benchmark_summary.json").read_text())
    deferred = next(item for item in summary["runs"] if item["status"] == "deferred")
    assert deferred["run_id"] == plans[1].run_id
    assert deferred["deferred_reason"] == "source_revision_changed"
    assert summary["runner"]["new_runs_started"] == 1
    assert summary["runner"]["settings"]["runtime_versions"]["code_revision"] == (
        expected_revision
    )


def test_full_accounting_keeps_existing_metrics_with_explicit_counts(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "suite")
    _manifest, cases = runner._suite(suite)
    plan = runner._plans(cases, tmp_path / "output", 1)[0]
    _bundle(plan.bundle, plan.case.public_input, plan.run_id)
    result = runner.summarize_benchmark(
        [plan.bundle],
        run_metadata={plan.run_id: {"condition": plan.condition, "repeat": 1}},
    )
    before = cast(dict[str, Any], result["summary"]).copy()

    runner._initialize_counts(result, 1)

    summary = cast(dict[str, Any], result["summary"])
    assert summary["planned_count"] == 1
    assert summary["attempted_count"] == 1
    assert summary["bundle_run_count"] == 1
    assert summary["deferred_count"] == 0
    assert summary["failed_count"] == before["failed_count"]
    assert summary["refused_count"] == before["refused_count"]
    assert summary["failed_rate"] == before["failed_rate"]
    assert summary["refused_rate"] == before["refused_rate"]
    assert summary["diagnostic_output_rate_planned"] == before["diagnostic_output_rate"]
    assert summary["variation"] == before["variation"]


def test_mocked_matrix_merges_predictions_and_keeps_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _suite(tmp_path / "suite")
    output = tmp_path / "output"
    pricing = tmp_path / "pricing.json"
    _write(
        pricing,
        {"completion_per_million": 4.0, "prompt_per_million": 2.0},
    )
    failed_id = (
        "openrca-bank-d9-v1-bank-row-0128-"
        "6dba4b2068d16420091b25ccf0c898aca6cbebc79cba04cb217dfe71f7a4e33d-g1-r1"
    )
    commands: list[list[str]] = []
    monkeypatch.setenv("ARK_API_KEY", "SECRET_ARK_KEY")

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert "SECRET_ARK_KEY" not in command
        assert "env" not in _kwargs
        assert _kwargs["timeout"] in {60.0, 900.0}
        if command[1] == "run":
            run_id = _option(command, "--run-id")
            if run_id == failed_id:
                raise subprocess.TimeoutExpired(
                    command, 900, output="SECRET_ARK_KEY", stderr="SECRET_ARK_KEY"
                )
            _bundle(
                Path(_option(command, "--output")),
                Path(_option(command, "--data-root")),
                run_id,
                strategy=_option(command, "--context-strategy"),
                validator_mode=_option(command, "--validator-mode"),
                max_output_tokens=int(_option(command, "--max-output-tokens")),
                model_timeout_seconds=float(_option(command, "--model-timeout-s")),
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        bundle = Path(command[2])
        task = json.loads((bundle / "task.json").read_text())
        target = Path(_option(command, "--output"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["row_id", "prediction"])
            writer.writeheader()
            writer.writerow(
                {
                    "row_id": task["source_row_id"],
                    "prediction": '{"1":{"root cause component":"node-a"}}',
                }
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(output),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--repeats",
            "1",
            "--pricing-json",
            str(pricing),
            "--confirm-paid-run",
        ]
    )

    assert code == 1
    run_commands = [command for command in commands if command[1] == "run"]
    assert len(run_commands) == 54
    summary = json.loads((output / "benchmark_summary.json").read_text())
    assert summary["summary"]["planned_count"] == 54
    assert summary["summary"]["attempted_count"] == 54
    assert summary["summary"]["bundle_run_count"] == 53
    assert summary["summary"]["deferred_count"] == 0
    assert summary["summary"]["failed_count"] == 1
    assert summary["summary"]["failed_rate"] == 1 / 54
    assert {
        key: summary["runner"][key]
        for key in (
            "bundle_run_count",
            "complete",
            "planned_count",
            "attempted_count",
            "deferred_count",
            "runner_failure_count",
        )
    } == {
        "bundle_run_count": 53,
        "complete": False,
        "planned_count": 54,
        "attempted_count": 54,
        "deferred_count": 0,
        "runner_failure_count": 1,
    }
    assert summary["runner"]["settings"]["run_timeout_seconds"] == 900.0
    assert (
        summary["runner"]["settings"]["runtime_budget_defaults"]["max_artifact_bytes"]
        == 64 * 1024 * 1024
    )
    assert summary["runner"]["settings"]["runtime_versions"] == (
        runner.runtime_version_manifest()
    )
    assert summary["runner"]["settings"]["query_sandbox_bwrap_version"]
    assert (
        len(summary["runner"]["settings"]["query_sandbox_bwrap_executable_sha256"])
        == 64
    )
    assert summary["runner"]["settings"]["observability"] == (
        runner._DISABLED_OBSERVABILITY
    )
    assert summary["provider_models"] == ["fixed-model"]
    assert summary["provider_model_controlled"] is True
    missing = next(item for item in summary["runs"] if item["run_id"] == failed_id)
    assert missing["status"] == "runner_failure"
    assert missing["bundle"] is None
    failures = json.loads((output / "runner_failures.json").read_text())
    assert failures[0]["run_id"] == failed_id
    assert failures[0]["counted_as_run"] is True
    assert failures[0]["error_type"] == "TimeoutExpired"
    assert "SECRET_ARK_KEY" not in json.dumps(summary)
    assert "SECRET_ARK_KEY" not in json.dumps(failures)
    for name in ("c0-r1.csv", "c1-g0-r1.csv", "g1-r1.csv"):
        with (output / "predictions" / name).open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["row_id"]) for row in rows] == list(runner.DEFAULT_ROW_IDS)
    with (output / "predictions/g1-r1.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]["prediction"] == "{}"


def test_case_manifest_hash_is_checked_before_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _suite(tmp_path / "suite")
    (suite / "bank-row-0002/public_input/manifest.json").write_text("tampered")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid suite invoked subprocess")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    code = runner.main(
        [
            "--suite-root",
            str(suite),
            "--output-root",
            str(tmp_path / "output"),
            "--model",
            "fixed-model",
            "--base-url",
            "https://example.invalid/v3",
            "--dry-run",
        ]
    )

    assert code == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "wrong"),
        ("suite_id", "wrong"),
        ("selection_sha256", "0" * 64),
        ("case_plan_sha256", "0" * 64),
        ("upstream_commit", "wrong"),
        ("row_ids", list(reversed(runner.DEFAULT_ROW_IDS))),
    ],
)
def test_frozen_suite_identity_is_required(
    tmp_path: Path, field: str, value: object
) -> None:
    suite = _suite(tmp_path / "suite")
    manifest_path = suite / "suite_manifest.json"
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text()))
    manifest[field] = value
    _write(manifest_path, manifest)

    with pytest.raises(ValueError, match="suite|row_ids"):
        runner._suite(suite)


def test_suite_outputs_bind_every_file_and_case_id_is_safe(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "suite")
    (suite / "unbound.json").write_text("{}")
    with pytest.raises(ValueError, match="complete prepared tree"):
        runner._suite(suite)

    (suite / "unbound.json").unlink()
    manifest_path = suite / "suite_manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    manifest["cases"][0]["case_id"] = "../unsafe"
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="unsafe case_id"):
        runner._suite(suite)


def test_output_and_prediction_paths_reject_overlap_root_and_symlinks(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        runner._output_root(suite / "output", suite)
    with pytest.raises(ValueError, match="filesystem root"):
        runner._output_root(Path("/"), suite)
    target = tmp_path / "target"
    target.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runner._output_root(linked_output, suite)

    outside = tmp_path / "outside.csv"
    outside.write_text("unchanged")
    prediction = tmp_path / "prediction.csv"
    prediction.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        runner._placeholder(prediction, 2, tmp_path)
    assert outside.read_text() == "unchanged"

    outside_runs = tmp_path / "outside-runs"
    outside_runs.mkdir()
    runs = tmp_path / "runs"
    runs.symlink_to(outside_runs, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runner._safe_output_path(tmp_path, runs / "bundle")

    outside_predictions = tmp_path / "outside-predictions"
    (outside_predictions / "by-run").mkdir(parents=True)
    predictions = tmp_path / "predictions"
    predictions.symlink_to(outside_predictions, target_is_directory=True)
    nested = predictions / "by-run/nested.csv"
    with pytest.raises(ValueError, match="symlink"):
        runner._placeholder(nested, 2, tmp_path)
    assert not (outside_predictions / "by-run/nested.csv").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_provider", "old-provider"),
        ("model_id", "old-model"),
        ("model_endpoint", "https://old.invalid/v3"),
        ("model_timeout_seconds", 30.0),
        ("observability", {"enabled": True}),
        ("prompt_version", "old-prompt"),
        ("tool_version", "old-tools"),
        ("code_revision", "0" * 64),
        ("uv_lock_sha256", "1" * 64),
        ("direct_dependency_fingerprint", "2" * 64),
        ("direct_dependency_versions", {"dependency": "0"}),
        ("context_strategy", "evidence"),
        ("validator_mode", "enforce"),
        ("reasoning_effort", "low"),
        ("max_model_calls", 13),
        ("max_artifact_bytes", 1),
        ("max_validator_retries", 5),
        ("max_no_progress_rounds", 6),
        ("max_output_tokens_per_call", 4_000),
        ("query_sandbox_requested", "auto"),
        ("query_sandbox_effective", "spawn"),
        ("query_sandbox_bwrap_version", "old-bubblewrap"),
        ("query_sandbox_bwrap_executable_sha256", "0" * 64),
    ],
)
def test_bundle_reuse_requires_current_model_condition_and_budgets(
    tmp_path: Path, field: str, value: object
) -> None:
    suite = _suite(tmp_path / "suite")
    _manifest, cases = runner._suite(suite)
    plan = runner._plans(cases, tmp_path / "output", 1)[0]
    settings = runner.Settings(
        "fixed-model",
        "https://example.invalid/v3",
        None,
        60.0,
        2_000,
        900.0,
        60.0,
        *runner.required_bwrap_metadata(),
    )
    _bundle(plan.bundle, plan.case.public_input, plan.run_id)
    assert runner._bundle_error(plan, settings) is None

    manifest_path = plan.bundle / "manifest.json"
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text()))
    manifest[field] = value
    _write(manifest_path, manifest)

    assert "bundle config mismatch" in cast(str, runner._bundle_error(plan, settings))


def test_bundle_reuse_checks_report_sandbox_provenance(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "suite")
    _manifest, cases = runner._suite(suite)
    plan = runner._plans(cases, tmp_path / "output", 1)[0]
    version, sha256 = runner.required_bwrap_metadata()
    settings = runner.Settings(
        "fixed-model",
        "https://example.invalid/v3",
        None,
        60.0,
        2_000,
        900.0,
        60.0,
        version,
        sha256,
    )
    _bundle(plan.bundle, plan.case.public_input, plan.run_id)
    report_path = plan.bundle / "report.json"
    report = cast(dict[str, object], json.loads(report_path.read_text()))
    report["query_sandbox_effective"] = "spawn"
    _write(report_path, report)
    manifest_path = plan.bundle / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    cast(dict[str, str], manifest["file_hashes"])["report.json"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    _write(manifest_path, manifest)

    assert "bundle report config mismatch" in cast(
        str, runner._bundle_error(plan, settings)
    )
