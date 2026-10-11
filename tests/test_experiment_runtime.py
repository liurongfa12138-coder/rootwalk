import asyncio
import json
from pathlib import Path
from typing import Any, cast

from rootwalk.experiments import (
    build_experiment_matrix,
    compare_actual_model_inputs,
)
from rootwalk.model import RecordedModelAdapter
from rootwalk.openrca import BankEnvironment
from rootwalk.runtime import RootWalkRuntime, RuntimeConfig
from rootwalk.store import ArtifactStore, RunStore


def _decisions(run_id: str) -> list[dict[str, Any]]:
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0001:0"
    return [
        {
            "decision_summary": "record an ungrounded candidate",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "Tomcat01 may be causal",
                    "status": "open",
                }
            ],
        },
        {
            "decision_summary": "submit an ungrounded path",
            "operations": [
                {
                    "type": "submit_path",
                    "hypothesis_id": hypothesis_id,
                    "root_cause_entity_id": "bank:entity:Tomcat01",
                    "symptom_entity_ids": ["bank:entity:Tomcat02"],
                    "steps": [],
                }
            ],
        },
        {
            "decision_summary": "finish without a supported finding",
            "operations": [
                {"type": "finalize", "abstain_reason": "No grounded finding"}
            ],
        },
    ]


def _run_arm(
    root: Path,
    public_input: Path,
    run_id: str,
    config: RuntimeConfig,
) -> Path:
    store = RunStore(root / "run.sqlite3")
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(public_input),
            RecordedModelAdapter(_decisions(run_id)),
            store,
            ArtifactStore(root / "artifact-store"),
            config,
        ).run(run_id, root / "bundle")
    )
    assert result.status == "completed_partial"
    assert result.model_calls == 3
    assert store.seal_state(run_id) is not None
    store.close()
    return result.bundle_path


def _context(bundle: Path, call_number: int) -> dict[str, Any]:
    manifests = [
        cast(dict[str, Any], json.loads(line))["payload"]
        for line in (bundle / "context_manifests.jsonl").read_text().splitlines()
    ]
    manifest = next(item for item in manifests if item["call_number"] == call_number)
    messages = json.loads(
        (bundle / "artifacts" / manifest["model_input_ref"]).read_text()
    )
    return cast(dict[str, Any], json.loads(messages[1]["content"]))


def test_real_runtime_bundles_preserve_m3_experiment_controls(
    tmp_path: Path, synthetic_public_input: Path
) -> None:
    matrix = build_experiment_matrix(
        RuntimeConfig(model_provider="recorded", model_id="fixed-model")
    )
    c0 = _run_arm(
        tmp_path / "c0",
        synthetic_public_input,
        "experiment-runtime",
        matrix["C0"].runtime_config,
    )
    c1_g0 = _run_arm(
        tmp_path / "c1-g0",
        synthetic_public_input,
        "experiment-runtime",
        matrix["C1"].runtime_config,
    )
    g1 = _run_arm(
        tmp_path / "g1",
        synthetic_public_input,
        "experiment-runtime",
        matrix["G1"].runtime_config,
    )

    assert matrix["G0"].runtime_config is matrix["C1"].runtime_config
    assert matrix["G0"].reuse_of == "C1"
    context_comparison = compare_actual_model_inputs(
        c0, c1_g0, target_variable="context_strategy"
    )
    gate_comparison = compare_actual_model_inputs(
        c1_g0, g1, target_variable="validator_mode"
    )

    assert context_comparison["controlled_variables"]
    assert context_comparison["recorded_conditions"] == ["C0", "C1"]
    assert gate_comparison["controlled_variables"]
    assert gate_comparison["conditions"]["control"] == {
        "condition_id": "G0",
        "recorded_as": "C1",
        "reused": True,
    }
    assert "path_quality" not in json.dumps(_context(c1_g0, 3))
    assert "path_quality" in json.dumps(_context(g1, 3))
    assert gate_comparison["observe_only_feedback_check"]["control"]["passed"]
    assert (
        gate_comparison["observe_only_feedback_check"]["treatment"]["path_quality"] > 0
    )
