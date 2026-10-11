from pathlib import Path
from typing import cast

from rootwalk.environment import (
    ResultExporter,
    RuntimeEnvironment,
    TaskSource,
    ToolProvider,
    TopologyProvider,
)
from rootwalk.openrca import BankEnvironment
from rootwalk.store import ArtifactStore, BundleExporter, RunStore


def test_bank_components_satisfy_environment_protocols(
    synthetic_public_input: Path, tmp_path: Path
) -> None:
    bank = BankEnvironment(synthetic_public_input)
    task_source: TaskSource = bank
    topology_provider: TopologyProvider = bank
    tool_provider: ToolProvider = bank
    runtime_environment: RuntimeEnvironment = bank
    store = RunStore(tmp_path / "run.sqlite3")
    exporter: ResultExporter = BundleExporter(
        store, ArtifactStore(tmp_path / "artifacts")
    )

    assert isinstance(bank, TaskSource)
    assert isinstance(bank, TopologyProvider)
    assert isinstance(bank, ToolProvider)
    assert isinstance(bank, RuntimeEnvironment)
    assert runtime_environment.dataset_id == bank.dataset_id
    assert isinstance(exporter, ResultExporter)
    assert task_source.task == bank.task
    assert topology_provider.topology == bank.topology
    assert tool_provider.list_entities()["status"] == "success"
    assert callable(tool_provider.query_impact_profile)
    assert callable(tool_provider.query_trace_operation_profile)
    assert bank.capabilities.metrics
    assert bank.capabilities.logs
    assert bank.capabilities.trace
    assert bank.capabilities.topology
    assert not bank.capabilities.changes
    assert not bank.capabilities.source_code
    store.close()


def test_optional_profile_tool_defaults_return_unsupported_results() -> None:
    provider = cast(ToolProvider, object())

    impact = ToolProvider.query_impact_profile(provider)
    operation = ToolProvider.query_trace_operation_profile(provider, "entity", 3)

    assert impact["error_type"] == "unsupported_tool"
    assert operation["error_type"] == "unsupported_tool"
    assert impact["artifact_refs"] == operation["artifact_refs"] == []
