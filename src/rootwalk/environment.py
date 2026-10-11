"""Narrow environment boundaries; composition only, not a plugin system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .contracts import Task, TopologySnapshot

ToolResult = dict[str, Any]


def unsupported_tool_result(tool: str) -> ToolResult:
    """Return a stable result for optional provider capabilities."""
    return {
        "status": "error",
        "error_type": "unsupported_tool",
        "retryable": False,
        "summary": f"tool is not supported by this provider: {tool}",
        "artifact_refs": [],
    }


@runtime_checkable
class TaskSource(Protocol):
    task: Task
    dataset_id: str


@runtime_checkable
class TopologyProvider(Protocol):
    topology: TopologySnapshot


@runtime_checkable
class ToolProvider(Protocol):
    def list_entities(self) -> ToolResult: ...

    def list_metrics(self, entity_id: str | None = None) -> ToolResult: ...

    def query_metric(
        self,
        entity_id: str,
        metric_id: str,
        *,
        period: str = "incident",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> ToolResult: ...

    def query_metric_profile(
        self,
        entity_id: str,
        metric_ids: list[str],
        *,
        peer_entity_ids: list[str] | None = None,
        derived_ratios: list[dict[str, str]] | None = None,
        fault_type: str | None = None,
    ) -> ToolResult: ...

    def query_impact_profile(self) -> ToolResult:
        return unsupported_tool_result("query_impact_profile")

    def query_service_profile(
        self,
        service_id: str,
        fault_type: str,
        *,
        member_entity_ids: list[str] | None = None,
    ) -> ToolResult: ...

    def query_log_samples(
        self,
        entity_id: str,
        *,
        period: str = "incident",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        log_name: str | None = None,
        contains: str | None = None,
        limit: int = 20,
    ) -> ToolResult: ...

    def summarize_log_patterns(
        self,
        entity_id: str,
        *,
        log_name: str | None = None,
        limit: int = 20,
    ) -> ToolResult: ...

    def query_trace_neighbors(
        self, entity_id: str, *, direction: str = "both", limit: int = 20
    ) -> ToolResult: ...

    def query_trace_edges(
        self, entity_id: str, *, direction: str = "both", limit: int = 20
    ) -> ToolResult: ...

    def query_trace_operation_profile(
        self, entity_id: str, limit: int = 20
    ) -> ToolResult:
        del entity_id, limit
        return unsupported_tool_result("query_trace_operation_profile")

    def read_artifact_slice(
        self,
        artifact_id: str,
        *,
        start_line: int = 1,
        max_lines: int = 50,
        max_bytes: int = 16 * 1024,
    ) -> ToolResult: ...

    def summarize_anomalies(
        self, *, entity_ids: list[str] | None = None, limit: int = 20
    ) -> ToolResult: ...


@runtime_checkable
class ResultExporter(Protocol):
    def export(self, destination: Path) -> Path: ...


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    metrics: bool = False
    logs: bool = False
    trace: bool = False
    changes: bool = False
    topology: bool = False
    source_code: bool = False
    coverage_limits: tuple[str, ...] = ()


@runtime_checkable
class RuntimeEnvironment(TaskSource, TopologyProvider, ToolProvider, Protocol):
    capabilities: CapabilityManifest
