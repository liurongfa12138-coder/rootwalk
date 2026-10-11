import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import rootwalk.openrca as openrca_module
from rootwalk.openrca import BankEnvironment, MarketEnvironment, OpenRCAEnvironment

PUBLIC_INPUT = (
    Path(__file__).resolve().parents[2]
    / "rootwalk-handoff"
    / "rootwalk-development-handoff"
    / "rootwalk-data-prep"
    / "slices"
    / "bank-row-0001"
    / "public_input"
)

MINIMAL_ARTIFACTS = {
    "dataset_profile.json",
    "entity_mapping.json",
    "metric_catalog.json",
    "metric_window_stats.jsonl",
    "task.json",
    "topology.json",
    "vocabulary.json",
}


def test_burst_profile_exposes_sampling_lower_bound_for_onset() -> None:
    profile = openrca_module._burst_profile(  # pyright: ignore[reportPrivateUsage]
        [1.0, 1.0, 10.0, 11.0],
        [100.0, 160.0, 220.0, 280.0],
        reference_min=1.0,
        reference_max=2.0,
        direction="higher_is_more_pressure",
    )

    assert profile["onset_timestamp"] == 220.0
    assert profile["onset_lower_bound_timestamp"] == 160.0


def test_generic_profile_helpers_preserve_auditable_onset_and_effect_parts() -> None:
    markers = openrca_module._profile_onset_markers(  # pyright: ignore[reportPrivateUsage]
        [(100.0, 10.0, 5.0), (160.0, 11.0, 6.0)],
        [(220.0, 10.0, 5.0), (280.0, 4.0, 12.0)],
    )
    effect, parts = openrca_module._profile_effect_size(  # pyright: ignore[reportPrivateUsage]
        100.0, 50.0, 5.0, 10.0
    )

    assert markers == {
        "last_normal": 220.0,
        "first_abnormal": 280.0,
        "first_anomalous": 280.0,
        "onset_interval": {"start": 220.0, "end": 280.0},
        "segment_start": 280.0,
        "segment_end": 280.0,
        "consecutive_points": 1,
    }
    latency_effect = parts["latency"]
    count_effect = parts["count"]
    assert latency_effect is not None
    assert count_effect is not None
    assert abs(effect) == abs(latency_effect)
    assert count_effect < 0 < latency_effect


def test_harmful_onset_ignores_early_count_change_until_latency_is_sustained() -> None:
    markers = openrca_module._profile_onset_markers(  # pyright: ignore[reportPrivateUsage]
        [(0.0, 10.0, 1.0), (60.0, 10.0, 1.0)],
        [
            (120.0, 20.0, 1.0),
            (180.0, 30.0, 1.0),
            (240.0, 30.0, 1.4),
            (300.0, 30.0, 1.5),
        ],
        count_is_harmful=False,
        minimum_consecutive_points=2,
        minimum_relative_change=0.2,
    )

    assert markers["last_normal"] == 180.0
    assert markers["first_anomalous"] == 240.0
    assert markers["onset_interval"] == {"start": 180.0, "end": 240.0}


def test_count_change_and_latency_improvement_are_not_harmful_impact() -> None:
    markers = openrca_module._profile_onset_markers(  # pyright: ignore[reportPrivateUsage]
        [(0.0, 10.0, 2.0), (60.0, 10.0, 2.0)],
        [(120.0, 20.0, 1.5), (180.0, 30.0, 1.4)],
        count_is_harmful=False,
        minimum_consecutive_points=2,
        minimum_relative_change=0.2,
    )
    _effect, parts = openrca_module._profile_effect_size(  # pyright: ignore[reportPrivateUsage]
        20.0, 50.0, 2.0, 1.5
    )

    assert markers["first_anomalous"] is None
    assert parts["count"] is not None and parts["count"] > 0
    assert parts["latency"] is not None and parts["latency"] < 0


def test_success_rate_drop_is_a_harmful_onset() -> None:
    markers = openrca_module._profile_onset_markers(  # pyright: ignore[reportPrivateUsage]
        [(0.0, 0.0, 100.0), (60.0, 0.0, 99.0)],
        [(120.0, 0.0, 99.0), (180.0, 0.0, 70.0), (240.0, 0.0, 60.0)],
        count_is_harmful=False,
        value_direction="lower",
        minimum_consecutive_points=2,
        minimum_relative_change=0.1,
    )

    assert markers["last_normal"] == 120.0
    assert markers["first_anomalous"] == 180.0


def test_peer_isolated_trace_error_event_has_mechanism_rank() -> None:
    assert (
        openrca_module.mechanism_signature_rank(
            {
                "boundary_aligned": False,
                "mechanism_signature": "peer_isolated_trace_error_event",
            }
        )
        == 3
    )


def test_peer_isolated_queue_congestion_is_lower_rank_than_direct_mechanisms() -> None:
    queue_rank = openrca_module.mechanism_signature_rank(  # pyright: ignore[reportPrivateUsage]
        {
            "boundary_aligned": False,
            "mechanism_signature": "peer_isolated_queue_congestion_event",
        }
    )
    cpu_rank = openrca_module.mechanism_signature_rank(  # pyright: ignore[reportPrivateUsage]
        {
            "boundary_aligned": False,
            "mechanism_signature": "sustained_high_saturation",
        }
    )

    assert queue_rank == 1
    assert queue_rank < cpu_rank


def test_bank_business_impact_alignment_adds_audited_cross_layer_bonus() -> None:
    result: dict[str, Any] = {
        "artifact_refs": [],
        "finding_gate_summary": {
            "fault_type_ready": True,
            "signature_free_finding_ready": True,
            "earliest_strong_onset_timestamp": 1_000.0,
            "root_cause_score": 0.4,
        },
    }
    consensus: dict[str, Any] = {
        "policy": "independent-business-impact-consensus-v1",
        "ready": True,
        "consensus_onset_timestamp": 1_120.0,
        "source_entity_ids": ["test-1", "test-2"],
        "events": [],
        "tolerance_seconds": 300.0,
        "artifact_refs": [{"content_hash": "a" * 64, "selector": {"path": "app.csv"}}],
    }

    aligned = BankEnvironment._apply_business_impact_alignment(  # pyright: ignore[reportPrivateUsage]
        result, consensus
    )
    summary = aligned["finding_gate_summary"]

    assert summary["business_impact_aligned"] is True
    assert summary["business_impact_score_bonus"] == 0.5
    assert summary["root_cause_score"] == 0.9
    assert (
        summary["business_impact_alignment"]["candidate_onset_distance_seconds"]
        == 120.0
    )
    assert aligned["artifact_refs"] == consensus["artifact_refs"]

    far = BankEnvironment._apply_business_impact_alignment(  # pyright: ignore[reportPrivateUsage]
        {
            "artifact_refs": [],
            "finding_gate_summary": {
                "fault_type_ready": True,
                "signature_free_finding_ready": True,
                "earliest_strong_onset_timestamp": 2_000.0,
                "root_cause_score": 0.4,
            },
        },
        consensus,
    )
    assert far["finding_gate_summary"]["business_impact_aligned"] is False
    assert far["finding_gate_summary"]["root_cause_score"] == 0.4
    assert far["artifact_refs"] == []


def minimal_public_input(parent: Path) -> Path:
    root = parent / "public_input"
    root.mkdir(parents=True)
    manifest = json.loads((PUBLIC_INPUT / "manifest.json").read_text())
    manifest["artifacts"] = [
        item for item in manifest["artifacts"] if item["path"] in MINIMAL_ARTIFACTS
    ]
    for item in manifest["artifacts"]:
        shutil.copyfile(PUBLIC_INPUT / item["path"], root / item["path"])
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def environment() -> BankEnvironment:
    return BankEnvironment(PUBLIC_INPUT)


@pytest.mark.openrca_bank
def test_catalogs_use_canonical_entities_and_both_metric_shapes() -> None:
    bank = environment()

    entities = bank.list_entities()
    assert entities["status"] == "success"
    assert entities["count"] == 29
    assert entities["entities"][0]["canonical_id"].startswith("bank:")
    assert "cnt" in bank.list_metrics("bank:test:ServiceTest1")["metric_ids"]
    assert (
        "OSLinux-CPU_CPU_CPUCpuUtil"
        in bank.list_metrics("bank:entity:Tomcat01")["metric_ids"]
    )
    assert bank.list_metrics("Tomcat01")["error_type"] == "invalid_entity"


@pytest.mark.openrca_bank
def test_query_metric_reads_app_container_and_custom_ranges() -> None:
    bank = environment()

    container = bank.query_metric(
        "bank:entity:Tomcat01", "OSLinux-CPU_CPU_CPUCpuUtil", period="reference"
    )
    app = bank.query_metric("bank:test:ServiceTest1", "cnt", period="incident")
    custom = bank.query_metric(
        "bank:test:ServiceTest1",
        "cnt",
        start="2021-03-04T18:00:00+08:00",
        end="2021-03-04T18:05:00+08:00",
    )

    assert container["status"] == app["status"] == custom["status"] == "success"
    assert container["sample_count"] == 20
    assert app["sample_count"] == 30
    assert custom["period"] == "custom"
    assert custom["sample_count"] == 5
    assert app["points"][0]["source_record"] > 0
    paired = container["paired_metric_summary"]
    assert paired["counts_complete"] is True
    assert paired["reference"]["sample_count"] == container["sample_count"]
    assert paired["pressure_direction"] == "higher_is_more_pressure"
    assert paired["scale_hint"] == "unknown"
    assert paired["hint_basis"] == "heuristic from metric name"
    assert (
        0
        <= paired["incident"]["above_reference_max_count"]
        <= paired["incident"]["sample_count"]
    )
    assert app["paired_metric_summary"] == custom["paired_metric_summary"]
    assert bank.query_metric("Tomcat01", "cnt")["status"] == "error"


@pytest.mark.openrca_bank
def test_impact_profile_is_generic_typed_and_source_auditable() -> None:
    result = environment().query_impact_profile()

    assert result["evidence_role"] == "impact_only"
    if result["status"] == "no_data":
        assert result["profiles"] == []
        return
    assert result["status"] == "success"
    assert result["profiles"]
    required = {
        "entity_id",
        "metric_name",
        "evidence_role",
        "last_normal",
        "first_abnormal",
        "onset_interval",
        "reference_count",
        "incident_count",
        "reference_latency",
        "incident_latency",
        "effect_size",
        "artifact_refs",
    }
    for profile in result["profiles"]:
        assert required <= profile.keys()
        assert profile["evidence_role"] == "impact_only"
        assert profile["artifact_refs"]
        assert all(
            ref["selector"]["path"].split("/")[-1].casefold()
            not in {"query.csv", "record.csv"}
            for ref in profile["artifact_refs"]
        )


@pytest.mark.openrca_bank
def test_trace_operation_profile_returns_reason_when_source_has_no_operations() -> None:
    result = environment().query_trace_operation_profile(
        "bank:entity:Tomcat01", limit=3
    )

    assert result["status"] == "no_data"
    assert result["operations"] == []
    assert "missing" in result["reason"]
    assert result["artifact_refs"]


@pytest.mark.openrca_bank
def test_candidate_signals_require_metric_and_entity_root_cause_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = environment()
    entity_id = "bank:entity:Tomcat01"
    root_signal = {
        "entity_id": entity_id,
        "metric_id": "root_metric",
        "evidence_role": "root_cause_candidate",
    }
    impact_signal = {
        "entity_id": entity_id,
        "metric_id": "impact_metric",
        "evidence_role": "impact_only",
    }

    assert bank._root_cause_candidate_signals(  # pyright: ignore[reportPrivateUsage]
        [impact_signal, root_signal]
    ) == [root_signal]

    metric_id = bank._catalog_metrics_by_entity[entity_id][0]  # pyright: ignore[reportPrivateUsage]
    stats_by_pair = bank._stats_by_pair  # pyright: ignore[reportPrivateUsage]
    stats_by_pair[(entity_id, metric_id)]["evidence_role"] = "impact_only"
    assert (
        metric_id
        not in bank._root_cause_candidate_metrics()[  # pyright: ignore[reportPrivateUsage]
            entity_id
        ]
    )

    bank._entities[entity_id] = bank._entities[entity_id].model_copy(  # pyright: ignore[reportPrivateUsage]
        update={"evidence_role": "impact_only"}
    )
    assert (
        bank._root_cause_candidate_signals(  # pyright: ignore[reportPrivateUsage]
            [root_signal]
        )
        == []
    )

    def unexpected_consensus() -> dict[str, Any]:
        raise AssertionError(
            "impact-only profile must not request root-cause consensus"
        )

    monkeypatch.setattr(bank, "_business_impact_consensus", unexpected_consensus)
    impact_profile = bank.query_metric_profile(entity_id, [metric_id])
    assert impact_profile["status"] == "success"
    assert impact_profile["evidence_role"] == "impact_only"
    assert impact_profile["metric_profiles"][0]["evidence_role"] == "impact_only"
    gate = impact_profile["finding_gate_summary"]
    assert gate["finding_eligible"] is False
    assert gate["root_cause_score"] == 0.0
    assert gate["single_component_ready"] is False


@pytest.mark.openrca_bank
def test_sparse_metric_profile_cannot_pass_root_cause_finding_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = environment()
    entity_id = "bank:entity:Tomcat01"
    metric_id = "OSLinux-CPU_CPU_CPUCpuUtil"

    def sparse_scan(
        _entity_id: str, metric_ids: list[str]
    ) -> tuple[
        dict[tuple[str, str], tuple[list[float], list[float]]],
        dict[str, dict[str, Any]],
        set[str],
    ]:
        assert _entity_id == entity_id
        assert metric_ids == [metric_id]
        return (
            {
                (metric_id, "reference"): ([100.0], [0.0]),
                (metric_id, "incident"): ([200.0], [100.0]),
            },
            {},
            {"success"},
        )

    monkeypatch.setattr(bank, "_scan_metric_group", sparse_scan)
    result = OpenRCAEnvironment.query_metric_profile(
        bank,
        entity_id,
        [metric_id],
        peer_entity_ids=[],
        fault_type="high CPU usage",
    )

    profile = result["metric_profiles"][0]
    gate = result["finding_gate_summary"]
    assert profile["strong_incident_change"] is True
    assert profile["sample_count"] == {"reference": 1, "incident": 1}
    assert profile["finding_sample_ready"] is False
    assert profile["missing_evidence_reasons"]
    assert gate["fault_type_ready"] is False
    assert gate["single_component_ready"] is False
    assert gate["qualifying_metric_ids"] == []
    assert gate["mechanism_signature"] is None
    assert gate["insufficient_sample_metric_ids"] == [metric_id]


@pytest.mark.openrca_bank
def test_metric_name_hints_are_conservative() -> None:
    bank = environment()
    catalog = [
        (entity["canonical_id"], metric_id)
        for entity in bank.list_entities()["entities"]
        for metric_id in bank.list_metrics(entity["canonical_id"])["metric_ids"]
    ]
    percent_entity, percent_metric = next(
        item for item in catalog if "Percent" in item[1]
    )
    available_entity, available_metric = next(
        item for item in catalog if "Available" in item[1]
    )
    thread_entity, thread_metric = next(
        item for item in catalog if "ThreadCount" in item[1]
    )

    percent = bank.query_metric(percent_entity, percent_metric)["paired_metric_summary"]
    available = bank.query_metric(available_entity, available_metric)[
        "paired_metric_summary"
    ]
    thread = bank.query_metric(thread_entity, thread_metric)["paired_metric_summary"]

    assert percent["scale_hint"] == "0..100"
    assert available["pressure_direction"] == "lower_is_more_pressure"
    assert thread["pressure_direction"] == "unknown"
    assert BankEnvironment._metric_name_hints(  # pyright: ignore[reportPrivateUsage]
        "CPUUtilization"
    ) == ("unknown", "higher_is_more_pressure")
    assert (
        BankEnvironment._candidate_pressure_direction(  # pyright: ignore[reportPrivateUsage]
            "network latency", "OSLinux_NETWORK_NETBandwidthUtil"
        )
        == "lower_is_more_pressure"
    )
    assert (
        BankEnvironment._candidate_pressure_direction(  # pyright: ignore[reportPrivateUsage]
            "high CPU usage", "OSLinux_NETWORK_NETBandwidthUtil"
        )
        == "higher_is_more_pressure"
    )
    assert BankEnvironment._metric_name_hints("sr") == (  # pyright: ignore[reportPrivateUsage]
        "0..100",
        "lower_is_more_pressure",
    )


def test_legacy_business_entity_gets_effective_impact_role(
    synthetic_public_input: Path,
) -> None:
    bank = BankEnvironment(synthetic_public_input)
    entity_id = "bank:test:ServiceTest1"
    entity = next(
        item
        for item in bank.list_entities()["entities"]
        if item["canonical_id"] == entity_id
    )

    assert entity["evidence_role"] == "impact_only"
    profile = bank.query_metric_profile(entity_id, ["cnt"])
    assert profile["error_type"] == "ambiguous_evidence_role"
    assert all(
        candidate["entity_id"] != entity_id
        for candidate in bank.summarize_anomalies()["fault_type_candidates"]
    )
    assert (
        BankEnvironment._candidate_pressure_direction(  # pyright: ignore[reportPrivateUsage]
            "high memory OOM", "memoryBandwidthUtil"
        )
        == "higher_is_more_pressure"
    )
    assert (
        BankEnvironment._candidate_pressure_direction(  # pyright: ignore[reportPrivateUsage]
            "disk I/O error", "storageThroughputUtil"
        )
        == "higher_is_more_pressure"
    )


@pytest.mark.openrca_bank
def test_logs_are_bounded_and_keep_raw_source_locators() -> None:
    bank = environment()
    result = bank.query_log_samples(
        "bank:entity:Tomcat01", log_name="localhost_access_log", limit=2
    )

    assert result["status"] == "partial"
    assert len(result["samples"]) == 2
    assert result["samples"][0]["text"]
    assert result["samples"][0]["log_id"]
    assert result["samples"][0]["source_record"] > 0
    assert "limited to first 2" in result["coverage"]["truncation"]
    paired = result["paired_log_summary"]
    assert paired["counts_complete"] is True
    assert paired["samples_truncated"] is True
    assert paired["incident"]["matched_count"] > len(result["samples"])
    assert paired["incident"]["matched_count"] == sum(
        paired["incident"]["log_name_counts"].values()
    )
    assert paired["reference"]["matched_count"] == sum(
        paired["reference"]["log_name_counts"].values()
    )
    fragment = result["samples"][0]["text"][:12]
    filtered = bank.query_log_samples(
        "bank:entity:Tomcat01",
        log_name="localhost_access_log",
        contains=fragment,
        limit=2,
    )
    assert filtered["samples"]
    assert all(fragment in sample["text"] for sample in filtered["samples"])
    assert set(filtered["paired_log_summary"]["incident"]["log_name_counts"]) == {
        "localhost_access_log"
    }


@pytest.mark.openrca_bank
def test_trace_neighbors_only_return_source_backed_topology_edges() -> None:
    result = environment().query_trace_neighbors(
        "bank:entity:Tomcat02", direction="both", limit=2
    )

    assert result["status"] == "partial"
    assert len(result["edges"]) == 2
    assert all(edge["fact_kind"] == "observed" for edge in result["edges"])
    assert all(edge["samples"] for edge in result["edges"])
    assert result["edge_counts_scope"].startswith("full allowed query range")
    assert result["trace_duration_unit"].startswith("source-native;")


@pytest.mark.openrca_bank
def test_artifact_reads_require_manifest_content_hash_and_bounds() -> None:
    bank = environment()
    artifact_id = bank.list_entities()["artifact_refs"][0]["content_hash"]
    result = bank.read_artifact_slice(artifact_id, max_lines=1, max_bytes=4096)

    assert result["status"] == "partial"
    assert result["returned_lines"] == 1
    assert len(result["content"].encode()) <= 4096
    assert bank.read_artifact_slice("../hidden_eval/query.csv")["error_type"] == (
        "unknown_artifact"
    )
    assert bank.read_artifact_slice("entity_mapping.json")["error_type"] == (
        "unknown_artifact"
    )


@pytest.mark.openrca_bank
def test_anomaly_summary_is_deterministic_and_not_a_root_cause_ranking() -> None:
    bank = environment()
    first = bank.summarize_anomalies(limit=3)
    second = bank.summarize_anomalies(limit=3)
    default = bank.summarize_anomalies()
    filtered = bank.summarize_anomalies(entity_ids=["bank:entity:Tomcat01"], limit=3)

    assert first == second
    assert first["status"] == "partial"
    assert first["reference_health"] == "unverified"
    assert first["public_fault_types"]
    assert default["returned"] <= 12
    assert len(default["fault_type_candidates"]) <= min(
        32, len(default["public_fault_types"]) * 2
    )
    counts = Counter(
        candidate["fault_type"] for candidate in default["fault_type_candidates"]
    )
    assert all(count <= 2 for count in counts.values())
    assert len(
        {candidate["candidate_id"] for candidate in default["fault_type_candidates"]}
    ) == len(default["fault_type_candidates"])
    assert [
        candidate["candidate_rank"] for candidate in default["fault_type_candidates"]
    ] == sorted(
        candidate["candidate_rank"] for candidate in default["fault_type_candidates"]
    )
    for candidate in default["fault_type_candidates"]:
        assert set(candidate) - {
            "source_metric_id",
            "source_pressure_direction",
            "saturation_score",
        } == {
            "candidate_id",
            "candidate_rank",
            "fault_type",
            "entity_id",
            "metric_id",
            "resource_family",
            "score",
            "overview_score",
            "reference",
            "incident",
            "requires_paired_exact",
            "metric_role",
            "candidate_pressure_direction",
            "has_log_source",
            "has_trace_source",
            "log_names",
            "metric_group",
            "materiality",
            "materiality_basis",
        }
        if "source_metric_id" in candidate:
            assert candidate["source_metric_id"] != candidate["metric_id"]
            assert (
                candidate["source_metric_id"]
                in bank.list_metrics(candidate["entity_id"])["metric_ids"]
            )
            assert candidate["source_pressure_direction"] in {
                "higher_is_more_pressure",
                "lower_is_more_pressure",
                "unknown",
            }
        assert (
            candidate["metric_id"]
            in bank.list_metrics(candidate["entity_id"])["metric_ids"]
        )
        assert 0.0 <= candidate["score"] <= 1.0
        assert candidate["overview_score"] >= 0.0
        assert candidate["candidate_id"].startswith("fc1:")
        assert len(candidate["candidate_id"]) == 68
        assert candidate["candidate_rank"] in {0, 1}
        assert candidate["requires_paired_exact"] is True
        assert candidate["metric_role"] in {"direct", "ambiguous"}
        assert isinstance(candidate["has_log_source"], bool)
        assert set(candidate["reference"]) == {"min", "median", "max"}
        assert set(candidate["incident"]) == {"min", "median", "max"}
    compact_navigation = json.dumps(
        {
            "fault_type_candidates": default["fault_type_candidates"],
            "signals": default["signals"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert (len(compact_navigation) + 2) // 3 < 12_000
    assert all("resource_family" in signal for signal in first["signals"])
    assert all("relative_extreme_shift" in signal for signal in first["signals"])
    assert all("direct_metric_candidates" in signal for signal in first["signals"])
    assert "source_metric_id records the original nomination" in first["method"]
    assert "not a root-cause ranking" in first["limitations"][-1]
    assert first["unknown_counts"] == {
        "reference_only": 17,
        "incident_only": 8,
        "neither": 0,
    }
    assert len(first["unknown_series"]) == 25
    assert first["scanned_series"] == 1705
    assert first["scored_series"] == 1680
    assert first["artifact_refs"][0]["selector"]["signal_line_numbers"]
    assert all(
        signal["entity_id"] == "bank:entity:Tomcat01" for signal in filtered["signals"]
    )
    assert filtered["returned"] == 3


def test_anomaly_summary_detects_peaks_and_diversifies_families() -> None:
    bank = environment()
    entity_id = next(
        entity["canonical_id"]
        for entity in bank.list_entities()["entities"]
        if "synthetic_cpu_peak"
        in bank.list_metrics(entity["canonical_id"])["metric_ids"]
    )

    first = bank.summarize_anomalies(entity_ids=[entity_id], limit=6)
    second = bank.summarize_anomalies(entity_ids=[entity_id], limit=6)
    limited = bank.summarize_anomalies(entity_ids=[entity_id], limit=3)
    peak = next(
        signal
        for signal in first["signals"]
        if signal["metric_id"] == "synthetic_cpu_peak"
    )

    assert first == second
    assert first["status"] == "partial"
    assert first["public_fault_types"] == ["synthetic public fault type"]
    assert {signal["resource_family"] for signal in first["signals"]} == {
        "cpu",
        "memory",
        "network",
        "storage",
        "latency",
        "other",
    }
    assert sum(signal["resource_family"] == "other" for signal in first["signals"]) == 1
    assert peak["relative_median_shift"] == 0.0
    assert abs(peak["relative_extreme_shift"] - 0.9) < 1e-12
    assert abs(peak["overview_score"] - 0.9) < 1e-12
    assert limited["status"] == "partial"
    assert limited["returned"] == len(limited["signals"]) == 3
    assert "resource_family" in limited["coverage"]["truncation"]
    assert len(limited["artifact_refs"][0]["selector"]["signal_line_numbers"]) == 3
    assert limited["artifact_refs"][1]["selector"] == {
        "path": "vocabulary.json",
        "kind": "public_fault_types",
    }
    assert "relative_extreme_shift" in first["method"]
    assert "bounded positive peak" in first["method"]
    assert "not a root-cause ranking" in first["method"]


def test_direct_metric_candidates_are_bounded_catalog_siblings() -> None:
    bank = environment()
    result = bank.summarize_anomalies(limit=20)

    for signal in result["signals"]:
        candidates = signal["direct_metric_candidates"]
        catalog = set(bank.list_metrics(signal["entity_id"])["metric_ids"])
        assert len(candidates) <= 2
        assert len(candidates) == len(set(candidates))
        assert signal["metric_id"] not in candidates
        assert set(candidates) <= catalog
        assert all(
            bank._resource_family(candidate)  # pyright: ignore[reportPrivateUsage]
            == signal["resource_family"]
            for candidate in candidates
        )


def test_direct_metric_candidate_ranking_is_generic_and_stable() -> None:
    ranked = BankEnvironment._rank_direct_metric_candidates(  # pyright: ignore[reportPrivateUsage]
        [
            "synthetic_cpu_idle",
            "synthetic_cpu_usage_percent",
            "synthetic_cpu_used",
            "synthetic_cpu_load",
            "synthetic_cpu_counter",
            "synthetic_memory_used_percent",
            "synthetic_cpu_current",
        ],
        "synthetic_cpu_current",
        "cpu",
    )

    assert ranked == ["synthetic_cpu_usage_percent", "synthetic_cpu_load"]


def test_fault_type_candidates_use_public_semantics_and_positive_pressure() -> None:
    entity_id = "synthetic:entity"

    def signal(
        metric_id: str,
        resource_family: str,
        reference: tuple[float, float, float],
        incident: tuple[float, float, float],
    ) -> dict[str, object]:
        denominator = max(abs(reference[1]), abs(incident[1]), 1e-12)
        return {
            "entity_id": entity_id,
            "metric_id": metric_id,
            "resource_family": resource_family,
            "relative_median_shift": (incident[1] - reference[1]) / denominator,
            "reference": dict(zip(("min", "median", "max"), reference)),
            "incident": dict(zip(("min", "median", "max"), incident)),
        }

    signals = [
        signal("synthetic_cpu_idle", "cpu", (0.1, 0.2, 0.3), (0.2, 0.3, 0.5)),
        signal(
            "synthetic_cpu_usage_percent",
            "cpu",
            (10.0, 20.0, 30.0),
            (80.0, 82.0, 85.0),
        ),
        signal(
            "synthetic_mem_used_ratio",
            "memory",
            (0.2, 0.3, 0.4),
            (0.8, 0.81, 0.82),
        ),
        signal(
            "synthetic_network_response_latency",
            "latency",
            (10.0, 12.0, 14.0),
            (20.0, 22.0, 24.0),
        ),
        signal(
            "synthetic_net_packet_loss_percent",
            "network",
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 5.0),
        ),
        signal(
            "synthetic_dsk_read_util_percent",
            "storage",
            (10.0, 20.0, 30.0),
            (70.0, 80.0, 90.0),
        ),
        signal(
            "synthetic_fs_space_used_percent",
            "storage",
            (20.0, 30.0, 40.0),
            (80.0, 85.0, 90.0),
        ),
        signal(
            "synthetic_jvm_cpu_load_ratio",
            "cpu",
            (0.1, 0.2, 0.3),
            (0.8, 0.85, 0.9),
        ),
        signal(
            "synthetic_jvm_heap_used_percent",
            "memory",
            (20.0, 30.0, 40.0),
            (80.0, 85.0, 90.0),
        ),
    ]
    fault_types = [
        "high CPU usage",
        "high memory usage",
        "network latency",
        "network packet loss",
        "high disk read usage",
        "high disk space usage",
        "high JVM CPU load",
        "JVM OOM Heap",
    ]
    catalog = {entity_id: [str(signal["metric_id"]) for signal in signals]}

    first = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        fault_types, signals, catalog
    )
    second = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        fault_types, signals, catalog
    )
    selected = {
        candidate["fault_type"]: candidate
        for candidate in first
        if candidate["candidate_rank"] == 0
    }

    assert first == second
    counts = Counter(candidate["fault_type"] for candidate in first)
    assert all(count <= 2 for count in counts.values())
    assert [candidate["candidate_rank"] for candidate in first] == sorted(
        candidate["candidate_rank"] for candidate in first
    )
    assert selected["high CPU usage"]["metric_id"] == "synthetic_jvm_cpu_load_ratio"
    assert selected["high JVM CPU load"]["metric_id"] == (
        "synthetic_jvm_cpu_load_ratio"
    )
    assert selected["high disk read usage"]["metric_id"] == (
        "synthetic_dsk_read_util_percent"
    )
    assert selected["high disk space usage"]["metric_id"] == (
        "synthetic_dsk_read_util_percent"
    )
    assert 0.0 < selected["high memory usage"]["score"] <= 1.0
    assert 0.0 < selected["high CPU usage"]["score"] <= 1.0
    cpu_candidate = selected["high CPU usage"]
    canonical = json.dumps(
        {
            name: cpu_candidate[name]
            for name in (
                "fault_type",
                "entity_id",
                "resource_family",
                "metric_id",
            )
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert (
        cpu_candidate["candidate_id"] == "fc1:" + hashlib.sha256(canonical).hexdigest()
    )
    assert all(candidate["metric_id"] in catalog[entity_id] for candidate in first)
    assert all(candidate["requires_paired_exact"] is True for candidate in first)

    reversed_result = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        fault_types,
        list(reversed(signals)),
        {entity_id: list(reversed(catalog[entity_id]))},
    )
    assert first == reversed_result

    shared_metric = signal(
        "synthetic_network_response_latency",
        "latency",
        (10.0, 12.0, 14.0),
        (20.0, 22.0, 24.0),
    )
    shared = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["network latency", "response latency"],
        [shared_metric],
        {entity_id: [str(shared_metric["metric_id"])]},
    )
    assert len(shared) == 2
    assert shared[0]["metric_id"] == shared[1]["metric_id"]
    assert shared[0]["candidate_id"] != shared[1]["candidate_id"]
    assert all(candidate["candidate_rank"] == 0 for candidate in shared)


def test_fault_candidate_envelope_cross_beats_stable_high_level() -> None:
    stable_entity = "synthetic:stable"
    worsening_entity = "synthetic:worsening"
    stable_metric = "synthetic_memory_usage_percent"
    worsening_metric = "synthetic_memory_usage_percent_rising"
    common = {
        "resource_family": "memory",
        "incident": {"min": 80.0, "median": 85.0, "max": 90.0},
    }
    signals = [
        {
            **common,
            "entity_id": stable_entity,
            "metric_id": stable_metric,
            "relative_median_shift": 0.0,
            "reference": {"min": 80.0, "median": 85.0, "max": 90.0},
        },
        {
            **common,
            "entity_id": worsening_entity,
            "metric_id": worsening_metric,
            "relative_median_shift": (85.0 - 70.0) / 85.0,
            "reference": {"min": 60.0, "median": 70.0, "max": 80.0},
        },
    ]
    catalog = {
        stable_entity: [stable_metric],
        worsening_entity: [worsening_metric],
    }

    candidates = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high memory usage"], signals, catalog
    )

    assert len(candidates) == 2
    assert candidates[0]["entity_id"] == worsening_entity
    assert candidates[0]["metric_id"] == worsening_metric
    assert candidates[0]["score"] == 0.111111
    assert candidates[0]["candidate_rank"] == 0
    assert candidates[1]["entity_id"] == stable_entity
    assert candidates[1]["candidate_rank"] == 1
    assert candidates[1]["score"] == 0.0


def test_fault_candidate_keeps_one_best_metric_per_entity() -> None:
    first_entity = "synthetic:first"
    second_entity = "synthetic:second"
    first_best = "synthetic_memory_usage_percent_primary"
    first_other = "synthetic_memory_usage_percent_secondary"
    second_metric = "synthetic_memory_usage_percent_peer"
    signals = [
        {
            "entity_id": first_entity,
            "metric_id": first_other,
            "resource_family": "memory",
            "relative_median_shift": 0.2,
            "reference": {"min": 30.0, "median": 40.0, "max": 50.0},
            "incident": {"min": 40.0, "median": 50.0, "max": 60.0},
        },
        {
            "entity_id": first_entity,
            "metric_id": first_best,
            "resource_family": "memory",
            "relative_median_shift": 0.4,
            "reference": {"min": 20.0, "median": 30.0, "max": 40.0},
            "incident": {"min": 50.0, "median": 60.0, "max": 80.0},
        },
        {
            "entity_id": second_entity,
            "metric_id": second_metric,
            "resource_family": "memory",
            "relative_median_shift": 0.3,
            "reference": {"min": 30.0, "median": 40.0, "max": 50.0},
            "incident": {"min": 50.0, "median": 60.0, "max": 70.0},
        },
    ]
    catalog = {
        first_entity: [first_best, first_other],
        second_entity: [second_metric],
    }

    candidates = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high memory usage"], signals, catalog
    )

    assert [candidate["entity_id"] for candidate in candidates] == [
        first_entity,
        second_entity,
    ]
    assert [candidate["metric_id"] for candidate in candidates] == [
        first_best,
        second_metric,
    ]


@pytest.mark.parametrize(
    "source_metric",
    ["synthetic_memory_traffic", "synthetic_memory_current"],
)
def test_nondirect_candidate_uses_reachable_direct_metric_deterministically(
    source_metric: str,
) -> None:
    entity_id = "synthetic:reachable"
    direct_metric = "synthetic_memory_usage_percent"
    secondary_direct_metric = "synthetic_memory_used"
    source_reference = {"min": 10.0, "median": 20.0, "max": 30.0}
    source_incident = {"min": 80.0, "median": 90.0, "max": 100.0}
    direct_reference = {"min": 40.0, "median": 45.0, "max": 50.0}
    direct_incident = {"min": 39.0, "median": 44.0, "max": 49.0}
    signals = [
        {
            "entity_id": entity_id,
            "metric_id": source_metric,
            "resource_family": "memory",
            "relative_median_shift": 0.7,
            "reference": source_reference,
            "incident": source_incident,
        },
        {
            "entity_id": entity_id,
            "metric_id": secondary_direct_metric,
            "resource_family": "memory",
            "relative_median_shift": 0.0,
            "reference": {"min": 0.2, "median": 0.3, "max": 0.4},
            "incident": {"min": 0.2, "median": 0.3, "max": 0.4},
        },
        {
            "entity_id": entity_id,
            "metric_id": direct_metric,
            "resource_family": "memory",
            "relative_median_shift": -0.02,
            "reference": direct_reference,
            "incident": direct_incident,
        },
    ]
    catalog = {entity_id: [source_metric, secondary_direct_metric, direct_metric]}

    first = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high memory usage"], signals, catalog
    )
    reversed_result = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high memory usage"],
        list(reversed(signals)),
        {entity_id: list(reversed(catalog[entity_id]))},
    )

    assert first == reversed_result
    assert len(first) == 1
    assert first[0]["metric_id"] == direct_metric
    assert first[0]["source_metric_id"] == source_metric
    assert first[0]["reference"] == direct_reference
    assert first[0]["incident"] == direct_incident
    assert first[0]["score"] == 0.7
    assert first[0]["overview_score"] == 0.7
    assert first[0]["candidate_rank"] == 0
    assert first[0]["metric_role"] == "direct"


@pytest.mark.parametrize("metric_id", ["synthetic_cpu_idle", "synthetic_cpu_current"])
def test_fault_candidate_nondirect_fallback_and_empty_match(metric_id: str) -> None:
    entity_id = "synthetic:fallback"
    signal = {
        "entity_id": entity_id,
        "metric_id": metric_id,
        "resource_family": "cpu",
        "relative_median_shift": 0.5,
        "reference": {"min": 0.1, "median": 0.2, "max": 0.3},
        "incident": {"min": 0.2, "median": 0.4, "max": 0.5},
    }

    candidates = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high CPU usage", "unmatched public category"],
        [signal],
        {entity_id: [metric_id]},
    )

    assert len(candidates) == 1
    assert candidates[0]["fault_type"] == "high CPU usage"
    assert candidates[0]["metric_id"] == metric_id
    assert candidates[0]["candidate_rank"] == 0
    assert candidates[0]["metric_role"] == "ambiguous"
    assert candidates[0]["has_log_source"] is False
    assert "source_metric_id" not in candidates[0]


def test_remaining_candidate_uses_reachable_direct_metric() -> None:
    entity_id = "synthetic:remaining"
    remaining = {
        "entity_id": entity_id,
        "metric_id": "synthetic_disk_space_remaining",
        "resource_family": "storage",
        "relative_median_shift": -0.6,
        "reference": {"min": 40.0, "median": 50.0, "max": 60.0},
        "incident": {"min": 10.0, "median": 20.0, "max": 30.0},
    }
    used = {
        "entity_id": entity_id,
        "metric_id": "synthetic_disk_space_used_percent",
        "resource_family": "storage",
        "relative_median_shift": 0.375,
        "reference": {"min": 40.0, "median": 50.0, "max": 60.0},
        "incident": {"min": 70.0, "median": 80.0, "max": 90.0},
    }

    candidates = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high disk space usage"],
        [remaining, used],
        {entity_id: [str(remaining["metric_id"]), str(used["metric_id"])]},
    )

    assert candidates[0]["metric_id"] == used["metric_id"]
    assert candidates[0]["source_metric_id"] == remaining["metric_id"]
    assert candidates[0]["metric_role"] == "direct"


def test_fault_candidate_ranks_direct_evidence_before_stronger_proxy() -> None:
    def signal(
        entity_id: str,
        metric_id: str,
        reference_max: float,
        incident_max: float,
    ) -> dict[str, object]:
        return {
            "entity_id": entity_id,
            "metric_id": metric_id,
            "resource_family": "memory",
            "relative_median_shift": 0.0,
            "reference": {"min": 0.0, "median": 0.0, "max": reference_max},
            "incident": {"min": 0.0, "median": 0.0, "max": incident_max},
        }

    constant = "synthetic:constant-direct"
    strong = "synthetic:strong-direct"
    peer = "synthetic:peer-direct"
    signals = [
        signal(constant, "synthetic_memory_traffic", 0.0, 100.0),
        signal(constant, "synthetic_memory_usage_percent", 50.0, 50.0),
        signal(strong, "synthetic_memory_counter", 10.0, 100.0),
        signal(strong, "synthetic_memory_usage_percent", 20.0, 100.0),
        signal(peer, "synthetic_memory_usage_percent", 30.0, 100.0),
    ]
    catalog = {
        entity_id: [
            str(item["metric_id"]) for item in signals if item["entity_id"] == entity_id
        ]
        for entity_id in (constant, strong, peer)
    }

    first = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high memory usage"], signals, catalog
    )
    reversed_result = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high memory usage"],
        list(reversed(signals)),
        {key: list(reversed(value)) for key, value in catalog.items()},
    )

    assert first == reversed_result
    assert [candidate["entity_id"] for candidate in first] == [strong, peer]
    assert first[0]["metric_id"] == "synthetic_memory_usage_percent"
    assert first[0]["source_metric_id"] == "synthetic_memory_counter"
    assert first[0]["score"] == 0.9


def test_fault_candidate_lower_pressure_uses_incident_min_undercut() -> None:
    metric_id = "synthetic_network_bandwidth_utilization"
    severe = {
        "entity_id": "synthetic:severe",
        "metric_id": metric_id,
        "resource_family": "network",
        "reference": {"min": 80.0, "median": 90.0, "max": 100.0},
        "incident": {"min": 10.0, "median": 50.0, "max": 110.0},
    }
    mild = {
        "entity_id": "synthetic:mild",
        "metric_id": metric_id,
        "resource_family": "network",
        "reference": {"min": 80.0, "median": 90.0, "max": 100.0},
        "incident": {"min": 70.0, "median": 85.0, "max": 200.0},
    }

    candidates = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["network packet loss"],
        [mild, severe],
        {str(item["entity_id"]): [metric_id] for item in (mild, severe)},
    )

    assert candidates[0]["entity_id"] == severe["entity_id"]
    assert candidates[0]["candidate_pressure_direction"] == ("lower_is_more_pressure")
    assert candidates[0]["score"] == 0.875


def test_fault_candidate_does_not_replace_with_semantically_incompatible_direct() -> (
    None
):
    entity_id = "synthetic:semantic"
    source_metric = "synthetic_container_memory_failures_pgfault"
    incompatible_direct = "synthetic_container_memory_usage_percent"
    signals = [
        {
            "entity_id": entity_id,
            "metric_id": source_metric,
            "resource_family": "memory",
            "reference": {"min": 0.0, "median": 0.0, "max": 0.0},
            "incident": {"min": 0.0, "median": 0.0, "max": 10.0},
        },
        {
            "entity_id": entity_id,
            "metric_id": incompatible_direct,
            "resource_family": "memory",
            "reference": {"min": 20.0, "median": 30.0, "max": 40.0},
            "incident": {"min": 80.0, "median": 90.0, "max": 100.0},
        },
    ]

    candidates = BankEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["container packet loss"],
        signals,
        {entity_id: [source_metric, incompatible_direct]},
    )

    assert len(candidates) == 1
    assert candidates[0]["metric_id"] == source_metric
    assert candidates[0]["metric_role"] == "ambiguous"
    assert "source_metric_id" not in candidates[0]


@pytest.mark.parametrize(
    ("environment_type", "log_source"),
    [
        (BankEnvironment, "log/log_service.csv"),
        (MarketEnvironment, "log/log_proxy.csv"),
    ],
)
def test_candidate_log_availability_uses_provider_source_inventory(
    environment_type: type[OpenRCAEnvironment],
    log_source: str,
) -> None:
    with_log = "synthetic:with-log"
    without_log = "synthetic:without-log"
    metric_id = "synthetic_memory_usage_percent"
    signals = [
        {
            "entity_id": entity_id,
            "metric_id": metric_id,
            "resource_family": "memory",
            "relative_median_shift": 0.5,
            "reference": {"min": 10.0, "median": 20.0, "max": 30.0},
            "incident": {"min": 40.0, "median": 50.0, "max": 60.0},
        }
        for entity_id in (with_log, without_log)
    ]

    candidates = environment_type._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high memory usage"],
        signals,
        {with_log: [metric_id], without_log: [metric_id]},
        {
            with_log: [log_source],
            without_log: ["metric/metric_container.csv"],
        },
    )

    by_entity = {candidate["entity_id"]: candidate for candidate in candidates}
    assert by_entity[with_log]["has_log_source"] is True
    assert by_entity[without_log]["has_log_source"] is False
    assert all(candidate["metric_role"] == "direct" for candidate in candidates)


def test_market_candidate_keeps_log_source_priority_after_direct_crossing() -> None:
    with_log = "synthetic:with-log"
    without_log = "synthetic:without-log"
    metric_id = "synthetic_memory_usage_percent"
    signals = [
        {
            "entity_id": with_log,
            "metric_id": metric_id,
            "resource_family": "memory",
            "overview_score": 0.5,
            "reference": {"min": 10.0, "median": 20.0, "max": 30.0},
            "incident": {"min": 20.0, "median": 30.0, "max": 60.0},
        },
        {
            "entity_id": without_log,
            "metric_id": metric_id,
            "resource_family": "memory",
            "overview_score": 0.75,
            "reference": {"min": 10.0, "median": 20.0, "max": 30.0},
            "incident": {"min": 40.0, "median": 50.0, "max": 100.0},
        },
    ]

    candidates = MarketEnvironment._select_fault_type_candidates(  # pyright: ignore[reportPrivateUsage]
        ["high memory usage"],
        signals,
        {entity_id: [metric_id] for entity_id in (with_log, without_log)},
        {
            with_log: ["log/log_proxy.csv"],
            without_log: ["metric/metric_container.csv"],
        },
    )

    assert [candidate["entity_id"] for candidate in candidates] == [
        with_log,
        without_log,
    ]


def test_no_data_with_unknown_coverage_is_only_a_gap() -> None:
    result = environment().query_metric(
        "bank:entity:Tomcat01", "cnt", period="reference"
    )

    assert result["status"] == "no_data"
    assert result["coverage"]["completeness"] == "unknown"
    assert set(result["coverage"]["requested_range"]) == {
        "start",
        "end",
        "end_exclusive",
    }
    assert result["gap"]["kind"] == "coverage_gap"


def test_initialization_rejects_same_size_content_tampering(tmp_path: Path) -> None:
    root = minimal_public_input(tmp_path)
    target = root / "metric_catalog.json"
    content = target.read_bytes()
    target.write_bytes(bytes([content[0] ^ 1]) + content[1:])

    try:
        BankEnvironment(root)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered same-size artifact was accepted")


def test_initialization_rejects_duplicate_manifest_paths(tmp_path: Path) -> None:
    root = minimal_public_input(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][1]["path"] = manifest["artifacts"][0]["path"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="duplicate artifact path"):
        BankEnvironment(root)


def test_initialization_requires_exact_manifest_file_set(tmp_path: Path) -> None:
    root = minimal_public_input(tmp_path)
    (root / "extra.txt").write_text("not declared")

    with pytest.raises(ValueError, match="file set mismatch"):
        BankEnvironment(root)


@pytest.mark.parametrize("relative", ["entity_mapping.json", "dataset_profile.json"])
def test_initialization_requires_common_control_artifacts(
    tmp_path: Path, relative: str
) -> None:
    root = minimal_public_input(tmp_path)
    (root / relative).unlink()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = [
        item for item in manifest["artifacts"] if item["path"] != relative
    ]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="required public artifact is missing"):
        BankEnvironment(root)


def test_cross_environment_control_artifact_is_not_authorized(tmp_path: Path) -> None:
    root = minimal_public_input(tmp_path)
    artifact = root / "source_schema.json"
    artifact.write_text("{}")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = artifact.read_bytes()
    manifest["artifacts"].append(
        {
            "path": artifact.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact path is not authorized"):
        BankEnvironment(root)


def test_allowed_control_json_rejects_extra_answer_bearing_fields(
    tmp_path: Path,
) -> None:
    root = minimal_public_input(tmp_path)
    relative = "dataset_profile.json"
    artifact = root / relative
    profile = json.loads(artifact.read_text())
    profile["notes"] = "root cause is secret"
    artifact.write_text(json.dumps(profile))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = artifact.read_bytes()
    item = next(item for item in manifest["artifacts"] if item["path"] == relative)
    item["bytes"] = len(payload)
    item["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="dataset_profile.json schema mismatch"):
        BankEnvironment(root)


@pytest.mark.parametrize("name", ["query.csv", "nested/record.csv"])
def test_initialization_rejects_private_source_files(tmp_path: Path, name: str) -> None:
    root = minimal_public_input(tmp_path)
    private_source = root / name
    private_source.parent.mkdir(parents=True, exist_ok=True)
    private_source.write_text("private")

    with pytest.raises(ValueError, match="forbidden source file"):
        BankEnvironment(root)


@pytest.mark.parametrize(
    ("name", "content", "error"),
    [
        ("labels.csv", "value\nTOP_SECRET\n", "forbidden source file"),
        (
            "telemetry/metric/metric_container.csv",
            "timestamp,cmdb_id,kpi_name,value,root_cause,_source_record,_source_member\n1,x,y,2,TOP_SECRET,1,safe.csv\n",
            "forbidden evaluation header",
        ),
        (
            "telemetry/metric/metric_container.csv",
            "timestamp,cmdb_id,kpi_name,value,target_component,_source_record,_source_member\n1,x,y,2,TOP_SECRET,1,safe.csv\n",
            "forbidden evaluation header",
        ),
        (
            "telemetry/metric/metric_container.csv",
            "timestamp,cmdb_id,kpi_name,value,scoring_points,_source_record,_source_member\n1,x,y,2,TOP_SECRET,1,safe.csv\n",
            "forbidden evaluation header",
        ),
    ],
)
def test_registered_csv_evaluation_sentinel_fails_closed_before_tools(
    tmp_path: Path, name: str, content: str, error: str
) -> None:
    root = minimal_public_input(tmp_path)
    artifact = root / name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(content)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = artifact.read_bytes()
    manifest["artifacts"].append(
        {
            "path": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=error) as caught:
        BankEnvironment(root)
    assert "TOP_SECRET" not in str(caught.value)


def test_malicious_source_member_is_rejected_when_telemetry_is_queried(
    tmp_path: Path,
) -> None:
    root = minimal_public_input(tmp_path)
    relative = "telemetry/metric/metric_container.csv"
    artifact = root / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        "timestamp,cmdb_id,kpi_name,value,_source_record,_source_member\n"
        "1614852000,Tomcat01,OSLinux-CPU_CPU_CPUCpuUtil,90,1,"
        "hidden/selected_source.csv\n"
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = artifact.read_bytes()
    manifest["artifacts"].append(
        {
            "path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))

    bank = BankEnvironment(root)
    raw_slice = bank.read_artifact_slice(hashlib.sha256(payload).hexdigest())
    result = bank.query_metric("bank:entity:Tomcat01", "OSLinux-CPU_CPU_CPUCpuUtil")

    assert result["status"] == "error"
    assert result["error_type"] == "source_read_error"
    assert "forbidden evaluation source member" in result["summary"]
    assert raw_slice["error_type"] == "raw_telemetry_slice_forbidden"


@pytest.mark.parametrize(
    "name",
    ["selected_source.json", "selected-source.csv", "gold.csv", "alignment.json"],
)
def test_manifest_hash_does_not_authorize_extra_artifact_paths(
    tmp_path: Path, name: str
) -> None:
    root = minimal_public_input(tmp_path)
    artifact = root / name
    artifact.write_text("benign")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = artifact.read_bytes()
    manifest["artifacts"].append(
        {
            "path": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        BankEnvironment(root)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (
            "label_policy",
            "public instruction/task_index allowlist; evaluation labels excluded",
        ),
        (
            "label_policy",
            (
                "public field allowlist; scoring_points only in sibling hidden_eval, "
                "not referenced by public artifacts"
            ),
        ),
        (
            "label_policy",
            "label-free public case spec; evaluation labels excluded",
        ),
        ("label_policy", "excluded"),
        ("labels_included", False),
        ("labels_included", None),
        ("labels_included", "false"),
    ],
)
def test_safe_label_metadata_requires_explicit_negative_value(
    tmp_path: Path, key: str, value: object
) -> None:
    root = minimal_public_input(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[key] = value
    manifest_path.write_text(json.dumps(manifest))

    assert BankEnvironment(root).dataset_id


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("label_policy", True),
        ("label_policy", "embedded"),
        ("label_policy", "answer is component A"),
        ("label_policy", "unknown"),
        ("labels_included", True),
        ("labels_included", "embedded"),
        ("labels_included", 0),
        ("labels_included", {"answer": "secret"}),
    ],
)
def test_label_metadata_unknown_or_positive_values_fail_closed(
    tmp_path: Path, key: str, value: object
) -> None:
    root = minimal_public_input(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[key] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unsafe evaluation metadata"):
        BankEnvironment(root)


@pytest.mark.parametrize(
    ("suffix", "label"),
    [
        (".json", "scoring_points"),
        (".jsonl", "groundtruth"),
        (".json", "ground_truth"),
        (".jsonl", "answer"),
        (".json", "label"),
        (".jsonl", "labels"),
        (".json", "ScoringPoints"),
        (".jsonl", "rootCause"),
        (".json", "expected_answer"),
    ],
)
def test_initialization_recursively_rejects_evaluation_labels(
    tmp_path: Path, suffix: str, label: str
) -> None:
    root = minimal_public_input(tmp_path)
    relative = (
        "dataset_profile.json" if suffix == ".json" else "metric_window_stats.jsonl"
    )
    artifact = root / relative
    if suffix == ".json":
        payload = json.loads(artifact.read_text())
        payload["nested_evaluation_sentinel"] = [{label: "secret"}]
        artifact.write_text(json.dumps(payload))
    else:
        with artifact.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"nested": [{label: "secret"}]}) + "\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    content = artifact.read_bytes()
    manifest_item = next(
        item for item in manifest["artifacts"] if item["path"] == relative
    )
    manifest_item["bytes"] = len(content)
    manifest_item["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"evaluation label '{label}'"):
        BankEnvironment(root)


def test_artifact_read_rechecks_symlink_boundary(tmp_path: Path) -> None:
    root = minimal_public_input(tmp_path)
    bank = BankEnvironment(root)
    artifact_id = bank.list_metrics()["artifact_refs"][0]["content_hash"]
    original = root / "metric_catalog.json"
    moved = root / "metric_catalog.moved"
    original.rename(moved)
    original.symlink_to(moved.name)

    result = bank.read_artifact_slice(artifact_id)

    assert result["status"] == "error"
    assert result["error_type"] == "source_read_error"
    assert "became a symlink" in result["summary"]
