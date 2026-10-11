"""Read-only investigation inspection from hash-verified historical records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .context import has_strict_directional_metric_shift
from .openrca import market_output_component
from .store import _verified_bundle_files  # pyright: ignore[reportPrivateUsage]

INSPECTOR_VERSION = "1"
_STREAMS = (
    "context_manifests",
    "decisions",
    "attempts",
    "model_requests",
    "observations",
    "evidence",
    "investigation_events",
    "budget_ledger",
)


def _object(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _array(value: object) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _call(payload: Mapping[str, Any]) -> int | None:
    number = payload.get("call_number")
    if isinstance(number, int) and not isinstance(number, bool):
        return number
    decision_id = payload.get("decision_id")
    if not decision_id:
        operation_id = _array(payload.get("operation_id"))
        decision_id = operation_id[1] if len(operation_id) > 1 else None
    match = re.search(r":(\d+)$", str(decision_id or ""))
    return int(match[1]) if match else None


def _sum_known(items: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [item.get(field) for item in items]
    known = [
        v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return {
        "sum": sum(known) if known else None,
        "known_count": len(known),
        "missing_count": len(values) - len(known),
    }


def inspect_bundle(
    bundle: str | Path, *, call_number: int | None = None
) -> dict[str, Any]:
    """Inspect saved choices and effects; never execute tools or load external data."""
    root = Path(bundle).resolve(strict=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("inspection requires a regular bundle manifest")
    manifest = _object(json.loads(manifest_path.read_text()))
    files = _verified_bundle_files(root, manifest)
    required = {
        "task.json",
        "topology.json",
        "report.json",
        "artifact_index.json",
        "context_manifests.jsonl",
        "decisions.jsonl",
        "observations.jsonl",
        "investigation_events.jsonl",
    }
    if missing := required - files.keys():
        raise ValueError(f"bundle is missing declared records: {sorted(missing)}")
    task = _object(json.loads(files["task.json"].read_text()))
    report = _object(json.loads(files["report.json"].read_text()))
    topology = _object(json.loads(files["topology.json"].read_text()))
    nodes = [_object(node) for node in _array(topology.get("nodes"))]
    nodes_by_id = {
        str(node["canonical_id"]): node
        for node in nodes
        if isinstance(node.get("canonical_id"), str)
    }

    def output_component(node: Mapping[str, Any]) -> str:
        original = str(node.get("original_id", node.get("canonical_id", "")))
        return (
            market_output_component(original)
            if task.get("environment") == "openrca_market"
            else original
        )

    aliases: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        aliases.setdefault(output_component(node), []).append(node)
    index = {
        item["content_hash"]: item
        for raw in _array(json.loads(files["artifact_index.json"].read_text()))
        if (item := _object(raw))
    }

    streams: dict[str, list[dict[str, Any]]] = {}
    for name in _STREAMS:
        records: list[dict[str, Any]] = []
        path = files.get(name + ".jsonl")
        for line_number, line in enumerate(
            path.read_text().splitlines() if path else [], 1
        ):
            if not line.strip():
                continue
            record = _object(json.loads(line))
            if not isinstance(record.get("revision"), int) or not isinstance(
                record.get("payload"), dict
            ):
                raise ValueError(f"{name}:{line_number} requires revisioned records")
            if records and record["revision"] <= records[-1]["revision"]:
                raise ValueError(f"{name}:{line_number} has non-increasing revisions")
            records.append(
                {**record, "source": f"{root / (name + '.jsonl')}:{line_number}"}
            )
        streams[name] = records
    if not any(
        r["payload"].get("event_type") == "run_sealed"
        and r["payload"].get("status") == report.get("status")
        for r in streams["investigation_events"]
    ):
        raise ValueError("report has no matching run_sealed event")

    def artifact(reference: object) -> tuple[object, str | None]:
        entry = _object(index.get(reference)) if isinstance(reference, str) else {}
        if (
            entry.get("storage") != "embedded"
            or entry.get("visibility") != "model_readable"
        ):
            return None, None
        relative = entry.get("bundle_path")
        path = files.get(relative) if isinstance(relative, str) else None
        if path is None or hashlib.sha256(path.read_bytes()).hexdigest() != reference:
            raise ValueError(f"invalid embedded artifact reference: {reference}")
        return json.loads(path.read_text()), str(path)

    rounds: dict[int, dict[str, Any]] = {}

    def ensure(number: int) -> dict[str, Any]:
        return rounds.setdefault(
            number,
            {
                "call_number": number,
                "context": None,
                "decision": None,
                "model_attempts": [],
                "provider_requests": [],
                "tool_attempts": [],
                "observations": [],
                "evidence": [],
                "events": [],
                "budget_events": [],
            },
        )

    boundaries: list[tuple[int, int]] = []
    for record in streams["context_manifests"]:
        payload = record["payload"]
        number = _call(payload)
        if number is None:
            raise ValueError("context manifest has no call_number")
        messages, input_path = artifact(
            payload.get("model_input_ref", payload.get("message_hash"))
        )
        context: dict[str, Any] = {}
        section_bytes: dict[str, int] = {}
        system_bytes = 0
        for raw_message in _array(messages):
            message = _object(raw_message)
            if message.get("role") == "system":
                system_bytes += len(str(message.get("content", "")).encode())
            elif message.get("role") == "user":
                context = _object(_parse(message.get("content")))
                section_bytes = {
                    name: len(_json(value).encode()) for name, value in context.items()
                }
        view = {
            key: context.get(key)
            for key in (
                "current_focus",
                "convergence",
                "candidate_evidence_summary",
                "comparison_coverage",
                "feedback",
                "budget",
                "hypothesis_index",
                "terminal_hypothesis_summary",
            )
        }
        ensure(number)["context"] = {
            **payload,
            "source": record["source"],
            "input_path": input_path,
            "system_bytes": system_bytes,
            "section_bytes": section_bytes,
            "view": view,
        }
        boundaries.append((record["revision"], number))

    def owner(record: dict[str, Any]) -> int | None:
        explicit = _call(record["payload"])
        if explicit is not None:
            return explicit
        return next(
            (
                n
                for revision, n in reversed(boundaries)
                if revision <= record["revision"]
            ),
            None,
        )

    for record in streams["decisions"]:
        number = owner(record)
        if number is not None:
            payload = record["payload"]
            _, response_path = artifact(payload.get("model_response_ref"))
            ensure(number)["decision"] = {
                **payload,
                "source": record["source"],
                "response_path": response_path,
            }

    attempts: dict[tuple[str, str], dict[str, Any]] = {}
    provider_requests_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    unkeyed_provider_requests: list[tuple[int, dict[str, Any]]] = []
    for record in streams["attempts"]:
        payload = record["payload"]
        key = (
            str(payload.get("attempt_id", record.get("stable_key"))),
            str(payload.get("execution_attempt_id", "legacy")),
        )
        prior = attempts.setdefault(key, {"sources": []})
        prior.update(payload)
        prior["sources"].append(record["source"])
        prior["call_number"] = owner(record)
    for (attempt_id, _), payload in attempts.items():
        number = payload.get("call_number")
        if number is None:
            continue
        if re.fullmatch(r"model:\d+", attempt_id):
            ensure(number)["model_attempts"].append(payload)
        elif attempt_id.startswith("attempt:provider:"):
            request_index = payload.get("provider_request_index")
            if isinstance(request_index, int) and not isinstance(request_index, bool):
                provider_requests_by_key[number, request_index] = payload
            else:
                unkeyed_provider_requests.append((number, payload))
        elif not attempt_id.startswith("model:"):
            ensure(number)["tool_attempts"].append(payload)

    for record in streams["model_requests"]:
        number = owner(record)
        if number is None:
            continue
        payload = {
            **record["payload"],
            "source": record["source"],
            "revision": record["revision"],
        }
        request_index = payload.get("provider_request_index")
        if not isinstance(request_index, int) or isinstance(request_index, bool):
            request_index = payload.get("request_index")
        if isinstance(request_index, int) and not isinstance(request_index, bool):
            payload["provider_request_index"] = request_index
            provider_requests_by_key.setdefault((number, request_index), {}).update(
                payload
            )
        else:
            unkeyed_provider_requests.append((number, payload))

    for (number, _), payload in sorted(provider_requests_by_key.items()):
        for field in ("model_input_ref", "model_response_ref"):
            _, payload[field.replace("_ref", "_path")] = artifact(payload.get(field))
        ensure(number)["provider_requests"].append(payload)
    for number, payload in unkeyed_provider_requests:
        for field in ("model_input_ref", "model_response_ref"):
            _, payload[field.replace("_ref", "_path")] = artifact(payload.get(field))
        ensure(number)["provider_requests"].append(payload)

    for name, target in (
        ("observations", "observations"),
        ("evidence", "evidence"),
        ("investigation_events", "events"),
        ("budget_ledger", "budget_events"),
    ):
        for record in streams[name]:
            number = owner(record)
            if number is None:
                continue
            payload = {
                **record["payload"],
                "source": record["source"],
                "revision": record["revision"],
            }
            if target == "observations":
                payload["summary"] = _parse(payload.get("summary")) or payload.get(
                    "summary"
                )
                payload["result_paths"] = [
                    path
                    for ref in _array(payload.get("artifact_refs"))
                    if (path := artifact(_object(ref).get("content_hash"))[1])
                    is not None
                ]
            ensure(number)[target].append(payload)

    candidates: dict[str, dict[str, Any]] = {}
    series: dict[tuple[str, str], dict[str, Any]] = {}
    for record in streams["evidence"]:
        payload = record["payload"]
        paired = _object(_object(payload.get("value")).get("paired_metric_summary"))
        if not paired:
            continue
        for observation in streams["observations"]:
            if observation["payload"].get("observation_id") in _array(
                payload.get("observation_refs")
            ):
                args = _object(observation["payload"].get("arguments"))
                series[(str(args.get("entity_id")), str(args.get("metric_id")))] = (
                    paired
                )
                break
    for record in streams["observations"]:
        if record["payload"].get("tool") != "summarize_anomalies":
            continue
        overview = _object(_parse(record["payload"].get("summary")))
        for rank, raw in enumerate(_array(overview.get("fault_type_candidates")), 1):
            candidate = _object(raw)
            if not candidate.get("candidate_id"):
                continue
            pair = series.get(
                (str(candidate.get("entity_id")), str(candidate.get("metric_id")))
            )
            source_pair = series.get(
                (
                    str(candidate.get("entity_id")),
                    str(candidate.get("source_metric_id")),
                )
            )
            candidates[str(candidate["candidate_id"])] = {
                **candidate,
                "overview_position": rank,
                "source": record["source"],
                "direct_pair": pair,
                "source_pair": source_pair,
                "direct_constant": _constant(pair),
                "source_constant": _constant(source_pair),
                "direct_strict": has_strict_directional_metric_shift(
                    pair, candidate.get("candidate_pressure_direction")
                ),
                "source_strict": has_strict_directional_metric_shift(
                    source_pair, candidate.get("source_pressure_direction")
                ),
                "dispositions": [],
            }
    for record in streams["investigation_events"]:
        payload = record["payload"]
        candidate = candidates.get(str(payload.get("candidate_id")))
        if (
            candidate is not None
            and payload.get("event_type") == "candidate_disposition_applied"
        ):
            original = _object(payload.get("original_operation"))
            applied = _object(payload.get("applied_operation"))
            candidate["dispositions"].append(
                {
                    "call_number": owner(record),
                    "status": payload.get("status"),
                    "changed_fields": [
                        k
                        for k in sorted(original.keys() | applied.keys())
                        if original.get(k) != applied.get(k)
                    ],
                    "source": record["source"],
                }
            )

    for candidate in candidates.values():
        node = nodes_by_id.get(str(candidate.get("entity_id")), {})
        component = output_component(node)
        related = aliases.get(component, [])
        own_sources = {str(source) for source in _array(node.get("sources"))}
        sibling_sources = {
            str(source)
            for sibling in related
            if sibling.get("canonical_id") != node.get("canonical_id")
            for source in _array(sibling.get("sources"))
        }
        candidate["output_component"] = component
        candidate["component_level"] = (
            "node"
            if re.fullmatch(r"node-\d+", component)
            else "pod"
            if re.search(r"-\d+$", component)
            else "service"
        )
        candidate["identity_alias_count"] = len(related)
        candidate["own_source_kinds"] = sorted(
            {source.split("/", 1)[0] for source in own_sources}
        )
        candidate["sibling_source_kinds"] = sorted(
            {source.split("/", 1)[0] for source in sibling_sources}
        )
        candidate["sibling_log_source_available"] = not any(
            source.startswith("log/") for source in own_sources
        ) and any(source.startswith("log/") for source in sibling_sources)

    ordered = sorted(rounds.values(), key=lambda r: r["call_number"])
    for row in ordered:
        row["usage"] = (
            _object(row["model_attempts"][-1].get("usage"))
            if row["model_attempts"]
            else {}
        )
        row["new_observation_count"] = len(row["observations"])
        row["query_tools"] = dict(
            Counter(item.get("tool") for item in row["observations"])
        )
        row["bindings"] = [
            {
                "original": event.get("original_operation"),
                "applied": event.get("applied_operation"),
                "source": event["source"],
            }
            for event in row["events"]
            if event.get("event_type") == "candidate_disposition_applied"
        ]

    model_attempts = [a for row in ordered for a in row["model_attempts"]]
    provider_requests = [a for row in ordered for a in row["provider_requests"]]
    tool_attempts = [a for row in ordered for a in row["tool_attempts"]]
    contexts = [row["context"] for row in ordered if row["context"]]
    section_totals: Counter[str] = Counter()
    for context in contexts:
        section_totals.update(context["section_bytes"])
    no_query_rounds = [
        row for row in ordered if row["decision"] and not row["observations"]
    ]
    observations = [record["payload"] for record in streams["observations"]]
    recommendation = _object(report.get("benchmark_recommendation"))
    result = {
        "inspector_version": INSPECTOR_VERSION,
        "bundle": str(root),
        "integrity": {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "verified_files": len(files),
            "unmanifested_files": sorted(
                str(p.relative_to(root))
                for p in root.rglob("*")
                if (p.is_file() or p.is_symlink())
                and str(p.relative_to(root)) not in {*files, "manifest.json"}
            ),
        },
        "provenance": {
            k: manifest.get(k)
            for k in (
                "code_revision",
                "prompt_version",
                "tool_version",
                "model_id",
                "context_strategy",
                "validator_mode",
                "query_sandbox_effective",
                "max_model_calls",
                "max_tool_attempts",
            )
        },
        "record_availability": {name: name + ".jsonl" in files for name in _STREAMS},
        "task": {
            k: task.get(k)
            for k in ("task_id", "environment", "source_row_id", "question")
        },
        "summary": {
            "status": report.get("status"),
            "model_calls": report.get("model_calls"),
            "provider_requests": len(provider_requests)
            if "model_requests.jsonl" in files or provider_requests
            else None,
            "provider_status_counts": dict(
                Counter(request.get("status") for request in provider_requests)
            ),
            "tool_attempts": len(tool_attempts),
            "observations": len(observations),
            "tool_counts": dict(Counter(o.get("tool") for o in observations)),
            "context_builds": len(contexts),
            "rounds_without_new_observations": len(no_query_rounds),
            "usage_without_new_observations": _sum_known(
                [r["usage"] for r in no_query_rounds], "prompt_tokens"
            ),
            "model_usage": report.get("model_usage"),
            "budget": report.get("budget"),
            "model_duration_ms": _sum_known(model_attempts, "duration_ms"),
            "tool_duration_ms": _sum_known(tool_attempts, "duration_ms"),
            "wall_clock_seconds": report.get("wall_clock_seconds"),
            "context_section_bytes_total": dict(section_totals.most_common()),
            "system_bytes_total": sum(c["system_bytes"] for c in contexts),
            "finding_count": len(_array(report.get("findings"))),
            "recommendation_only": bool(recommendation) and not report.get("findings"),
            "final_selection_origin": "findings"
            if report.get("findings")
            else recommendation.get("policy", "abstention"),
            "abstain_reason": report.get("abstain_reason"),
        },
        "candidate_funnel": {
            "topology_nodes": len(nodes),
            "overview_candidates": len(candidates),
            "candidate_entities": len(
                {c.get("entity_id") for c in candidates.values()}
            ),
            "queried_direct_candidates": sum(
                c["direct_pair"] is not None for c in candidates.values()
            ),
            "constant_direct_candidates": sum(
                c["direct_constant"] is True for c in candidates.values()
            ),
            "strict_direct_candidates": sum(
                c["direct_strict"] is True for c in candidates.values()
            ),
            "strict_source_candidates": sum(
                c["source_strict"] is True for c in candidates.values()
            ),
            "strict_source_replaced_by_constant_direct": sum(
                c["source_strict"] is True and c["direct_constant"] is True
                for c in candidates.values()
            ),
            "declared_log_sources": sum(
                c.get("has_log_source") is True for c in candidates.values()
            ),
            "candidate_component_levels": dict(
                Counter(c.get("component_level") for c in candidates.values())
            ),
            "fragmented_candidate_identities": sum(
                int(c.get("identity_alias_count", 0)) > 1 for c in candidates.values()
            ),
            "candidates_missing_own_log_with_sibling_log": sum(
                c.get("sibling_log_source_available") is True
                for c in candidates.values()
            ),
            "fragmented_output_components": sum(
                len(group) > 1 for group in aliases.values()
            ),
            "candidate_status_counts": dict(
                Counter(
                    c["dispositions"][-1]["status"]
                    for c in candidates.values()
                    if c["dispositions"]
                )
            ),
            "report_hypothesis_status_counts": dict(
                Counter(
                    _object(_object(c).get("hypothesis")).get(
                        "status", _object(c).get("status")
                    )
                    for c in _array(report.get("candidates"))
                )
            ),
        },
        "candidates": list(candidates.values()),
        "report": report,
        "rounds": ordered,
        "limitations": [
            "Historical records only; missing timings and provider requests "
            "remain unknown.",
            "Section sizes are serialized UTF-8 bytes, not billed tokens; "
            "no-query rounds are not automatically wasted.",
            "Only manifest-bound embedded model-readable artifacts are opened; "
            "no external URIs or evaluation labels are read.",
            "Current strategy is not replayed over historical decisions; "
            "manifest hashes do not prove external authenticity.",
        ],
    }
    result["diagnostic_signals"] = _diagnostic_signals(result)
    if call_number is not None:
        if call_number not in rounds:
            raise ValueError(f"call {call_number} is absent")
        result["rounds"] = [rounds[call_number]]
    return result


def inspect_many(bundles: Sequence[str | Path]) -> dict[str, Any]:
    """Aggregate comparable inspection facts without merging per-run histories."""
    reports = [inspect_bundle(bundle) for bundle in bundles]
    if not reports:
        raise ValueError("inspection requires at least one bundle")

    def values(path: tuple[str, ...]) -> list[int | float]:
        result: list[int | float] = []
        for report in reports:
            value: object = report
            for name in path:
                value = _object(value).get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result.append(value)
        return result

    def distribution(path: tuple[str, ...]) -> dict[str, Any]:
        raw = values(path)
        return {
            "sum": sum(raw) if raw else None,
            "mean": sum(raw) / len(raw) if raw else None,
            "min": min(raw) if raw else None,
            "max": max(raw) if raw else None,
            "known_count": len(raw),
            "missing_count": len(reports) - len(raw),
        }

    recommendations: Counter[str] = Counter()
    candidate_presence: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    diagnostic_signals: Counter[str] = Counter()
    for report in reports:
        tool_counts.update(_object(report["summary"].get("tool_counts")))
        candidate_presence.update(
            set(
                str(candidate.get("candidate_id"))
                for candidate in report["candidates"]
                if candidate.get("candidate_id")
            )
        )
        recommendation = _object(report["report"].get("benchmark_recommendation"))
        if recommendation.get("candidate_id"):
            recommendations[str(recommendation["candidate_id"])] += 1
        diagnostic_signals.update(
            str(signal.get("code"))
            for signal in report["diagnostic_signals"]
            if signal.get("code")
        )

    return {
        "inspector_version": INSPECTOR_VERSION,
        "run_count": len(reports),
        "statuses": dict(
            Counter(report["summary"].get("status") for report in reports)
        ),
        "final_selection_origins": dict(
            Counter(
                report["summary"].get("final_selection_origin") for report in reports
            )
        ),
        "model_calls": distribution(("summary", "model_calls")),
        "provider_requests": distribution(("summary", "provider_requests")),
        "prompt_tokens": distribution(("summary", "model_usage", "prompt_tokens")),
        "completion_tokens": distribution(
            ("summary", "model_usage", "completion_tokens")
        ),
        "tool_attempts": distribution(("summary", "tool_attempts")),
        "rounds_without_new_observations": distribution(
            ("summary", "rounds_without_new_observations")
        ),
        "prompt_tokens_without_new_observations": distribution(
            ("summary", "usage_without_new_observations", "sum")
        ),
        "overview_candidates": distribution(
            ("candidate_funnel", "overview_candidates")
        ),
        "queried_direct_candidates": distribution(
            ("candidate_funnel", "queried_direct_candidates")
        ),
        "constant_direct_candidates": distribution(
            ("candidate_funnel", "constant_direct_candidates")
        ),
        "strict_direct_candidates": distribution(
            ("candidate_funnel", "strict_direct_candidates")
        ),
        "strict_source_candidates": distribution(
            ("candidate_funnel", "strict_source_candidates")
        ),
        "strict_source_replaced_by_constant_direct": distribution(
            ("candidate_funnel", "strict_source_replaced_by_constant_direct")
        ),
        "declared_log_sources": distribution(
            ("candidate_funnel", "declared_log_sources")
        ),
        "fragmented_candidate_identities": distribution(
            ("candidate_funnel", "fragmented_candidate_identities")
        ),
        "candidates_missing_own_log_with_sibling_log": distribution(
            ("candidate_funnel", "candidates_missing_own_log_with_sibling_log")
        ),
        "tool_counts": dict(tool_counts),
        "candidate_presence": dict(candidate_presence.most_common()),
        "recommendation_candidate_counts": dict(recommendations),
        "diagnostic_signal_counts": dict(diagnostic_signals),
        "runs": [
            {
                "bundle": report["bundle"],
                "task_id": report["task"].get("task_id"),
                "status": report["summary"].get("status"),
                "selection": report["summary"].get("final_selection_origin"),
                "model_calls": report["summary"].get("model_calls"),
                "prompt_tokens": _object(report["summary"].get("model_usage")).get(
                    "prompt_tokens"
                ),
                "tool_attempts": report["summary"].get("tool_attempts"),
                "constant_direct_candidates": report["candidate_funnel"].get(
                    "constant_direct_candidates"
                ),
                "declared_log_sources": report["candidate_funnel"].get(
                    "declared_log_sources"
                ),
                "fragmented_candidate_identities": report["candidate_funnel"].get(
                    "fragmented_candidate_identities"
                ),
                "candidates_missing_own_log_with_sibling_log": report[
                    "candidate_funnel"
                ].get("candidates_missing_own_log_with_sibling_log"),
            }
            for report in reports
        ],
        "limitations": [
            "The aggregate contains historical observations only and does not score "
            "or replay the runs.",
            "Candidate presence counts identity reuse across runs; it is not accuracy.",
        ],
    }


def _diagnostic_signals(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = _object(report.get("summary"))
    funnel = _object(report.get("candidate_funnel"))
    candidates = [_object(item) for item in _array(report.get("candidates"))]
    signals: list[dict[str, Any]] = []

    def add(code: str, severity: str, evidence: Mapping[str, Any]) -> None:
        signals.append({"code": code, "severity": severity, "evidence": dict(evidence)})

    candidate_count = int(funnel.get("overview_candidates", 0))
    constant_count = int(funnel.get("constant_direct_candidates", 0))
    if candidate_count and constant_count * 2 >= candidate_count:
        add(
            "candidate_direct_signal_collapse",
            "high",
            {
                "constant_direct_candidates": constant_count,
                "candidates": candidate_count,
            },
        )
    scores = Counter(
        candidate.get("score")
        for candidate in candidates
        if isinstance(candidate.get("score"), (int, float))
        and not isinstance(candidate.get("score"), bool)
    )
    if candidate_count and scores:
        score, count = scores.most_common(1)[0]
        if count * 2 >= candidate_count:
            add(
                "candidate_score_saturation",
                "high",
                {"score": score, "count": count, "candidates": candidate_count},
            )
    hidden_logs = int(funnel.get("candidates_missing_own_log_with_sibling_log", 0))
    if hidden_logs:
        add(
            "identity_fragmentation_hides_logs",
            "high",
            {
                "affected_candidates": hidden_logs,
                "fragmented_output_components": funnel.get(
                    "fragmented_output_components"
                ),
            },
        )
    tool_counts = _object(summary.get("tool_counts"))
    if candidate_count and not tool_counts.get("query_trace_neighbors"):
        add(
            "candidate_investigation_has_no_trace_queries",
            "medium",
            {
                "candidate_count": candidate_count,
                "paths": len(_array(_object(report.get("report")).get("paths"))),
            },
        )
    prompt = _object(summary.get("model_usage")).get("prompt_tokens")
    no_observation = _object(summary.get("usage_without_new_observations")).get("sum")
    if (
        isinstance(prompt, (int, float))
        and prompt > 0
        and isinstance(no_observation, (int, float))
        and no_observation * 2 >= prompt
    ):
        add(
            "state_transition_prompt_dominance",
            "medium",
            {
                "prompt_tokens": prompt,
                "prompt_tokens_without_new_observations": no_observation,
                "share": no_observation / prompt,
            },
        )
    if summary.get("recommendation_only") is True:
        add(
            "prediction_comes_from_low_confidence_recommendation",
            "medium",
            {"selection_origin": summary.get("final_selection_origin")},
        )
    provider_requests = summary.get("provider_requests")
    model_calls = summary.get("model_calls")
    if (
        isinstance(provider_requests, int)
        and isinstance(model_calls, int)
        and provider_requests > model_calls
    ):
        add(
            "provider_repairs_present",
            "low",
            {"provider_requests": provider_requests, "model_calls": model_calls},
        )
    return signals


def _constant(paired: object) -> bool | None:
    pair = _object(paired)
    if pair.get("counts_complete") is not True:
        return None
    reference, incident = _object(pair.get("reference")), _object(pair.get("incident"))
    if not reference.get("sample_count") or not incident.get("sample_count"):
        return None
    values = [window.get(k) for window in (reference, incident) for k in ("min", "max")]
    return all(v is not None for v in values) and len(set(values)) == 1


def render_inspection(report: dict[str, Any]) -> str:
    """Render a portable Markdown audit with raw artifact and record links."""
    summary = report["summary"]
    lines = [
        f"# Investigation: {report['task']['task_id']}",
        "",
        f"Bundle: {report['bundle']}",
        "",
        f"Status: {summary['status']}. Selection: {summary['final_selection_origin']}.",
        "",
        "```json",
        json.dumps(
            {
                "provenance": report["provenance"],
                "summary": summary,
                "candidate_funnel": report["candidate_funnel"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "| Call | Requested operation | Required next operation | "
        "New observations | Input tokens | Context estimate |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in report["rounds"]:
        decision = _object(row["decision"])
        context = _object(row["context"])
        operations = ", ".join(
            str(_object(o).get("tool", _object(o).get("type")))
            for o in _array(decision.get("operations"))
        )
        required = _object(_object(context.get("view")).get("convergence")).get(
            "required_next_operation", "unknown"
        )
        lines.append(
            f"| {row['call_number']} | {operations} | {required} | "
            f"{row['new_observation_count']} | "
            f"{row['usage'].get('prompt_tokens', 'unknown')} | "
            f"{context.get('estimated_tokens', 'unknown')} |"
        )
    for row in report["rounds"]:
        lines += ["", f"## Call {row['call_number']}", ""]
        for name, path in (
            ("Compiled input", _object(row["context"]).get("input_path")),
            ("Raw response", _object(row["decision"]).get("response_path")),
        ):
            if path:
                lines += [f"[{name}](<{path}>)", ""]
        detail = {
            k: row[k]
            for k in (
                "context",
                "decision",
                "provider_requests",
                "tool_attempts",
                "bindings",
                "events",
            )
        }
        lines += [
            "<details><summary>Saved context, requests, effects and events</summary>",
            "",
            "```json",
            json.dumps(detail, ensure_ascii=False, indent=2),
            "```",
            "",
            "</details>",
        ]
    lines += [
        "",
        "## Candidate investigation",
        "",
        "```json",
        json.dumps(report["candidates"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)
