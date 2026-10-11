from datetime import UTC, datetime, timedelta

from rootwalk.occurrence import (
    OccurrenceAtom,
    build_occurrence_groups,
    matching_occurrence_group,
)

START = datetime(2020, 5, 23, 3, 30, tzinfo=UTC)


def atom(
    evidence_id: str,
    subject: str,
    second: int,
    *,
    duration: int = 1,
    fault_ready: bool = False,
    material_ready: bool = False,
) -> OccurrenceAtom:
    start = START + timedelta(seconds=second)
    return OccurrenceAtom(
        evidence_id=evidence_id,
        subject_id=subject,
        start=start,
        end=start + timedelta(seconds=duration),
        fault_type_ready=fault_ready,
        material_event_ready=material_ready,
    )


def test_same_subject_overlapping_profiles_form_one_eligible_group() -> None:
    groups = build_occurrence_groups(
        [
            atom("delay", "node-a", 120, duration=301, material_ready=True),
            atom("loss", "node-a", 420),
        ]
    )

    assert len(groups) == 1
    assert groups[0].evidence_refs == ("delay", "loss")
    assert groups[0].start == START + timedelta(seconds=420)
    assert groups[0].end == START + timedelta(seconds=421)


def test_runtime_matches_model_choice_without_selecting_between_groups() -> None:
    groups = build_occurrence_groups(
        [
            atom("a-1", "node-a", 100, duration=10),
            atom("a-2", "node-a", 105, duration=10),
            atom("b-1", "node-b", 500, fault_ready=True),
        ]
    )

    selected = matching_occurrence_group(groups, ["a-1", "a-2"])

    assert len(groups) == 2
    assert selected is not None
    assert selected.subject_id == "node-a"
    assert matching_occurrence_group(groups, ["a-1", "b-1"]) is None


def test_single_material_event_does_not_bypass_occurrence_corroboration() -> None:
    assert (
        build_occurrence_groups([atom("material", "node-a", 100, material_ready=True)])
        == []
    )
