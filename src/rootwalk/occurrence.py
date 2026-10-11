"""Source-backed occurrence groups without choosing an RCA conclusion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True, slots=True)
class OccurrenceAtom:
    evidence_id: str
    subject_id: str
    start: datetime
    end: datetime
    fault_type_ready: bool = False
    material_event_ready: bool = False


@dataclass(frozen=True, slots=True)
class OccurrenceGroup:
    evidence_refs: tuple[str, ...]
    subject_id: str
    start: datetime
    end: datetime
    fault_type_ready: bool
    material_event_ready: bool

    @property
    def eligible(self) -> bool:
        """A low-confidence occurrence needs corroboration or a typed fault gate."""
        return len(self.evidence_refs) >= 2 or self.fault_type_ready


def build_occurrence_groups(atoms: Iterable[OccurrenceAtom]) -> list[OccurrenceGroup]:
    """Merge same-subject Evidence only while every onset interval overlaps."""
    by_subject: dict[str, list[OccurrenceAtom]] = {}
    for atom in atoms:
        if atom.start >= atom.end:
            continue
        by_subject.setdefault(atom.subject_id, []).append(atom)

    result: list[OccurrenceGroup] = []
    for subject_id, subject_atoms in sorted(by_subject.items()):
        groups: list[OccurrenceGroup] = []
        for atom in sorted(
            subject_atoms,
            key=lambda item: (item.start, item.end, item.evidence_id),
        ):
            merged = False
            for index, group in enumerate(groups):
                overlap_start = max(group.start, atom.start)
                overlap_end = min(group.end, atom.end)
                if overlap_start >= overlap_end:
                    continue
                groups[index] = OccurrenceGroup(
                    evidence_refs=tuple(
                        sorted({*group.evidence_refs, atom.evidence_id})
                    ),
                    subject_id=subject_id,
                    start=overlap_start,
                    end=overlap_end,
                    fault_type_ready=(group.fault_type_ready or atom.fault_type_ready),
                    material_event_ready=(
                        group.material_event_ready or atom.material_event_ready
                    ),
                )
                merged = True
                break
            if not merged:
                groups.append(
                    OccurrenceGroup(
                        evidence_refs=(atom.evidence_id,),
                        subject_id=subject_id,
                        start=atom.start,
                        end=atom.end,
                        fault_type_ready=atom.fault_type_ready,
                        material_event_ready=atom.material_event_ready,
                    )
                )
        result.extend(group for group in groups if group.eligible)
    return sorted(
        result,
        key=lambda item: (
            not item.fault_type_ready,
            not item.material_event_ready,
            item.end,
            item.start,
            item.subject_id,
            item.evidence_refs,
        ),
    )


def matching_occurrence_group(
    groups: Iterable[OccurrenceGroup], evidence_refs: Iterable[str]
) -> OccurrenceGroup | None:
    """Return the exact model-selected eligible group; never choose among groups."""
    target = set(evidence_refs)
    matches = [group for group in groups if set(group.evidence_refs) == target]
    return matches[0] if len(matches) == 1 else None
