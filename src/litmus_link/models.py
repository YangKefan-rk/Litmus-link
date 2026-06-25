from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


GENERATED = "generated"
EXCLUDED_ILLEGAL = "excluded_illegal"
EXCLUDED_UNSUPPORTED = "excluded_unsupported"
HAND_REQUIRED = "hand_required"
MISSING = "missing"

DECISION_STATUSES = {
    GENERATED,
    EXCLUDED_ILLEGAL,
    EXCLUDED_UNSUPPORTED,
    HAND_REQUIRED,
    MISSING,
}


def sanitize_token(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"_", "+", ".", "-"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "x"


@dataclass(frozen=True)
class Combination:
    profile: str
    category: str
    skeleton: str
    memory_event: str
    attribute: str
    tlb: str = "no_tlb"
    cmo: str = "no_cmo"
    vector: str = "none"
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        tokens = [
            "LL",
            self.category,
            self.skeleton,
            self.memory_event,
            self.attribute,
            self.tlb,
            self.cmo,
            self.vector,
        ]
        return sanitize_token("_".join(tokens))

    def axes(self) -> Dict[str, str]:
        return {
            "category": self.category,
            "skeleton": self.skeleton,
            "memory_event": self.memory_event,
            "attribute": self.attribute,
            "tlb": self.tlb,
            "cmo": self.cmo,
            "vector": self.vector,
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "category": self.category,
            "skeleton": self.skeleton,
            "memory_event": self.memory_event,
            "attribute": self.attribute,
            "tlb": self.tlb,
            "cmo": self.cmo,
            "vector": self.vector,
            "params": dict(self.params),
            "name": self.name,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Combination":
        return cls(
            profile=str(data["profile"]),
            category=str(data["category"]),
            skeleton=str(data["skeleton"]),
            memory_event=str(data["memory_event"]),
            attribute=str(data["attribute"]),
            tlb=str(data.get("tlb", "no_tlb")),
            cmo=str(data.get("cmo", "no_cmo")),
            vector=str(data.get("vector", "none")),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    rvwmo_class: str
    expected_kind: str
    requires: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    hand_category: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        if self.status not in DECISION_STATUSES:
            raise ValueError(f"unknown decision status: {self.status}")
        return {
            "status": self.status,
            "reason": self.reason,
            "rvwmo_class": self.rvwmo_class,
            "expected_kind": self.expected_kind,
            "requires": list(self.requires),
            "notes": list(self.notes),
            "hand_category": self.hand_category,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GeneratedCase:
    combination: Combination
    decision: Decision
    litmus: str

    def meta(self) -> Dict[str, Any]:
        return {
            "schema": "litmus-link.meta.v1",
            "name": self.combination.name,
            "combination": self.combination.to_json(),
            "axes": self.combination.axes(),
            "decision": self.decision.to_json(),
            "requires": list(self.decision.requires),
            "rvwmo_class": self.decision.rvwmo_class,
            "legality_status": self.decision.status,
            "expected_kind": self.decision.expected_kind,
            "generated_from": {
                "profile": self.combination.profile,
                "category": self.combination.category,
            },
        }


def count_by_status(decisions: Iterable[Decision]) -> Dict[str, int]:
    counts = {status: 0 for status in sorted(DECISION_STATUSES)}
    for decision in decisions:
        counts[decision.status] = counts.get(decision.status, 0) + 1
    return counts
