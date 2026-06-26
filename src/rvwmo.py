from __future__ import annotations

"""Native axiomatic RVWMO checker for cycle-shaped scalar litmus tests.

herd7 is the reference oracle for RISC-V's RVWMO model, but it is not always
installed and it cannot reason about the vector/CMO/PBMT/TLB fusion scenarios
this project also generates. This module is a small, faithful checker for the
*pure scalar* RVWMO cases the IR builds as canonical dependency cycles
(MP/LB/SB/WRC/RWC/IRIW and their ordering variants).

Principle
---------
Each scalar case IR encodes exactly one critical cycle that alternates local
program-order (``po``) edges with external communication edges
(``rfe``/``fre``/``co``). Under RVWMO's Global Memory Order acyclicity rule the
test's ``exists`` (weak / "bad") outcome is *forbidden* iff every ``po`` edge of
that cycle is a *preserved* program-order (``ppo``) edge -- because then
``ppo u comm`` is cyclic, which RVWMO disallows. If any ``po`` edge is not
preserved the cycle is broken and the outcome is *allowed*.

The whole problem therefore reduces to a PPO oracle over consecutive same-hart
memory operations. We implement the subset of RVWMO PPO rules the generated
variants exercise, each tagged with its rule number from the RISC-V
unprivileged spec ("RVWMO Memory Consistency Model", the PPO list):

* rules 1-3 -- overlapping-address ordering
* rule 4    -- FENCE with matching predecessor/successor sets
* rule 9    -- address dependency from a load
* rule 10   -- data dependency from a load to a store
* rule 11   -- control dependency from a load to a store (store only)

``fence.i`` is deliberately NOT treated as a data fence: RVWMO gives it no
data-memory ordering power, so "control dep + fence.i" orders exactly what a
bare control dependency orders.

This checker is specialized for the cycle-shaped IR; it is not a general herd7
replacement and only renders a verdict for ``model == "rvwmo"`` cases.
"""

from dataclasses import dataclass
from typing import Any

from litmus_ir import LitmusCaseIR, LitmusEvent

MEMORY_KINDS = {"load", "store", "amo"}
NATIVE_MODEL = "rvwmo-native"

# __APPEND_MARKER__


@dataclass(frozen=True)
class EdgeJudgement:
    """One edge of the critical cycle and whether RVWMO keeps it."""

    src: str
    dst: str
    kind: str          # "po" for local edges, otherwise rfe/fre/co/...
    preserved: bool    # is this edge part of the global ordering?
    rule: str          # spec rule that preserved it, or why it was not
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "preserved": self.preserved,
            "rule": self.rule,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RvwmoResult:
    status: str                  # "verified" | "not_applicable"
    verdict: str                 # "forbidden" | "allowed" | "unmodeled"
    allowed: bool | None
    reason: str
    edges: list[EdgeJudgement]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "litmus-link.rvwmo-native.v1",
            "status": self.status,
            "verdict": self.verdict,
            "allowed": self.allowed,
            "model": NATIVE_MODEL,
            "reason": self.reason,
            "edges": [edge.to_json() for edge in self.edges],
        }


def check_rvwmo(case_ir: LitmusCaseIR) -> RvwmoResult:
    """Decide allowed/forbidden for a pure scalar RVWMO cycle case."""
    if case_ir.model != "rvwmo":
        return RvwmoResult(
            status="not_applicable",
            verdict="unmodeled",
            allowed=None,
            reason="Native RVWMO checker only models pure scalar main-memory cases.",
            edges=[],
        )

    events = case_ir.event_map()
    judgements: list[EdgeJudgement] = []
    for relation in case_ir.relations:
        if relation.local:
            preserved, rule, detail = _ppo_preserved(case_ir, events, relation.src, relation.dst)
            judgements.append(EdgeJudgement(relation.src, relation.dst, "po", preserved, rule, detail))
        else:
            judgements.append(
                EdgeJudgement(
                    relation.src,
                    relation.dst,
                    relation.kind,
                    True,
                    f"comm:{relation.kind}",
                    "External communication edge is always part of global order.",
                )
            )

    po_edges = [j for j in judgements if j.kind == "po"]
    broken = [j for j in po_edges if not j.preserved]
    if broken:
        names = ", ".join(f"{j.src}->{j.dst}" for j in broken)
        return RvwmoResult(
            status="verified",
            verdict="allowed",
            allowed=True,
            reason=f"Cycle is broken: program-order edge(s) {names} are not preserved by RVWMO PPO.",
            edges=judgements,
        )
    return RvwmoResult(
        status="verified",
        verdict="forbidden",
        allowed=False,
        reason="Every program-order edge in the cycle is a preserved (ppo) edge, so ppo u comm is cyclic; RVWMO forbids it.",
        edges=judgements,
    )


def _ppo_preserved(
    case_ir: LitmusCaseIR,
    events: dict[str, LitmusEvent],
    src_id: str,
    dst_id: str,
) -> tuple[bool, str, str]:
    """Is the same-hart program-order edge src->dst preserved as ppo?"""
    a = events.get(src_id)
    b = events.get(dst_id)
    if a is None or b is None:
        return False, "unknown-event", "Edge endpoint missing from event map."
    if a.kind not in MEMORY_KINDS or b.kind not in MEMORY_KINDS:
        return False, "non-memory", "Edge does not connect two memory operations."

    between = _between(case_ir, src_id, dst_id)

    # Rules 1-3: overlapping addresses are ordered (loads see program order to
    # same-address accesses). Our scalar skeletons use distinct locations, so
    # this only fires when the two ops actually touch the same location.
    if a.location and a.location == b.location:
        return True, "rvwmo-1..3", f"Same address {a.location}; overlapping-address ordering applies."

    # Rule 4: a data FENCE between them whose predecessor/successor sets cover
    # the two access types.
    for event in between:
        if _is_data_fence(event) and _fence_orders(event, a, b):
            return True, "rvwmo-4", f"FENCE '{event.instruction}' orders {_ty_label(a)}->{_ty_label(b)}."

    # Rule 9: address dependency from the first (a load) to b.
    if a.kind == "load" and _has_role(between, "addr-dep"):
        return True, "rvwmo-9", "Address dependency from the load preserves order to the dependent access."

    # Rule 10: data dependency from a load to a store.
    if a.kind == "load" and b.kind == "store" and _has_role(between, "data-dep"):
        return True, "rvwmo-10", "Data dependency from the load to the store preserves order."

    # Rule 11: control dependency from a load orders only subsequent *stores*.
    if a.kind == "load" and b.kind == "store" and _has_role(between, "ctrl-dep"):
        return True, "rvwmo-11", "Control dependency from the load preserves order to the later store."

    return False, "no-ppo-rule", _why_unordered(a, b, between)


def _why_unordered(a: LitmusEvent, b: LitmusEvent, between: list[LitmusEvent]) -> str:
    if a.kind == "load" and b.kind != "store" and _has_role(between, "ctrl-dep"):
        return "Control dependency orders later stores only (rule 11); this successor is not a store."
    return f"No fence or dependency preserves the {_ty_label(a)}->{_ty_label(b)} program-order edge."


def _between(case_ir: LitmusCaseIR, src_id: str, dst_id: str) -> list[LitmusEvent]:
    """Events strictly between src and dst within their shared hart."""
    for hart in case_ir.harts:
        ids = [event.event_id for event in hart]
        if src_id in ids and dst_id in ids:
            i, j = ids.index(src_id), ids.index(dst_id)
            lo, hi = (i, j) if i < j else (j, i)
            return hart[lo + 1 : hi]
    return []


def _is_data_fence(event: LitmusEvent) -> bool:
    # "fence rw,rw" / "fence w,w" / "fence r,rw" -- but NOT "fence.i".
    return event.kind == "fence" and event.instruction.startswith("fence ") and "," in event.instruction


def _fence_orders(fence: LitmusEvent, a: LitmusEvent, b: LitmusEvent) -> bool:
    pred, succ = _fence_sets(fence.instruction)
    return bool(_ty(a) & pred) and bool(_ty(b) & succ)


def _fence_sets(instruction: str) -> tuple[set[str], set[str]]:
    body = instruction.split(None, 1)[1] if " " in instruction else ""
    pred, _, succ = body.partition(",")
    return _mode_set(pred), _mode_set(succ)


def _mode_set(token: str) -> set[str]:
    token = token.strip().lower()
    modes: set[str] = set()
    if "r" in token:
        modes.add("r")
    if "w" in token:
        modes.add("w")
    return modes


def _ty(event: LitmusEvent) -> set[str]:
    if event.kind == "load":
        return {"r"}
    if event.kind == "store":
        return {"w"}
    if event.kind == "amo":
        return {"r", "w"}
    return set()


def _ty_label(event: LitmusEvent) -> str:
    return {"load": "R", "store": "W", "amo": "RW"}.get(event.kind, event.kind)


def _has_role(events: list[LitmusEvent], role: str) -> bool:
    return any(event.role == role for event in events)

