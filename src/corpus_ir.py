from __future__ import annotations

"""Bridge real corpus litmus tests into Litmus-link's display IR.

A CorpusTest (parsed from a real diy7-generated .litmus file) is turned into a
LitmusCaseIR so the existing diagram renderer can draw it, and the herd7 verdict
is attached as the per-outcome (observable/forbidden) result.

No memory-model reasoning happens here. The relations drawn are program-order
within each hart (always sound) plus the authoritative cycle string from the
tool; herd7 owns the actual verdict. Cross-hart comm edges are intentionally not
fabricated -- a wrong edge is worse than none.
"""

import re

from corpus_riscv import CorpusTest
from litmus_ir import LitmusCaseIR, LitmusEvent, LitmusRelation
from toolchain import HerdVerdict

_STORE = {"sw", "sd", "sh", "sb", "sc.w", "sc.d"}
_LOAD = {"lw", "ld", "lh", "lb", "lbu", "lhu", "lwu", "lr.w", "lr.d"}
_BRANCH = ("beq", "bne", "blt", "bge", "bltu", "bgeu", "bnez", "beqz")


def _mnemonic(instr: str) -> str:
    parts = instr.split()
    return parts[0] if parts else ""


def _kind(instr: str) -> str:
    m = _mnemonic(instr)
    if instr.strip().endswith(":") or m.endswith(":"):
        return "label"
    if m.startswith("amo"):
        return "amo"
    if m.startswith("fence"):
        return "fence"
    if m in _STORE:
        return "store"
    if m in _LOAD:
        return "load"
    if m.startswith(_BRANCH):
        return "branch"
    return "dep"  # ori/xor/add/... value setup or dependency arithmetic


def _addr_map(init_lines) -> dict:
    """Map (hart, reg) -> symbolic address from init entries like '0:x6=x'."""
    out: dict = {}
    for entry in init_lines:
        mm = re.match(r"(\d+):(x\d+)\s*=\s*(.+)", str(entry).strip())
        if not mm:
            continue
        hart, reg, val = int(mm.group(1)), mm.group(2), mm.group(3).strip()
        if re.fullmatch(r"[a-zA-Z]\w*", val) and not re.fullmatch(r"x\d+", val):
            out[(hart, reg)] = val
    return out


def _location(instr: str, hart: int, amap: dict) -> str:
    mm = re.search(r"\((x\d+)\)", instr)
    if not mm:
        return ""
    return amap.get((hart, mm.group(1)), "")


def _dest_reg(instr: str) -> str:
    # "lw x5,0(x6)" -> x5 ; "amoor.w.aq x5,x0,(x6)" -> x5
    mm = re.match(r"\S+\s+(x\d+)", instr)
    return mm.group(1) if mm else ""


def corpus_to_ir(test: CorpusTest, verdict: HerdVerdict | None = None) -> LitmusCaseIR:
    """Build a display LitmusCaseIR from a parsed real corpus test."""
    amap = _addr_map(test.init_lines)
    harts: list[list[LitmusEvent]] = []
    for hart_idx, instrs in enumerate(test.harts):
        events: list[LitmusEvent] = []
        for pos, instr in enumerate(instrs):
            kind = _kind(instr)
            events.append(
                LitmusEvent(
                    event_id=f"p{hart_idx}_{pos}",
                    hart=hart_idx,
                    kind=kind,
                    instruction=instr,
                    location=_location(instr, hart_idx, amap),
                    register=_dest_reg(instr),
                    value="",
                    role="",
                )
            )
        harts.append(events)

    # Program-order relations within each hart between successive memory/fence
    # events (always sound; the cycle string carries the precise edge names).
    relations: list[LitmusRelation] = []
    for events in harts:
        prev = None
        for ev in events:
            if ev.kind in ("load", "store", "amo", "fence"):
                if prev is not None:
                    relations.append(
                        LitmusRelation(prev.event_id, ev.event_id, "po", "po", local=True)
                    )
                prev = ev

    if verdict is not None and verdict.outcome != "unknown":
        outcome = "forbidden" if verdict.outcome == "forbidden" else "observable"
    else:
        outcome = "solver_required"

    return LitmusCaseIR(
        name=test.name,
        display_name=test.name,
        combination_name=test.name,
        skeleton=test.skeleton,
        variant=test.name,
        cycle=test.cycle,
        init_lines=list(test.init_lines),
        harts=harts,
        relations=relations,
        exists=test.exists,
        expected_outcome=outcome,
        model="rvwmo-herd7",
        description=f"Real RVWMO litmus ({test.skeleton} family), cycle: {test.cycle}",
        tags=["rvwmo", "corpus", test.skeleton, test.family],
    )
