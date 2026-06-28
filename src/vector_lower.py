from __future__ import annotations

"""Lower a vector (RVV) litmus case to an equivalent scalar RVWMO twin herd7 judges.

The RISC-V spec (§30.9) defines vector memory ordering at the ELEMENT level: a
vector memory instruction is a set of per-element load/store operations that obey
RVWMO at the instruction level, and -- except for indexed-ordered forms -- the
element operations are UNORDERED within the instruction. Nanhu's VSplit.scala
lowers each vector memory instruction to exactly such per-element flows feeding
the scalar load/store pipeline. So a vector access is faithfully represented as N
scalar element accesses carrying the same po/dep/fence edges -- which stock
herd7's scalar RISC-V front-end CAN parse and judge. This module performs that
lowering so vector cases get a REAL herd verdict instead of extension prose.

Correctness rules (each validated against herd7):
  * element COUNT is bounded (default 2, configurable): vle/vse with vl>>1 would
    blow up herd's state space combinatorially, so we lower a small witness set.
  * element ORDER follows the form. Every non-indexed-ordered form is unordered:
    plain sequential scalar accesses, which RVWMO ppo does NOT order for
    independent addresses (experiment-confirmed). For the cyclic shapes generated
    here only ONE element sits at the shared location, so inter-element order is
    verdict-irrelevant and is recorded in metadata rather than emitted as a fence
    -- a fence between siblings would wrongly order the data access before the
    flag access and contaminate the cross-hart cycle. Faithful ordered-element
    encoding (a dependency chain, not a fence) is only needed for shapes that
    observe two elements of one instruction across harts -- noted for future work.
  * overlapping/duplicate addresses (stride 0, repeated index) are left to herd
    coherence, which orders same-address accesses (po-loc) automatically.
"""

import re
from dataclasses import replace

from litmus_ir import LitmusCaseIR, LitmusEvent

ELEM_BYTES = 4  # e32 elements
_SCRATCH = ["x28", "x29", "x30", "x31"]

# vector form -> are element ops ordered within the instruction?
VECTOR_FORM_ORDERED = {
    "indexed_ordered_load": True,
    "indexed_ordered_store": True,
}


def _base_reg(instruction: str) -> str:
    m = re.search(r"\((x\d+)\)", instruction)
    return m.group(1) if m else "x6"


def _is_vector_mem(event: LitmusEvent) -> bool:
    if event.kind not in ("store", "load"):
        return False
    mnem = event.instruction.split()[0] if event.instruction else ""
    return mnem.startswith("v")


def _broadcast_src(hart: list[LitmusEvent]) -> str:
    for ev in hart:
        m = re.match(r"vmv\.v\.x\s+v\d+,(x\d+)", ev.instruction)
        if m:
            return m.group(1)
    return "x5"


def _extract_dst(hart: list[LitmusEvent]) -> str:
    for ev in hart:
        m = re.match(r"vmv\.x\.s\s+(x\d+),v\d+", ev.instruction)
        if m:
            return m.group(1)
    return "x7"


def _sibling(i: int, loc: str, hart: int) -> tuple[str, str]:
    """Address register + fresh symbol for sibling element i (>=1).

    Siblings use a FRESH distinct symbol, NOT a byte offset into the shared
    symbol: stock herd7 is non-mixed-size and rejects sub-word offsets like x+4
    unless that exact location is written/declared (loads at x+4 with x+4 never
    written are rejected). For RVWMO only the same-vs-different ADDRESS relation
    matters, not physical contiguity, so a distinct symbol is a faithful and
    parseable model of a separate element location. (Overlap/duplicate-address
    forms -- stride 0, repeated index -- would instead reuse the shared symbol so
    herd coherence orders them; relevant only for multi-element-observed shapes,
    which this generator does not yet emit.)"""
    return f"x{19 + i}", f"{loc or 'v'}e{i}h{hart}"


def _expand_store(event: LitmusEvent, src_reg: str, n: int) -> tuple[list[LitmusEvent], list[str]]:
    base = _base_reg(event.instruction)
    out: list[LitmusEvent] = []
    init: list[str] = []
    for i in range(n):
        if i == 0:
            out.append(replace(event, instruction=f"sw {src_reg},0({base})", register=src_reg))
            continue
        areg, sym = _sibling(i, event.location, event.hart)
        out.append(LitmusEvent(
            event_id=f"{event.event_id}_e{i}", hart=event.hart, kind="store",
            instruction=f"sw {src_reg},0({areg})", location=sym,
            register=src_reg, value="", role="vector-elem"))
        init.append(f"{event.hart}:{areg}={sym};")
    return out, init


def _expand_load(event: LitmusEvent, dst_reg: str, n: int) -> tuple[list[LitmusEvent], list[str]]:
    base = _base_reg(event.instruction)
    out: list[LitmusEvent] = []
    init: list[str] = []
    for i in range(n):
        if i == 0:
            out.append(replace(event, instruction=f"lw {dst_reg},0({base})", register=dst_reg))
            continue
        areg, sym = _sibling(i, event.location, event.hart)
        dreg = _SCRATCH[(i - 1) % len(_SCRATCH)]
        out.append(LitmusEvent(
            event_id=f"{event.event_id}_e{i}", hart=event.hart, kind="load",
            instruction=f"lw {dreg},0({areg})", location=sym,
            register=dreg, value="", role="vector-elem"))
        init.append(f"{event.hart}:{areg}={sym};")
    return out, init


def lower_vector_case(case_ir: LitmusCaseIR, n_elems: int = 2) -> LitmusCaseIR:
    """Return a scalar RVWMO twin of a vector case: each vector memory access is
    expanded to n_elems per-element scalar accesses; vector-only setup (vsetvli,
    vmv.v.x, vid.v) and extract (vmv.x.s) are dropped. Element 0 keeps the
    original event id and the shared location so the cross-hart cycle is intact."""
    n = max(1, n_elems)
    ordered = any(t in VECTOR_FORM_ORDERED for t in case_ir.tags)
    new_harts: list[list[LitmusEvent]] = []
    extra_init: list[str] = []
    for hart in case_ir.harts:
        src_reg = _broadcast_src(hart)
        dst_reg = _extract_dst(hart)
        out: list[LitmusEvent] = []
        for ev in hart:
            if ev.kind in ("setup", "extract"):
                continue
            if _is_vector_mem(ev):
                if ev.kind == "store":
                    events, init = _expand_store(ev, src_reg, n)
                else:
                    events, init = _expand_load(ev, dst_reg, n)
                out.extend(events)
                extra_init.extend(init)
            else:
                out.append(ev)
        new_harts.append(out)
    note = "indexed-ordered: element order verdict-irrelevant here (only e0 shared)" if ordered else "unordered elements"
    return replace(
        case_ir,
        name=f"{case_ir.name}__lowered",
        model="rvwmo",
        init_lines=list(case_ir.init_lines) + extra_init,
        harts=new_harts,
        description=f"Scalar RVWMO lowering of vector case ({note}); herd7-judgeable.",
        tags=list(case_ir.tags) + ["lowered-scalar"],
    )


def lower_vector_to_litmus(case_ir: LitmusCaseIR, n_elems: int = 2) -> str:
    """Render the scalar-lowered twin to herd7-parseable litmus text."""
    from renderer import render_ir
    return render_ir(lower_vector_case(case_ir, n_elems))
