from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from models import Combination, Decision


DEFAULT_SCALAR_VARIANTS = [
    "base",
    "fence_rw_rw",
    "fence_w_w_r_rw",
    "addr_dep",
    "ctrl_dep",
    "ctrl_fencei",
]


@dataclass(frozen=True)
class LitmusEvent:
    event_id: str
    hart: int
    kind: str
    instruction: str
    location: str = ""
    register: str = ""
    value: str = ""
    role: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "hart": self.hart,
            "kind": self.kind,
            "instruction": self.instruction,
            "location": self.location,
            "register": self.register,
            "value": self.value,
            "role": self.role,
        }


@dataclass(frozen=True)
class LitmusRelation:
    src: str
    dst: str
    kind: str
    label: str = ""
    local: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "label": self.label or self.kind,
            "local": self.local,
        }


@dataclass(frozen=True)
class LitmusCaseIR:
    name: str
    display_name: str
    combination_name: str
    skeleton: str
    variant: str
    cycle: str
    init_lines: list[str]
    harts: list[list[LitmusEvent]]
    relations: list[LitmusRelation]
    exists: str
    expected_outcome: str
    model: str
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def events(self) -> list[LitmusEvent]:
        return [event for hart in self.harts for event in hart]

    def event_map(self) -> dict[str, LitmusEvent]:
        return {event.event_id: event for event in self.events()}

    def hart_names(self) -> list[str]:
        return [f"P{index}" for index in range(len(self.harts))]

    def memory_locations(self) -> list[str]:
        seen: list[str] = []
        for event in self.events():
            if event.location and event.location not in seen:
                seen.append(event.location)
        return seen

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "litmus-link.case-ir.v1",
            "name": self.name,
            "display_name": self.display_name,
            "combination_name": self.combination_name,
            "skeleton": self.skeleton,
            "variant": self.variant,
            "cycle": self.cycle,
            "init_lines": list(self.init_lines),
            "harts": [[event.to_json() for event in hart] for hart in self.harts],
            "relations": [relation.to_json() for relation in self.relations],
            "exists": self.exists,
            "expected_outcome": self.expected_outcome,
            "model": self.model,
            "description": self.description,
            "tags": list(self.tags),
        }


def case_count(combination: Combination, decision: Decision) -> int:
    if not _is_scalar_rvwmo(combination, decision):
        return 1
    return len(_scalar_variant_ids(combination))


def build_litmus_ir_cases(combination: Combination, decision: Decision) -> list[LitmusCaseIR]:
    if _is_scalar_rvwmo(combination, decision):
        variants = _scalar_variant_ids(combination)
        expanded = len(variants) > 1
        return [_scalar_case(combination, variant, expanded) for variant in variants]
    return [_observation_case(combination, decision)]


def _is_scalar_rvwmo(combination: Combination, decision: Decision) -> bool:
    return (
        decision.status == "generated"
        and decision.rvwmo_class in {"rvwmo-herd", "rvwmo-nc"}
        and combination.memory_event == "scalar_pair"
        and combination.attribute in {"cacheable", "pbmt_nc"}
        and combination.vector == "none"
        and combination.cmo == "no_cmo"
        and combination.tlb == "no_tlb"
    )


def _scalar_variant_ids(combination: Combination) -> list[str]:
    if not combination.params:
        return list(DEFAULT_SCALAR_VARIANTS)
    if "variant" in combination.params:
        return [str(combination.params["variant"])]
    tokens = []
    for key in ["dep", "width", "outcome", "stress"]:
        if key in combination.params:
            tokens.append(f"{key}-{combination.params[key]}")
    return ["_".join(tokens) or "base"]


def _scalar_case(combination: Combination, variant: str, expanded: bool) -> LitmusCaseIR:
    builder = {
        "MP": _mp_case,
        "LB": _lb_case,
        "SB": _sb_case,
        "WRC": _wrc_case,
        "RWC": _rwc_case,
        "IRIW": _iriw_case,
    }.get(combination.skeleton, _generic_scalar_case)
    name = _case_name(combination, variant, expanded)
    return builder(combination, variant, name)


def _case_name(combination: Combination, variant: str, expanded: bool) -> str:
    if not expanded:
        return combination.name
    return f"{combination.name}_var-{_sanitize_variant(variant)}"


def _sanitize_variant(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "base"


def _ordering_events(hart: int, variant: str, prefix: str, role: str) -> list[LitmusEvent]:
    if variant == "fence_rw_rw":
        return [_event(f"{prefix}_fence_rw_rw", hart, "fence", "fence rw,rw", role=role)]
    if variant == "fence_w_w_r_rw":
        instruction = "fence w,w" if role == "writer" else "fence r,rw"
        return [_event(f"{prefix}_fence", hart, "fence", instruction, role=role)]
    if role == "reader" and variant == "addr_dep":
        return [
            _event(f"{prefix}_xor", hart, "dep", "xor x9,x5,x5", register="x9", role="addr-dep"),
            _event(f"{prefix}_add", hart, "dep", "add x8,x8,x9", register="x8", role="addr-dep"),
        ]
    if role == "reader" and variant == "ctrl_dep":
        return [
            _event(f"{prefix}_branch", hart, "dep", "beq x5,x5,1f", role="ctrl-dep"),
            _event(f"{prefix}_label", hart, "label", "1:", role="ctrl-dep"),
        ]
    if role == "reader" and variant == "ctrl_fencei":
        return [
            _event(f"{prefix}_branch", hart, "dep", "beq x5,x5,1f", role="ctrl-dep"),
            _event(f"{prefix}_label", hart, "label", "1:", role="ctrl-dep"),
            _event(f"{prefix}_fencei", hart, "fence", "fence.i", role="ctrl-fencei"),
        ]
    return []


def _event(
    event_id: str,
    hart: int,
    kind: str,
    instruction: str,
    location: str = "",
    register: str = "",
    value: str = "",
    role: str = "",
) -> LitmusEvent:
    return LitmusEvent(event_id, hart, kind, instruction, location, register, value, role)


def _relation(src: str, dst: str, kind: str, label: str | None = None, local: bool = False) -> LitmusRelation:
    return LitmusRelation(src, dst, kind, label or kind, local)


def _case(
    combination: Combination,
    name: str,
    variant: str,
    cycle: str,
    init_lines: Iterable[str],
    harts: list[list[LitmusEvent]],
    relations: list[LitmusRelation],
    exists: str,
    description: str,
) -> LitmusCaseIR:
    expected = _expected_outcome(combination)
    return LitmusCaseIR(
        name=name,
        display_name=f"{combination.skeleton}.{variant}",
        combination_name=combination.name,
        skeleton=combination.skeleton,
        variant=variant,
        cycle=cycle,
        init_lines=list(init_lines),
        harts=harts,
        relations=relations,
        exists=exists,
        expected_outcome=expected,
        model="rvwmo",
        description=description,
        tags=["scalar", "rvwmo", combination.skeleton, variant],
    )


def _expected_outcome(combination: Combination) -> str:
    requested = str(combination.params.get("outcome", "solver_required"))
    if requested in {"allowed", "forbidden", "mixed_size"}:
        return requested
    return "solver_required"


def _mp_case(combination: Combination, variant: str, name: str) -> LitmusCaseIR:
    p0 = [
        _event("p0_wx", 0, "store", "sw x5,0(x6)", "x", value="1", role="data-write"),
        *_ordering_events(0, variant, "p0", "writer"),
        _event("p0_wy", 0, "store", "sw x5,0(x7)", "y", value="1", role="flag-write"),
    ]
    p1 = [
        _event("p1_ry", 1, "load", "lw x5,0(x6)", "y", register="x5", value="1", role="flag-read"),
        *_ordering_events(1, variant, "p1", "reader"),
        _event("p1_rx", 1, "load", "lw x7,0(x8)", "x", register="x7", value="0", role="data-read"),
    ]
    return _case(
        combination,
        name,
        variant,
        _cycle_label("PodWW -> Rfe -> PodRR -> Fre", variant),
        ["0:x5=1; 0:x6=x; 0:x7=y;", "1:x6=y; 1:x8=x;"],
        [p0, p1],
        [
            _relation("p0_wx", "p0_wy", "po", _variant_po_label("PodWW", variant), True),
            _relation("p0_wy", "p1_ry", "rfe", "Rfe"),
            _relation("p1_ry", "p1_rx", "po", _variant_po_label("PodRR", variant), True),
            _relation("p1_rx", "p0_wx", "fre", "Fre"),
        ],
        "(1:x5=1 /\\ 1:x7=0)",
        "Message passing: reader observes the flag write but may still observe old data unless ordering forbids it.",
    )


def _lb_case(combination: Combination, variant: str, name: str) -> LitmusCaseIR:
    p0 = [
        _event("p0_rx", 0, "load", "lw x5,0(x6)", "x", register="x5", value="0"),
        *_ordering_events(0, variant, "p0", "reader"),
        _event("p0_wy", 0, "store", "sw x9,0(x7)", "y", value="1"),
    ]
    p1 = [
        _event("p1_ry", 1, "load", "lw x5,0(x6)", "y", register="x5", value="0"),
        *_ordering_events(1, variant, "p1", "reader"),
        _event("p1_wx", 1, "store", "sw x9,0(x7)", "x", value="1"),
    ]
    return _case(
        combination,
        name,
        variant,
        _cycle_label("PodRW -> Fre -> PodRW -> Fre", variant),
        ["0:x6=x; 0:x7=y; 0:x9=1;", "1:x6=y; 1:x7=x; 1:x9=1;"],
        [p0, p1],
        [
            _relation("p0_rx", "p0_wy", "po", _variant_po_label("PodRW", variant), True),
            _relation("p0_wy", "p1_ry", "fre", "Fre"),
            _relation("p1_ry", "p1_wx", "po", _variant_po_label("PodRW", variant), True),
            _relation("p1_wx", "p0_rx", "fre", "Fre"),
        ],
        "(0:x5=0 /\\ 1:x5=0)",
        "Load buffering: both harts read before publishing their writes.",
    )


def _sb_case(combination: Combination, variant: str, name: str) -> LitmusCaseIR:
    p0 = [
        _event("p0_wx", 0, "store", "sw x5,0(x6)", "x", value="1"),
        *_ordering_events(0, variant, "p0", "writer"),
        _event("p0_ry", 0, "load", "lw x7,0(x8)", "y", register="x7", value="0"),
    ]
    p1 = [
        _event("p1_wy", 1, "store", "sw x5,0(x6)", "y", value="1"),
        *_ordering_events(1, variant, "p1", "writer"),
        _event("p1_rx", 1, "load", "lw x7,0(x8)", "x", register="x7", value="0"),
    ]
    return _case(
        combination,
        name,
        variant,
        _cycle_label("PodWR -> Fre -> PodWR -> Fre", variant),
        ["0:x5=1; 0:x6=x; 0:x8=y;", "1:x5=1; 1:x6=y; 1:x8=x;"],
        [p0, p1],
        [
            _relation("p0_wx", "p0_ry", "po", _variant_po_label("PodWR", variant), True),
            _relation("p0_ry", "p1_wy", "fre", "Fre"),
            _relation("p1_wy", "p1_rx", "po", _variant_po_label("PodWR", variant), True),
            _relation("p1_rx", "p0_wx", "fre", "Fre"),
        ],
        "(0:x7=0 /\\ 1:x7=0)",
        "Store buffering: both harts publish stores then read the other location.",
    )


def _wrc_case(combination: Combination, variant: str, name: str) -> LitmusCaseIR:
    p0 = [_event("p0_wx", 0, "store", "sw x5,0(x6)", "x", value="1")]
    p1 = [
        _event("p1_rx", 1, "load", "lw x5,0(x6)", "x", register="x5", value="1"),
        *_ordering_events(1, variant, "p1", "reader"),
        _event("p1_wy", 1, "store", "sw x5,0(x7)", "y", value="1"),
    ]
    p2 = [
        _event("p2_ry", 2, "load", "lw x5,0(x6)", "y", register="x5", value="1"),
        *_ordering_events(2, variant, "p2", "reader"),
        _event("p2_rx", 2, "load", "lw x7,0(x8)", "x", register="x7", value="0"),
    ]
    return _case(
        combination,
        name,
        variant,
        _cycle_label("Rfe -> PodRW -> Rfe -> PodRR -> Fre", variant),
        ["0:x5=1; 0:x6=x;", "1:x6=x; 1:x7=y;", "2:x6=y; 2:x8=x;"],
        [p0, p1, p2],
        [
            _relation("p0_wx", "p1_rx", "rfe", "Rfe"),
            _relation("p1_rx", "p1_wy", "po", _variant_po_label("PodRW", variant), True),
            _relation("p1_wy", "p2_ry", "rfe", "Rfe"),
            _relation("p2_ry", "p2_rx", "po", _variant_po_label("PodRR", variant), True),
            _relation("p2_rx", "p0_wx", "fre", "Fre"),
        ],
        "(1:x5=1 /\\ 2:x5=1 /\\ 2:x7=0)",
        "Write-read causality: an observed write is propagated through a second hart before a third hart reads old data.",
    )


def _rwc_case(combination: Combination, variant: str, name: str) -> LitmusCaseIR:
    p0 = [_event("p0_wx", 0, "store", "sw x5,0(x6)", "x", value="1")]
    p1 = [
        _event("p1_rx", 1, "load", "lw x5,0(x6)", "x", register="x5", value="1"),
        *_ordering_events(1, variant, "p1", "reader"),
        _event("p1_ry", 1, "load", "lw x7,0(x8)", "y", register="x7", value="0"),
    ]
    p2 = [_event("p2_wy", 2, "store", "sw x5,0(x6)", "y", value="1")]
    return _case(
        combination,
        name,
        variant,
        _cycle_label("Rfe -> PodRR -> Fre -> Wse", variant),
        ["0:x5=1; 0:x6=x;", "1:x6=x; 1:x8=y;", "2:x5=1; 2:x6=y;"],
        [p0, p1, p2],
        [
            _relation("p0_wx", "p1_rx", "rfe", "Rfe"),
            _relation("p1_rx", "p1_ry", "po", _variant_po_label("PodRR", variant), True),
            _relation("p1_ry", "p2_wy", "fre", "Fre"),
            _relation("p2_wy", "p0_wx", "co", "Wse"),
        ],
        "(1:x5=1 /\\ 1:x7=0)",
        "Read-write causality: a read of one write is paired with an old read of another location.",
    )


def _iriw_case(combination: Combination, variant: str, name: str) -> LitmusCaseIR:
    p0 = [_event("p0_wx", 0, "store", "sw x5,0(x6)", "x", value="1")]
    p1 = [_event("p1_wy", 1, "store", "sw x5,0(x6)", "y", value="1")]
    p2 = [
        _event("p2_rx", 2, "load", "lw x5,0(x6)", "x", register="x5", value="1"),
        *_ordering_events(2, variant, "p2", "reader"),
        _event("p2_ry", 2, "load", "lw x7,0(x8)", "y", register="x7", value="0"),
    ]
    p3 = [
        _event("p3_ry", 3, "load", "lw x5,0(x6)", "y", register="x5", value="1"),
        *_ordering_events(3, variant, "p3", "reader"),
        _event("p3_rx", 3, "load", "lw x7,0(x8)", "x", register="x7", value="0"),
    ]
    return _case(
        combination,
        name,
        variant,
        _cycle_label("Rfe -> PodRR -> Fre -> Rfe -> PodRR -> Fre", variant),
        ["0:x5=1; 0:x6=x;", "1:x5=1; 1:x6=y;", "2:x6=x; 2:x8=y;", "3:x6=y; 3:x8=x;"],
        [p0, p1, p2, p3],
        [
            _relation("p0_wx", "p2_rx", "rfe", "Rfe"),
            _relation("p2_rx", "p2_ry", "po", _variant_po_label("PodRR", variant), True),
            _relation("p2_ry", "p1_wy", "fre", "Fre"),
            _relation("p1_wy", "p3_ry", "rfe", "Rfe"),
            _relation("p3_ry", "p3_rx", "po", _variant_po_label("PodRR", variant), True),
            _relation("p3_rx", "p0_wx", "fre", "Fre"),
        ],
        "(2:x5=1 /\\ 2:x7=0 /\\ 3:x5=1 /\\ 3:x7=0)",
        "Independent reads of independent writes: two readers observe independent writers in opposite orders.",
    )


def _generic_scalar_case(combination: Combination, variant: str, name: str) -> LitmusCaseIR:
    generic = Combination(
        combination.profile,
        combination.category,
        "MP",
        combination.memory_event,
        combination.attribute,
        combination.tlb,
        combination.cmo,
        combination.vector,
        combination.params,
    )
    base = _mp_case(generic, variant, name)
    return LitmusCaseIR(
        name=base.name,
        display_name=f"{combination.skeleton}.{variant}",
        combination_name=combination.name,
        skeleton=combination.skeleton,
        variant=base.variant,
        cycle=f"{combination.skeleton}: {base.cycle}",
        init_lines=base.init_lines,
        harts=base.harts,
        relations=base.relations,
        exists=base.exists,
        expected_outcome=base.expected_outcome,
        model=base.model,
        description=f"{combination.skeleton} scalar RVWMO variant represented with the MP two-hart fallback topology.",
        tags=["scalar", "rvwmo", combination.skeleton, variant],
    )


def _observation_case(combination: Combination, decision: Decision) -> LitmusCaseIR:
    events = _observation_events(combination)
    relations = [_relation(events[0][0].event_id, events[-1][-1].event_id, "obs", "observation")]
    return LitmusCaseIR(
        name=combination.name,
        display_name=combination.name,
        combination_name=combination.name,
        skeleton=combination.skeleton,
        variant="observation",
        cycle=_observation_cycle(combination),
        init_lines=["0:x5=1; 0:x6=x; 0:x7=y;", "1:x6=y; 1:x8=x;"],
        harts=events,
        relations=relations,
        exists="(1:x5=1)",
        expected_outcome=decision.expected_kind,
        model=decision.rvwmo_class,
        description="Specification-constrained or hardware-observation case; no formal RVWMO forbidden claim is made.",
        tags=[combination.category, combination.memory_event, combination.attribute, combination.vector, combination.cmo, combination.tlb],
    )


def _observation_events(combination: Combination) -> list[list[LitmusEvent]]:
    if combination.vector != "none" and combination.cmo != "no_cmo":
        p0 = [
            _event("p0_vset", 0, "setup", "vsetvli x10,x11,e32,m1,ta,ma"),
            _event("p0_vec", 0, "vector", _vector_instruction(combination), "x"),
            *_cmo_events(combination, 0, "p0"),
        ]
        p1 = [_event("p1_r", 1, "load", "lw x5,0(x6)", "y", register="x5", value="1")]
        return [p0, p1]
    if combination.vector != "none":
        hart = 0 if combination.memory_event == "vector_store" else 1
        vector_event = _event(f"p{hart}_vec", hart, "vector", _vector_instruction(combination), "x")
        if hart == 0:
            return [[_event("p0_vset", 0, "setup", "vsetvli x10,x11,e32,m1,ta,ma"), vector_event], [_event("p1_r", 1, "load", "lw x5,0(x6)", "y")]]
        return [[_event("p0_w", 0, "store", "sw x5,0(x6)", "x")], [_event("p1_vset", 1, "setup", "vsetvli x10,x11,e32,m1,ta,ma"), vector_event]]
    if combination.cmo != "no_cmo":
        return [[_event("p0_w", 0, "store", "sw x5,0(x6)", "x"), *_cmo_events(combination, 0, "p0")], [_event("p1_r", 1, "load", "lw x5,0(x6)", "y")]]
    return [[_event("p0_w", 0, "store", "sw x5,0(x6)", "x")], [_event("p1_r", 1, "load", "lw x5,0(x6)", "y")]]


def _cmo_events(combination: Combination, hart: int, prefix: str) -> list[LitmusEvent]:
    sync = str(combination.params.get("sync", "none"))
    op = {
        "clean": "cbo.clean 0(x6)",
        "flush": "cbo.flush 0(x6)",
        "inval": "cbo.inval 0(x6)",
        "zero": "cbo.zero 0(x6)",
    }.get(combination.cmo, "fence rw,rw")
    if sync == "full_alias_sync":
        instructions = ["fence iorw,iorw", "cbo.flush 0(x6)", "fence iorw,iorw"]
    elif sync == "pre_fence":
        instructions = ["fence iorw,iorw", op]
    elif sync == "post_fence":
        instructions = [op, "fence iorw,iorw"]
    elif sync == "fence_i_after":
        instructions = [op, "fence.i"]
    else:
        instructions = [op]
    return [_event(f"{prefix}_cmo{index}", hart, "cmo" if instruction.startswith("cbo") else "fence", instruction, "x") for index, instruction in enumerate(instructions)]


def _vector_instruction(combination: Combination) -> str:
    table = {
        "unit_load": "vle32.v v8,(x6)",
        "unit_store": "vse32.v v8,(x6)",
        "strided_load": "vlse32.v v8,(x6),x9",
        "strided_store": "vsse32.v v8,(x6),x9",
        "indexed_ordered_load": "vloxei32.v v8,(x6),v4",
        "indexed_unordered_load": "vluxei32.v v8,(x6),v4",
        "indexed_ordered_store": "vsoxei32.v v8,(x6),v4",
        "indexed_unordered_store": "vsuxei32.v v8,(x6),v4",
        "segment_load": "vlseg2e32.v v8,(x6)",
        "segment_store": "vsseg2e32.v v8,(x6)",
        "fof_load": "vle32ff.v v8,(x6)",
        "fof_segment_load": "vlseg2e32ff.v v8,(x6)",
    }
    return table.get(combination.vector, "vle32.v v8,(x6)")


def _observation_cycle(combination: Combination) -> str:
    features = [combination.skeleton, combination.memory_event, combination.attribute]
    for value in [combination.vector, combination.cmo, combination.tlb]:
        if value not in {"none", "no_cmo", "no_tlb"}:
            features.append(value)
    return " + ".join(features)


def _cycle_label(base: str, variant: str) -> str:
    if variant == "base":
        return base
    return f"{base} [{variant}]"


def _variant_po_label(base: str, variant: str) -> str:
    mapping = {
        "fence_rw_rw": f"{base}+fence.rw.rw",
        "fence_w_w_r_rw": f"{base}+fence",
        "addr_dep": f"{base}+addr",
        "ctrl_dep": f"{base}+ctrl",
        "ctrl_fencei": f"{base}+ctrlfencei",
    }
    if variant.startswith("dep-"):
        return f"{base}+{variant[4:]}"
    return mapping.get(variant, base)
