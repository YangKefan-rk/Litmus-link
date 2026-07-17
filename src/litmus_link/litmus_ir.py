from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .models import Combination, Decision


DEFAULT_SCALAR_VARIANTS = [
    "base",
    "fence_rw_rw",
    "fence_w_w_r_rw",
    "addr_dep",
    "ctrl_dep",
    "ctrl_fencei",
]

# Vector memory ordering reduces per-element to RVWMO, and a FENCE orders those
# element accesses by its predecessor/successor sets exactly like scalar
# accesses. So vector cycles get the FENCE-orderable variants only -- address /
# control dependencies feeding vector-register element-address computation are a
# subtlety we do not assert a verdict on.
VECTOR_CYCLE_VARIANTS = [
    "base",
    "fence_rw_rw",
    "fence_w_w_r_rw",
]


@dataclass(frozen=True)
class MemoryAccess:
    """Byte-addressed footprint of one scalar memory instruction.

    ``byte_level_no_mag`` is the only misaligned atomicity model currently
    supported.  It deliberately carries no whole-instruction atomicity
    guarantee; ordering relations still belong to the parent instruction.
    """

    base_symbol: str
    offset_bytes: int
    size_bytes: int
    covered_bytes: tuple[int, ...]
    natural_aligned: bool
    boundary: str
    atomicity_model: str

    def __post_init__(self) -> None:
        if self.size_bytes not in {1, 2, 4, 8}:
            raise ValueError(f"unsupported scalar access size: {self.size_bytes}")
        if self.offset_bytes < 0:
            raise ValueError("memory access offset must be non-negative")
        expected = tuple(range(self.offset_bytes, self.offset_bytes + self.size_bytes))
        if self.covered_bytes != expected:
            raise ValueError("covered_bytes must exactly match offset_bytes and size_bytes")
        if self.natural_aligned != (self.offset_bytes % self.size_bytes == 0):
            raise ValueError("natural_aligned does not match offset and access size")
        if self.atomicity_model not in {"aligned_atomic", "byte_level_no_mag"}:
            raise ValueError(f"unknown scalar atomicity model: {self.atomicity_model}")
        if self.atomicity_model == "byte_level_no_mag" and self.natural_aligned:
            raise ValueError("byte_level_no_mag is reserved for misaligned accesses")
        if self.atomicity_model == "aligned_atomic" and not self.natural_aligned:
            raise ValueError("aligned_atomic requires a naturally aligned access")
        if self.boundary != _access_boundary(self.offset_bytes, self.size_bytes):
            raise ValueError("memory access boundary does not match its byte range")

    @classmethod
    def create(cls, base_symbol: str, offset_bytes: int, size_bytes: int) -> "MemoryAccess":
        aligned = offset_bytes % size_bytes == 0
        return cls(
            base_symbol=base_symbol,
            offset_bytes=offset_bytes,
            size_bytes=size_bytes,
            covered_bytes=tuple(range(offset_bytes, offset_bytes + size_bytes)),
            natural_aligned=aligned,
            boundary=_access_boundary(offset_bytes, size_bytes),
            atomicity_model="aligned_atomic" if aligned else "byte_level_no_mag",
        )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "MemoryAccess":
        return cls(
            base_symbol=str(data["base_symbol"]),
            offset_bytes=int(data["offset_bytes"]),
            size_bytes=int(data["size_bytes"]),
            covered_bytes=tuple(int(value) for value in data["covered_bytes"]),
            natural_aligned=bool(data["natural_aligned"]),
            boundary=str(data["boundary"]),
            atomicity_model=str(data["atomicity_model"]),
        )

    def byte_location(self, byte_offset: int) -> str:
        if byte_offset not in self.covered_bytes:
            raise ValueError(f"byte {byte_offset} is outside this memory access")
        return f"{self.base_symbol}[{byte_offset}]"

    def to_json(self) -> dict[str, Any]:
        return {
            "base_symbol": self.base_symbol,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
            "covered_bytes": list(self.covered_bytes),
            "natural_aligned": self.natural_aligned,
            "boundary": self.boundary,
            "atomicity_model": self.atomicity_model,
            "whole_access_atomic": self.atomicity_model == "aligned_atomic",
            "mag_bytes": None,
        }


def _access_boundary(offset: int, size: int) -> str:
    end = offset + size - 1
    if offset // 64 != end // 64:
        return "cross64"
    if offset // 16 != end // 16:
        return "cross16_same_line"
    return "same16"


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
    memory_access: MemoryAccess | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LitmusEvent":
        access = data.get("memory_access")
        return cls(
            event_id=str(data["event_id"]),
            hart=int(data["hart"]),
            kind=str(data["kind"]),
            instruction=str(data["instruction"]),
            location=str(data.get("location", "")),
            register=str(data.get("register", "")),
            value=str(data.get("value", "")),
            role=str(data.get("role", "")),
            memory_access=(
                MemoryAccess.from_json(access)
                if isinstance(access, Mapping)
                else None
            ),
        )

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
            "memory_access": self.memory_access.to_json() if self.memory_access else None,
        }


@dataclass(frozen=True)
class LitmusRelation:
    src: str
    dst: str
    kind: str
    label: str = ""
    local: bool = False

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LitmusRelation":
        return cls(
            src=str(data["src"]),
            dst=str(data["dst"]),
            kind=str(data["kind"]),
            label=str(data.get("label", "")),
            local=bool(data.get("local", False)),
        )

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

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LitmusCaseIR":
        return cls(
            name=str(data["name"]),
            display_name=str(data.get("display_name", data["name"])),
            combination_name=str(data.get("combination_name", data["name"])),
            skeleton=str(data.get("skeleton", "Native")),
            variant=str(data.get("variant", "native")),
            cycle=str(data.get("cycle", "")),
            init_lines=[str(value) for value in data.get("init_lines", [])],
            harts=[
                [LitmusEvent.from_json(event) for event in hart]
                for hart in data.get("harts", [])
            ],
            relations=[
                LitmusRelation.from_json(relation)
                for relation in data.get("relations", [])
            ],
            exists=str(data.get("exists", "")),
            expected_outcome=str(data.get("expected_outcome", "solver_required")),
            model=str(data.get("model", "rvwmo")),
            description=str(data.get("description", "")),
            tags=[str(value) for value in data.get("tags", [])],
        )

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
    if _is_scalar_rvwmo(combination, decision):
        return len(_scalar_variant_ids(combination))
    if _is_vector_rvwmo(combination, decision):
        return len(_vector_variant_ids(combination))
    return 1


def build_litmus_ir_cases(combination: Combination, decision: Decision) -> list[LitmusCaseIR]:
    if _is_scalar_rvwmo(combination, decision):
        variants = _scalar_variant_ids(combination)
        expanded = len(variants) > 1
        return [_scalar_case(combination, variant, expanded) for variant in variants]
    if _is_vector_rvwmo(combination, decision):
        variants = _vector_variant_ids(combination)
        expanded = len(variants) > 1
        return [_vector_mp_case(combination, variant, _case_name(combination, variant, expanded)) for variant in variants]
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


def _is_vector_rvwmo(combination: Combination, decision: Decision) -> bool:
    return (
        decision.status == "generated"
        and decision.rvwmo_class == "rvwmo-vector"
        and combination.memory_event in {"vector_load", "vector_store"}
    )


def _vector_variant_ids(combination: Combination) -> list[str]:
    if "variant" in combination.params:
        return [str(combination.params["variant"])]
    return list(VECTOR_CYCLE_VARIANTS)


def _vector_setup(combination: Combination, hart: int, prefix: str) -> tuple[list[LitmusEvent], list[str]]:
    events = [_event(f"{prefix}_vset", hart, "setup", _vector_vset_instruction(combination))]
    extra_init: list[str] = []
    if combination.params.get("mask") == "masked":
        events.append(_event(f"{prefix}_mask", hart, "setup", "vmset.m v0"))
    if "indexed" in combination.vector:
        events.append(_event(f"{prefix}_vid", hart, "setup", "vid.v v4"))
    if "strided" in combination.vector:
        extra_init.append(f"{hart}:x9=4;")
    return events, extra_init


def _rebase_vector(instruction: str, base_reg: str) -> str:
    return re.sub(r"\(x\d+\)", f"({base_reg})", instruction, count=1)


def _vector_mp_case(combination: Combination, variant: str, name: str) -> LitmusCaseIR:
    # Vector message passing: the data-carrying access is a vector load/store,
    # the flag stays scalar. The RVV spec reduces vector memory ordering to
    # per-element RVWMO, and a FENCE orders those element accesses exactly like
    # scalar -- so the data-carrying event keeps kind store/load and the cycle
    # verdict equals the scalar MP twin.
    init = ["0:x5=1; 0:x6=x; 0:x7=y;", "1:x6=y; 1:x8=x;"]
    if combination.memory_event == "vector_store":
        setup, extra_init = _vector_setup(combination, 0, "p0")
        vinstr = _rebase_vector(_vector_instruction(combination), "x6")
        p0 = [
            *setup,
            _event("p0_vbcast", 0, "setup", "vmv.v.x v8,x5"),
            _event("p0_wx", 0, "store", vinstr, "x", value="1", role="data-write"),
            *_ordering_events(0, variant, "p0", "writer"),
            _event("p0_wy", 0, "store", "sw x5,0(x7)", "y", value="1", role="flag-write"),
        ]
        p1 = [
            _event("p1_ry", 1, "load", "lw x5,0(x6)", "y", register="x5", value="1", role="flag-read"),
            *_ordering_events(1, variant, "p1", "reader"),
            _event("p1_rx", 1, "load", "lw x7,0(x8)", "x", register="x7", value="0", role="data-read"),
        ]
        if extra_init:
            init[0] = init[0] + " " + " ".join(extra_init)
    else:
        setup, extra_init = _vector_setup(combination, 1, "p1")
        vinstr = _rebase_vector(_vector_instruction(combination), "x8")
        p0 = [
            _event("p0_wx", 0, "store", "sw x5,0(x6)", "x", value="1", role="data-write"),
            *_ordering_events(0, variant, "p0", "writer"),
            _event("p0_wy", 0, "store", "sw x5,0(x7)", "y", value="1", role="flag-write"),
        ]
        p1 = [
            *setup,
            _event("p1_ry", 1, "load", "lw x5,0(x6)", "y", register="x5", value="1", role="flag-read"),
            *_ordering_events(1, variant, "p1", "reader"),
            _event("p1_rx", 1, "load", vinstr, "x", register="v8", value="0", role="data-read"),
            _event("p1_extract", 1, "extract", "vmv.x.s x7,v8", register="x7", role="vector-extract"),
        ]
        if extra_init:
            init[1] = init[1] + " " + " ".join(extra_init)

    relations = [
        _relation("p0_wx", "p0_wy", "po", _variant_po_label("PodWW", variant), True),
        _relation("p0_wy", "p1_ry", "rfe", "Rfe"),
        _relation("p1_ry", "p1_rx", "po", _variant_po_label("PodRR", variant), True),
        _relation("p1_rx", "p0_wx", "fre", "Fre"),
    ]
    return _case(
        combination,
        name,
        variant,
        _cycle_label("PodWW -> Rfe -> PodRR -> Fre", variant),
        init,
        [p0, p1],
        relations,
        "(1:x5=1 /\\ 1:x7=0)",
        f"Vector message passing ({combination.vector}): per-element RVWMO reduction; FENCE orders the vector element accesses exactly like scalar.",
        tags=["vector", "rvwmo", combination.skeleton, variant, combination.vector],
    )


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
    return builder(combination, _effective_ordering_variant(variant), name)


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


# Stress profiles pack the ordering mechanism into a `dep-<shape>` token inside a
# composite variant id (e.g. "dep-addr_width-w32_outcome-forbidden_stress-none").
# Map the renderable shapes onto the scalar ordering renderers so the body honours
# what the file name claims. none/data/aq/rl/aqrl have no scalar lowering yet and
# render as the bare (base) body; width/outcome/stress are cross-product axis
# labels that never affect the body (they stay in the file name for cell identity).
_DEP_TO_VARIANT = {"addr": "addr_dep", "ctrl": "ctrl_dep", "ctrl_fence": "ctrl_fencei"}


def _effective_ordering_variant(variant: str) -> str:
    match = re.match(r"dep-(.+?)(?:_width-|_outcome-|_stress-|$)", variant)
    if not match:
        return variant
    return _DEP_TO_VARIANT.get(match.group(1), "base")


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
            _event(f"{prefix}_branch", hart, "dep", f"beq x5,x5,LC{hart}", role="ctrl-dep"),
            _event(f"{prefix}_label", hart, "label", f"LC{hart}:", role="ctrl-dep"),
        ]
    if role == "reader" and variant == "ctrl_fencei":
        return [
            _event(f"{prefix}_branch", hart, "dep", f"beq x5,x5,LC{hart}", role="ctrl-dep"),
            _event(f"{prefix}_label", hart, "label", f"LC{hart}:", role="ctrl-dep"),
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
    tags: list[str] | None = None,
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
        tags=tags if tags is not None else ["scalar", "rvwmo", combination.skeleton, variant],
    )


def _expected_outcome(combination: Combination) -> str:
    # The `outcome` param (allowed/forbidden/mixed_size) is a stress cross-product
    # axis LABEL, not a verified fact: the body is built from skeleton + dep only,
    # so a requested "forbidden" routinely renders a body the authoritative solver
    # judges "allowed" -- baking it in produced metadata that contradicted the
    # verdict. The solver is the source of truth, so always defer to it. (The
    # requested label is preserved in the file name / GUI prose for cell identity.)
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
        _event("p0_rx", 0, "load", "lw x5,0(x6)", "x", register="x5", value="1"),
        *_ordering_events(0, variant, "p0", "reader"),
        _event("p0_wy", 0, "store", "sw x9,0(x7)", "y", value="1"),
    ]
    p1 = [
        _event("p1_ry", 1, "load", "lw x5,0(x6)", "y", register="x5", value="1"),
        *_ordering_events(1, variant, "p1", "reader"),
        _event("p1_wx", 1, "store", "sw x9,0(x7)", "x", value="1"),
    ]
    return _case(
        combination,
        name,
        variant,
        _cycle_label("PodRW -> Rfe -> PodRW -> Rfe", variant),
        ["0:x6=x; 0:x7=y; 0:x9=1;", "1:x6=y; 1:x7=x; 1:x9=1;"],
        [p0, p1],
        [
            _relation("p0_rx", "p0_wy", "po", _variant_po_label("PodRW", variant), True),
            _relation("p0_wy", "p1_ry", "rfe", "Rfe"),
            _relation("p1_ry", "p1_wx", "po", _variant_po_label("PodRW", variant), True),
            _relation("p1_wx", "p0_rx", "rfe", "Rfe"),
        ],
        "(0:x5=1 /\\ 1:x5=1)",
        "Load buffering: each hart's load reads the other's later store (rf+po cycle); RVWMO allows it unless load->store is ordered.",
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
    init_lines = ["0:x5=1; 0:x6=x; 0:x7=y;", "1:x6=y; 1:x8=x;"]
    if "strided" in combination.vector:
        hart = 0 if combination.memory_event == "vector_store" else 1
        init_lines[hart] = init_lines[hart] + f" {hart}:x9=4;"
    return LitmusCaseIR(
        name=combination.name,
        display_name=combination.name,
        combination_name=combination.name,
        skeleton=combination.skeleton,
        variant="observation",
        cycle=_observation_cycle(combination),
        init_lines=init_lines,
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
        setup, _extra_init = _vector_setup(combination, 0, "p0")
        p0 = [
            *setup,
            _event("p0_vec", 0, "vector", _vector_instruction(combination), "x"),
            *_cmo_events(combination, 0, "p0"),
        ]
        p1 = [_event("p1_r", 1, "load", "lw x5,0(x6)", "y", register="x5", value="1")]
        return [p0, p1]
    if combination.vector != "none":
        hart = 0 if combination.memory_event == "vector_store" else 1
        setup, _extra_init = _vector_setup(combination, hart, f"p{hart}")
        base_reg = "x6" if hart == 0 else "x8"
        vector_event = _event(f"p{hart}_vec", hart, "vector", _rebase_vector(_vector_instruction(combination), base_reg), "x")
        if hart == 0:
            return [[*setup, vector_event], [_event("p1_r", 1, "load", "lw x5,0(x6)", "y")]]
        return [[_event("p0_w", 0, "store", "sw x5,0(x6)", "x")], [*setup, vector_event]]
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
    width = _vector_width(combination)
    mask = _vector_mask_suffix(combination)
    table = {
        "unit_load": f"vle{width}.v v8,(x6){mask}",
        "unit_store": f"vse{width}.v v8,(x6){mask}",
        "strided_load": f"vlse{width}.v v8,(x6),x9{mask}",
        "strided_store": f"vsse{width}.v v8,(x6),x9{mask}",
        "indexed_ordered_load": f"vloxei32.v v8,(x6),v4{mask}",
        "indexed_unordered_load": f"vluxei32.v v8,(x6),v4{mask}",
        "indexed_ordered_store": f"vsoxei32.v v8,(x6),v4{mask}",
        "indexed_unordered_store": f"vsuxei32.v v8,(x6),v4{mask}",
        "segment_load": f"vlseg2e{width}.v v8,(x6){mask}",
        "segment_store": f"vsseg2e{width}.v v8,(x6){mask}",
        "fof_load": f"vle{width}ff.v v8,(x6){mask}",
        "fof_segment_load": f"vlseg2e{width}ff.v v8,(x6){mask}",
    }
    return table.get(combination.vector, f"vle{width}.v v8,(x6){mask}")


def _vector_width(combination: Combination) -> str:
    sew = str(combination.params.get("sew", "e32"))
    return {"e8": "8", "e16": "16", "e32": "32", "e64": "64"}.get(sew, "32")


def _vector_policy(combination: Combination) -> str:
    tail = str(combination.params.get("tail", "ta_ma"))
    return {"ta_ma": "ta,ma", "ta_mu": "ta,mu", "tu_ma": "tu,ma", "tu_mu": "tu,mu"}.get(tail, "ta,ma")


def _vector_vset_instruction(combination: Combination) -> str:
    sew = str(combination.params.get("sew", "e32"))
    lmul = str(combination.params.get("lmul", "m1"))
    policy = _vector_policy(combination)
    vl = str(combination.params.get("vl", "vlmax"))
    if vl == "vl1":
        return f"vsetivli x10,1,{sew},{lmul},{policy}"
    if vl == "vl2":
        return f"vsetivli x10,2,{sew},{lmul},{policy}"
    if vl == "vl_random":
        return f"vsetvli x10,x11,{sew},{lmul},{policy}"
    return f"vsetvli x10,x0,{sew},{lmul},{policy}"


def _vector_mask_suffix(combination: Combination) -> str:
    return ",v0.t" if combination.params.get("mask") == "masked" else ""


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
