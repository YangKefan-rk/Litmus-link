from __future__ import annotations

"""Vector-aware RVWMO frontend for the supported Nanhu RVV memory subset.

The frontend expands active vector elements into explicit memory events while
preserving the distinction between instruction-level order and element-level
order.  It then delegates execution-graph enumeration and the RVWMO axioms to
``rvwmo_solver``.  This is not the old scalar-witness lowering: every element
has its architectural address, mask/vl determine whether it exists, unordered
forms have no sibling po, and ordered-indexed forms add an explicit preserved
element-order relation.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .litmus_ir import LitmusCaseIR, LitmusEvent, MemoryAccess
from .profiles import NANHU_VLEN_BITS, VECTOR_INDEX_EEWS, VECTOR_LENGTHS, VECTOR_LMUL_FACTORS, vector_vlmax
from .rvwmo_solver import EmbeddedVerdict, OrderingOverrides, Pair, solve_rvwmo


SUPPORTED_VECTOR_FORMS = frozenset(
    {
        "unit_load",
        "unit_store",
        "strided_load",
        "strided_store",
        "indexed_unordered_load",
        "indexed_unordered_store",
        "indexed_ordered_load",
        "indexed_ordered_store",
    }
)

class VectorSolverError(ValueError):
    pass


@dataclass(frozen=True)
class VectorConfig:
    form: str
    vlen_bits: int
    sew_bits: int
    lmul: str
    avl: str
    index_eew: str | None
    mask: str
    mask_pattern: str
    tail_policy: str
    footprint: str
    stride_bytes: int | None
    index_pattern: str | None
    ordered_elements: bool
    vlmax: int
    effective_vl: int
    base_offset_bytes: int
    alignment: str
    atomicity_model: str

    @classmethod
    def from_case(cls, case: LitmusCaseIR) -> "VectorConfig":
        raw = case.metadata.get("vector")
        if not isinstance(raw, Mapping):
            raise VectorSolverError("vector-aware verification requires case_ir.metadata.vector")

        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "VectorConfig":

        form = str(raw.get("form", ""))
        if form not in SUPPORTED_VECTOR_FORMS:
            raise VectorSolverError(f"unsupported vector solver form: {form or '<missing>'}")
        vlen_bits = _integer(raw.get("vlen_bits", 0), "vlen_bits")
        if vlen_bits != NANHU_VLEN_BITS:
            raise VectorSolverError(
                f"the Nanhu vector solver profile requires VLEN={NANHU_VLEN_BITS}"
            )
        sew_bits = _integer(raw.get("sew_bits", 0), "sew_bits")
        if sew_bits not in {8, 16, 32, 64}:
            raise VectorSolverError(f"unsupported SEW: {sew_bits}")
        lmul = str(raw.get("lmul", ""))
        if lmul not in VECTOR_LMUL_FACTORS:
            raise VectorSolverError(f"unsupported LMUL: {lmul}")
        vlmax = vector_vlmax(f"e{sew_bits}", lmul)
        if vlmax is None:
            raise VectorSolverError(f"illegal VLMAX for VLEN={vlen_bits}, SEW={sew_bits}, LMUL={lmul}")

        avl = str(raw.get("avl", ""))
        if avl == "vlmax":
            effective_vl = vlmax
        elif avl in VECTOR_LENGTHS and avl.startswith("vl") and avl[2:].isdigit():
            effective_vl = min(int(avl[2:]), vlmax)
        else:
            raise VectorSolverError(f"AVL {avl!r} is not statically supported")

        mask = str(raw.get("mask", "unmasked"))
        if mask not in {"unmasked", "masked"}:
            raise VectorSolverError(f"unsupported mask mode: {mask}")
        mask_pattern = str(raw.get("mask_pattern", "all-elements"))
        expected_pattern = "even-elements" if mask == "masked" else "all-elements"
        if mask_pattern != expected_pattern:
            raise VectorSolverError(
                f"mask metadata does not match generated assembly: expected {expected_pattern}, got {mask_pattern}"
            )

        tail_policy = str(raw.get("tail_policy", ""))
        if tail_policy not in {"ta_ma", "ta_mu", "tu_ma", "tu_mu"}:
            raise VectorSolverError(f"unsupported tail policy: {tail_policy}")
        footprint = str(raw.get("footprint", ""))
        if footprint not in {"same_line", "cross_line"}:
            raise VectorSolverError(
                f"footprint {footprint!r} is outside the current formal vector solver scope"
            )

        stride_bytes = raw.get("stride_bytes")
        if "strided" in form:
            stride_bytes = _integer(stride_bytes, "stride_bytes")
        elif stride_bytes is not None:
            raise VectorSolverError("stride_bytes is only valid for strided vector memory")

        index_pattern = raw.get("index_pattern")
        if "indexed" in form:
            if index_pattern != "scaled-element-index":
                raise VectorSolverError("complex indexed aliases are outside the current solver scope")
            index_eew = str(raw.get("index_eew", "ei32"))
            if index_eew not in VECTOR_INDEX_EEWS:
                raise VectorSolverError(f"unsupported indexed EEW: {index_eew}")
        elif index_pattern is not None:
            raise VectorSolverError("index_pattern is only valid for indexed vector memory")
        else:
            index_eew = None

        ordered = form.startswith("indexed_ordered")
        metadata_ordered = raw.get("ordered_elements", False)
        if not isinstance(metadata_ordered, bool):
            raise VectorSolverError("ordered_elements metadata must be boolean")
        if metadata_ordered != ordered:
            raise VectorSolverError("ordered_elements metadata does not match the vector instruction form")
        if _integer(raw.get("vstart", 0), "vstart") != 0:
            raise VectorSolverError("vstart/restart semantics are outside the current solver scope")
        base_offset_bytes = _integer(raw.get("base_offset_bytes", 0), "base_offset_bytes")
        if base_offset_bytes < 0:
            raise VectorSolverError("base_offset_bytes cannot be negative")
        alignment = str(raw.get("alignment", "aligned"))
        if alignment not in {
            "aligned",
            "misalign_same16",
            "misalign_cross16",
            "misalign_cross64",
        }:
            raise VectorSolverError(f"unsupported Vector alignment: {alignment}")
        atomicity_model = str(
            raw.get(
                "atomicity_model",
                "aligned_atomic" if base_offset_bytes % (sew_bits // 8) == 0 else "byte_level_no_mag",
            )
        )
        if atomicity_model not in {"aligned_atomic", "byte_level_no_mag"}:
            raise VectorSolverError(f"unsupported Vector element atomicity model: {atomicity_model}")
        if alignment == "aligned" and base_offset_bytes % (sew_bits // 8):
            raise VectorSolverError("aligned Vector metadata carries a misaligned base offset")
        if alignment != "aligned" and base_offset_bytes % (sew_bits // 8) == 0:
            raise VectorSolverError("misaligned Vector metadata carries a naturally aligned base offset")

        config = cls(
            form=form,
            vlen_bits=vlen_bits,
            sew_bits=sew_bits,
            lmul=lmul,
            avl=avl,
            index_eew=index_eew,
            mask=mask,
            mask_pattern=mask_pattern,
            tail_policy=tail_policy,
            footprint=footprint,
            stride_bytes=stride_bytes,
            index_pattern=str(index_pattern) if index_pattern is not None else None,
            ordered_elements=ordered,
            vlmax=vlmax,
            effective_vl=effective_vl,
            base_offset_bytes=base_offset_bytes,
            alignment=alignment,
            atomicity_model=atomicity_model,
        )
        active_offsets = [
            config.offset(index)
            for index in range(config.effective_vl)
            if config.active(index)
        ]
        if not active_offsets:
            raise VectorSolverError("vector instruction has no active elements")
        if config.base_offset_bytes + max(active_offsets) + config.element_bytes > 4096:
            raise VectorSolverError(
                "active Vector element footprint crosses the current 4 KiB formal object"
            )
        return config

    @property
    def element_bytes(self) -> int:
        return self.sew_bits // 8

    def active(self, index: int) -> bool:
        return index < self.effective_vl and (self.mask == "unmasked" or index % 2 == 0)

    def offset(self, index: int) -> int:
        if self.form.startswith("unit_"):
            return index * self.element_bytes
        if self.form.startswith("strided_"):
            assert self.stride_bytes is not None
            return index * self.stride_bytes
        if self.form.startswith("indexed_"):
            return index * self.element_bytes
        raise VectorSolverError(f"no address function for {self.form}")

    def to_json(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "vlen_bits": self.vlen_bits,
            "sew_bits": self.sew_bits,
            "element_bytes": self.element_bytes,
            "lmul": self.lmul,
            "avl": self.avl,
            "index_eew": self.index_eew,
            "vlmax": self.vlmax,
            "effective_vl": self.effective_vl,
            "mask": self.mask,
            "mask_pattern": self.mask_pattern,
            "tail_policy": self.tail_policy,
            "footprint": self.footprint,
            "stride_bytes": self.stride_bytes,
            "index_pattern": self.index_pattern,
            "ordered_elements": self.ordered_elements,
            "base_offset_bytes": self.base_offset_bytes,
            "alignment": self.alignment,
            "atomicity_model": self.atomicity_model,
            "vstart": 0,
        }


@dataclass(frozen=True)
class VectorElement:
    parent_event: str
    event_id: str
    hart: int
    index: int
    within_vl: bool
    mask_enabled: bool
    active: bool
    offset_bytes: int
    size_bytes: int
    location: str
    read: bool
    write: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "parent_event": self.parent_event,
            "event_id": self.event_id,
            "hart": self.hart,
            "index": self.index,
            "within_vl": self.within_vl,
            "mask_enabled": self.mask_enabled,
            "active": self.active,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
            "location": self.location,
            "read": self.read,
            "write": self.write,
        }


@dataclass(frozen=True)
class VectorInstruction:
    event_id: str
    hart: int
    form: str
    instruction_order: int
    elements: tuple[VectorElement, ...]

    @property
    def active_elements(self) -> tuple[VectorElement, ...]:
        return tuple(element for element in self.elements if element.active)

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "hart": self.hart,
            "form": self.form,
            "instruction_order": self.instruction_order,
            "active_element_count": len(self.active_elements),
            "elements": [element.to_json() for element in self.elements],
        }


@dataclass(frozen=True)
class VectorExpansion:
    case: LitmusCaseIR
    ordering: OrderingOverrides
    config: VectorConfig
    configs: Mapping[str, VectorConfig]
    instructions: tuple[VectorInstruction, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "litmus-link.vector-element-ir.v1",
            "config": self.config.to_json(),
            "configs": {
                event_id: config.to_json()
                for event_id, config in sorted(self.configs.items())
            },
            "instructions": [instruction.to_json() for instruction in self.instructions],
            "preserved_element_order": [list(pair) for pair in sorted(self.ordering.preserved_order)],
        }


@dataclass(frozen=True)
class VectorSolverVerdict:
    status: str
    verdict: str
    allowed: bool | None
    reason: str
    expansion: VectorExpansion | None
    embedded: EmbeddedVerdict | None
    external: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "litmus-link.vector-solver.v1",
            "status": self.status,
            "verdict": self.verdict,
            "allowed": self.allowed,
            "backend": (
                "vector-aware-embedded+herd-scalar-projection"
                if self.external is not None
                else "vector-aware-embedded"
            ),
            "tool": (
                "litmus-link-vector-rvwmo+herd7-reference"
                if self.external is not None
                else "litmus-link-vector-rvwmo"
            ),
            "model": "riscv.cat+rvv-elements",
            "model_revision": "rvv-element-order-v1",
            "reason": self.reason,
            "vector_ir": self.expansion.to_json() if self.expansion else None,
            "embedded": self.embedded.to_json() if self.embedded else None,
            "external": dict(self.external) if self.external is not None else None,
        }


def is_vector_case(case: LitmusCaseIR) -> bool:
    return isinstance(case.metadata.get("vector"), Mapping) or any(
        _is_vector_memory(event) for event in case.events()
    )


def expand_vector_case(case: LitmusCaseIR) -> VectorExpansion:
    config_by_event = _vector_configs(case)
    _validate_aligned_mixed_size_scope(case, config_by_event)
    harts: list[list[LitmusEvent]] = []
    order_by_event: dict[str, int] = {}
    instruction_by_event: dict[str, str] = {}
    preserved: set[Pair] = set()
    instructions: list[VectorInstruction] = []

    for hart_id, sequence in enumerate(case.harts):
        expanded_hart: list[LitmusEvent] = []
        instruction_order = 0
        for event in sequence:
            if event.kind not in {"load", "store", "amo"}:
                expanded_hart.append(event)
                continue
            if not _is_vector_memory(event):
                expanded_hart.append(event)
                order_by_event[event.event_id] = instruction_order
                instruction_by_event[event.event_id] = event.event_id
                instruction_order += 1
                continue

            if event.kind not in {"load", "store"}:
                raise VectorSolverError(f"vector event {event.event_id} is not a load/store")
            try:
                config = config_by_event[event.event_id]
            except KeyError as exc:
                raise VectorSolverError(
                    f"vector memory event {event.event_id} has no per-instruction configuration"
                ) from exc
            elements: list[VectorElement] = []
            active_ids: list[str] = []
            for index in range(config.vlmax):
                within_vl = index < config.effective_vl
                mask_enabled = config.mask == "unmasked" or index % 2 == 0
                active = config.active(index)
                offset = config.base_offset_bytes + config.offset(index)
                location = _offset_location(event.location, offset)
                event_id = f"{event.event_id}.e{index}"
                element = VectorElement(
                    parent_event=event.event_id,
                    event_id=event_id,
                    hart=hart_id,
                    index=index,
                    within_vl=within_vl,
                    mask_enabled=mask_enabled,
                    active=active,
                    offset_bytes=offset,
                    size_bytes=config.element_bytes,
                    location=location,
                    read=event.kind == "load",
                    write=event.kind == "store",
                )
                elements.append(element)
                if not active:
                    continue
                active_ids.append(event_id)
                expanded_event = LitmusEvent(
                    event_id=event_id,
                    hart=hart_id,
                    kind=event.kind,
                    instruction=_scalar_element_instruction(event.kind, config.sew_bits),
                    location=location,
                    register="x28" if event.kind == "load" else "x5",
                    value=event.value if event.kind == "store" or index == 0 else "",
                    role=f"vector-element-active:{index}",
                    memory_access=MemoryAccess.create(
                        event.location,
                        offset,
                        config.element_bytes,
                        config.atomicity_model,
                        transaction_kind="vector_element",
                        parent_instruction=event.event_id,
                        element_index=index,
                    ),
                )
                expanded_hart.append(expanded_event)
                order_by_event[event_id] = instruction_order
                instruction_by_event[event_id] = event.event_id

            if not active_ids:
                raise VectorSolverError(f"vector instruction {event.event_id} has no active elements")
            if config.ordered_elements:
                preserved.update(
                    (left, right)
                    for left_index, left in enumerate(active_ids)
                    for right in active_ids[left_index + 1 :]
                )
            instructions.append(
                VectorInstruction(
                    event_id=event.event_id,
                    hart=hart_id,
                    form=config.form,
                    instruction_order=instruction_order,
                    elements=tuple(elements),
                )
            )
            instruction_order += 1
        harts.append(expanded_hart)

    if not instructions:
        raise VectorSolverError("case does not contain a supported vector memory instruction")

    metadata = dict(case.metadata)
    metadata["vector_solver"] = {
        "schema": "litmus-link.vector-solver-expansion.v1",
        "source_case": case.name,
    }
    expanded_case = replace(
        case,
        name=f"{case.name}__vector_elements",
        model="rvwmo-vector-elements",
        harts=harts,
        description="Vector element execution graph for the supported Nanhu RVV subset.",
        metadata=metadata,
    )
    return VectorExpansion(
        case=expanded_case,
        ordering=OrderingOverrides(
            order_by_event=order_by_event,
            instruction_by_event=instruction_by_event,
            preserved_order=frozenset(preserved),
        ),
        config=next(iter(config_by_event.values())),
        configs=config_by_event,
        instructions=tuple(instructions),
    )


def solve_vector_case(
    case: LitmusCaseIR,
    *,
    max_candidates: int = 100_000,
    timeout_seconds: float = 10.0,
    max_search_steps: int = 1_000_000,
    max_memory_events: int | None = None,
    external_check: bool = False,
    external_max_projections: int = 64,
    external_timeout: int = 30,
) -> VectorSolverVerdict:
    try:
        expansion = expand_vector_case(case)
    except (TypeError, ValueError) as exc:
        return VectorSolverVerdict(
            status="not_applicable",
            verdict="unmodeled",
            allowed=None,
            reason=str(exc),
            expansion=None,
            embedded=None,
            external=None,
        )

    memory_event_count = sum(
        event.kind in {"load", "store", "amo"}
        for event in expansion.case.events()
    )
    if max_memory_events is not None and memory_event_count > max_memory_events:
        return VectorSolverVerdict(
            status="inconclusive",
            verdict="unknown",
            allowed=None,
            reason=(
                f"Interactive verification skipped a {memory_event_count}-transaction "
                f"execution graph; the current limit is {max_memory_events}. "
                "Use a higher verification effort for this case."
            ),
            expansion=expansion,
            embedded=None,
            external=None,
        )

    embedded = solve_rvwmo(
        expansion.case,
        max_candidates=max_candidates,
        timeout_seconds=timeout_seconds,
        max_search_steps=max_search_steps,
        ordering=expansion.ordering,
    )
    if external_check and embedded.status == "verified":
        from .herd_reference import crosscheck_vector_projection

        external = crosscheck_vector_projection(
            case,
            expansion,
            embedded,
            max_projections=external_max_projections,
            timeout=external_timeout,
        )
    else:
        external = None
    if external is not None and external.get("status") == "conflict":
        status = "conflict"
        verdict = "conflict"
        allowed = None
        reason = str(external.get("reason", "Embedded/herd projection conflict"))
    elif embedded.status == "verified":
        status = embedded.status
        verdict = embedded.verdict
        allowed = embedded.allowed
        reason = (
            "Active RVV elements were solved as one instruction-level event set under RVWMO; "
            "mask/vl/address generation and ordered-indexed element PPO are explicit in vector_ir. "
            f"External reference status: {external.get('status', 'not_run') if external is not None else 'not_run'}."
        )
    else:
        status = embedded.status
        verdict = embedded.verdict
        allowed = embedded.allowed
        reason = embedded.reason
    return VectorSolverVerdict(
        status=status,
        verdict=verdict,
        allowed=allowed,
        reason=reason,
        expansion=expansion,
        embedded=embedded,
        external=external,
    )


def _is_vector_memory(event: LitmusEvent) -> bool:
    instruction = event.instruction.lstrip().lower()
    return event.kind in {"load", "store"} and instruction.startswith("v")


def _vector_configs(case: LitmusCaseIR) -> dict[str, VectorConfig]:
    vector_events = [event for event in case.events() if _is_vector_memory(event)]
    if not vector_events:
        raise VectorSolverError("case does not contain a supported vector memory instruction")
    raw_configs = case.metadata.get("vectors")
    if isinstance(raw_configs, Mapping):
        configs: dict[str, VectorConfig] = {}
        for event in vector_events:
            raw = raw_configs.get(event.event_id)
            if not isinstance(raw, Mapping):
                raise VectorSolverError(
                    f"case_ir.metadata.vectors has no configuration for {event.event_id}"
                )
            configs[event.event_id] = VectorConfig.from_mapping(raw)
        unknown = set(str(key) for key in raw_configs) - {event.event_id for event in vector_events}
        if unknown:
            raise VectorSolverError(
                f"case_ir.metadata.vectors references non-Vector events: {', '.join(sorted(unknown))}"
            )
        return configs
    legacy = VectorConfig.from_case(case)
    return {event.event_id: legacy for event in vector_events}


def _offset_location(base: str, offset: int) -> str:
    if not base:
        raise VectorSolverError("vector memory event has no base location")
    return base if offset == 0 else f"{base}[{offset}]"


def _scalar_element_instruction(kind: str, sew_bits: int) -> str:
    load = {8: "lb", 16: "lh", 32: "lw", 64: "ld"}
    store = {8: "sb", 16: "sh", 32: "sw", 64: "sd"}
    mnemonic = load[sew_bits] if kind == "load" else store[sew_bits]
    register = "x28" if kind == "load" else "x5"
    return f"{mnemonic} {register},0(x31)"


def _validate_aligned_mixed_size_scope(
    case: LitmusCaseIR,
    configs: Mapping[str, VectorConfig],
) -> None:
    for event_id, config in configs.items():
        if config.atomicity_model != "aligned_atomic" or config.alignment != "aligned":
            raise VectorSolverError(
                f"Vector event {event_id} is misaligned; aligned Vector/scalar/AMO fusion only"
            )
        for index in range(config.effective_vl):
            if not config.active(index):
                continue
            offset = config.base_offset_bytes + config.offset(index)
            if offset % config.element_bytes:
                raise VectorSolverError(
                    f"Vector event {event_id} element {index} is not naturally aligned"
                )
    for event in case.events():
        if event.kind != "amo" or event.memory_access is None:
            continue
        if not event.memory_access.natural_aligned:
            raise VectorSolverError(f"AMO event {event.event_id} is not naturally aligned")


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise VectorSolverError(f"{field} must be an integer")
    try:
        return int(str(value), 0)
    except (TypeError, ValueError) as exc:
        raise VectorSolverError(f"{field} must be an integer: {value!r}") from exc
