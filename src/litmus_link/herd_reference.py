from __future__ import annotations

"""External herd7 capability probes and scalar RVV projections.

Stock herd7 does not parse RISC-V Vector instructions.  This module never
feeds RVV assembly to herd.  It projects the supported active elements to
ordinary scalar memory instructions, preserving the surrounding scalar
dependency/fence structure, and uses those projections only as a differential
reference for Litmus-link's Vector-aware RVWMO solver.
"""

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from itertools import permutations, product
from math import factorial
from typing import Any, Mapping, Sequence

from .amo import (
    AMO_OPERATIONS,
    AMO_ORDERINGS,
    AmoError,
    AmoSpec,
    amo_read_result,
    apply_amo,
    parse_amo_mnemonic,
)
from .litmus_ir import LitmusCaseIR, LitmusEvent, MemoryAccess
from .renderer import render_ir
from . import toolchain


class HerdProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class HerdCapability:
    supported: bool
    reason: str
    observation: str = ""
    variants: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "observation": self.observation or None,
            "variants": list(self.variants),
        }


@dataclass(frozen=True)
class HerdCapabilities:
    available: bool
    scalar: HerdCapability
    mixed_size: HerdCapability
    mixed_amo: HerdCapability
    amo: Mapping[str, HerdCapability]
    tool_path: str
    tool_version: str
    model_path: str

    def amo_capability(
        self, operation: str, width_bytes: int, ordering: str
    ) -> HerdCapability:
        return self.amo.get(
            _amo_capability_key(operation, width_bytes, ordering),
            HerdCapability(False, "AMO combination was not probed"),
        )

    def to_json(self) -> dict[str, Any]:
        supported_amo = sorted(
            key for key, capability in self.amo.items() if capability.supported
        )
        unsupported_amo = {
            key: capability.reason
            for key, capability in sorted(self.amo.items())
            if not capability.supported
        }
        return {
            "schema": "litmus-link.herd-capabilities.v1",
            "available": self.available,
            "tool": {
                "path": self.tool_path,
                "version": self.tool_version,
            },
            "model": self.model_path,
            "scalar": self.scalar.to_json(),
            "mixed_size": self.mixed_size.to_json(),
            "mixed_amo": self.mixed_amo.to_json(),
            "amo_supported": supported_amo,
            "amo_unsupported": unsupported_amo,
        }


@dataclass(frozen=True)
class ScalarProjection:
    name: str
    source: str
    variants: tuple[str, ...]
    storage_mode: str
    element_orders: Mapping[str, tuple[int, ...]]

    def to_json(self, *, include_source: bool = False) -> dict[str, Any]:
        out = {
            "name": self.name,
            "variants": list(self.variants),
            "storage_mode": self.storage_mode,
            "element_orders": {
                parent: list(order)
                for parent, order in sorted(self.element_orders.items())
            },
        }
        if include_source:
            out["source"] = self.source
        return out


@dataclass(frozen=True)
class ProjectionBundle:
    status: str
    reason: str
    projections: tuple[ScalarProjection, ...] = ()
    requested_projections: int = 0
    storage_mode: str = ""
    exact: bool = False
    oracle_kind: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "requested_projections": self.requested_projections,
            "generated_projections": len(self.projections),
            "storage_mode": self.storage_mode or None,
            "exact": self.exact,
            "oracle_kind": self.oracle_kind or None,
            "projections": [projection.to_json() for projection in self.projections],
        }


@lru_cache(maxsize=1)
def probe_herd_capabilities() -> HerdCapabilities:
    path = str(toolchain.HERD)
    model = str(toolchain.RISCV_CAT)
    version = toolchain.tool_version(path)
    if not toolchain.HERD.exists() or not toolchain.RISCV_CAT.exists():
        missing_parts = []
        if not toolchain.HERD.exists():
            missing_parts.append(f"herd7 ({toolchain.HERD})")
        if not toolchain.RISCV_CAT.exists():
            missing_parts.append(f"riscv.cat ({toolchain.RISCV_CAT})")
        missing = ", ".join(missing_parts) or "herd7/riscv.cat unavailable"
        unsupported = HerdCapability(False, missing)
        return HerdCapabilities(
            False,
            unsupported,
            unsupported,
            unsupported,
            {},
            path,
            version,
            model,
        )

    scalar = _probe(
        """RISCV LLProbeScalar
{
x=0;
0:x1=x;
}
 P0;
 lw x2,0(x1);
exists (0:x2=0)
"""
    )
    mixed = _probe(_mixed_probe_source(), variants=("mixed",))
    amo: dict[str, HerdCapability] = {}
    for operation, width_bytes, ordering in product(
        AMO_OPERATIONS, (4, 8), AMO_ORDERINGS
    ):
        key = _amo_capability_key(operation, width_bytes, ordering)
        amo[key] = _probe(_amo_probe_source(operation, width_bytes, ordering))
    mixed_amo = (
        _probe(_mixed_amo_probe_source(), variants=("mixed",))
        if mixed.supported
        else HerdCapability(False, "mixed-size RISC-V semantics are unavailable", variants=("mixed",))
    )
    return HerdCapabilities(
        scalar.supported,
        scalar,
        mixed,
        mixed_amo,
        amo,
        path,
        version,
        model,
    )


def probe_case_herd_capabilities(
    case: LitmusCaseIR,
    *,
    mixed_size: bool,
) -> HerdCapabilities:
    """Probe only the herd features required by one scalar projection.

    The full public capability report intentionally checks every supported AMO
    combination. Doing that eagerly in an interactive preview is unnecessary:
    one generated case can use only a small subset, and every probe starts a
    separate herd7 process.
    """

    path = str(toolchain.HERD)
    model = str(toolchain.RISCV_CAT)
    version = toolchain.tool_version(path)
    if not toolchain.HERD.exists() or not toolchain.RISCV_CAT.exists():
        missing_parts = []
        if not toolchain.HERD.exists():
            missing_parts.append(f"herd7 ({toolchain.HERD})")
        if not toolchain.RISCV_CAT.exists():
            missing_parts.append(f"riscv.cat ({toolchain.RISCV_CAT})")
        unsupported = HerdCapability(False, ", ".join(missing_parts))
        return HerdCapabilities(
            False,
            unsupported,
            unsupported,
            unsupported,
            {},
            path,
            version,
            model,
        )

    scalar = _probe(_scalar_probe_source())
    not_required = HerdCapability(False, "not required by this projection")
    mixed = _probe(_mixed_probe_source(), variants=("mixed",)) if mixed_size else not_required
    requirements = _case_amo_requirements(case)
    amo = {
        _amo_capability_key(operation, width, ordering): _probe(
            _amo_probe_source(operation, width, ordering)
        )
        for operation, width, ordering in requirements
    }
    mixed_amo = (
        _probe(_mixed_amo_probe_source(), variants=("mixed",))
        if mixed_size and requirements and mixed.supported
        else not_required
    )
    return HerdCapabilities(
        scalar.supported,
        scalar,
        mixed,
        mixed_amo,
        amo,
        path,
        version,
        model,
    )


def _scalar_probe_source() -> str:
    return """RISCV LLProbeScalar
{
x=0;
0:x1=x;
}
 P0;
 lw x2,0(x1);
exists (0:x2=0)
"""


def _probe(source: str, variants: tuple[str, ...] = ()) -> HerdCapability:
    return _probe_cached(source, variants)


@lru_cache(maxsize=256)
def _probe_cached(source: str, variants: tuple[str, ...]) -> HerdCapability:
    try:
        verdict = toolchain.herd_judge(source, timeout=20, variants=variants)
    except (OSError, toolchain.ToolchainError) as exc:
        return HerdCapability(False, _stable_probe_reason(str(exc)), variants=variants)
    if verdict.allowed is not True:
        return HerdCapability(
            False,
            (
                "herd7 parsed the probe but did not admit its exact expected "
                f"outcome ({verdict.outcome})"
            ),
            verdict.observation,
            variants,
        )
    return HerdCapability(
        True,
        "herd7 parsed and evaluated the probe",
        verdict.observation,
        variants,
    )


def _stable_probe_reason(reason: str) -> str:
    return re.sub(
        r'File "/tmp/ll-herd-[^/"]+/t\.litmus"',
        "File <probe>",
        reason,
    )


def _amo_capability_key(operation: str, width_bytes: int, ordering: str) -> str:
    return f"{operation}.{'w' if width_bytes == 4 else 'd'}.{ordering}"


def _amo_probe_source(operation: str, width_bytes: int, ordering: str) -> str:
    spec = AmoSpec(operation, width_bytes, ordering)
    old, operand = _amo_probe_values(operation, width_bytes)
    new = apply_amo(operation, width_bytes, old, operand)
    rd = amo_read_result(width_bytes, old)
    ctype = f"uint{width_bytes * 8}_t"
    old_literal = _memory_literal(old, width_bytes)
    operand_literal = _register_literal(operand)
    rd_literal = _register_literal(rd)
    new_literal = _memory_literal(new, width_bytes)
    return f"""RISCV LLProbeAmo{operation.title()}{width_bytes * 8}{ordering.title()}
{{
{ctype} x={old_literal};
0:x1=x;
0:x2={operand_literal};
}}
 P0;
 {spec.mnemonic} x3,x2,(x1);
exists (0:x3={rd_literal} /\\ [x]={new_literal})
"""


def _amo_probe_values(operation: str, width_bytes: int) -> tuple[int, int]:
    bits = width_bytes * 8
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    values = {
        "swap": (0x11, 0x22),
        "add": ((mask, 2) if width_bytes == 4 else (0x1234, 0x10)),
        "xor": (0xAAAA & mask, 0x0F),
        "and": (0xF3, 0x3F),
        "or": (0x30, 0x0F),
        "min": (sign - 1, sign | 4),
        "max": (sign | 4, sign - 1),
        "minu": (mask - 1, 1),
        "maxu": (1, mask - 1),
    }
    return values[operation]


def _register_literal(value: int) -> str:
    masked = value & ((1 << 64) - 1)
    if masked & (1 << 63):
        return str(masked - (1 << 64))
    return f"0x{masked:x}"


def _memory_literal(value: int, width_bytes: int) -> str:
    masked = value & ((1 << (width_bytes * 8)) - 1)
    return _register_literal(masked) if width_bytes == 8 else f"0x{masked:x}"


def _mixed_probe_source() -> str:
    values = ",".join("0" for _ in range(8))
    return f"""RISCV LLProbeMixed
{{
uint8_t x[8]={{{values}}};
0:x1=x;
0:x2=0x11223344;
}}
 P0;
 sw x2,0(x1);
exists (x[0]=0x44 /\\ x[1]=0x33 /\\ x[2]=0x22 /\\ x[3]=0x11)
"""


def _mixed_amo_probe_source() -> str:
    values = ",".join("0" for _ in range(8))
    return f"""RISCV LLProbeMixedAmo
{{
uint8_t x[8]={{{values}}};
0:x1=x;
0:x2=1;
}}
 P0;
 amoadd.d x3,x2,(x1);
exists (0:x3=0 /\\ x[0]=1)
"""


def build_scalar_projections(
    case: LitmusCaseIR,
    expansion: Any,
    *,
    max_projections: int = 64,
) -> ProjectionBundle:
    if max_projections < 1:
        raise ValueError("max_projections must be positive")
    value_plan = case.metadata.get("value_plan")
    if not isinstance(value_plan, Mapping):
        return ProjectionBundle(
            "external_unsupported",
            "Exact scalar projection requires fusion value_plan metadata.",
        )

    order_domains: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
    requested = 1
    exact = True
    oracle_kind = "vl1-exact"
    for instruction in expansion.instructions:
        active = tuple(element.index for element in instruction.active_elements)
        if instruction.form.startswith("indexed_ordered") or len(active) < 2:
            orders = (active,)
            if instruction.form.startswith("indexed_ordered") and len(active) > 1:
                exact = False
                oracle_kind = "ordered-fixed-advisory"
        else:
            if oracle_kind == "vl1-exact":
                oracle_kind = "unordered-permutation-exact"
            requested *= factorial(len(active))
            if requested > max_projections:
                return ProjectionBundle(
                    "external_unsupported",
                    f"Scalar projection needs {requested} element permutations; limit is {max_projections}.",
                    requested_projections=requested,
                )
            orders = tuple(permutations(active))
        order_domains.append((instruction.event_id, orders))

    projections: list[ScalarProjection] = []
    for selected in product(*(orders for _parent, orders in order_domains)):
        order_map = {
            parent: tuple(order)
            for (parent, _orders), order in zip(order_domains, selected)
        }
        try:
            projected, storage_mode = _project_case(
                case, expansion, order_map, value_plan
            )
        except HerdProjectionError as exc:
            return ProjectionBundle(
                "external_unsupported",
                str(exc),
                requested_projections=requested,
            )
        source = render_ir(projected)
        projections.append(
            ScalarProjection(
                projected.name,
                source,
                ("mixed",) if storage_mode == "mixed" else (),
                storage_mode,
                order_map,
            )
        )
    storage_modes = {projection.storage_mode for projection in projections}
    if len(storage_modes) != 1:
        raise AssertionError("projection storage mode changed across element permutations")
    return ProjectionBundle(
        "ready",
        "Generated exact scalar element projection(s).",
        tuple(projections),
        requested_projections=requested,
        storage_mode=next(iter(storage_modes), ""),
        exact=exact,
        oracle_kind=oracle_kind,
    )


def capability_for_scalar_case(
    case: LitmusCaseIR,
    *,
    mixed_size: bool = False,
    capabilities: HerdCapabilities | None = None,
) -> HerdCapability:
    caps = capabilities or probe_herd_capabilities()
    if not caps.scalar.supported:
        return caps.scalar
    amo_specs: list[AmoSpec] = []
    for event in case.events():
        if event.kind != "amo":
            continue
        try:
            parsed = parse_amo_mnemonic(event.instruction)
            spec = AmoSpec(
                event.amo_op or parsed.operation,
                event.amo_width_bytes or parsed.width_bytes,
                event.amo_ordering or parsed.ordering,
            )
        except (AmoError, ValueError) as exc:
            return HerdCapability(False, str(exc))
        capability = caps.amo_capability(
            spec.operation, spec.width_bytes, spec.ordering
        )
        if not capability.supported:
            return capability
        amo_specs.append(spec)
    if mixed_size:
        if not caps.mixed_size.supported:
            return caps.mixed_size
        if amo_specs and not caps.mixed_amo.supported:
            return caps.mixed_amo
    return HerdCapability(True, "Required scalar/AMO herd capabilities passed")


def _project_case(
    case: LitmusCaseIR,
    expansion: Any,
    order_map: Mapping[str, tuple[int, ...]],
    value_plan: Mapping[str, Any],
) -> tuple[LitmusCaseIR, str]:
    instructions = {instruction.event_id: instruction for instruction in expansion.instructions}
    configs = expansion.configs
    vector_only_init: set[tuple[int, str]] = set()
    projected_harts: list[list[LitmusEvent]] = []

    for hart, sequence in enumerate(case.harts):
        by_id = {event.event_id: event for event in sequence}
        projected: list[LitmusEvent] = []
        for event in sequence:
            parent = _vector_aux_parent(event, instructions)
            if parent is not None:
                if event.event_id.startswith(f"{parent}_vector_"):
                    vector_only_init.update(
                        (hart, register)
                        for register in _vector_setup_input_registers(event.instruction)
                    )
                    continue
                if event.role in {"vector-broadcast", "vector-extract-element0"}:
                    continue

            if event.event_id not in instructions:
                projected.append(_normalize_event_literals(event))
                continue

            instruction = instructions[event.event_id]
            config = configs[event.event_id]
            base = _memory_base_register(event.instruction)
            base_is_adjusted = any(
                candidate.event_id == f"{event.event_id}_offset_base"
                and candidate.role == "vector-fusion-base"
                for candidate in sequence
            )
            if event.kind == "load":
                data_register = _vector_extract_register(by_id, event.event_id)
            else:
                data_register = _vector_broadcast_register(by_id, event.event_id)
            if "strided" in instruction.form:
                vector_only_init.update(
                    (hart, register)
                    for register in _strided_registers(event.instruction)
                )

            for element_index in order_map[event.event_id]:
                relative = config.offset(element_index)
                immediate = relative if base_is_adjusted else config.base_offset_bytes + relative
                absolute = config.base_offset_bytes + relative
                register = (
                    data_register
                    if event.kind == "store" or element_index == 0
                    else "x0"
                )
                mnemonic = _scalar_mnemonic(event.kind, config.sew_bits)
                projected.append(
                    replace(
                        event,
                        event_id=f"{event.event_id}.e{element_index}",
                        instruction=f"{mnemonic} {register},{immediate}({base})",
                        register=register,
                        value=(
                            event.value
                            if event.kind == "store" or element_index == 0
                            else ""
                        ),
                        role=f"herd-scalar-projection-element:{element_index}",
                        memory_access=MemoryAccess.create(
                            event.location,
                            absolute,
                            config.element_bytes,
                            "mixed_size_atomic",
                        ),
                    )
                )
        projected_harts.append(projected)

    filtered_init = _filter_vector_init(case.init_lines, vector_only_init)
    relations = [
        replace(
            relation,
            src=f"{relation.src}.e0" if relation.src in instructions else relation.src,
            dst=f"{relation.dst}.e0" if relation.dst in instructions else relation.dst,
        )
        for relation in case.relations
    ]
    canonical_orders = {
        parent: list(order) for parent, order in sorted(order_map.items())
    }
    digest = sha256(
        repr((case.name, canonical_orders)).encode("utf-8")
    ).hexdigest()[:20]
    projected_name = f"LLP-{case.skeleton}-{digest}"
    projected_case = replace(
        case,
        name=projected_name,
        display_name=projected_name,
        variant="herd-scalar-projection",
        cycle=case.cycle,
        init_lines=filtered_init,
        harts=projected_harts,
        relations=relations,
        model="rvwmo",
        description="Exact scalar projection used only for herd7 differential checking.",
        tags=[*case.tags, "herd-scalar-projection"],
        metadata={
            **dict(case.metadata),
            "herd_projection": {
                "source_case": case.name,
                "element_orders": canonical_orders,
            },
        },
    )
    storage_mode = _projection_storage_mode(projected_case)
    if storage_mode == "typed":
        projected_case = _rewrite_typed_storage(projected_case, value_plan)
    return projected_case, storage_mode


def _vector_aux_parent(
    event: LitmusEvent, instructions: Mapping[str, Any]
) -> str | None:
    for parent in instructions:
        if event.event_id.startswith(f"{parent}_"):
            return parent
    return None


def _vector_setup_input_registers(instruction: str) -> tuple[str, ...]:
    normalized = instruction.strip().lower()
    if not normalized.startswith("vsetvli"):
        return ()
    registers = re.findall(r"\bx(?:[12]?\d|3[01]|[0-9])\b", normalized)
    return tuple(registers[1:2])


def _strided_registers(instruction: str) -> tuple[str, ...]:
    match = re.search(r"\)\s*,\s*(x(?:[12]?\d|3[01]|[0-9]))\s*$", instruction)
    return (match.group(1),) if match else ()


def _memory_base_register(instruction: str) -> str:
    match = re.search(r"\((x(?:[12]?\d|3[01]|[0-9]))\)", instruction.lower())
    if match is None:
        raise HerdProjectionError(f"cannot find Vector base register in {instruction!r}")
    return match.group(1)


def _vector_extract_register(
    by_id: Mapping[str, LitmusEvent], parent: str
) -> str:
    event = by_id.get(f"{parent}_extract")
    if event is None:
        raise HerdProjectionError(f"Vector load {parent} has no element-0 extract")
    match = re.fullmatch(r"\s*vmv\.x\.s\s+(x\d+)\s*,.*", event.instruction.lower())
    if match is None:
        raise HerdProjectionError(f"cannot parse Vector extract {event.instruction!r}")
    return match.group(1)


def _vector_broadcast_register(
    by_id: Mapping[str, LitmusEvent], parent: str
) -> str:
    event = by_id.get(f"{parent}_broadcast")
    if event is None:
        raise HerdProjectionError(f"Vector store {parent} has no scalar broadcast")
    match = re.fullmatch(r"\s*vmv\.v\.x\s+v\d+\s*,\s*(x\d+)\s*", event.instruction.lower())
    if match is None:
        raise HerdProjectionError(f"cannot parse Vector broadcast {event.instruction!r}")
    return match.group(1)


def _scalar_mnemonic(kind: str, sew_bits: int) -> str:
    loads = {8: "lb", 16: "lh", 32: "lw", 64: "ld"}
    stores = {8: "sb", 16: "sh", 32: "sw", 64: "sd"}
    try:
        return (loads if kind == "load" else stores)[sew_bits]
    except KeyError as exc:
        raise HerdProjectionError(
            f"cannot project {kind} with SEW={sew_bits}"
        ) from exc


def _filter_vector_init(
    lines: Sequence[str], remove: set[tuple[int, str]]
) -> list[str]:
    out: list[str] = []
    assignment = re.compile(r"\s*(\d+):(x\d+)\s*=.*")
    for line in lines:
        if ":" not in line:
            out.append(line)
            continue
        retained: list[str] = []
        for raw_term in line.split(";"):
            term = raw_term.strip()
            if not term:
                continue
            match = assignment.fullmatch(term)
            if match and (int(match.group(1)), match.group(2)) in remove:
                continue
            retained.append(_normalize_register_assignment(term) + ";")
        if retained:
            out.append(" ".join(retained))
    return out


def _projection_storage_mode(case: LitmusCaseIR) -> str:
    widths: dict[str, set[int]] = {}
    for event in case.events():
        if event.kind not in {"load", "store", "amo"} or event.memory_access is None:
            continue
        widths.setdefault(event.memory_access.base_symbol, set()).add(
            event.memory_access.size_bytes
        )
    if not widths:
        raise HerdProjectionError("projection contains no memory transactions")
    return "typed" if all(len(values) == 1 for values in widths.values()) else "mixed"


def _rewrite_typed_storage(
    case: LitmusCaseIR, value_plan: Mapping[str, Any]
) -> LitmusCaseIR:
    initial_raw = value_plan.get("initial_bytes")
    final_raw = value_plan.get("final_bytes")
    observed_raw = value_plan.get("observed_final_bytes")
    if not all(isinstance(value, Mapping) for value in (initial_raw, final_raw, observed_raw)):
        raise HerdProjectionError("fusion value_plan has no byte images for typed projection")
    initial = dict(initial_raw)
    final = dict(final_raw)
    observed = dict(observed_raw)
    widths: dict[str, int] = {}
    for event in case.events():
        access = event.memory_access
        if event.kind in {"load", "store", "amo"} and access is not None:
            previous = widths.setdefault(access.base_symbol, access.size_bytes)
            if previous != access.size_bytes:
                raise HerdProjectionError("typed projection found a mixed-width object")

    declarations: list[str] = []
    for name, width in sorted(widths.items()):
        byte_values = [int(value) & 0xFF for value in initial.get(name, ())]
        if not byte_values or len(byte_values) % width:
            raise HerdProjectionError(
                f"initial byte image for {name} is not divisible by {width}"
            )
        values = [
            _bytes_value(byte_values[offset : offset + width])
            for offset in range(0, len(byte_values), width)
        ]
        declarations.append(
            f"uint{width * 8}_t {name}[{len(values)}]={{"
            + ",".join(_memory_literal(value, width) for value in values)
            + "};"
        )

    declaration_names = set(widths)
    old_declaration = re.compile(
        r"\s*(?:u?int\d+_t|char)\s+([A-Za-z_]\w*)\s*\[.*"
    )
    retained = [
        line
        for line in case.init_lines
        if not (
            (match := old_declaration.fullmatch(line))
            and match.group(1) in declaration_names
        )
    ]

    register_terms = [
        _normalize_exists_register(term)
        for term in _exists_terms(case.exists)
        if re.fullmatch(r"\d+:x\d+\s*=.*", term)
    ]
    memory_terms: list[str] = []
    for name, width in sorted(widths.items()):
        observed_offsets = {int(value) for value in observed.get(name, ())}
        final_image = {
            int(offset): int(value) & 0xFF
            for offset, value in dict(final.get(name, {})).items()
        }
        for index in sorted({offset // width for offset in observed_offsets}):
            offsets = set(range(index * width, (index + 1) * width))
            if not offsets <= observed_offsets:
                raise HerdProjectionError(
                    f"typed final observation for {name}[{index}] covers only part of the object"
                )
            if not offsets <= set(final_image):
                raise HerdProjectionError(
                    f"typed final image for {name}[{index}] is incomplete"
                )
            value = _bytes_value(final_image[offset] for offset in sorted(offsets))
            memory_terms.append(
                f"{name}[{index}]={_memory_literal(value, width)}"
            )
    terms = [*register_terms, *memory_terms]
    if not terms:
        raise HerdProjectionError("typed projection has no exists outcome")
    return replace(
        case,
        init_lines=[*declarations, *retained],
        exists="(" + " /\\ ".join(terms) + ")",
    )


def _exists_terms(exists: str) -> list[str]:
    text = exists.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return [term.strip() for term in re.split(r"\s*/\\\s*", text) if term.strip()]


def _bytes_value(values: Sequence[int] | Any) -> int:
    return sum(
        (int(value) & 0xFF) << (8 * index)
        for index, value in enumerate(values)
    )


def _normalize_event_literals(event: LitmusEvent) -> LitmusEvent:
    match = re.fullmatch(
        r"(\s*li\s+x\d+\s*,\s*)(-?(?:0x[0-9a-fA-F]+|\d+))(\s*)",
        event.instruction,
    )
    if match is None:
        return event
    value = int(match.group(2), 0)
    return replace(
        event,
        instruction=f"{match.group(1)}{_register_literal(value)}{match.group(3)}",
    )


def _normalize_register_assignment(term: str) -> str:
    match = re.fullmatch(
        r"(\d+:x\d+\s*=\s*)(-?(?:0x[0-9a-fA-F]+|\d+))",
        term,
    )
    if match is None:
        return term
    return f"{match.group(1)}{_register_literal(int(match.group(2), 0))}"


def _normalize_exists_register(term: str) -> str:
    match = re.fullmatch(
        r"(\d+:x\d+\s*=\s*)(-?(?:0x[0-9a-fA-F]+|\d+))",
        term,
    )
    if match is None:
        raise HerdProjectionError(f"cannot parse register outcome {term!r}")
    return f"{match.group(1)}{_register_literal(int(match.group(2), 0))}"


def crosscheck_vector_projection(
    case: LitmusCaseIR,
    expansion: Any,
    embedded: Any,
    *,
    max_projections: int = 64,
    timeout: int = 30,
    capabilities: HerdCapabilities | None = None,
) -> dict[str, Any]:
    bundle = build_scalar_projections(
        case, expansion, max_projections=max_projections
    )
    caps = capabilities or (
        probe_case_herd_capabilities(
            case,
            mixed_size=bundle.storage_mode == "mixed",
        )
        if bundle.status == "ready"
        else _unprobed_case_capabilities(bundle.reason)
    )
    base = {
        "schema": "litmus-link.vector-herd-reference.v1",
        "capabilities": _case_capabilities(case, caps),
        "projection": bundle.to_json(),
    }
    if bundle.status != "ready":
        return {
            **base,
            "status": "external_unsupported",
            "verdict": "unknown",
            "allowed": None,
            "reason": bundle.reason,
            "results": [],
        }
    if not caps.scalar.supported:
        return {
            **base,
            "status": "external_unsupported",
            "verdict": "unknown",
            "allowed": None,
            "reason": caps.scalar.reason,
            "results": [],
        }
    if bundle.storage_mode == "mixed" and not caps.mixed_size.supported:
        return {
            **base,
            "status": "external_unsupported",
            "verdict": "unknown",
            "allowed": None,
            "reason": caps.mixed_size.reason,
            "results": [],
        }

    amo_requirements = _case_amo_requirements(case)
    unsupported_amo = [
        (operation, width, ordering, caps.amo_capability(operation, width, ordering))
        for operation, width, ordering in sorted(amo_requirements)
        if not caps.amo_capability(operation, width, ordering).supported
    ]
    if unsupported_amo:
        operation, width, ordering, capability = unsupported_amo[0]
        return {
            **base,
            "status": "external_unsupported",
            "verdict": "unknown",
            "allowed": None,
            "reason": (
                f"herd7 cannot evaluate {operation}.{width * 8}.{ordering}: "
                f"{capability.reason}"
            ),
            "results": [],
        }
    if bundle.storage_mode == "mixed" and amo_requirements and not caps.mixed_amo.supported:
        return {
            **base,
            "status": "external_unsupported",
            "verdict": "unknown",
            "allowed": None,
            "reason": caps.mixed_amo.reason,
            "results": [],
        }

    results: list[dict[str, Any]] = []
    for projection in bundle.projections:
        try:
            verdict = _judge_projection(
                projection.source, projection.variants, timeout
            )
        except (OSError, toolchain.ToolchainError) as exc:
            return {
                **base,
                "status": "external_unsupported",
                "verdict": "unknown",
                "allowed": None,
                "reason": _stable_probe_reason(str(exc)),
                "results": results,
            }
        result = {
            **projection.to_json(),
            "status": (
                "verified"
                if verdict.allowed is not None
                else "external_unsupported"
            ),
            "verdict": verdict.outcome,
            "allowed": verdict.allowed,
            "observation": verdict.observation,
            "positive": verdict.positive,
            "negative": verdict.negative,
            "states": verdict.states,
            "condition": verdict.condition,
        }
        results.append(result)
        if verdict.allowed is None:
            return {
                **base,
                "status": "external_unsupported",
                "verdict": "unknown",
                "allowed": None,
                "reason": "herd7 did not return a verdict for a scalar projection",
                "results": results,
            }

    external_allowed = any(bool(result["allowed"]) for result in results)
    external_verdict = "observable" if external_allowed else "forbidden"
    if embedded.status != "verified":
        status = "external_only"
        reason = (
            "herd7 completed the scalar projection oracle, but the embedded "
            f"result is {embedded.status}."
        )
    elif bool(embedded.allowed) != external_allowed:
        if bundle.exact:
            status = "conflict"
            reason = "Embedded RVWMO and exact herd7 scalar projection disagree."
        else:
            status = "advisory_disagree"
            reason = (
                "Embedded RVWMO and the ordered fixed-sequence advisory projection disagree; "
                "the advisory result does not override embedded semantics."
            )
    else:
        status = "agree" if bundle.exact else "advisory_agree"
        reason = (
            "Embedded RVWMO and exact herd7 scalar projection agree."
            if bundle.exact
            else "Embedded RVWMO and the ordered fixed-sequence advisory projection agree."
        )
    return {
        **base,
        "status": status,
        "verdict": external_verdict,
        "allowed": external_allowed,
        "reason": reason,
        "results": results,
    }


def _case_amo_requirements(
    case: LitmusCaseIR,
) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        sorted(
            {
                (
                    str(event.amo_op),
                    int(event.amo_width_bytes or 0),
                    str(event.amo_ordering),
                )
                for event in case.events()
                if event.kind == "amo"
            }
        )
    )


def _unprobed_case_capabilities(reason: str) -> HerdCapabilities:
    capability = HerdCapability(
        False,
        f"Capability probes skipped because no scalar projection was built: {reason}",
    )
    return HerdCapabilities(
        False,
        capability,
        capability,
        capability,
        {},
        str(toolchain.HERD),
        toolchain.tool_version(str(toolchain.HERD)),
        str(toolchain.RISCV_CAT),
    )


def _case_capabilities(
    case: LitmusCaseIR, capabilities: HerdCapabilities
) -> dict[str, Any]:
    required: dict[str, Any] = {}
    for event in case.events():
        if event.kind != "amo":
            continue
        try:
            parsed = parse_amo_mnemonic(event.instruction)
            operation = event.amo_op or parsed.operation
            width = event.amo_width_bytes or parsed.width_bytes
            ordering = event.amo_ordering or parsed.ordering
        except AmoError:
            continue
        key = _amo_capability_key(operation, width, ordering)
        required[key] = capabilities.amo_capability(
            operation, width, ordering
        ).to_json()
    return {
        "schema": "litmus-link.herd-case-capabilities.v1",
        "available": capabilities.available,
        "tool": {
            "path": capabilities.tool_path,
            "version": capabilities.tool_version,
        },
        "model": capabilities.model_path,
        "scalar": capabilities.scalar.to_json(),
        "mixed_size": capabilities.mixed_size.to_json(),
        "mixed_amo": capabilities.mixed_amo.to_json(),
        "required_amo": dict(sorted(required.items())),
    }


@lru_cache(maxsize=4096)
def _judge_projection(
    source: str, variants: tuple[str, ...], timeout: int
) -> toolchain.HerdVerdict:
    return toolchain.herd_judge(source, timeout=timeout, variants=variants)
