from __future__ import annotations

"""Native scalar litmus generation and RISC-V lowering.

Generation in this module does not invoke diy7/diycross7 and does not read an
existing litmus corpus.  The optional herd7 call is an independent model check
performed only after Litmus-link has constructed the complete test source.
"""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice, product
from math import gcd
from pathlib import Path
from typing import Iterable, Sequence

from .diagram import render_diagram
from .litmus_ir import LitmusCaseIR, LitmusEvent, LitmusRelation, MemoryAccess
from .memory_layout import ALIGNED_LAYOUT, MemoryLayoutConfig
from .native_cycles import (
    EnumerationReport,
    NativeCycle,
    enumerate_relation_cycles,
    enumerate_template_cycles,
    location_ids,
    process_ids,
    validate_cycle,
    vertex_directions,
)
from .native_diy import DiyConfig, enumerate_diy_cycles
from .native_edges import EXTERNAL, LOCAL, READ, SAME, WRITE, NativeEdge, edge_by_label, edge_catalog, edges_for_shape
from .rvwmo_solver import solve_rvwmo
from .toolchain import RISCV_CAT, ToolchainError, herd_judge


class NativeGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativePreset:
    name: str
    description: str
    axes: tuple[str, ...]
    nprocs: int

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "axes": list(self.axes),
            "nprocs": self.nprocs,
        }


NATIVE_PRESETS: dict[str, NativePreset] = {
    "MP": NativePreset("MP", "Message Passing", ("Rfe", "RR", "Fre", "WW"), 2),
    "LB": NativePreset("LB", "Load Buffering", ("Rfe", "RW", "Rfe", "RW"), 2),
    "SB": NativePreset("SB", "Store Buffering", ("Fre", "WR", "Fre", "WR"), 2),
    "WRC": NativePreset("WRC", "Write-Read Causality", ("Rfe", "RW", "Rfe", "RR", "Fre"), 3),
    "RWC": NativePreset("RWC", "Read-Write Causality", ("Rfe", "RR", "Fre", "WR", "Fre"), 3),
    "IRIW": NativePreset("IRIW", "Independent Reads of Independent Writes", ("Rfe", "RR", "Fre", "Rfe", "RR", "Fre"), 4),
    "ISA2": NativePreset("ISA2", "Three-hart causality shape", ("Fre", "WW", "Rfe", "RW", "Rfe", "RR"), 3),
    "R": NativePreset("R", "Read/coherence shape", ("Fre", "WW", "Wse", "WR"), 2),
    "S": NativePreset("S", "Store/coherence shape", ("Rfe", "RW", "Wse", "WW"), 2),
    "CoRR": NativePreset("CoRR", "Single-location read-read coherence shape", ("Rfe", "PosRR", "Fre"), 2),
}

DEFAULT_NATIVE_MECHANISMS = ("po", "fence", "dependency")
NATIVE_ANNOTATIONS = ("P", "Aq", "Rl", "AR")


@dataclass(frozen=True)
class NativeLoweredCase:
    name: str
    litmus: str
    cycle: NativeCycle
    case_ir: LitmusCaseIR
    metadata: dict


def native_catalog() -> dict:
    plain_counts = {
        name: native_template_audit([name], DEFAULT_NATIVE_MECHANISMS)["accepted"]
        for name in NATIVE_PRESETS
    }
    return {
        "presets": {name: preset.to_json() for name, preset in NATIVE_PRESETS.items()},
        "mechanisms": {
            "po": "program-order edges (different and same location)",
            "fence": "all nonempty R/W predecessor/successor subsets relevant to main-memory events",
            "dependency": "address, data, control, and control+fence.i chains",
        },
        "annotations": list(NATIVE_ANNOTATIONS),
        "template_counts": plain_counts,
        "template_counts_all_annotations": {
            name: _annotated_count(native_template_cycles([name], DEFAULT_NATIVE_MECHANISMS)[0], NATIVE_ANNOTATIONS)
            for name in NATIVE_PRESETS
        },
    }


def native_template_cycles(
    presets: Sequence[str],
    mechanisms: Sequence[str] = DEFAULT_NATIVE_MECHANISMS,
    *,
    include_same: bool = True,
) -> tuple[list[NativeCycle], dict]:
    selected_presets = tuple(dict.fromkeys(presets or ("MP",)))
    selected_mechanisms = _validate_mechanisms(mechanisms)
    cycles: list[NativeCycle] = []
    reports: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for name in selected_presets:
        try:
            preset = NATIVE_PRESETS[name]
        except KeyError as exc:
            raise NativeGenerationError(f"unknown native scalar preset: {name}") from exc
        axes: list[NativeEdge | Sequence[NativeEdge]] = []
        for token in preset.axes:
            if token in {"RR", "RW", "WR", "WW"}:
                alternatives = edges_for_shape(token, selected_mechanisms, include_same=include_same)
                if not alternatives:
                    raise NativeGenerationError(f"preset {name} has no legal alternatives for {token}")
                axes.append(alternatives)
            else:
                axes.append(edge_by_label(token))
        family_cycles, report = enumerate_template_cycles(
            name,
            axes,
            max_procs=preset.nprocs,
            exact_procs=True,
        )
        reports.append({"family": name, **report.to_json()})
        for cycle in family_cycles:
            key = (name, cycle.canonical_key)
            if key not in seen:
                seen.add(key)
                cycles.append(cycle)
    cycles.sort(key=lambda cycle: (cycle.family, cycle.canonical_key))
    excluded: Counter[str] = Counter()
    for report in reports:
        excluded.update(report["excluded"])
    audit = {
        "schema": "litmus-link.native-enumeration.v1",
        "mode": "templates",
        "presets": list(selected_presets),
        "mechanisms": list(selected_mechanisms),
        "include_same_location": include_same,
        "candidates": sum(report["candidates"] for report in reports),
        "accepted": len(cycles),
        "duplicate": sum(report["duplicate"] for report in reports),
        "excluded": dict(sorted(excluded.items())),
        "families": reports,
    }
    return cycles, audit


def native_template_audit(
    presets: Sequence[str],
    mechanisms: Sequence[str] = DEFAULT_NATIVE_MECHANISMS,
    *,
    include_same: bool = True,
    annotations: Sequence[str] = ("P",),
) -> dict:
    cycles, base_audit = native_template_cycles(presets, mechanisms, include_same=include_same)
    selected_annotations = _validate_annotations(annotations)
    audit = dict(base_audit)
    audit["annotations"] = list(selected_annotations)
    audit["base_cycles"] = len(cycles)
    audit["accepted"] = _annotated_count(cycles, selected_annotations)
    return audit


def annotated_native_cycles(
    cycles: Sequence[NativeCycle],
    annotations: Sequence[str],
) -> Iterable[NativeCycle]:
    modes = _validate_annotations(annotations)
    for cycle in cycles:
        symmetries = _edge_symmetries(cycle)
        for assignment in product(modes, repeat=cycle.size):
            if assignment != min(_rotate(assignment, offset) for offset in symmetries):
                continue
            yield NativeCycle(cycle.edges, cycle.family, tuple(assignment)).canonical()


def native_relation_cycles(
    *,
    mechanisms: Sequence[str] = ("communication", "po", "fence", "dependency"),
    include_same: bool = True,
    include_internal: bool = True,
    min_size: int = 2,
    max_size: int = 4,
    max_procs: int = 2,
    exact_procs: bool = False,
    max_accesses_per_proc: int | None = None,
) -> list[NativeCycle]:
    domain = edge_catalog(tuple(dict.fromkeys(mechanisms)), include_same, include_internal)
    return list(
        enumerate_relation_cycles(
            domain,
            min_size=min_size,
            max_size=max_size,
            max_procs=max_procs,
            exact_procs=exact_procs,
            max_accesses_per_proc=max_accesses_per_proc,
        )
    )


def lower_native_cycle(
    cycle: NativeCycle,
    *,
    realdep: bool = False,
    memory_layout: MemoryLayoutConfig = ALIGNED_LAYOUT,
) -> NativeLoweredCase:
    decision = validate_cycle(cycle.edges)
    if not decision.accepted:
        raise NativeGenerationError(f"cannot lower invalid native cycle: {decision.reason}")
    directions = vertex_directions(cycle.edges)
    procs = process_ids(cycle.edges)
    locations = location_ids(cycle.edges)
    nvertices = len(cycle.edges)
    annotations = _cycle_annotations(cycle)
    if not memory_layout.is_aligned and any(annotation != "P" for annotation in annotations):
        raise NativeGenerationError("misaligned layouts support ordinary scalar load/store events only")
    proc_orders = _program_orders(cycle.edges, procs)
    write_values, read_values, final_values, rf_source = _memory_values(cycle.edges, directions, locations)
    location_names = {index: _location_name(index) for index in sorted(set(locations))}
    event_ordinals: dict[int, int] = {}
    for location in sorted(set(locations)):
        for ordinal, vertex in enumerate(
            vertex for vertex in range(nvertices) if locations[vertex] == location
        ):
            event_ordinals[vertex] = ordinal
    memory_accesses = {
        vertex: memory_layout.access_for(
            location_names[locations[vertex]], event_ordinals[vertex]
        )
        for vertex in range(nvertices)
    }
    if memory_layout.is_aligned:
        actual_write_values = write_values
        actual_read_values = read_values
        final_byte_values: dict[str, int] = {}
    else:
        actual_write_values, actual_read_values, final_byte_values = _mixed_values(
            directions,
            locations,
            memory_accesses,
            write_values,
            rf_source,
            location_names,
        )

    address_regs: dict[tuple[int, int], str] = {}
    memory_regs: dict[int, str] = {}
    init_lines: list[str] = [
        (
            f"{location_names[index]}=0;"
            if memory_layout.is_aligned
            else f"uint8_t {location_names[index]}[128];"
        )
        for index in sorted(location_names)
    ]
    hart_events: list[list[LitmusEvent]] = []
    instruction_rows: list[list[str]] = []

    for proc in range(max(procs) + 1):
        vertices = proc_orders[proc]
        register_pool = _RegisterPool()
        for location in sorted({locations[vertex] for vertex in vertices}):
            register = register_pool.address()
            address_regs[(proc, location)] = register
            init_lines.append(f"{proc}:{register}={location_names[location]};")
        for vertex in vertices:
            register = register_pool.value()
            memory_regs[vertex] = register
            if directions[vertex] == WRITE:
                init_lines.append(f"{proc}:{register}={_hex_value(actual_write_values[vertex])};")

        events: list[LitmusEvent] = []
        instructions: list[str] = []
        for position, vertex in enumerate(vertices):
            location = locations[vertex]
            address = address_regs[(proc, location)]
            register = memory_regs[vertex]
            annotation = annotations[vertex]
            access = memory_accesses[vertex]
            if directions[vertex] == READ:
                instruction = _load_instruction(register, address, annotation, access)
                kind = "load" if annotation == "P" else "amo"
                value = _hex_value(actual_read_values[vertex])
            else:
                instruction = _store_instruction(register, address, annotation, access)
                kind = "store" if annotation == "P" else "amo"
                value = _hex_value(actual_write_values[vertex])
            events.append(
                LitmusEvent(
                    event_id=f"v{vertex}",
                    hart=proc,
                    kind=kind,
                    instruction=instruction,
                    location=location_names[location],
                    register=register,
                    value=value,
                    role="cycle-event",
                    memory_access=access,
                )
            )
            instructions.append(instruction)
            outgoing = cycle.edges[vertex]
            target = (vertex + 1) % nvertices
            if outgoing.scope == LOCAL:
                if target not in vertices or position + 1 >= len(vertices) or vertices[position + 1] != target:
                    raise NativeGenerationError(
                        f"local edge {outgoing.label} is inconsistent with hart P{proc} program order"
                    )
                middle = _lower_local_edge(
                    outgoing,
                    proc=proc,
                    source_register=register,
                    target_address=address_regs[(proc, locations[target])],
                    target_data=memory_regs[target],
                    registers=register_pool,
                    label_index=vertex,
                    realdep=realdep,
                )
                for middle_index, (middle_kind, middle_instruction, role) in enumerate(middle):
                    events.append(
                        LitmusEvent(
                            event_id=f"v{vertex}_m{middle_index}",
                            hart=proc,
                            kind=middle_kind,
                            instruction=middle_instruction,
                            role=role,
                        )
                    )
                    instructions.append(middle_instruction)
        hart_events.append(events)
        instruction_rows.append(instructions)

    relations = [
        LitmusRelation(
            src=f"v{index}",
            dst=f"v{(index + 1) % nvertices}",
            kind=edge.relation,
            label=edge.label,
            local=edge.scope == LOCAL,
        )
        for index, edge in enumerate(cycle.edges)
    ]
    exists_terms = [
        f"{procs[vertex]}:{memory_regs[vertex]}={_hex_value(actual_read_values[vertex])}"
        for vertex in range(nvertices)
        if directions[vertex] == READ
    ]
    if memory_layout.is_aligned:
        exists_terms.extend(
            f"{location_names[location]}={value}" for location, value in sorted(final_values.items())
        )
    else:
        exists_terms.extend(
            f"{location}=0x{value:02x}" for location, value in sorted(final_byte_values.items())
        )
    if not exists_terms:
        raise NativeGenerationError("native cycle has no observable read or final-memory outcome")
    exists = "(" + " /\\ ".join(exists_terms) + ")"
    base_name = _native_name(cycle)
    name = base_name if memory_layout.is_aligned else f"{base_name}_{_safe_name(memory_layout.id)}"
    cycle_text = " ".join(cycle.labels)
    case_ir = LitmusCaseIR(
        name=name,
        display_name=name,
        combination_name=name,
        skeleton=cycle.family or "Native",
        variant=f"native-exhaustive_{memory_layout.id}",
        cycle=cycle_text,
        init_lines=init_lines,
        harts=hart_events,
        relations=relations,
        exists=exists,
        expected_outcome="solver_required",
        model="rvwmo-herd7",
        description=(
            f"Native exhaustive scalar cycle ({cycle.family or 'unclassified'}), "
            f"memory layout {memory_layout.id}: {cycle_text}"
        ),
        tags=[
            "native",
            "scalar",
            "rvwmo",
            cycle.family or "unclassified",
            memory_layout.mode,
            memory_layout.boundary,
            "no-mag" if not memory_layout.is_aligned else "aligned",
        ],
    )
    litmus = _render_native_litmus(name, cycle_text, init_lines, instruction_rows, exists)
    return NativeLoweredCase(
        name=name,
        litmus=litmus,
        cycle=cycle,
        case_ir=case_ir,
        metadata={
            "process_ids": list(procs),
            "location_ids": list(locations),
            "write_values": {str(key): value for key, value in sorted(write_values.items())},
            "read_values": {str(key): value for key, value in sorted(read_values.items())},
            "actual_write_values": {str(key): value for key, value in sorted(actual_write_values.items())},
            "actual_read_values": {str(key): value for key, value in sorted(actual_read_values.items())},
            "final_values": {location_names[key]: value for key, value in sorted(final_values.items())},
            "final_byte_values": dict(sorted(final_byte_values.items())),
            "annotations": list(annotations),
            "real_dependencies": realdep,
            "memory_layout": memory_layout.to_json(),
        },
    )


def generate_native_templates(
    *,
    out_dir: Path,
    presets: Sequence[str],
    mechanisms: Sequence[str] = DEFAULT_NATIVE_MECHANISMS,
    include_same: bool = True,
    annotations: Sequence[str] = NATIVE_ANNOTATIONS,
    limit: int | None = None,
    judge: bool = True,
    solver_backend: str = "embedded",
    diagrams: bool = False,
    timeout: int = 180,
    memory_layouts: Sequence[MemoryLayoutConfig] = (ALIGNED_LAYOUT,),
) -> dict:
    base_cycles, audit = native_template_cycles(presets, mechanisms, include_same=include_same)
    selected_annotations = _validate_annotations(annotations)
    selected_layouts = _validate_memory_layouts(memory_layouts)
    annotated_count = _annotated_count(base_cycles, selected_annotations)
    available, excluded_atomic = _memory_layout_count(
        len(base_cycles), annotated_count, selected_annotations, selected_layouts
    )
    audit = dict(audit)
    audit["annotations"] = list(selected_annotations)
    audit["base_cycles"] = len(base_cycles)
    audit["annotated_cycles"] = annotated_count
    audit["memory_layouts"] = [layout.to_json() for layout in selected_layouts]
    audit["excluded_misaligned_atomic_annotations"] = excluded_atomic
    audit["accepted"] = available
    return _write_native_cases(
        annotated_native_cycles(base_cycles, selected_annotations),
        out_dir,
        audit=audit,
        available=available,
        limit=limit,
        judge=judge,
        solver_backend=solver_backend,
        diagrams=diagrams,
        timeout=timeout,
        memory_layouts=selected_layouts,
    )


def generate_native_relations(
    *,
    out_dir: Path,
    mechanisms: Sequence[str] = ("communication", "po", "fence", "dependency"),
    include_same: bool = True,
    include_internal: bool = True,
    min_size: int = 2,
    max_size: int = 4,
    max_procs: int = 2,
    exact_procs: bool = False,
    max_accesses_per_proc: int | None = None,
    annotations: Sequence[str] = ("P",),
    limit: int | None = None,
    judge: bool = True,
    solver_backend: str = "embedded",
    diagrams: bool = False,
    timeout: int = 180,
    memory_layouts: Sequence[MemoryLayoutConfig] = (ALIGNED_LAYOUT,),
) -> dict:
    base_cycles = native_relation_cycles(
        mechanisms=mechanisms,
        include_same=include_same,
        include_internal=include_internal,
        min_size=min_size,
        max_size=max_size,
        max_procs=max_procs,
        exact_procs=exact_procs,
        max_accesses_per_proc=max_accesses_per_proc,
    )
    audit = {
        "schema": "litmus-link.native-enumeration.v1",
        "mode": "cycles",
        "mechanisms": list(mechanisms),
        "include_same_location": include_same,
        "include_internal_communication": include_internal,
        "min_size": min_size,
        "max_size": max_size,
        "max_procs": max_procs,
        "exact_procs": exact_procs,
        "annotations": list(_validate_annotations(annotations)),
        "base_cycles": len(base_cycles),
    }
    selected_annotations = _validate_annotations(annotations)
    selected_layouts = _validate_memory_layouts(memory_layouts)
    annotated_count = _annotated_count(base_cycles, selected_annotations)
    available, excluded_atomic = _memory_layout_count(
        len(base_cycles), annotated_count, selected_annotations, selected_layouts
    )
    audit["annotated_cycles"] = annotated_count
    audit["memory_layouts"] = [layout.to_json() for layout in selected_layouts]
    audit["excluded_misaligned_atomic_annotations"] = excluded_atomic
    audit["accepted"] = available
    return _write_native_cases(
        annotated_native_cycles(base_cycles, selected_annotations),
        out_dir,
        audit=audit,
        available=available,
        limit=limit,
        judge=judge,
        solver_backend=solver_backend,
        diagrams=diagrams,
        timeout=timeout,
        memory_layouts=selected_layouts,
    )


def generate_native_diy(
    *,
    out_dir: Path,
    config: DiyConfig,
    annotations: Sequence[str] = ("P",),
    limit: int | None = None,
    judge: bool = True,
    solver_backend: str = "embedded",
    diagrams: bool = False,
    timeout: int = 180,
    memory_layouts: Sequence[MemoryLayoutConfig] = (ALIGNED_LAYOUT,),
) -> dict:
    base_cycles, audit = enumerate_diy_cycles(config)
    selected_annotations = _validate_annotations(annotations)
    selected_layouts = _validate_memory_layouts(memory_layouts)
    annotated_count = _annotated_count(base_cycles, selected_annotations)
    available, excluded_atomic = _memory_layout_count(
        len(base_cycles), annotated_count, selected_annotations, selected_layouts
    )
    audit = dict(audit)
    audit["annotations"] = list(selected_annotations)
    audit["base_cycles"] = len(base_cycles)
    audit["annotated_cycles"] = annotated_count
    audit["memory_layouts"] = [layout.to_json() for layout in selected_layouts]
    audit["excluded_misaligned_atomic_annotations"] = excluded_atomic
    audit["accepted"] = available
    return _write_native_cases(
        annotated_native_cycles(base_cycles, selected_annotations),
        out_dir,
        audit=audit,
        available=available,
        limit=limit,
        judge=judge,
        solver_backend=solver_backend,
        diagrams=diagrams,
        timeout=timeout,
        realdep=config.realdep,
        memory_layouts=selected_layouts,
    )


def _write_native_cases(
    cycles: Iterable[NativeCycle],
    out_dir: Path,
    *,
    audit: dict,
    available: int,
    limit: int | None,
    judge: bool,
    solver_backend: str,
    diagrams: bool,
    timeout: int,
    realdep: bool = False,
    memory_layouts: Sequence[MemoryLayoutConfig] = (ALIGNED_LAYOUT,),
) -> dict:
    if limit is not None and limit < 1:
        raise NativeGenerationError("native generation limit must be at least 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    lowered_domain = _cycles_with_layouts(cycles, memory_layouts)
    selected = islice(lowered_domain, limit) if limit is not None else lowered_domain
    filenames: list[str] = []
    verdicts: Counter[str] = Counter()
    backend = _validate_solver_backend(solver_backend)
    for cycle, memory_layout in selected:
        case = lower_native_cycle(cycle, realdep=realdep, memory_layout=memory_layout)
        solver = _judge_native(case, judge=judge, backend=backend, timeout=timeout)
        verdicts[solver["status"]] += 1
        (out_dir / f"{case.name}.litmus").write_text(case.litmus, encoding="utf-8")
        (out_dir / f"{case.name}.solver.json").write_text(
            json.dumps(solver, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        diagram = render_diagram(case.case_ir, solver, out_dir).summary if diagrams else None
        meta = {
            "schema": "litmus-link.native-scalar-meta.v1",
            "name": case.name,
            "architecture": "RISCV",
            "requires": ["RV64I", *(["A"] if any(mode != "P" for mode in _cycle_annotations(case.cycle)) else [])],
            "cycle": " ".join(case.cycle.labels),
            "exists": case.case_ir.exists,
            "nprocs": len(case.case_ir.harts),
            "generator": {"engine": "litmus-link-native", "audit": audit},
            "generated_from": "litmus-link-native",
            "native": case.metadata,
            "case_ir": case.case_ir.to_json(),
            "solver": solver,
        }
        if diagram is not None:
            meta["diagram"] = diagram
        (out_dir / f"{case.name}.meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        filenames.append(f"{case.name}.litmus")
    (out_dir / "@all").write_text("\n".join(filenames) + ("\n" if filenames else ""), encoding="utf-8")
    report = {
        "schema": "litmus-link.native-generation.v1",
        "architecture": "RISCV",
        "generator": {"engine": "litmus-link-native"},
        "audit": audit,
        "available_litmus": available,
        "generated_litmus": len(filenames),
        "generation_limit": limit,
        "generation_limited": len(filenames) < available,
        "judge": judge,
        "solver_backend": backend if judge else "none",
        "verdicts": dict(sorted(verdicts.items())),
        "output": str(out_dir),
        "atfile": str(out_dir / "@all"),
    }
    (out_dir / "generation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "audit-report.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _cycles_with_layouts(
    cycles: Iterable[NativeCycle],
    memory_layouts: Sequence[MemoryLayoutConfig],
) -> Iterable[tuple[NativeCycle, MemoryLayoutConfig]]:
    for cycle in cycles:
        plain = all(annotation == "P" for annotation in _cycle_annotations(cycle))
        for layout in memory_layouts:
            if layout.is_aligned or plain:
                yield cycle, layout


def _validate_memory_layouts(
    memory_layouts: Sequence[MemoryLayoutConfig],
) -> tuple[MemoryLayoutConfig, ...]:
    selected = tuple(dict.fromkeys(memory_layouts or (ALIGNED_LAYOUT,)))
    if not all(isinstance(layout, MemoryLayoutConfig) for layout in selected):
        raise NativeGenerationError("memory_layouts must contain MemoryLayoutConfig values")
    return selected


def _memory_layout_count(
    base_cycle_count: int,
    annotated_cycle_count: int,
    annotations: Sequence[str],
    memory_layouts: Sequence[MemoryLayoutConfig],
) -> tuple[int, int]:
    aligned = sum(layout.is_aligned for layout in memory_layouts)
    misaligned = len(memory_layouts) - aligned
    plain_cycles = base_cycle_count if "P" in annotations else 0
    accepted = annotated_cycle_count * aligned + plain_cycles * misaligned
    excluded_atomic = (annotated_cycle_count - plain_cycles) * misaligned
    if accepted == 0:
        raise NativeGenerationError(
            "selected misaligned layouts require the ordinary P annotation"
        )
    return accepted, excluded_atomic


def _program_orders(edges: Sequence[NativeEdge], procs: Sequence[int]) -> list[list[int]]:
    vertices_by_proc: dict[int, set[int]] = defaultdict(set)
    outgoing: dict[int, int] = {}
    indegree: dict[int, int] = defaultdict(int)
    for vertex, proc in enumerate(procs):
        vertices_by_proc[proc].add(vertex)
    for vertex, edge in enumerate(edges):
        if edge.scope != LOCAL:
            continue
        target = (vertex + 1) % len(edges)
        if procs[vertex] != procs[target]:
            raise NativeGenerationError(f"local edge {edge.label} crosses harts")
        outgoing[vertex] = target
        indegree[target] += 1
    orders: list[list[int]] = []
    for proc in range(max(procs) + 1):
        vertices = vertices_by_proc[proc]
        ready = sorted(vertex for vertex in vertices if indegree[vertex] == 0)
        order: list[int] = []
        while ready:
            vertex = ready.pop(0)
            order.append(vertex)
            target = outgoing.get(vertex)
            if target is not None:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(order) != len(vertices):
            raise NativeGenerationError(f"local program-order graph for P{proc} is cyclic")
        orders.append(order)
    return orders


def _memory_values(
    edges: Sequence[NativeEdge],
    directions: Sequence[str],
    locations: Sequence[int],
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    writes = [vertex for vertex, direction in enumerate(directions) if direction == WRITE]
    reads = [vertex for vertex, direction in enumerate(directions) if direction == READ]
    rf_source: dict[int, int] = {}
    co_edges: dict[int, set[int]] = defaultdict(set)
    indegree: dict[int, int] = defaultdict(int)

    def add_co(before: int, after: int) -> None:
        if before == after or after in co_edges[before]:
            return
        if directions[before] != WRITE or directions[after] != WRITE:
            raise NativeGenerationError("coherence constraint does not connect two writes")
        if locations[before] != locations[after]:
            raise NativeGenerationError("coherence constraint crosses locations")
        co_edges[before].add(after)
        indegree[after] += 1

    for vertex, edge in enumerate(edges):
        target = (vertex + 1) % len(edges)
        if edge.relation == "rf":
            if target in rf_source and rf_source[target] != vertex:
                raise NativeGenerationError(f"read v{target} has multiple rf sources")
            rf_source[target] = vertex
        elif edge.relation == "co":
            add_co(vertex, target)

    for vertex, edge in enumerate(edges):
        if edge.relation != "fr":
            continue
        target = (vertex + 1) % len(edges)
        source = rf_source.get(vertex)
        if source is not None:
            add_co(source, target)

    write_values: dict[int, int] = {}
    final_values: dict[int, int] = {}
    for location in sorted(set(locations)):
        location_writes = sorted(vertex for vertex in writes if locations[vertex] == location)
        ready = sorted(vertex for vertex in location_writes if indegree[vertex] == 0)
        order: list[int] = []
        while ready:
            vertex = ready.pop(0)
            order.append(vertex)
            for target in sorted(co_edges.get(vertex, ())):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(order) != len(location_writes):
            raise NativeGenerationError(f"coherence constraints for location {location} are cyclic")
        for value, vertex in enumerate(order, start=1):
            write_values[vertex] = value
        if order:
            final_values[location] = write_values[order[-1]]
    read_values = {
        vertex: write_values[rf_source[vertex]] if vertex in rf_source else 0
        for vertex in reads
    }
    return write_values, read_values, final_values, rf_source


def _mixed_values(
    directions: Sequence[str],
    locations: Sequence[int],
    accesses: dict[int, MemoryAccess],
    write_ordinals: dict[int, int],
    rf_source: dict[int, int],
    location_names: dict[int, str],
) -> tuple[dict[int, int], dict[int, int], dict[str, int]]:
    byte_tags = {
        vertex: _write_byte_tag(ordinal)
        for vertex, ordinal in write_ordinals.items()
    }
    write_values = {
        vertex: _repeat_byte(byte_tags[vertex], accesses[vertex].size_bytes)
        for vertex in write_ordinals
    }
    read_values: dict[int, int] = {}
    for vertex, direction in enumerate(directions):
        if direction != READ:
            continue
        access = accesses[vertex]
        source = rf_source.get(vertex)
        source_bytes = set(accesses[source].covered_bytes) if source is not None else set()
        source_tag = byte_tags[source] if source is not None else 0
        value = 0
        for byte_index, absolute_byte in enumerate(access.covered_bytes):
            if absolute_byte in source_bytes:
                value |= source_tag << (8 * byte_index)
        read_values[vertex] = value

    final_values: dict[str, int] = {}
    for location in sorted(set(locations)):
        writes = sorted(
            (
                vertex
                for vertex, direction in enumerate(directions)
                if direction == WRITE and locations[vertex] == location
            ),
            key=lambda vertex: write_ordinals[vertex],
        )
        image: dict[int, int] = {}
        for vertex in writes:
            for absolute_byte in accesses[vertex].covered_bytes:
                image[absolute_byte] = byte_tags[vertex]
        base = location_names[location]
        final_values.update({f"{base}[{offset}]": value for offset, value in image.items()})

    for vertex, source in rf_source.items():
        overlap = set(accesses[vertex].covered_bytes) & set(accesses[source].covered_bytes)
        if not overlap:
            raise NativeGenerationError(
                f"memory layout removed required rf overlap v{source}->v{vertex}"
            )
    return write_values, read_values, final_values


def _write_byte_tag(ordinal: int) -> int:
    # Native cycle lowering currently emits only a handful of writes per
    # location.  Keep each byte nonzero and unique so target read values select
    # their intended rf source rather than an initial byte.
    if ordinal < 1 or ordinal > 255:
        raise NativeGenerationError("mixed-size lowering supports at most 255 writes per location")
    return ordinal


def _repeat_byte(byte: int, size: int) -> int:
    return sum(byte << (8 * index) for index in range(size))


def _hex_value(value: int) -> str:
    return f"0x{value:x}"


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


class _RegisterPool:
    def __init__(self) -> None:
        self._value = iter([f"x{index}" for index in range(5, 16)])
        self._address = iter([f"x{index}" for index in range(20, 29)])
        self._temp = iter([f"x{index}" for index in range(16, 20)] + ["x29", "x30", "x31"])

    def value(self) -> str:
        return _next_register(self._value, "value")

    def address(self) -> str:
        return _next_register(self._address, "address")

    def temp(self) -> str:
        return _next_register(self._temp, "temporary")


def _next_register(pool, purpose: str) -> str:
    try:
        return next(pool)
    except StopIteration as exc:
        raise NativeGenerationError(f"native lowering exhausted {purpose} registers") from exc


def _lower_local_edge(
    edge: NativeEdge,
    *,
    proc: int,
    source_register: str,
    target_address: str,
    target_data: str,
    registers: _RegisterPool,
    label_index: int,
    realdep: bool,
) -> list[tuple[str, str, str]]:
    if edge.relation in {"po", "rf", "fr", "co"}:
        return []
    if edge.relation == "fence":
        return [("fence", edge.lowering, "fence")]
    if edge.relation != "dependency":
        raise NativeGenerationError(f"no native lowering for local edge {edge.label}")
    if edge.mechanism == "addr":
        temporary = registers.temp()
        calculation = (
            f"andi {temporary},{source_register},128"
            if realdep
            else f"xor {temporary},{source_register},{source_register}"
        )
        return [
            ("dep", calculation, "addr-dep"),
            ("dep", f"add {target_address},{target_address},{temporary}", "addr-dep"),
        ]
    if edge.mechanism == "data":
        temporary = registers.temp()
        calculation = (
            f"andi {temporary},{source_register},128"
            if realdep
            else f"xor {temporary},{source_register},{source_register}"
        )
        return [
            ("dep", calculation, "data-dep"),
            ("dep", f"add {target_data},{target_data},{temporary}", "data-dep"),
        ]
    if edge.mechanism in {"ctrl", "ctrl_fencei"}:
        label = f"LC{proc}_{label_index}"
        out = [
            ("branch", f"bne {source_register},x0,{label}", "ctrl-dep"),
            ("label", f"{label}:", "ctrl-dep"),
        ]
        if edge.mechanism == "ctrl_fencei":
            out.append(("fence", "fence.i", "ctrl-fencei"))
        return out
    raise NativeGenerationError(f"unknown dependency mechanism: {edge.mechanism}")


def _render_native_litmus(
    name: str,
    cycle: str,
    init_lines: Sequence[str],
    harts: Sequence[Sequence[str]],
    exists: str,
) -> str:
    width = max(18, max((len(instruction) for hart in harts for instruction in hart), default=0) + 2)
    rows = [" " + " | ".join(f"P{proc:<{width - 1}}" for proc in range(len(harts))) + ";"]
    for row in range(max((len(hart) for hart in harts), default=0)):
        rows.append(
            " "
            + " | ".join(
                f"{(hart[row] if row < len(hart) else ''):<{width}}" for hart in harts
            )
            + ";"
        )
    return "\n".join(
        [
            f"RISCV {name}",
            f'"{cycle}"',
            f"Cycle={cycle}",
            "Generator=Litmus-link-native",
            "{",
            *init_lines,
            "}",
            *rows,
            "exists",
            exists,
            "",
        ]
    )


def _judge_native(
    case: NativeLoweredCase,
    *,
    judge: bool,
    backend: str,
    timeout: int,
) -> dict:
    if not judge:
        return {
            "schema": "litmus-link.native-solver.v1",
            "status": "unchecked",
            "tool": "herd7",
            "model": "riscv.cat",
            "allowed": None,
            "verdict": "unchecked",
            "reason": "RVWMO outcome checking disabled by the user",
        }
    if backend == "embedded":
        return solve_rvwmo(
            case.case_ir,
            timeout_seconds=float(timeout),
        ).to_json()
    if backend == "crosscheck":
        embedded = solve_rvwmo(
            case.case_ir,
            timeout_seconds=float(timeout),
        ).to_json()
        external = _judge_native_herd7(case, timeout=timeout)
        agree = (
            embedded.get("status") == "verified"
            and external.get("status") == "verified"
            and embedded.get("allowed") == external.get("allowed")
        )
        if not agree:
            return {
                "schema": "litmus-link.native-solver.v2",
                "status": "conflict",
                "tool": "litmus-link-rvwmo+herd7",
                "model": "riscv.cat",
                "backend": "crosscheck",
                "allowed": None,
                "verdict": "conflict",
                "reason": "Embedded RVWMO and herd7 did not produce the same verified verdict.",
                "embedded": embedded,
                "herd7": external,
            }
        return {
            "schema": "litmus-link.native-solver.v2",
            "status": "verified",
            "tool": "litmus-link-rvwmo+herd7",
            "model": "riscv.cat",
            "backend": "crosscheck",
            "allowed": embedded["allowed"],
            "verdict": embedded["verdict"],
            "reason": "Embedded RVWMO and herd7 agree.",
            "embedded": embedded,
            "herd7": external,
        }
    return _judge_native_herd7(case, timeout=timeout)


def _judge_native_herd7(case: NativeLoweredCase, *, timeout: int) -> dict:
    try:
        mixed = any(
            event.memory_access is not None
            and event.memory_access.atomicity_model == "byte_level_no_mag"
            for event in case.case_ir.events()
        )
        verdict = herd_judge(
            case.litmus,
            timeout=timeout,
            variants=("mixed", "unaligned") if mixed else (),
        )
    except ToolchainError as exc:
        raise NativeGenerationError(f"herd7 cross-check failed for {case.name}: {exc}") from exc
    status = "verified" if verdict.outcome in {"observable", "forbidden"} else "unknown"
    return {
        "schema": "litmus-link.native-solver.v1",
        "status": status,
        "tool": "herd7",
        "model": "riscv.cat",
        "model_path": str(RISCV_CAT),
        "variants": ["mixed", "unaligned"] if mixed else [],
        "allowed": verdict.allowed,
        "verdict": verdict.outcome,
        "observation": verdict.observation,
        "positive": verdict.positive,
        "negative": verdict.negative,
        "states": verdict.states,
        "condition": verdict.condition,
        "raw_output": verdict.raw,
    }


def _validate_solver_backend(value: str) -> str:
    backend = str(value or "embedded").lower()
    if backend not in {"embedded", "herd7", "crosscheck"}:
        raise NativeGenerationError(f"unknown native solver backend: {value}")
    return backend


def _native_name(cycle: NativeCycle) -> str:
    family = cycle.family or "Cycle"
    readable = "+".join(_short_edge(edge) for edge in cycle.edges if edge.scope == LOCAL)
    digest = hashlib.sha256("\0".join(cycle.labels).encode("utf-8")).hexdigest()[:12]
    readable = readable[:96].strip("+") or "comm"
    return f"NATIVE_{family}_{readable}_{digest}"


def _short_edge(edge: NativeEdge) -> str:
    return edge.label.replace("Fence.", "F.").replace("Dp", "")


def _location_name(index: int) -> str:
    alphabet = "xyzabcdefghijklmnopqrstuvw"
    if index < len(alphabet):
        return alphabet[index]
    return f"loc{index}"


def _cycle_annotations(cycle: NativeCycle) -> tuple[str, ...]:
    return cycle.annotations or ("P",) * cycle.size


def _load_instruction(
    destination: str,
    address: str,
    annotation: str,
    access: MemoryAccess,
) -> str:
    if annotation == "P":
        opcode = {1: "lbu", 2: "lhu", 4: "lwu", 8: "ld"}[access.size_bytes]
        return f"{opcode} {destination},{access.offset_bytes}({address})"
    suffix = _amo_suffix(annotation)
    return f"amoor.w{suffix} {destination},x0,({address})"


def _store_instruction(
    data: str,
    address: str,
    annotation: str,
    access: MemoryAccess,
) -> str:
    if annotation == "P":
        opcode = {1: "sb", 2: "sh", 4: "sw", 8: "sd"}[access.size_bytes]
        return f"{opcode} {data},{access.offset_bytes}({address})"
    suffix = _amo_suffix(annotation)
    return f"amoswap.w{suffix} x0,{data},({address})"


def _amo_suffix(annotation: str) -> str:
    try:
        return {"Aq": ".aq", "Rl": ".rl", "AR": ".aq.rl"}[annotation]
    except KeyError as exc:
        raise NativeGenerationError(f"invalid AMO annotation: {annotation}") from exc


def _validate_annotations(annotations: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(annotations or ("P",)))
    unknown = set(selected) - set(NATIVE_ANNOTATIONS)
    if unknown:
        raise NativeGenerationError(f"unknown native annotations: {', '.join(sorted(unknown))}")
    return selected


def _edge_symmetries(cycle: NativeCycle) -> tuple[int, ...]:
    edge_labels = tuple(edge.label for edge in cycle.edges)
    return tuple(
        offset
        for offset in range(cycle.size)
        if _rotate(edge_labels, offset) == edge_labels
    )


def _annotated_count(cycles: Sequence[NativeCycle], annotations: Sequence[str]) -> int:
    modes = _validate_annotations(annotations)
    count = 0
    for cycle in cycles:
        symmetries = _edge_symmetries(cycle)
        count += sum(len(modes) ** gcd(cycle.size, offset) for offset in symmetries) // len(symmetries)
    return count


def _rotate(values: Sequence[str], offset: int) -> tuple[str, ...]:
    values = tuple(values)
    return values[offset:] + values[:offset]


def _validate_mechanisms(mechanisms: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(mechanisms or DEFAULT_NATIVE_MECHANISMS))
    unknown = set(selected) - set(DEFAULT_NATIVE_MECHANISMS)
    if unknown:
        raise NativeGenerationError(f"unknown native mechanisms: {', '.join(sorted(unknown))}")
    return selected
