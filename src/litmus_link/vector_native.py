from __future__ import annotations

"""Relation-cycle driven Vector Litmus generation.

The original Vector profile replaced one scalar endpoint in a handwritten
skeleton.  This module instead starts from the same NativeCycle objects used by
the diy-compatible scalar generator.  Every cycle vertex independently selects
an ISA-legal plain, AMO, or Vector memory operation, while the relation edges
remain the source of fences and dependencies.
"""

import bisect
import json
import random
import re
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .litmus_ir import (
    LitmusCaseIR,
    LitmusEvent,
    MemoryAccess,
    _rebase_vector,
    _scalar_memory_operands,
    _vector_instruction,
    _vector_metadata,
    _vector_setup,
)
from .models import Combination, Decision, GENERATED, GeneratedCase
from .native_cycles import NativeCycle, location_ids
from .native_edges import READ, WRITE
from .native_scalar import (
    DEFAULT_NATIVE_MECHANISMS,
    NATIVE_ANNOTATIONS,
    lower_native_cycle,
    native_template_cycles,
)
from .profiles import (
    NANHU_VECTOR_ATTRIBUTES,
    VECTOR_INDEX_EEWS,
    VECTOR_LENGTHS,
    VECTOR_LMULS,
    VECTOR_MASKS,
    VECTOR_OPS,
    VECTOR_TAILS,
    VECTOR_WIDTHS,
    vector_same_line_footprint,
)
from .renderer import render_ir
from .solver import solve_generated_case


ProgressCallback = Callable[[int, int, str], None]

VECTOR_ALIGNMENTS = (
    "aligned",
    "misalign_same16",
    "misalign_cross16",
    "misalign_cross64",
)


@dataclass(frozen=True)
class EndpointChoice:
    choice_id: str
    category: str
    direction: str
    annotation: str = "P"
    vector_form: str = ""
    params: Mapping[str, str] | None = None

    @property
    def is_vector(self) -> bool:
        return self.category == "vector"

    @property
    def alignment(self) -> str:
        return "aligned"

    def to_json(self) -> dict[str, Any]:
        return {
            "choice_id": self.choice_id,
            "category": self.category,
            "direction": self.direction,
            "annotation": self.annotation,
            "vector_form": self.vector_form or None,
            "params": dict(self.params or {}),
        }


@dataclass(frozen=True)
class VectorAssignment:
    cycle: NativeCycle
    choices: tuple[EndpointChoice, ...]
    alignment: str = "aligned"

    def __post_init__(self) -> None:
        if self.alignment not in VECTOR_ALIGNMENTS:
            raise ValueError(f"unknown Vector assignment alignment: {self.alignment}")
        directions = tuple(edge.src for edge in self.cycle.edges)
        if len(self.choices) != len(directions):
            raise ValueError("Vector assignment must provide one endpoint choice per cycle vertex")
        if any(choice.direction != direction for choice, direction in zip(self.choices, directions)):
            raise ValueError("Vector endpoint choice direction does not match its cycle vertex")
        if not any(choice.is_vector for choice in self.choices):
            raise ValueError("Vector assignment must contain at least one Vector memory endpoint")
        if self.alignment != "aligned":
            illegal = [
                choice.choice_id
                for choice in self.choices
                if not _choice_allowed_alignment(choice, self.alignment)
            ]
            if illegal:
                raise ValueError(
                    "misaligned Vector assignments exclude AMO and byte-sized Vector endpoints: "
                    + ", ".join(illegal)
                )

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.cycle.family,
            self.cycle.canonical_key,
            self.alignment,
            *(choice.choice_id for choice in self.choices),
        )


@dataclass(frozen=True)
class VectorNativeDomain:
    cycles: tuple[NativeCycle, ...]
    read_choices: tuple[EndpointChoice, ...]
    write_choices: tuple[EndpointChoice, ...]
    relation_audit: Mapping[str, Any]
    read_nonvector_choices: int
    write_nonvector_choices: int
    read_plain_choices: int
    write_plain_choices: int
    read_no_amo_choices: int
    write_no_amo_choices: int
    read_misaligned_choices: int
    write_misaligned_choices: int
    alignments: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "VectorNativeDomain":
        skeletons = _selected(payload, "skeletons", ())
        if not skeletons:
            raise ValueError("select at least one Vector skeleton")
        native_skeletons = tuple("CoRR" if skeleton == "Co" else skeleton for skeleton in skeletons)
        mechanisms = _selected(payload, "mechanisms", DEFAULT_NATIVE_MECHANISMS)
        cycles, relation_audit = native_template_cycles(
            native_skeletons,
            mechanisms,
            include_same=bool(payload.get("include_same", True)),
        )
        if not cycles:
            raise ValueError("the selected relation mechanisms produce no legal cycles")

        endpoint_modes = _selected(payload, "endpoint_modes", NATIVE_ANNOTATIONS)
        unknown_modes = set(endpoint_modes) - set(NATIVE_ANNOTATIONS)
        if unknown_modes:
            raise ValueError(f"unknown endpoint mode(s): {', '.join(sorted(unknown_modes))}")

        vector_choices = _vector_choices(payload)
        alignments = _selected(payload, "alignments", ("aligned",))
        unknown_alignments = set(alignments) - set(VECTOR_ALIGNMENTS)
        if unknown_alignments:
            raise ValueError(f"unknown Vector alignment(s): {', '.join(sorted(unknown_alignments))}")
        plain = tuple(
            EndpointChoice(f"scalar:{mode}", "scalar" if mode == "P" else "amo", direction, mode)
            for direction in (READ, WRITE)
            for mode in endpoint_modes
        )
        read_choices = tuple(choice for choice in plain if choice.direction == READ) + tuple(
            choice for choice in vector_choices if choice.direction == READ
        )
        write_choices = tuple(choice for choice in plain if choice.direction == WRITE) + tuple(
            choice for choice in vector_choices if choice.direction == WRITE
        )
        if not any(choice.is_vector for choice in read_choices + write_choices):
            raise ValueError("the Vector relation domain must include at least one Vector form")
        return cls(
            tuple(cycles),
            read_choices,
            write_choices,
            relation_audit,
            sum(not choice.is_vector for choice in read_choices),
            sum(not choice.is_vector for choice in write_choices),
            sum(not choice.is_vector and choice.annotation == "P" for choice in read_choices),
            sum(not choice.is_vector and choice.annotation == "P" for choice in write_choices),
            sum(choice.is_vector or choice.annotation == "P" for choice in read_choices),
            sum(choice.is_vector or choice.annotation == "P" for choice in write_choices),
            sum(_choice_allowed_alignment(choice, "misalign_same16") for choice in read_choices),
            sum(_choice_allowed_alignment(choice, "misalign_same16") for choice in write_choices),
            alignments,
        )

    def choices_for(self, direction: str) -> tuple[EndpointChoice, ...]:
        return self.read_choices if direction == READ else self.write_choices

    def count_for_cycle(self, cycle: NativeCycle) -> int:
        directions = tuple(edge.src for edge in cycle.edges)
        total = 1
        nonvector = 1
        plain_nonvector = 1
        misaligned_total = 1
        for direction in directions:
            choices = self.choices_for(direction)
            total *= len(choices)
            nonvector *= (
                self.read_nonvector_choices
                if direction == READ
                else self.write_nonvector_choices
            )
            misaligned_total *= (
                self.read_misaligned_choices
                if direction == READ
                else self.write_misaligned_choices
            )
            plain_nonvector *= (
                self.read_plain_choices if direction == READ else self.write_plain_choices
            )
        aligned = total - nonvector if "aligned" in self.alignments else 0
        misaligned = (misaligned_total - plain_nonvector) * sum(
            alignment != "aligned" for alignment in self.alignments
        )
        return aligned + misaligned

    @property
    def total_cases(self) -> int:
        return sum(self.count_for_cycle(cycle) for cycle in self.cycles)

    def random_assignments(self, limit: int, seed: int) -> list[VectorAssignment]:
        if limit < 1:
            return []
        total = self.total_cases
        if total < 1:
            return []
        target = min(limit, total)
        weights = [self.count_for_cycle(cycle) for cycle in self.cycles]
        cumulative: list[int] = []
        running = 0
        for weight in weights:
            running += weight
            cumulative.append(running)

        rng = random.Random(seed)
        selected: dict[tuple[Any, ...], VectorAssignment] = {}

        def add_stratum(
            *,
            forced_choices: Sequence[EndpointChoice] = (),
            mechanism: str | None = None,
            exact_vector_count: int | None = None,
            alignment: str | None = None,
        ) -> None:
            if len(selected) >= target:
                return
            eligible_cycles = [
                cycle
                for cycle in self.cycles
                if mechanism is None or any(edge.mechanism == mechanism for edge in cycle.edges)
            ]
            rng.shuffle(eligible_cycles)
            for cycle in eligible_cycles[:100]:
                selected_alignment = alignment or rng.choice(self.alignments)
                directions = tuple(edge.src for edge in cycle.edges)
                available_vertices = list(range(len(directions)))
                assignment: list[EndpointChoice | None] = [None] * len(directions)
                ok = True
                for forced in forced_choices:
                    if not _choice_allowed_alignment(forced, selected_alignment):
                        ok = False
                        break
                    candidates = [
                        vertex
                        for vertex in available_vertices
                        if directions[vertex] == forced.direction
                    ]
                    if not candidates:
                        ok = False
                        break
                    vertex = rng.choice(candidates)
                    assignment[vertex] = forced
                    available_vertices.remove(vertex)
                if not ok:
                    continue

                forced_vector_count = sum(
                    choice is not None and choice.is_vector for choice in assignment
                )
                desired_vectors = exact_vector_count
                if desired_vectors is None and forced_vector_count == 0:
                    desired_vectors = 1
                if desired_vectors is not None:
                    needed = desired_vectors - forced_vector_count
                    candidates = [
                        vertex
                        for vertex in available_vertices
                        if any(
                            choice.is_vector
                            and _choice_allowed_alignment(choice, selected_alignment)
                            for choice in self.choices_for(directions[vertex])
                        )
                    ]
                    if needed < 0 or len(candidates) < needed:
                        continue
                    for vertex in rng.sample(candidates, needed):
                        vector_domain = [
                            choice
                            for choice in self.choices_for(directions[vertex])
                            if choice.is_vector
                            and _choice_allowed_alignment(choice, selected_alignment)
                        ]
                        assignment[vertex] = rng.choice(vector_domain)
                        available_vertices.remove(vertex)

                for vertex in available_vertices:
                    domain = self.choices_for(directions[vertex])
                    if selected_alignment != "aligned":
                        domain = tuple(
                            choice
                            for choice in domain
                            if _choice_allowed_alignment(choice, selected_alignment)
                        )
                    if exact_vector_count is not None:
                        domain = tuple(choice for choice in domain if not choice.is_vector)
                        if not domain:
                            ok = False
                            break
                    assignment[vertex] = rng.choice(domain)
                concrete = tuple(choice for choice in assignment if choice is not None)
                if not ok or len(concrete) != len(directions) or not any(choice.is_vector for choice in concrete):
                    continue
                if selected_alignment != "aligned" and any(
                    choice.category == "amo" for choice in concrete
                ):
                    continue
                candidate = VectorAssignment(cycle, concrete, selected_alignment)
                selected.setdefault(candidate.key, candidate)
                return

        nonvector = self.read_choices + self.write_choices
        for annotation in ("AMO", "Aq", "Rl", "AR", "P"):
            choices = [
                choice
                for choice in nonvector
                if not choice.is_vector and choice.annotation == annotation
            ]
            if choices:
                add_stratum(
                    forced_choices=(rng.choice(choices),),
                    alignment="aligned" if annotation != "P" else None,
                )
        add_stratum(exact_vector_count=1)
        add_stratum(exact_vector_count=2)

        vectors = [choice for choice in self.read_choices + self.write_choices if choice.is_vector]
        for form in dict.fromkeys(choice.vector_form for choice in vectors):
            choices = [choice for choice in vectors if choice.vector_form == form]
            selected_choice = rng.choice(choices)
            add_stratum(
                forced_choices=(selected_choice,),
                alignment=(
                    "aligned"
                    if str((selected_choice.params or {}).get("sew")) == "e8"
                    else None
                ),
            )
        for alignment in self.alignments:
            add_stratum(alignment=alignment)
        for mechanism in ("po", "fence", "dependency"):
            if any(edge.mechanism == mechanism for cycle in self.cycles for edge in cycle.edges):
                add_stratum(mechanism=mechanism)

        attempts = 0
        max_attempts = max(target * 40, 1000)
        while len(selected) < target and attempts < max_attempts:
            attempts += 1
            cycle_index = bisect.bisect_right(cumulative, rng.randrange(total))
            cycle = self.cycles[cycle_index]
            directions = tuple(edge.src for edge in cycle.edges)
            alignment = rng.choice(self.alignments)
            while True:
                choices = tuple(
                    rng.choice(
                        tuple(
                            choice
                            for choice in self.choices_for(direction)
                            if alignment == "aligned"
                            or _choice_allowed_alignment(choice, alignment)
                        )
                    )
                    for direction in directions
                )
                if any(choice.is_vector for choice in choices):
                    break
            assignment = VectorAssignment(cycle, choices, alignment)
            selected.setdefault(assignment.key, assignment)

        if len(selected) < target:
            for assignment in self.assignments():
                selected.setdefault(assignment.key, assignment)
                if len(selected) >= target:
                    break
        return list(selected.values())

    def assignments(self) -> Iterator[VectorAssignment]:
        for cycle in self.cycles:
            domains = [self.choices_for(edge.src) for edge in cycle.edges]
            for choices in product(*domains):
                if not any(choice.is_vector for choice in choices):
                    continue
                for alignment in self.alignments:
                    if any(not _choice_allowed_alignment(choice, alignment) for choice in choices):
                        continue
                    yield VectorAssignment(cycle, tuple(choices), alignment)

    def audit(self) -> dict[str, Any]:
        return {
            "schema": "litmus-link.vector-native-audit.v1",
            "relation_cycles": len(self.cycles),
            "read_endpoint_choices": len(self.read_choices),
            "write_endpoint_choices": len(self.write_choices),
            "alignments": list(self.alignments),
            "total_cases": self.total_cases,
            "relation_audit": dict(self.relation_audit),
        }


def lower_vector_assignment(assignment: VectorAssignment) -> GeneratedCase:
    annotations = tuple(choice.annotation if not choice.is_vector else "P" for choice in assignment.choices)
    annotated = NativeCycle(assignment.cycle.edges, assignment.cycle.family, annotations)
    scalar = lower_native_cycle(annotated)
    case = scalar.case_ir
    harts: list[list[LitmusEvent]] = []
    init_lines = list(case.init_lines)
    vector_metadata: dict[str, dict[str, Any]] = {}
    source_events = case.event_map()
    vector_locations = {
        source_events[f"v{vertex}"].location
        for vertex, choice in enumerate(assignment.choices)
        if choice.is_vector
    }
    base_offset = _alignment_offset(assignment.alignment)
    location_sizes: dict[str, int] = {}

    for hart_id, sequence in enumerate(case.harts):
        expanded: list[LitmusEvent] = []
        for event in sequence:
            vertex = _event_vertex(event.event_id)
            if vertex is None or not assignment.choices[vertex].is_vector:
                if (
                    assignment.alignment != "aligned"
                    and event.kind in {"load", "store"}
                    and event.location in vector_locations
                ):
                    access = event.memory_access
                    size = access.size_bytes if access is not None else 4
                    location_sizes[event.location] = max(location_sizes.get(event.location, 0), size)
                    expanded.append(
                        replace(
                            event,
                            instruction=_replace_memory_offset(event.instruction, base_offset),
                            role="scalar-misaligned-cycle-event",
                            memory_access=MemoryAccess.create(
                                event.location, base_offset, size, "byte_level_no_mag"
                            ),
                        )
                    )
                else:
                    expanded.append(event)
                continue
            choice = assignment.choices[vertex]
            combination = _choice_combination(assignment.cycle.family, event, choice)
            base_register, data_register = _scalar_memory_operands(event)
            setup, extra_init = _vector_setup(combination, hart_id, f"{event.event_id}_vector")
            large_avl = str(choice.params.get("vl")) in {"vl32", "vl64"}
            strided = "strided" in choice.vector_form
            misaligned = assignment.alignment != "aligned"
            temporary_registers = iter(
                _free_temp_registers(sequence, 1 + int(large_avl) + int(strided) + int(misaligned))
            )
            config_register = next(temporary_registers)
            register_map = {"x10": config_register}
            if large_avl:
                register_map["x11"] = next(temporary_registers)
            if strided:
                register_map["x20"] = next(temporary_registers)
            setup = [
                replace(setup_event, instruction=_replace_registers(setup_event.instruction, register_map))
                for setup_event in setup
            ]
            extra_init = [_replace_registers(line, register_map) for line in extra_init]
            expanded.extend(setup)
            vector_base = base_register
            element_bytes = int(str(choice.params["sew"])[1:]) // 8
            location_sizes[event.location] = max(location_sizes.get(event.location, 0), element_bytes)
            if base_offset:
                vector_base = next(temporary_registers)
                expanded.append(
                    LitmusEvent(
                        f"{event.event_id}_misalign_base",
                        hart_id,
                        "setup",
                        f"addi {vector_base},{base_register},{base_offset}",
                        role="vector-misaligned-base",
                    )
                )
            instruction = _replace_registers(
                _rebase_vector(_vector_instruction(combination), vector_base),
                register_map,
            )
            vector_access = MemoryAccess.create(
                event.location,
                base_offset,
                element_bytes,
                "aligned_atomic" if base_offset == 0 else "byte_level_no_mag",
            )
            if event.kind == "store":
                expanded.append(
                    LitmusEvent(
                        f"{event.event_id}_broadcast",
                        hart_id,
                        "setup",
                        f"vmv.v.x v8,{data_register}",
                        role="vector-broadcast",
                    )
                )
                expanded.append(
                    replace(
                        event,
                        instruction=instruction,
                        register="v8",
                        role="vector-store",
                        memory_access=vector_access,
                    )
                )
            else:
                expanded.append(
                    replace(
                        event,
                        instruction=instruction,
                        register="v8",
                        role="vector-load",
                        memory_access=vector_access,
                    )
                )
                expanded.append(
                    LitmusEvent(
                        f"{event.event_id}_extract",
                        hart_id,
                        "extract",
                        f"vmv.x.s {data_register},v8",
                        register=data_register,
                        role="vector-extract-element0",
                    )
                )
            if extra_init:
                init_lines[hart_id] = init_lines[hart_id] + " " + " ".join(extra_init)
            metadata = dict(_vector_metadata(combination))
            metadata.update(
                {
                    "base_offset_bytes": base_offset,
                    "alignment": assignment.alignment,
                    "atomicity_model": "aligned_atomic" if base_offset == 0 else "byte_level_no_mag",
                }
            )
            vector_metadata[event.event_id] = metadata
        harts.append(expanded)

    if assignment.alignment != "aligned":
        init_lines, exists = _rewrite_misaligned_storage(
            init_lines, case.exists, vector_locations, location_sizes, base_offset
        )
    else:
        exists = case.exists
    cycle_labels = _decorated_cycle_labels(assignment)
    cycle_text = " ".join(cycle_labels)
    name = _vector_cycle_name(assignment, cycle_labels)
    relations = [
        replace(relation, label=cycle_labels[index])
        for index, relation in enumerate(case.relations)
    ]
    transformed = replace(case, harts=harts, init_lines=init_lines, relations=relations, exists=exists)
    formal, formal_reason = _formal_scope(transformed, assignment)
    metadata = dict(case.metadata)
    metadata.update(
        {
            "vectors": vector_metadata,
            "endpoint_choices": [choice.to_json() for choice in assignment.choices],
            "source_cycle": assignment.cycle.to_json(),
            "memory_layout": {
                "alignment": assignment.alignment,
                "base_offset_bytes": base_offset,
                "locations": sorted(vector_locations),
            },
            "formal_scope": formal_reason,
        }
    )
    case = replace(
        transformed,
        name=name,
        display_name=name,
        combination_name=name,
        variant="vector-native-cycle",
        cycle=cycle_text,
        model="rvwmo-vector-elements" if formal else "rvwmo-vector-mixed-observation",
        expected_outcome="solver_required" if formal else "manual_oracle_required",
        description="Relation-cycle driven scalar/AMO/Vector Litmus case.",
        tags=[*case.tags, "vector-native", "multi-endpoint"],
        metadata=metadata,
    )
    combination = Combination(
        "vector-native",
        "vector_mem",
        assignment.cycle.family,
        "multi_endpoint",
        NANHU_VECTOR_ATTRIBUTES[0],
        vector="relation_cycle",
        params={
            "cycle": cycle_text,
            "endpoint_modes": [choice.choice_id for choice in assignment.choices],
        },
    )
    decision = Decision(
        GENERATED,
        formal_reason,
        "rvwmo-vector" if formal else "prose-spec",
        "rvwmo-vector" if formal else "hardware-observation",
        [
            "RV64I",
            "V",
            *(("A",) if any(choice.category == "amo" for choice in assignment.choices) else ()),
        ],
        ["generator:vector-native", "multi-endpoint", "relation-cycle"],
        metadata={"formal_forbidden_claim": str(formal).lower()},
    )
    case = replace(case, combination_name=combination.name)
    generated = GeneratedCase(combination, decision, render_ir(case), case_ir=case)
    return generated


def sample_vector_cases(
    payload: Mapping[str, Any],
    *,
    compute_verdicts: bool,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GeneratedCase], dict[str, Any]]:
    domain = VectorNativeDomain.from_payload(payload)
    limit = int(payload.get("sample_limit", 1000))
    seed = int(payload.get("random_seed", 1))
    assignments = domain.random_assignments(limit, seed)
    cases: list[GeneratedCase] = []
    total = max(len(assignments), 1)
    for index, assignment in enumerate(assignments, start=1):
        case = lower_vector_assignment(assignment)
        if compute_verdicts:
            solver = solve_generated_case(case).to_json()
        else:
            solver = _unchecked_solver()
        cases.append(replace(case, solver=solver))
        if progress_callback is not None:
            progress_callback(index, total, f"Sampled {index:,}/{len(assignments):,} random Vector cases")
    audit = domain.audit()
    audit.update(
        {
            "sample_seed": seed,
            "sample_requested": limit,
            "sampled_cases": len(cases),
            "sampling": "coverage-stratified-random-without-replacement-by-assignment-key",
        }
    )
    return cases, audit


def generate_vector_cases(
    payload: Mapping[str, Any],
    out_dir: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    domain = VectorNativeDomain.from_payload(payload)
    requested = int(payload.get("generate_limit", 10000))
    if requested < 1:
        raise ValueError("generate_limit must be positive; the complete relation/endpoint domain is audit-only")
    seed = int(payload.get("random_seed", 1))
    sampled = domain.random_assignments(requested, seed)
    assignments: Iterable[VectorAssignment] = sampled
    target = len(sampled)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = bool(payload.get("compute_verdicts", True))
    filenames: list[str] = []
    verdicts: dict[str, int] = {}
    for index, assignment in enumerate(assignments, start=1):
        case = lower_vector_assignment(assignment)
        solver = solve_generated_case(case).to_json() if judge else _unchecked_solver()
        case = replace(case, solver=solver)
        litmus_path = out_dir / f"{case.name}.litmus"
        litmus_path.write_text(case.litmus, encoding="utf-8")
        (out_dir / f"{case.name}.meta.json").write_text(
            json.dumps(case.meta(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out_dir / f"{case.name}.solver.json").write_text(
            json.dumps(solver, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        filenames.append(litmus_path.name)
        status = str(solver.get("status", "unknown"))
        verdicts[status] = verdicts.get(status, 0) + 1
        if progress_callback is not None:
            progress_callback(index, max(target, 1), f"Generated {index:,}/{target:,}: {case.name}")
    (out_dir / "@all").write_text("\n".join(filenames) + ("\n" if filenames else ""), encoding="utf-8")
    audit = domain.audit()
    report = {
        "schema": "litmus-link.vector-native-generation.v1",
        "profile": "vector-native",
        "available_litmus": domain.total_cases,
        "generated_litmus": len(filenames),
        "generation_limit": requested,
        "generation_limited": len(filenames) < domain.total_cases,
        "random_seed": seed,
        "sampling": "coverage-stratified-random-without-replacement",
        "solver": verdicts,
        "output": str(out_dir),
        "atfile": str(out_dir / "@all"),
        "audit": audit,
    }
    (out_dir / "audit-report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "generation-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _vector_choices(payload: Mapping[str, Any]) -> tuple[EndpointChoice, ...]:
    forms = _selected(payload, "forms", VECTOR_OPS)
    sews = _selected(payload, "sew", VECTOR_WIDTHS)
    lmuls = _selected(payload, "lmul", VECTOR_LMULS)
    index_eews = _selected(payload, "index_eew", VECTOR_INDEX_EEWS)
    masks = _selected(payload, "mask", VECTOR_MASKS)
    tails = _selected(payload, "tail", VECTOR_TAILS)
    vls = _selected(payload, "vl", VECTOR_LENGTHS)
    out: list[EndpointChoice] = []
    for form, sew, lmul, mask, tail, vl in product(
        forms, sews, lmuls, masks, tails, vls
    ):
        if form not in VECTOR_OPS:
            raise ValueError(f"unsupported Vector form: {form}")
        if not vector_same_line_footprint(form, sew, lmul, mask, vl):
            continue
        selected_indexes: Sequence[str | None] = index_eews if "indexed" in form else (None,)
        for index_eew in selected_indexes:
            params = {
                "sew": sew,
                "lmul": lmul,
                "mask": mask,
                "tail": tail,
                "vl": vl,
                "footprint": "same_line",
            }
            if index_eew is not None:
                params["index_eew"] = index_eew
            direction = WRITE if form.endswith("store") else READ
            choice_id = "vector:" + ":".join(
                [form, sew, lmul, str(index_eew or "-"), mask, tail, vl]
            )
            out.append(EndpointChoice(choice_id, "vector", direction, "P", form, params))
    if not out:
        raise ValueError("the selected Vector axes contain no ISA-legal endpoint choices")
    return tuple(out)


def _choice_allowed_alignment(choice: EndpointChoice, alignment: str) -> bool:
    if alignment == "aligned":
        return True
    if choice.is_vector:
        return str((choice.params or {}).get("sew")) != "e8"
    return choice.annotation == "P"


def _choice_combination(family: str, event: LitmusEvent, choice: EndpointChoice) -> Combination:
    params = dict(choice.params or {})
    return Combination(
        "vector-native",
        "vector_mem",
        family,
        "vector_store" if event.kind == "store" else "vector_load",
        "cacheable",
        vector=choice.vector_form,
        params=params,
    )


def _event_vertex(event_id: str) -> int | None:
    match = re.fullmatch(r"v(\d+)", event_id)
    return int(match.group(1)) if match else None


def _alignment_offset(alignment: str) -> int:
    if alignment == "aligned":
        return 0
    return {
        "misalign_same16": 1,
        "misalign_cross16": 15,
        "misalign_cross64": 63,
    }[alignment]


def _replace_memory_offset(instruction: str, offset: int) -> str:
    updated, count = re.subn(r"-?\d+\((x\d+)\)", rf"{offset}(\1)", instruction, count=1)
    if count != 1:
        raise ValueError(f"cannot apply a misaligned offset to {instruction!r}")
    return updated


def _rewrite_misaligned_storage(
    init_lines: Sequence[str],
    exists: str,
    locations: set[str],
    sizes: Mapping[str, int],
    offset: int,
) -> tuple[list[str], str]:
    rewritten_init: list[str] = []
    location_pattern = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*0\s*;\s*$")
    for line in init_lines:
        match = location_pattern.fullmatch(line)
        if match and match.group(1) in locations:
            rewritten_init.append(f"uint8_t {match.group(1)}[128];")
        else:
            rewritten_init.append(line)

    body = exists.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    terms = [term.strip() for term in body.split("/\\") if term.strip()]
    rewritten_terms: list[str] = []
    final_pattern = re.compile(
        r"^([A-Za-z_]\w*)\s*=\s*(-?(?:0x[0-9a-fA-F]+|\d+))$"
    )
    for term in terms:
        match = final_pattern.fullmatch(term)
        if not match or match.group(1) not in locations:
            rewritten_terms.append(term)
            continue
        location = match.group(1)
        value = int(match.group(2), 0)
        for byte in range(sizes.get(location, 4)):
            rewritten_terms.append(f"{location}[{offset + byte}]=0x{(value >> (8 * byte)) & 0xff:02x}")
    return rewritten_init, "(" + " /\\ ".join(rewritten_terms) + ")"


def _free_temp_registers(sequence: Sequence[LitmusEvent], count: int) -> tuple[str, ...]:
    used = {register for event in sequence for register in re.findall(r"\bx(?:[12]?\d|3[01]|[0-9])\b", event.instruction)}
    available = tuple(
        candidate
        for candidate in ("x31", "x30", "x29", "x19", "x18", "x17", "x16", "x4", "x3")
        if candidate not in used
    )
    if len(available) < count:
        raise ValueError(f"only {len(available)} temporary registers are available; {count} are required")
    return available[:count]


def _replace_registers(instruction: str, replacements: Mapping[str, str]) -> str:
    return re.sub(
        r"\bx(?:[12]?\d|3[01]|[0-9])\b",
        lambda match: replacements.get(match.group(0), match.group(0)),
        instruction,
    )


def _endpoint_code(choice: EndpointChoice, alignment: str = "aligned") -> str:
    if not choice.is_vector:
        code = choice.annotation
        if alignment != "aligned":
            code += _alignment_code(alignment)
        return code
    params = choice.params or {}
    sew = str(params.get("sew", "e32"))[1:]
    index = str(params.get("index_eew", ""))[2:]
    forms = {
        "unit_load": f"Vle{sew}",
        "unit_store": f"Vse{sew}",
        "strided_load": f"Vlse{sew}",
        "strided_store": f"Vsse{sew}",
        "indexed_ordered_load": f"Vlo{index}e{sew}",
        "indexed_unordered_load": f"Vlu{index}e{sew}",
        "indexed_ordered_store": f"Vso{index}e{sew}",
        "indexed_unordered_store": f"Vsu{index}e{sew}",
    }
    code = forms[choice.vector_form]
    code += str(params.get("lmul", "m1")).upper()
    code += str(params.get("vl", "vlmax")).upper()
    if params.get("mask") == "masked":
        code += "M"
    if alignment != "aligned":
        code += _alignment_code(alignment)
    return code


def _alignment_code(alignment: str) -> str:
    return {
        "misalign_same16": "U16",
        "misalign_cross16": "X16",
        "misalign_cross64": "X64",
    }[alignment]


def _decorated_cycle_labels(assignment: VectorAssignment) -> tuple[str, ...]:
    locations = location_ids(assignment.cycle.edges)
    vector_locations = {
        locations[index]
        for index, choice in enumerate(assignment.choices)
        if choice.is_vector
    }

    def code(index: int) -> str:
        alignment = (
            assignment.alignment
            if locations[index] in vector_locations
            else "aligned"
        )
        return _endpoint_code(assignment.choices[index], alignment)

    labels: list[str] = []
    for index, edge in enumerate(assignment.cycle.edges):
        source = code(index)
        target = code((index + 1) % len(assignment.choices))
        suffix = "" if source == "P" and target == "P" else source + target
        labels.append(edge.label + suffix)
    return tuple(labels)


def _vector_cycle_name(assignment: VectorAssignment, labels: Sequence[str]) -> str:
    raw = "+".join([assignment.cycle.family or "Cycle", *labels])
    if len(raw) <= 220:
        return raw
    import hashlib

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return raw[: 220 - len(digest) - 4].rstrip("+._-") + "+ID." + digest


def _formal_scope(case: LitmusCaseIR, assignment: VectorAssignment) -> tuple[bool, str]:
    if assignment.alignment != "aligned":
        return (
            False,
            "Legal cacheable Vector misalignment is generated as a no-MAG byte-level observation; "
            "mixed scalar/Vector partial-overlap atomicity is outside the current formal solver.",
        )
    widths_by_location: dict[str, set[int]] = {}
    for event in case.events():
        if event.kind not in {"load", "store", "amo"} or not event.location:
            continue
        width = event.memory_access.size_bytes if event.memory_access is not None else 4
        widths_by_location.setdefault(event.location, set()).add(width)
    vector_widths: dict[str, set[int]] = {}
    for vertex, choice in enumerate(assignment.choices):
        if choice.is_vector:
            event = case.event_map()[f"v{vertex}"]
            vector_widths.setdefault(event.location, set()).add(int(str(choice.params["sew"])[1:]) // 8)
    if any(len(widths_by_location.get(location, set()) | widths) > 1 for location, widths in vector_widths.items()):
        return False, "Mixed-size scalar/AMO/Vector overlap is generated for observation, but is outside the current formal solver."
    return True, "ISA-legal aligned scalar/AMO/Vector relation cycle supported by the Vector-aware RVWMO solver."


def _selected(payload: Mapping[str, Any], key: str, default: Sequence[str]) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return tuple(default)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list")
    return tuple(dict.fromkeys(str(item) for item in value))


def _unchecked_solver() -> dict[str, Any]:
    return {
        "schema": "litmus-link.solver.v1",
        "status": "unchecked",
        "verdict": "unchecked",
        "allowed": None,
        "model": "rvwmo-vector-elements",
        "tool": "none",
        "reason": "Fast preview skips outcome solving; use Verify Preview to calculate the verdict.",
        "cross_check": "not_run",
        "edges": [],
        "fusion": None,
        "observation": "",
        "raw_output": "",
        "command": [],
    }
