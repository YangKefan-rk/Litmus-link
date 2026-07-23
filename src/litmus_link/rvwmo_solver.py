from __future__ import annotations

"""Small, self-contained RVWMO execution-graph solver.

The solver targets explicit main-memory execution graphs. Scalar LitmusCaseIR
programs feed it directly; the vector-aware frontend supplies expanded active
elements plus instruction/element ordering overrides. It does not parse or
execute ``riscv.cat`` dynamically; instead it implements the relations and
three axioms from the riscv.cat shipped with herdtools7 7.58:

* acyclic co | rf | fr | po-loc
* acyclic co | rfe | fr | ppo
* empty rmw & (fre ; coe)

The implementation is independent of herd7 and is therefore usable on closed
machines.  ``herd7`` remains a valuable differential oracle and the native
generation path can request both backends.
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Mapping, Sequence

from .amo import AmoError, AmoSpec, apply_amo, parse_amo_mnemonic
from .litmus_ir import LitmusCaseIR


Pair = tuple[str, str]
Relation = set[Pair]
ByteEdge = tuple[str, str, str]
ByteRelation = set[ByteEdge]


class RvwmoSolverError(ValueError):
    pass


@dataclass(frozen=True)
class OrderingOverrides:
    """Structured ordering supplied by a frontend that expands one ISA
    instruction into multiple memory events.

    Events with the same ``order_by_event`` value belong to the same
    instruction and therefore have no ordinary po edge between them.
    ``preserved_order`` is reserved for architecturally ordered sub-events,
    currently ordered-indexed RVV elements.
    """

    order_by_event: Mapping[str, int] = field(default_factory=dict)
    instruction_by_event: Mapping[str, str] = field(default_factory=dict)
    preserved_order: frozenset[Pair] = frozenset()


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    hart: int | None
    order: int
    location: str
    read: bool
    write: bool
    read_value: int | None
    write_value: int | None
    initial: bool = False
    aq: bool = False
    rl: bool = False
    amo: bool = False
    instruction_id: str = ""
    byte_offset: int | None = None
    access_size: int = 0
    atomicity_model: str = "location_atomic"
    footprint: tuple[str, ...] = ()
    read_bytes: tuple[tuple[str, int | None], ...] = ()
    write_bytes: tuple[tuple[str, int | None], ...] = ()
    transaction_kind: str = "legacy"
    amo_op: str = ""
    amo_operand: int | None = None
    amo_width_bytes: int | None = None
    amo_ordering: str = ""

    @property
    def rcsc(self) -> bool:
        return self.amo and (self.aq or self.rl)

    @property
    def byte_locations(self) -> frozenset[str]:
        return frozenset(self.footprint or (self.location,))

    def reads_byte(self, location: str) -> bool:
        return any(byte == location for byte, _value in self.read_bytes)

    def writes_byte(self, location: str) -> bool:
        return any(byte == location for byte, _value in self.write_bytes)

    def read_byte_value(self, location: str) -> int | None:
        return next((value for byte, value in self.read_bytes if byte == location), None)

    def write_byte_value(self, location: str) -> int | None:
        return next((value for byte, value in self.write_bytes if byte == location), None)

    def to_json(self) -> dict:
        return {
            "event_id": self.event_id,
            "hart": self.hart,
            "order": self.order,
            "location": self.location,
            "read": self.read,
            "write": self.write,
            "read_value": self.read_value,
            "write_value": self.write_value,
            "initial": self.initial,
            "aq": self.aq,
            "rl": self.rl,
            "amo": self.amo,
            "instruction_id": self.instruction_id or self.event_id,
            "byte_offset": self.byte_offset,
            "access_size": self.access_size,
            "atomicity_model": self.atomicity_model,
            "footprint": list(self.footprint or (self.location,)),
            "read_bytes": [
                {"location": location, "value": value}
                for location, value in self.read_bytes
            ],
            "write_bytes": [
                {"location": location, "value": value}
                for location, value in self.write_bytes
            ],
            "transaction_kind": self.transaction_kind,
            "amo_operation": self.amo_op or None,
            "amo_operand": self.amo_operand,
            "amo_width_bytes": self.amo_width_bytes,
            "amo_ordering": self.amo_ordering or None,
        }


@dataclass(frozen=True)
class Execution:
    rf: Relation
    co: Relation
    fr: Relation
    po: Relation
    po_loc: Relation
    ppo: Relation
    ppo_rules: Mapping[str, Relation]
    rf_bytes: ByteRelation = field(default_factory=set)
    co_bytes: ByteRelation = field(default_factory=set)
    fr_bytes: ByteRelation = field(default_factory=set)
    partial: bool = False

    def to_json(self) -> dict:
        return {
            "rf": _pairs_json(self.rf),
            "co": _pairs_json(self.co),
            "fr": _pairs_json(self.fr),
            "po": _pairs_json(self.po),
            "po_loc": _pairs_json(self.po_loc),
            "ppo": _pairs_json(self.ppo),
            "ppo_rules": {
                rule: _pairs_json(edges)
                for rule, edges in sorted(self.ppo_rules.items())
                if edges
            },
            "byte_relations": {
                "rf": _byte_pairs_json(self.rf_bytes),
                "co": _byte_pairs_json(self.co_bytes),
                "fr": _byte_pairs_json(self.fr_bytes),
            },
            "partial": self.partial,
        }


@dataclass(frozen=True)
class _CandidateRelation:
    relation: Relation
    bytes: ByteRelation
    partial: bool = False


@dataclass(frozen=True)
class _CoCandidate:
    relation: Relation
    bytes: ByteRelation
    successors: Mapping[tuple[str, str], frozenset[str]]
    immediate_predecessors: Mapping[tuple[str, str], str]
    latest_by_location: Mapping[str, str]


@dataclass
class _SearchBudget:
    deadline: float
    max_steps: int
    steps: int = 0
    next_deadline_check: int = 0

    def checkpoint(self, units: int = 1) -> None:
        if units < 1:
            raise ValueError("search budget units must be positive")
        self.steps += units
        if self.steps > self.max_steps:
            raise _SearchLimit("search_step_limit")
        # monotonic() is comparatively expensive in the solver's innermost
        # graph loops.  Checking every 64 logical steps keeps timeout drift
        # negligible while avoiding hundreds of thousands of system calls.
        if self.steps >= self.next_deadline_check:
            if time.monotonic() >= self.deadline:
                raise _SearchLimit("timeout")
            self.next_deadline_check = self.steps + 64


@dataclass(frozen=True)
class EmbeddedVerdict:
    status: str
    verdict: str
    allowed: bool | None
    candidates: int
    consistent_candidates: int
    reason: str
    elapsed_seconds: float
    events: tuple[MemoryEvent, ...]
    execution: Execution | None = None
    violation_counts: Mapping[str, int] | None = None
    example_cycles: Mapping[str, tuple[str, ...]] | None = None
    search_steps: int = 0

    def to_json(self) -> dict:
        no_mag = any(event.atomicity_model == "byte_level_no_mag" for event in self.events)
        return {
            "schema": "litmus-link.embedded-rvwmo.v1",
            "status": self.status,
            "tool": "litmus-link-rvwmo",
            "backend": "embedded",
            "model": "riscv.cat+byte_level_no_mag" if no_mag else "riscv.cat",
            "model_revision": (
                "litmus-link-byte-level-no-mag-v1"
                if no_mag
                else "herdtools7-7.58-riscv-cat"
            ),
            "model_extensions": (
                ["mixed-size", "unaligned", "byte-level-no-mag"] if no_mag else []
            ),
            "verdict": self.verdict,
            "allowed": self.allowed,
            "candidates": self.candidates,
            "search_steps": self.search_steps,
            "consistent_candidates": self.consistent_candidates,
            "reason": self.reason,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "events": [event.to_json() for event in self.events],
            "execution": self.execution.to_json() if self.execution else None,
            "violation_counts": dict(sorted((self.violation_counts or {}).items())),
            "example_cycles": {
                key: list(value)
                for key, value in sorted((self.example_cycles or {}).items())
            },
        }


def solve_rvwmo(
    case: LitmusCaseIR,
    *,
    max_candidates: int = 100_000,
    timeout_seconds: float = 10.0,
    max_search_steps: int = 1_000_000,
    ordering: OrderingOverrides | None = None,
) -> EmbeddedVerdict:
    """Decide whether ``case.exists`` has an RVWMO-consistent execution.

    A positive result may return as soon as a witness is found.  A forbidden
    result is emitted only after the complete bounded candidate space has been
    exhausted.  Candidate or time limits therefore produce ``inconclusive``.
    """

    if max_candidates < 1:
        raise RvwmoSolverError("max_candidates must be at least 1")
    if timeout_seconds <= 0:
        raise RvwmoSolverError("timeout_seconds must be positive")
    if max_search_steps < 1:
        raise RvwmoSolverError("max_search_steps must be at least 1")
    started = time.monotonic()
    budget = _SearchBudget(started + timeout_seconds, max_search_steps)
    events: tuple[MemoryEvent, ...] = ()
    try:
        selected_ordering = ordering or OrderingOverrides()
        events = _memory_events(case, selected_ordering)
        budget.checkpoint(max(1, len(events)))
        final_values = _final_values(case.exists)
        static = _static_relations(
            case,
            events,
            selected_ordering.preserved_order,
            budget,
        )
        invariant_ppo_rules = _ppo_invariant_relations(events, static, budget)
        invariant_ppo = set().union(*invariant_ppo_rules.values())
    except RvwmoSolverError as exc:
        return EmbeddedVerdict(
            status="not_applicable",
            verdict="unmodeled",
            allowed=None,
            candidates=0,
            consistent_candidates=0,
            reason=str(exc),
            elapsed_seconds=time.monotonic() - started,
            events=(),
        )
    except _SearchLimit as exc:
        return _limited_verdict(exc, started, budget, events)

    violations: dict[str, int] = defaultdict(int)
    examples: dict[str, tuple[str, ...]] = {}
    candidates = 0
    consistent = 0
    last_execution: Execution | None = None
    last_resolved_events = events
    try:
        for co in _co_candidates(
            events,
            budget,
            (static.po_loc, invariant_ppo),
        ):
            budget.checkpoint()
            cycle = _find_cycle(co.relation | static.po_loc, budget)
            if cycle:
                violations["Coherence"] += 1
                examples.setdefault("Coherence", cycle)
                continue
            cycle = _find_cycle(co.relation | invariant_ppo, budget)
            if cycle:
                violations["Model"] += 1
                examples.setdefault("Model", cycle)
                continue
            resolved_events = _resolve_amo_transactions(events, co, budget)
            if resolved_events is None:
                continue
            last_resolved_events = resolved_events
            if not _final_values_match(resolved_events, co, final_values, budget):
                continue
            for rf in _rf_candidates(
                resolved_events,
                co,
                budget,
                static,
                invariant_ppo,
            ):
                budget.checkpoint()
                if rf.partial:
                    execution, failure, cycle = _check_execution(
                        resolved_events,
                        static,
                        rf,
                        co,
                        budget,
                        invariant_ppo_rules,
                    )
                    last_execution = execution
                    if failure is None:
                        raise RvwmoSolverError(
                            "partial reads-from pruning did not reproduce its cycle"
                        )
                    violations[failure] += 1
                    if cycle and failure not in examples:
                        examples[failure] = cycle
                    continue
                candidates += 1
                if candidates > max_candidates:
                    raise _SearchLimit("candidate_limit")
                execution, failure, cycle = _check_execution(
                    resolved_events,
                    static,
                    rf,
                    co,
                    budget,
                    invariant_ppo_rules,
                )
                last_execution = execution
                if failure is None:
                    consistent += 1
                    return EmbeddedVerdict(
                        status="verified",
                        verdict="observable",
                        allowed=True,
                        candidates=candidates,
                        consistent_candidates=consistent,
                        reason="At least one candidate execution satisfies riscv.cat.",
                        elapsed_seconds=time.monotonic() - started,
                        events=resolved_events,
                        execution=execution,
                        violation_counts=violations,
                        example_cycles=examples,
                        search_steps=budget.steps,
                    )
                violations[failure] += 1
                if cycle and failure not in examples:
                    examples[failure] = cycle
    except _SearchLimit as exc:
        return _limited_verdict(
            exc,
            started,
            budget,
            last_resolved_events,
            candidates=min(candidates, max_candidates),
            consistent=consistent,
            violations=violations,
            examples=examples,
        )

    return EmbeddedVerdict(
        status="verified",
        verdict="forbidden",
        allowed=False,
        candidates=candidates,
        consistent_candidates=0,
        reason="Every candidate execution violates at least one riscv.cat axiom.",
        elapsed_seconds=time.monotonic() - started,
        events=last_resolved_events,
        execution=last_execution,
        violation_counts=violations,
        example_cycles=examples,
        search_steps=budget.steps,
    )


def _limited_verdict(
    limit: _SearchLimit,
    started: float,
    budget: _SearchBudget,
    events: Sequence[MemoryEvent],
    *,
    candidates: int = 0,
    consistent: int = 0,
    violations: Mapping[str, int] | None = None,
    examples: Mapping[str, tuple[str, ...]] | None = None,
) -> EmbeddedVerdict:
    return EmbeddedVerdict(
        status="inconclusive",
        verdict="unknown",
        allowed=None,
        candidates=candidates,
        consistent_candidates=consistent,
        reason=(
            f"Embedded RVWMO search stopped at {limit.reason}; "
            "a forbidden verdict requires exhaustive search."
        ),
        elapsed_seconds=time.monotonic() - started,
        events=tuple(events),
        violation_counts=violations,
        example_cycles=examples,
        search_steps=budget.steps,
    )


@dataclass(frozen=True)
class _StaticRelations:
    event_map: Mapping[str, MemoryEvent]
    po: Relation
    po_loc: Relation
    fence: Relation
    addr: Relation
    data: Relation
    ctrl: Relation
    preserved_order: Relation


class _SearchLimit(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _memory_events(case: LitmusCaseIR, ordering: OrderingOverrides) -> tuple[MemoryEvent, ...]:
    memory: list[MemoryEvent] = []
    byte_locations: set[str] = set()
    for hart, sequence in enumerate(case.harts):
        order = 0
        for event in sequence:
            if event.kind not in {"load", "store", "amo"}:
                continue
            if not event.location:
                raise RvwmoSolverError(f"memory event {event.event_id} has no location")
            if event.register.startswith("v") or event.instruction.lstrip().startswith("v"):
                raise RvwmoSolverError("embedded RVWMO solver currently supports scalar memory events only")
            event_order = ordering.order_by_event.get(event.event_id, order)
            instruction_id = ordering.instruction_by_event.get(event.event_id, event.event_id)
            classified = _classify_events(event, hart, event_order, instruction_id)
            memory.extend(classified)
            byte_locations.update(
                location
                for item in classified
                for location in item.byte_locations
            )
            order += 1
    if not memory:
        raise RvwmoSolverError("case has no scalar memory events")
    init = _initial_values(case.init_lines)
    for location in sorted(byte_locations):
        memory.append(
            MemoryEvent(
                event_id=f"init:{location}",
                hart=None,
                order=-1,
                location=location,
                read=False,
                write=True,
                read_value=None,
                write_value=_initial_byte_value(init, location),
                initial=True,
                instruction_id=f"init:{location}",
                byte_offset=_location_byte_offset(location),
                access_size=1 if _location_byte_offset(location) is not None else 0,
                atomicity_model="initial",
                footprint=(location,),
                write_bytes=((location, _initial_byte_value(init, location)),),
                transaction_kind="initial",
            )
        )
    ids = [event.event_id for event in memory]
    if len(ids) != len(set(ids)):
        raise RvwmoSolverError("case contains duplicate memory event ids")
    return tuple(memory)


def _classify_events(
    event: LitmusEvent,
    hart: int,
    order: int,
    instruction_id: str,
) -> tuple[MemoryEvent, ...]:
    access = event.memory_access
    if access is not None and access.atomicity_model == "byte_level_no_mag":
        if event.kind not in {"load", "store"}:
            raise RvwmoSolverError(
                f"byte-level no-MAG access {event.event_id} must be a plain load/store"
            )
        value = _integer(event.value, f"event {event.event_id} value") if event.value else None
        if value is None:
            raise RvwmoSolverError(f"memory event {event.event_id} has no target value")
        out: list[MemoryEvent] = []
        for byte_index, absolute_byte in enumerate(access.covered_bytes):
            byte_value = (value >> (8 * byte_index)) & 0xFF
            out.append(
                MemoryEvent(
                    event_id=f"{event.event_id}.b{byte_index}",
                    hart=hart,
                    order=order,
                    location=access.byte_location(absolute_byte),
                    read=event.kind == "load",
                    write=event.kind == "store",
                    read_value=byte_value if event.kind == "load" else None,
                    write_value=byte_value if event.kind == "store" else None,
                    instruction_id=instruction_id,
                    byte_offset=absolute_byte,
                    access_size=access.size_bytes,
                    atomicity_model="byte_level_no_mag",
                    footprint=(access.byte_location(absolute_byte),),
                    read_bytes=(
                        ((access.byte_location(absolute_byte), byte_value),)
                        if event.kind == "load"
                        else ()
                    ),
                    write_bytes=(
                        ((access.byte_location(absolute_byte), byte_value),)
                        if event.kind == "store"
                        else ()
                    ),
                    transaction_kind="byte_split",
                )
            )
        return tuple(out)
    return (_classify_event(event, hart, order, instruction_id),)


def _classify_event(event: LitmusEvent, hart: int, order: int, instruction_id: str) -> MemoryEvent:
    instruction = event.instruction.lower().replace(" ", "")
    value_text = event.read_value if event.kind == "amo" and event.read_value else event.value
    value = _integer(value_text, f"event {event.event_id} value") if value_text else None
    access = event.memory_access
    inferred_size, inferred_offset = _instruction_access_shape(event.instruction)
    access_metadata = {
        "byte_offset": (
            access.offset_bytes
            if access is not None
            else inferred_offset
        ),
        "access_size": (
            access.size_bytes
            if access is not None
            else inferred_size or 0
        ),
        "atomicity_model": access.atomicity_model if access is not None else "location_atomic",
    }
    footprint = _event_footprint(event)
    read_bytes = _value_bytes(footprint, value) if value is not None else tuple(
        (location, None) for location in footprint
    )
    transaction_kind = access.transaction_kind if access is not None else "scalar_plain"
    if event.kind == "load":
        if value is None and not event.role.startswith("vector-element"):
            raise RvwmoSolverError(f"load {event.event_id} has no target read value")
        return MemoryEvent(
            event.event_id, hart, order, event.location, True, False, value, None,
            instruction_id=instruction_id,
            footprint=footprint, read_bytes=read_bytes,
            transaction_kind=transaction_kind, **access_metadata,
        )
    if event.kind == "store":
        if value is None:
            raise RvwmoSolverError(f"store {event.event_id} has no write value")
        return MemoryEvent(
            event.event_id, hart, order, event.location, False, True, None, value,
            instruction_id=instruction_id,
            footprint=footprint, write_bytes=_value_bytes(footprint, value),
            transaction_kind=transaction_kind, **access_metadata,
        )
    if event.kind != "amo":
        raise RvwmoSolverError(
            f"unsupported memory event kind in {event.event_id}: {event.kind}"
        )

    try:
        parsed = parse_amo_mnemonic(event.instruction)
        spec = AmoSpec(
            event.amo_op or parsed.operation,
            event.amo_width_bytes or parsed.width_bytes,
            event.amo_ordering or parsed.ordering,
        )
    except AmoError as exc:
        raise RvwmoSolverError(str(exc)) from exc
    if spec != parsed:
        raise RvwmoSolverError(
            f"AMO metadata does not match instruction in {event.event_id}: "
            f"metadata={spec}, instruction={parsed}"
        )
    if len(footprint) != spec.width_bytes:
        raise RvwmoSolverError(
            f"AMO footprint width in {event.event_id} is {len(footprint)} bytes, "
            f"instruction requires {spec.width_bytes}"
        )
    offset = access.offset_bytes if access is not None else inferred_offset
    if offset % spec.width_bytes:
        raise RvwmoSolverError(
            f"misaligned AMO {event.event_id} is outside Nanhu formal scope"
        )

    explicit_metadata = any(
        (
            event.read_value,
            event.write_value,
            event.amo_op,
            event.amo_operand,
            event.amo_width_bytes,
            event.amo_ordering,
        )
    )
    if explicit_metadata:
        if not event.amo_operand:
            raise RvwmoSolverError(f"AMO {event.event_id} has no operand value")
        operand = _integer(event.amo_operand, f"event {event.event_id} AMO operand")
        expected_write = (
            _integer(event.write_value, f"event {event.event_id} AMO write value")
            if event.write_value
            else None
        )
    elif parsed.operation == "or" and ",x0," in instruction:
        # case_ir.v1 compatibility: amoor with rs2=x0 is a read endpoint.
        operand = 0
        expected_write = value
    elif parsed.operation == "swap":
        # case_ir.v1 compatibility: amoswap write endpoints stored rs2's value
        # in ``value`` and discarded rd.
        if value is None:
            raise RvwmoSolverError(f"legacy AMO {event.event_id} has no operand value")
        operand = value
        value = None
        expected_write = operand
        read_bytes = tuple((location, None) for location in footprint)
    else:
        raise RvwmoSolverError(
            f"AMO {event.event_id} requires explicit operand metadata"
        )

    return MemoryEvent(
        event.event_id,
        hart,
        order,
        event.location,
        True,
        True,
        value,
        expected_write,
        aq=spec.ordering in {"aq", "aqrl"},
        rl=spec.ordering in {"rl", "aqrl"},
        amo=True,
        instruction_id=instruction_id,
        footprint=footprint,
        read_bytes=read_bytes,
        write_bytes=(
            _value_bytes(footprint, expected_write)
            if expected_write is not None
            else tuple((location, None) for location in footprint)
        ),
        transaction_kind="amo_rmw",
        amo_op=spec.operation,
        amo_operand=operand,
        amo_width_bytes=spec.width_bytes,
        amo_ordering=spec.ordering,
        **access_metadata,
    )


def _initial_values(lines: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in lines:
        array = re.fullmatch(
            r"\s*(?:u?int8_t|char)\s+([A-Za-z_]\w*)\s*\[(\d+)\]\s*"
            r"(?:=\s*\{([^}]*)\})?\s*;\s*",
            line,
        )
        if array:
            name, size_text, values_text = array.groups()
            size = int(size_text)
            values = (
                [
                    int(token.strip(), 0) & 0xFF
                    for token in values_text.split(",")
                    if token.strip()
                ]
                if values_text is not None
                else []
            )
            if len(values) > size:
                raise RvwmoSolverError(
                    f"array initializer for {name} has {len(values)} values, size is {size}"
                )
            for offset in range(size):
                out[f"{name}[{offset}]"] = values[offset] if offset < len(values) else 0
            continue
        for term in line.split(";"):
            match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*(-?(?:0x[0-9a-fA-F]+|\d+))\s*", term)
            if match:
                out[match.group(1)] = int(match.group(2), 0)
                continue
            byte = re.fullmatch(
                r"\s*([A-Za-z_]\w*\[\d+\])\s*=\s*(-?(?:0x[0-9a-fA-F]+|\d+))\s*",
                term,
            )
            if byte:
                out[byte.group(1)] = int(byte.group(2), 0) & 0xFF
    return out


def _initial_byte_value(initial: Mapping[str, int], byte_location: str) -> int:
    if byte_location in initial:
        return initial[byte_location] & 0xFF
    base, offset = _split_byte_location(byte_location)
    return (initial.get(base, 0) >> (8 * offset)) & 0xFF


def _event_footprint(event: LitmusEvent) -> tuple[str, ...]:
    access = event.memory_access
    if access is not None:
        return tuple(access.byte_location(offset) for offset in access.covered_bytes)
    size, offset = _instruction_access_shape(event.instruction)
    if size is None:
        return (event.location,)
    return tuple(f"{event.location}[{offset + index}]" for index in range(size))


def _instruction_access_shape(instruction: str) -> tuple[int | None, int]:
    normalized = instruction.strip().lower()
    mnemonic = normalized.split(maxsplit=1)[0]
    size = {
        "lb": 1,
        "lbu": 1,
        "sb": 1,
        "lh": 2,
        "lhu": 2,
        "sh": 2,
        "lw": 4,
        "lwu": 4,
        "sw": 4,
        "ld": 8,
        "sd": 8,
    }.get(mnemonic)
    if size is None and mnemonic.startswith("amo"):
        try:
            size = parse_amo_mnemonic(mnemonic).width_bytes
        except AmoError:
            pass
    match = re.search(r"(-?\d+)\(x\d+\)", normalized)
    return size, int(match.group(1)) if match else 0


def _value_bytes(
    footprint: Sequence[str],
    value: int,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (location, (value >> (8 * index)) & 0xFF)
        for index, location in enumerate(footprint)
    )


def _split_byte_location(location: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z_]\w*)\[(\d+)\]", location)
    if match:
        return match.group(1), int(match.group(2))
    return location, 0


def _final_values(exists: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for location, value in re.findall(
        r"(?<![:\w])([A-Za-z_]\w*(?:\[\d+\])?)\s*=\s*(-?(?:0x[0-9a-fA-F]+|\d+))",
        exists,
    ):
        out[location] = int(value, 0)
    return out


def _final_byte_value(final_values: Mapping[str, int], byte_location: str) -> int | None:
    if byte_location in final_values:
        return final_values[byte_location] & 0xFF
    base, offset = _split_byte_location(byte_location)
    if base not in final_values:
        return None
    return (final_values[base] >> (8 * offset)) & 0xFF


def _location_byte_offset(location: str) -> int | None:
    match = re.fullmatch(r"[A-Za-z_]\w*\[(\d+)\]", location)
    return int(match.group(1)) if match else None


def _overlap(left: MemoryEvent, right: MemoryEvent) -> bool:
    return bool(left.byte_locations & right.byte_locations)


def _static_relations(
    case: LitmusCaseIR,
    events: Sequence[MemoryEvent],
    preserved_order: Iterable[Pair] = (),
    budget: _SearchBudget | None = None,
) -> _StaticRelations:
    event_map = {event.event_id: event for event in events}
    po: Relation = set()
    by_hart: dict[int, list[MemoryEvent]] = defaultdict(list)
    for event in events:
        if budget is not None:
            budget.checkpoint()
        if event.hart is not None:
            by_hart[event.hart].append(event)
    for sequence in by_hart.values():
        for left in sequence:
            for right in sequence:
                if budget is not None:
                    budget.checkpoint()
                if left.order < right.order:
                    po.add((left.event_id, right.event_id))
    explicit_order = set(preserved_order)
    for left, right in explicit_order:
        if left not in event_map or right not in event_map:
            raise RvwmoSolverError(f"preserved order references unknown event: {left}->{right}")
        if event_map[left].hart != event_map[right].hart:
            raise RvwmoSolverError(f"preserved order must be hart-local: {left}->{right}")
        if left == right:
            raise RvwmoSolverError(f"preserved order cannot be a self edge: {left}")
    po.update(explicit_order)
    po_loc = {
        pair for pair in po
        if _overlap(event_map[pair[0]], event_map[pair[1]])
    }
    addr: Relation = set()
    data: Relation = set()
    ctrl: Relation = set()
    by_instruction: dict[str, list[MemoryEvent]] = defaultdict(list)
    for event in events:
        by_instruction[event.instruction_id or event.event_id].append(event)
    for relation in case.relations:
        if budget is not None:
            budget.checkpoint()
        sources = by_instruction.get(relation.src, ())
        targets = by_instruction.get(relation.dst, ())
        if not sources or not targets:
            continue
        label = relation.label.lower()
        dependency = relation.kind == "dependency" or any(
            token in label for token in ("addr", "data", "ctrl")
        )
        if not dependency:
            continue
        # Relations are recorded between architectural parent instructions.
        # Every active memory element belongs to that instruction, so lift both
        # endpoints over the complete parent event set. Element-level ordering
        # inside an unordered Vector instruction remains absent; this only
        # preserves the parent dependency to another instruction.
        lifted = {
            (source.event_id, target.event_id)
            for source in sources
            for target in targets
        }
        if "addr" in label:
            addr.update(lifted)
        if "data" in label:
            data.update(lifted)
        if "ctrl" in label:
            ctrl.update(lifted)
    fence = _fence_relation(case, events, budget)
    return _StaticRelations(event_map, po, po_loc, fence, addr, data, ctrl, explicit_order)


def _fence_relation(
    case: LitmusCaseIR,
    events: Sequence[MemoryEvent],
    budget: _SearchBudget | None = None,
) -> Relation:
    by_instruction: dict[str, list[MemoryEvent]] = defaultdict(list)
    by_event = {event.event_id: event for event in events}
    for event in events:
        by_instruction[event.instruction_id or event.event_id].append(event)
    out: Relation = set()
    for sequence in case.harts:
        if budget is not None:
            budget.checkpoint()
        memory_positions = [
            (index, memory)
            for index, event in enumerate(sequence)
            for memory in (
                (by_event[event.event_id],)
                if event.event_id in by_event
                else by_instruction.get(event.event_id, ())
            )
        ]
        for index, event in enumerate(sequence):
            if budget is not None:
                budget.checkpoint()
            if event.kind != "fence":
                continue
            pred, succ = _fence_modes(event.instruction)
            if pred is None or succ is None:
                continue
            before = [memory for position, memory in memory_positions if position < index]
            after = [memory for position, memory in memory_positions if position > index]
            out.update(
                (left.event_id, right.event_id)
                for left in before for right in after
                if _fence_orders(pred, succ, left, right)
            )
    return out


def _fence_modes(instruction: str) -> tuple[str | None, str | None]:
    normalized = instruction.strip().lower().replace(" ", "")
    if normalized == "fence.tso":
        return "tso", "tso"
    if normalized == "fence.i":
        return None, None
    match = re.fullmatch(r"fence([^,]+),([^,]+)", normalized)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _mode_covers(mode: str, event: MemoryEvent) -> bool:
    return (event.read and "r" in mode) or (event.write and "w" in mode)


def _fence_orders(pred: str, succ: str, left: MemoryEvent, right: MemoryEvent) -> bool:
    if pred == "tso" and succ == "tso":
        return (left.write and right.write) or (left.read and (right.read or right.write))
    return _mode_covers(pred, left) and _mode_covers(succ, right)


def _rf_candidates(
    events: Sequence[MemoryEvent],
    co: _CoCandidate,
    budget: _SearchBudget,
    static: _StaticRelations | None = None,
    invariant_ppo: Relation | None = None,
) -> Iterator[_CandidateRelation]:
    writes = [event for event in events if event.write]
    reads = [event for event in events if event.read and not event.initial]
    writes_by_location: dict[str, list[MemoryEvent]] = defaultdict(list)
    for write in writes:
        for location, _value in write.write_bytes:
            writes_by_location[location].append(write)

    # Keep each transaction's byte choices together.  The old Cartesian
    # product built every byte assignment and only afterwards discarded
    # assignments containing rf/fr two-cycles.  Rejecting that condition as
    # soon as it becomes inevitable removes the dominant exponential waste in
    # mixed-size and Vector cases without removing any legal execution.
    groups: list[tuple[str, list[list[tuple[ByteEdge, frozenset[str]]]]]] = []
    for read in reads:
        byte_choices: list[list[tuple[ByteEdge, frozenset[str]]]] = []
        for location, read_value in read.read_bytes:
            budget.checkpoint()
            if read.amo:
                predecessor = co.immediate_predecessors.get(
                    (read.event_id, location)
                )
                sources = [
                    write
                    for write in writes_by_location.get(location, ())
                    if write.event_id == predecessor
                    and write.write_byte_value(location) == read_value
                ]
            else:
                sources = [
                    write
                    for write in writes_by_location.get(location, ())
                    if write.event_id != read.event_id
                    and (
                        read_value is None
                        or write.write_byte_value(location) == read_value
                    )
                ]
            budget.checkpoint(max(1, len(writes_by_location.get(location, ()))))
            if not sources:
                return
            byte_choices.append(
                [
                    (
                        (write.event_id, read.event_id, location),
                        co.successors.get(
                            (write.event_id, location), frozenset()
                        ) - {read.event_id},
                    )
                    for write in sources
                ]
            )
        byte_choices.sort(key=len)
        groups.append((read.event_id, byte_choices))
    groups.sort(
        key=lambda item: (
            _choice_product_size(item[1]),
            item[0],
        )
    )

    selected: list[ByteEdge] = []

    def partial_execution_has_cycle() -> bool:
        if static is None:
            return False
        rf_pairs = {(write, read) for write, read, _location in selected}
        fr_pairs = {
            (read, later)
            for source, read, location in selected
            for later in co.successors.get((source, location), ())
            if later != read
        }
        if _find_cycle(
            co.relation | static.po_loc | rf_pairs | fr_pairs,
            budget,
        ) is not None:
            return True
        rfe = {
            (write, read)
            for write, read in rf_pairs
            if static.event_map[write].hart != static.event_map[read].hart
        }
        return _find_cycle(
            co.relation | (invariant_ppo or set()) | rfe | fr_pairs,
            budget,
        ) is not None

    def assign_group(group_index: int) -> Iterator[_CandidateRelation]:
        budget.checkpoint()
        if group_index == len(groups):
            byte_edges = set(selected)
            yield _CandidateRelation(
                {(write, read) for write, read, _location in byte_edges},
                byte_edges,
            )
            return

        _read_id, byte_domains = groups[group_index]
        sources: set[str] = set()
        fr_targets: set[str] = set()

        def assign_byte(byte_index: int) -> Iterator[_CandidateRelation]:
            budget.checkpoint()
            if byte_index == len(byte_domains):
                if partial_execution_has_cycle():
                    byte_edges = set(selected)
                    yield _CandidateRelation(
                        {
                            (write, read)
                            for write, read, _location in byte_edges
                        },
                        byte_edges,
                        partial=True,
                    )
                    return
                yield from assign_group(group_index + 1)
                return
            for edge, later_writes in byte_domains[byte_index]:
                budget.checkpoint()
                source = edge[0]
                if source in fr_targets or sources & later_writes:
                    continue
                old_sources = set(sources)
                old_targets = set(fr_targets)
                sources.add(source)
                fr_targets.update(later_writes)
                selected.append(edge)
                yield from assign_byte(byte_index + 1)
                selected.pop()
                sources.clear()
                sources.update(old_sources)
                fr_targets.clear()
                fr_targets.update(old_targets)

        yield from assign_byte(0)

    yield from assign_group(0)


def _choice_product_size(
    choices: Sequence[Sequence[object]],
) -> int:
    size = 1
    for choice in choices:
        size *= len(choice)
    return size


def _co_candidates(
    events: Sequence[MemoryEvent],
    budget: _SearchBudget,
    acyclic_bases: Sequence[Relation] = (),
) -> Iterator[_CoCandidate]:
    event_map = {event.event_id: event for event in events}
    by_location: dict[str, list[str]] = defaultdict(list)
    initial_by_location: dict[str, str] = {}
    normal: list[str] = []
    for event in events:
        budget.checkpoint()
        if event.write and not event.initial:
            normal.append(event.event_id)
        for location, _value in event.write_bytes:
            by_location[location].append(event.event_id)
            if event.initial:
                initial_by_location[location] = event.event_id
    for location, writers in by_location.items():
        initial = [writer for writer in writers if event_map[writer].initial]
        if len(initial) != 1:
            raise RvwmoSolverError(f"location {location} does not have exactly one initial write")
    overlap_pairs = tuple(
        (left, right)
        for index, left in enumerate(normal)
        for right in normal[index + 1 :]
        if _overlap(event_map[left], event_map[right])
    )
    budget.checkpoint(max(1, len(normal) * max(len(normal) - 1, 0) // 2))
    components = [
        component
        for component in _write_components(normal, overlap_pairs)
        if len(component) > 1
    ]
    components.sort(
        key=lambda component: sum(
            left in component and right in component
            for left, right in overlap_pairs
        ),
        reverse=True,
    )

    def build_candidate(normal_order: Relation) -> _CoCandidate:
        budget.checkpoint()
        byte_edges: ByteRelation = set()
        for location, writers in sorted(by_location.items()):
            budget.checkpoint()
            initial = initial_by_location[location]
            active = [writer for writer in writers if writer != initial]
            byte_edges.update((initial, writer, location) for writer in active)
            byte_edges.update(
                (left, right, location)
                for left, right in normal_order
                if left in active and right in active
            )
        successors: dict[tuple[str, str], frozenset[str]] = {}
        immediate_predecessors: dict[tuple[str, str], str] = {}
        latest_by_location: dict[str, str] = {}
        for location, writers in sorted(by_location.items()):
            initial = initial_by_location[location]
            active = [writer for writer in writers if writer != initial]
            ordered = [initial] + sorted(
                active,
                key=lambda writer: sum(
                    (other, writer) in normal_order
                    for other in active
                    if other != writer
                ),
            )
            for index, writer in enumerate(ordered):
                successors[(writer, location)] = frozenset(ordered[index + 1 :])
                if index:
                    immediate_predecessors[(writer, location)] = ordered[index - 1]
            latest_by_location[location] = ordered[-1]
        return _CoCandidate(
            {(left, right) for left, right, _location in byte_edges},
            byte_edges,
            successors,
            immediate_predecessors,
            latest_by_location,
        )

    def assign_component(
        component_index: int,
        normal_order: Relation,
    ) -> Iterator[_CoCandidate]:
        budget.checkpoint()
        if component_index == len(components):
            yield build_candidate(normal_order)
            return
        component = components[component_index]
        for orientation in _component_orientations(
            component, overlap_pairs, budget
        ):
            combined = normal_order | orientation
            if any(
                _find_cycle(base | combined, budget) is not None
                for base in acyclic_bases
            ):
                continue
            yield from assign_component(component_index + 1, combined)

    yield from assign_component(0, set())


def _resolve_amo_transactions(
    events: Sequence[MemoryEvent],
    co: _CoCandidate,
    budget: _SearchBudget,
) -> tuple[MemoryEvent, ...] | None:
    """Evaluate AMOs from their immediate coherence predecessors.

    Each byte reads its immediate coherence predecessor. Different bytes may
    legitimately have different source transactions after a narrower
    partial-overlap write. The AMO still writes its whole W/D footprint as one
    transaction, so no observer can see an intermediate AMO state.
    """

    event_map = {event.event_id: event for event in events}
    resolved = dict(event_map)
    amo_ids = {event.event_id for event in events if event.amo}
    unresolved = set(amo_ids)
    while unresolved:
        budget.checkpoint()
        progressed = False
        for event_id in sorted(unresolved):
            budget.checkpoint()
            event = resolved[event_id]
            predecessors = tuple(
                co.immediate_predecessors.get((event_id, location))
                for location in event.footprint
            )
            if any(source is None for source in predecessors):
                return None
            if any(source_id in unresolved for source_id in predecessors):
                continue
            old_bytes = [
                resolved[source_id].write_byte_value(location)
                for source_id, location in zip(predecessors, event.footprint)
                if source_id is not None
            ]
            if any(value is None for value in old_bytes):
                return None
            old = _bytes_value(int(value) for value in old_bytes)
            if event.amo_operand is None or event.amo_width_bytes is None:
                return None
            new = apply_amo(event.amo_op, event.amo_width_bytes, old, event.amo_operand)
            if event.read_value is not None and event.read_value != old:
                return None
            if event.write_value is not None and event.write_value != new:
                return None
            resolved[event_id] = replace(
                event,
                read_value=old,
                write_value=new,
                read_bytes=_value_bytes(event.footprint, old),
                write_bytes=_value_bytes(event.footprint, new),
            )
            unresolved.remove(event_id)
            progressed = True
        if not progressed:
            return None
    return tuple(resolved[event.event_id] for event in events)


def _immediate_co_predecessor(
    event_id: str,
    location: str,
    co: ByteRelation,
    budget: _SearchBudget | None = None,
) -> str | None:
    predecessors: set[str] = set()
    for before, after, byte in co:
        if budget is not None:
            budget.checkpoint()
        if after == event_id and byte == location:
            predecessors.add(before)
    immediate: set[str] = set()
    for candidate in predecessors:
        if budget is not None:
            budget.checkpoint()
        blocked = False
        for middle in predecessors:
            if budget is not None:
                budget.checkpoint()
            if (
                middle != candidate
                and (candidate, middle, location) in co
                and (middle, event_id, location) in co
            ):
                blocked = True
                break
        if not blocked:
            immediate.add(candidate)
    return next(iter(immediate)) if len(immediate) == 1 else None


def _final_values_match(
    events: Sequence[MemoryEvent],
    co: _CoCandidate,
    final_values: Mapping[str, int],
    budget: _SearchBudget,
) -> bool:
    event_map = {event.event_id: event for event in events}
    locations = {location for event in events for location in event.byte_locations}
    for location in locations:
        budget.checkpoint()
        target = _final_byte_value(final_values, location)
        if target is None:
            continue
        latest = co.latest_by_location.get(location)
        if latest is None or event_map[latest].write_byte_value(location) != target:
            return False
    return True


def _latest_byte_write(
    writes: Sequence[str],
    location: str,
    ordering: ByteRelation,
    budget: _SearchBudget | None = None,
) -> str | None:
    latest: list[str] = []
    for write in writes:
        if budget is not None:
            budget.checkpoint()
        superseded = False
        for other in writes:
            if budget is not None:
                budget.checkpoint()
            if other != write and (write, other, location) in ordering:
                superseded = True
                break
        if not superseded:
            latest.append(write)
    return latest[0] if len(latest) == 1 else None


def _bytes_value(values: Iterable[int]) -> int:
    return sum((value & 0xFF) << (8 * index) for index, value in enumerate(values))


def _write_components(
    writes: Sequence[str],
    overlap_pairs: Sequence[Pair],
) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, set[str]] = {write: set() for write in writes}
    for left, right in overlap_pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[tuple[str, ...]] = []
    unseen = set(writes)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(adjacency[current] - component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _component_orientations(
    component: Sequence[str],
    overlap_pairs: Sequence[Pair],
    budget: _SearchBudget,
) -> Iterator[Relation]:
    pairs = tuple(
        pair
        for pair in overlap_pairs
        if pair[0] in component and pair[1] in component
    )
    if not pairs:
        yield set()
        return
    relation: Relation = set()

    def orient(pair_index: int) -> Iterator[Relation]:
        budget.checkpoint()
        if pair_index == len(pairs):
            yield set(relation)
            return
        left, right = pairs[pair_index]
        for edge in ((left, right), (right, left)):
            relation.add(edge)
            if _find_cycle(relation, budget) is None:
                yield from orient(pair_index + 1)
            relation.remove(edge)

    yield from orient(0)


def _latest_write(writes: Sequence[str], ordering: Relation) -> str:
    latest = [
        write
        for write in writes
        if not any((write, other) in ordering for other in writes if other != write)
    ]
    if len(latest) != 1:
        raise RvwmoSolverError("overlapping writes do not have a unique latest transaction")
    return latest[0]


def _check_execution(
    events: Sequence[MemoryEvent],
    static: _StaticRelations,
    rf_candidate: _CandidateRelation,
    co_candidate: _CoCandidate,
    budget: _SearchBudget,
    invariant_ppo_rules: Mapping[str, Relation],
) -> tuple[Execution, str | None, tuple[str, ...] | None]:
    event_map = static.event_map
    rf = rf_candidate.relation
    co = co_candidate.relation
    fr_bytes = _from_read_bytes(
        rf_candidate.bytes,
        co_candidate.bytes,
        budget,
        co_candidate.successors,
    )
    fr = {(read, write) for read, write, _location in fr_bytes}
    rfe = {
        (write, read) for write, read in rf
        if event_map[write].hart != event_map[read].hart
    }
    rfi = rf - rfe
    fre = {
        (read, write) for read, write in fr
        if event_map[read].hart != event_map[write].hart
    }
    coe = {
        (left, right) for left, right in co
        if event_map[left].hart != event_map[right].hart
    }
    ppo_rules = _ppo_relations(
        events,
        static,
        rf,
        rfi,
        rf_candidate.bytes,
        budget,
        invariant_ppo_rules,
    )
    ppo = set().union(*ppo_rules.values()) if ppo_rules else set()
    execution = Execution(
        set(rf),
        set(co),
        fr,
        set(static.po),
        set(static.po_loc),
        ppo,
        ppo_rules,
        set(rf_candidate.bytes),
        set(co_candidate.bytes),
        fr_bytes,
        rf_candidate.partial,
    )

    coherence = co | rf | fr | static.po_loc
    cycle = _find_cycle(coherence, budget)
    if cycle:
        return execution, "Coherence", cycle
    model = co | rfe | fr | ppo
    cycle = _find_cycle(model, budget)
    if cycle:
        return execution, "Model", cycle
    if not _atomicity_ok(events, rf_candidate.bytes, co_candidate, budget):
        # Report the equivalent cat relation when possible.
        atomic_pairs = _compose(fre, coe)
        cycle = tuple(next(iter(atomic_pairs))) if atomic_pairs else None
        return execution, "Atomic", cycle
    return execution, None, None


def _from_read_bytes(
    rf: ByteRelation,
    co: ByteRelation,
    budget: _SearchBudget | None = None,
    co_successors: Mapping[tuple[str, str], frozenset[str]] | None = None,
) -> ByteRelation:
    if co_successors is None:
        built_successors: dict[tuple[str, str], set[str]] = defaultdict(set)
        for before, after, location in co:
            if budget is not None:
                budget.checkpoint()
            built_successors[(before, location)].add(after)
        successors: Mapping[tuple[str, str], Iterable[str]] = built_successors
    else:
        successors = co_successors
    result: ByteRelation = set()
    for source, read, location in rf:
        if budget is not None:
            budget.checkpoint()
        result.update(
            (read, later, location)
            for later in successors.get((source, location), ())
            if later != read
        )
    return result


def _ppo_invariant_relations(
    events: Sequence[MemoryEvent],
    static: _StaticRelations,
    budget: _SearchBudget,
) -> dict[str, Relation]:
    """Build PPO rules that do not depend on the selected reads-from edges."""

    budget.checkpoint(max(1, len(events)))
    reads = {event.event_id for event in events if event.read and not event.initial}
    writes = {event.event_id for event in events if event.write and not event.initial}
    memory = reads | writes
    aq = {event.event_id for event in events if event.aq}
    rl = {event.event_id for event in events if event.rl}
    rcsc = {event.event_id for event in events if event.rcsc}
    addr_to_memory = {
        (left, right)
        for left, right in static.addr
        if left in memory and right in memory
    }
    return {
        "r1": {
            (left, right)
            for left, right in static.po_loc
            if left in memory and right in writes
        },
        "r2": set(),
        "r3": set(),
        "r4": set(static.fence),
        "r5": {
            (left, right)
            for left, right in static.po
            if left in aq and right in memory
        },
        "r6": {
            (left, right)
            for left, right in static.po
            if left in memory and right in rl
        },
        "r7": {
            (left, right)
            for left, right in static.po
            if left in rcsc and right in rcsc
        },
        # The internal R->W half of an RMW is represented by the immediate-co
        # atomicity check rather than a self edge.
        "r8": set(),
        "r9": addr_to_memory,
        "r10": {
            (left, right)
            for left, right in static.data
            if left in memory and right in writes
        },
        "r11": {
            (left, right)
            for left, right in static.ctrl
            if left in memory and right in writes
        },
        "r12": set(),
        "r13": _compose(
            addr_to_memory,
            {
                (left, right)
                for left, right in static.po
                if right in writes
            },
            budget,
        ),
        "vector-element-order": set(static.preserved_order),
    }


def _ppo_relations(
    events: Sequence[MemoryEvent],
    static: _StaticRelations,
    rf: Relation,
    rfi: Relation,
    rf_bytes: ByteRelation,
    budget: _SearchBudget,
    invariant: Mapping[str, Relation] | None = None,
) -> dict[str, Relation]:
    budget.checkpoint(max(1, len(events)))
    reads = {event.event_id for event in events if event.read and not event.initial}
    writes = {event.event_id for event in events if event.write and not event.initial}
    memory = reads | writes
    amo = {event.event_id for event in events if event.amo}
    po_loc_no_w = {
        (left, right)
        for left, right in static.po_loc
        if left in reads and right in reads
        and not any(
            middle in writes
            and (left, middle) in static.po_loc
            and (middle, right) in static.po_loc
            for middle in memory
        )
    }
    r2 = {
        (left, right)
        for left, right in po_loc_no_w
        if _loads_read_different_writes(
            static.event_map[left],
            static.event_map[right],
            rf_bytes,
            static.event_map,
            static.po,
            budget,
        )
    }
    r3 = {(left, right) for left, right in rfi if left in amo and right in reads}
    base = {
        key: set(value)
        for key, value in (
            invariant or _ppo_invariant_relations(events, static, budget)
        ).items()
    }
    dep_to_write = {
        (left, right)
        for left, right in base["r9"] | base["r10"]
        if right in writes
    }
    r12 = _compose(
        dep_to_write,
        {(left, right) for left, right in rfi if right in reads},
        budget,
    )
    base["r2"] = r2
    base["r3"] = r3
    base["r12"] = r12
    return base


def _loads_read_different_writes(
    left: MemoryEvent,
    right: MemoryEvent,
    rf_bytes: ByteRelation,
    event_map: Mapping[str, MemoryEvent],
    po: Relation,
    budget: _SearchBudget,
) -> bool:
    common = left.byte_locations & right.byte_locations
    if not common:
        return False
    sources = {
        (read, location): write
        for write, read, location in rf_bytes
    }
    writes = {
        event.event_id
        for event in event_map.values()
        if event.write and not event.initial
    }
    for location in common:
        budget.checkpoint()
        if sources.get((left.event_id, location)) == sources.get((right.event_id, location)):
            continue
        intervening = False
        for middle in writes:
            budget.checkpoint()
            middle_event = event_map[middle]
            if (
                (left.event_id, middle) in po
                and (middle, right.event_id) in po
                and middle_event.writes_byte(location)
            ):
                intervening = True
                break
        if not intervening:
            return True
    return False


def _atomicity_ok(
    events: Sequence[MemoryEvent],
    rf: ByteRelation,
    co: _CoCandidate,
    budget: _SearchBudget,
) -> bool:
    source_for_read = {
        (read, location): write
        for write, read, location in rf
    }
    for event in events:
        budget.checkpoint()
        if not event.amo:
            continue
        for location in event.footprint:
            budget.checkpoint()
            source = source_for_read.get((event.event_id, location))
            if source is None or source != co.immediate_predecessors.get(
                (event.event_id, location)
            ):
                return False
    return True


def _compose(
    left: Relation,
    right: Relation,
    budget: _SearchBudget | None = None,
) -> Relation:
    by_start: dict[str, set[str]] = defaultdict(set)
    for middle, target in right:
        if budget is not None:
            budget.checkpoint()
        by_start[middle].add(target)
    result: Relation = set()
    for source, middle in left:
        if budget is not None:
            budget.checkpoint()
        result.update((source, target) for target in by_start.get(middle, ()))
    return result


def _find_cycle(
    relation: Relation,
    budget: _SearchBudget | None = None,
) -> tuple[str, ...] | None:
    graph: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for source, target in relation:
        if budget is not None:
            budget.checkpoint()
        graph[source].add(target)
        nodes.update((source, target))
    state: dict[str, int] = {}
    stack: list[str] = []
    position: dict[str, int] = {}

    def visit(node: str) -> tuple[str, ...] | None:
        if budget is not None:
            budget.checkpoint()
        state[node] = 1
        position[node] = len(stack)
        stack.append(node)
        for target in sorted(graph.get(node, ())):
            if budget is not None:
                budget.checkpoint()
            if state.get(target, 0) == 0:
                cycle = visit(target)
                if cycle:
                    return cycle
            elif state.get(target) == 1:
                start = position[target]
                return tuple(stack[start:] + [target])
        stack.pop()
        position.pop(node, None)
        state[node] = 2
        return None

    for node in sorted(nodes):
        if state.get(node, 0) == 0:
            cycle = visit(node)
            if cycle:
                return cycle
    return None


def _integer(value: str, field: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise RvwmoSolverError(f"{field} is not an integer: {value}") from exc


def _pairs_json(relation: Iterable[Pair]) -> list[list[str]]:
    return [[left, right] for left, right in sorted(relation)]


def _byte_pairs_json(relation: Iterable[ByteEdge]) -> list[dict[str, str]]:
    return [
        {"src": left, "dst": right, "byte": location}
        for left, right, location in sorted(relation)
    ]
