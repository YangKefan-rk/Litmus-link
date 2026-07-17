from __future__ import annotations

"""Small, self-contained RVWMO execution-graph solver.

The solver intentionally targets scalar main-memory LitmusCaseIR programs.  It
does not parse or execute ``riscv.cat`` dynamically; instead it implements the
relations and three axioms from the riscv.cat shipped with herdtools7 7.58:

* acyclic co | rf | fr | po-loc
* acyclic co | rfe | fr | ppo
* empty rmw & (fre ; coe)

The implementation is independent of herd7 and is therefore usable on closed
machines.  ``herd7`` remains a valuable differential oracle and the native
generation path can request both backends.
"""

import itertools
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

from .litmus_ir import LitmusCaseIR


Pair = tuple[str, str]
Relation = set[Pair]


class RvwmoSolverError(ValueError):
    pass


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

    @property
    def rcsc(self) -> bool:
        return self.amo and (self.aq or self.rl)

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
        }


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

    def to_json(self) -> dict:
        return {
            "schema": "litmus-link.embedded-rvwmo.v1",
            "status": self.status,
            "tool": "litmus-link-rvwmo",
            "backend": "embedded",
            "model": "riscv.cat",
            "model_revision": "herdtools7-7.58-riscv-cat",
            "verdict": self.verdict,
            "allowed": self.allowed,
            "candidates": self.candidates,
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
    started = time.monotonic()
    try:
        events = _memory_events(case)
        final_values = _final_values(case.exists)
        static = _static_relations(case, events)
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

    violations: dict[str, int] = defaultdict(int)
    examples: dict[str, tuple[str, ...]] = {}
    candidates = 0
    consistent = 0
    try:
        for rf in _rf_candidates(events):
            for co in _co_candidates(events, final_values):
                if time.monotonic() - started > timeout_seconds:
                    raise _SearchLimit("timeout")
                candidates += 1
                if candidates > max_candidates:
                    raise _SearchLimit("candidate_limit")
                execution, failure, cycle = _check_execution(events, static, rf, co)
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
                        events=events,
                        execution=execution,
                        violation_counts=violations,
                        example_cycles=examples,
                    )
                violations[failure] += 1
                if cycle and failure not in examples:
                    examples[failure] = cycle
    except _SearchLimit as exc:
        return EmbeddedVerdict(
            status="inconclusive",
            verdict="unknown",
            allowed=None,
            candidates=min(candidates, max_candidates),
            consistent_candidates=consistent,
            reason=(
                f"Embedded RVWMO search stopped at {exc.reason}; "
                "a forbidden verdict requires exhaustive search."
            ),
            elapsed_seconds=time.monotonic() - started,
            events=events,
            violation_counts=violations,
            example_cycles=examples,
        )

    return EmbeddedVerdict(
        status="verified",
        verdict="forbidden",
        allowed=False,
        candidates=candidates,
        consistent_candidates=0,
        reason="Every candidate execution violates at least one riscv.cat axiom.",
        elapsed_seconds=time.monotonic() - started,
        events=events,
        violation_counts=violations,
        example_cycles=examples,
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


class _SearchLimit(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _memory_events(case: LitmusCaseIR) -> tuple[MemoryEvent, ...]:
    memory: list[MemoryEvent] = []
    locations: set[str] = set()
    for hart, sequence in enumerate(case.harts):
        order = 0
        for event in sequence:
            if event.kind not in {"load", "store", "amo"}:
                continue
            if not event.location:
                raise RvwmoSolverError(f"memory event {event.event_id} has no location")
            if event.register.startswith("v") or event.instruction.lstrip().startswith("v"):
                raise RvwmoSolverError("embedded RVWMO solver currently supports scalar memory events only")
            classified = _classify_events(event, hart, order)
            memory.extend(classified)
            locations.update(item.location for item in classified)
            order += 1
    if not memory:
        raise RvwmoSolverError("case has no scalar memory events")
    init = _initial_values(case.init_lines)
    for location in sorted(locations):
        memory.append(
            MemoryEvent(
                event_id=f"init:{location}",
                hart=None,
                order=-1,
                location=location,
                read=False,
                write=True,
                read_value=None,
                write_value=init.get(location, 0),
                initial=True,
                instruction_id=f"init:{location}",
                byte_offset=_location_byte_offset(location),
                access_size=1 if _location_byte_offset(location) is not None else 0,
                atomicity_model="initial",
            )
        )
    ids = [event.event_id for event in memory]
    if len(ids) != len(set(ids)):
        raise RvwmoSolverError("case contains duplicate memory event ids")
    return tuple(memory)


def _classify_events(event: LitmusEvent, hart: int, order: int) -> tuple[MemoryEvent, ...]:
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
                    instruction_id=event.event_id,
                    byte_offset=absolute_byte,
                    access_size=access.size_bytes,
                    atomicity_model="byte_level_no_mag",
                )
            )
        return tuple(out)
    return (_classify_event(event, hart, order),)


def _classify_event(event: LitmusEvent, hart: int, order: int) -> MemoryEvent:
    instruction = event.instruction.lower().replace(" ", "")
    aq = ".aq" in instruction or ".aq.rl" in instruction
    rl = ".rl" in instruction or ".aq.rl" in instruction
    value = _integer(event.value, f"event {event.event_id} value") if event.value else None
    if event.kind == "load":
        if value is None:
            raise RvwmoSolverError(f"load {event.event_id} has no target read value")
        return MemoryEvent(
            event.event_id, hart, order, event.location, True, False, value, None,
            aq=aq, rl=rl, instruction_id=event.event_id,
        )
    if event.kind == "store":
        if value is None:
            raise RvwmoSolverError(f"store {event.event_id} has no write value")
        return MemoryEvent(
            event.event_id, hart, order, event.location, False, True, None, value,
            aq=aq, rl=rl, instruction_id=event.event_id,
        )
    if "amoor" in instruction:
        if value is None:
            raise RvwmoSolverError(f"AMO load {event.event_id} has no target read value")
        return MemoryEvent(
            event.event_id, hart, order, event.location, True, True, value, value,
            aq=aq, rl=rl, amo=True, instruction_id=event.event_id,
        )
    if "amoswap" in instruction:
        if value is None:
            raise RvwmoSolverError(f"AMO store {event.event_id} has no write value")
        return MemoryEvent(
            event.event_id, hart, order, event.location, True, True, None, value,
            aq=aq, rl=rl, amo=True, instruction_id=event.event_id,
        )
    raise RvwmoSolverError(f"unsupported AMO instruction in {event.event_id}: {event.instruction}")


def _initial_values(lines: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in lines:
        for term in line.split(";"):
            match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*(-?(?:0x[0-9a-fA-F]+|\d+))\s*", term)
            if match:
                out[match.group(1)] = int(match.group(2), 0)
    return out


def _final_values(exists: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for location, value in re.findall(
        r"(?<![:\w])([A-Za-z_]\w*(?:\[\d+\])?)\s*=\s*(-?(?:0x[0-9a-fA-F]+|\d+))",
        exists,
    ):
        out[location] = int(value, 0)
    return out


def _location_byte_offset(location: str) -> int | None:
    match = re.fullmatch(r"[A-Za-z_]\w*\[(\d+)\]", location)
    return int(match.group(1)) if match else None


def _static_relations(
    case: LitmusCaseIR,
    events: Sequence[MemoryEvent],
) -> _StaticRelations:
    event_map = {event.event_id: event for event in events}
    po: Relation = set()
    by_hart: dict[int, list[MemoryEvent]] = defaultdict(list)
    for event in events:
        if event.hart is not None:
            by_hart[event.hart].append(event)
    for sequence in by_hart.values():
        po.update(
            (left.event_id, right.event_id)
            for left in sequence
            for right in sequence
            if left.order < right.order
        )
    po_loc = {
        pair for pair in po
        if event_map[pair[0]].location == event_map[pair[1]].location
    }
    addr: Relation = set()
    data: Relation = set()
    ctrl: Relation = set()
    by_instruction: dict[str, list[MemoryEvent]] = defaultdict(list)
    for event in events:
        by_instruction[event.instruction_id or event.event_id].append(event)
    for relation in case.relations:
        sources = by_instruction.get(relation.src, ())
        targets = by_instruction.get(relation.dst, ())
        if not sources or not targets:
            continue
        lifted = {
            (source.event_id, target.event_id)
            for source in sources
            for target in targets
        }
        label = relation.label.lower()
        if "addr" in label:
            addr.update(lifted)
        if "data" in label:
            data.update(lifted)
        if "ctrl" in label:
            ctrl.update(lifted)
    fence = _fence_relation(case, events)
    return _StaticRelations(event_map, po, po_loc, fence, addr, data, ctrl)


def _fence_relation(
    case: LitmusCaseIR,
    events: Sequence[MemoryEvent],
) -> Relation:
    by_instruction: dict[str, list[MemoryEvent]] = defaultdict(list)
    for event in events:
        by_instruction[event.instruction_id or event.event_id].append(event)
    out: Relation = set()
    for sequence in case.harts:
        memory_positions = [
            (index, memory)
            for index, event in enumerate(sequence)
            for memory in by_instruction.get(event.event_id, ())
        ]
        for index, event in enumerate(sequence):
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


def _rf_candidates(events: Sequence[MemoryEvent]) -> Iterator[Relation]:
    writes = [event for event in events if event.write]
    reads = [event for event in events if event.read and not event.initial]
    choices: list[list[Pair]] = []
    for read in reads:
        sources = [
            write
            for write in writes
            if write.location == read.location
            and write.event_id != read.event_id
            and (read.read_value is None or write.write_value == read.read_value)
        ]
        if not sources:
            return
        choices.append([(write.event_id, read.event_id) for write in sources])
    for selected in itertools.product(*choices):
        yield set(selected)


def _co_candidates(events: Sequence[MemoryEvent], final_values: Mapping[str, int]) -> Iterator[Relation]:
    by_location: dict[str, list[MemoryEvent]] = defaultdict(list)
    for event in events:
        if event.write:
            by_location[event.location].append(event)
    orders: list[list[tuple[str, ...]]] = []
    for location in sorted(by_location):
        initial = [event for event in by_location[location] if event.initial]
        normal = [event for event in by_location[location] if not event.initial]
        if len(initial) != 1:
            raise RvwmoSolverError(f"location {location} does not have exactly one initial write")
        location_orders: list[tuple[str, ...]] = []
        for permutation in itertools.permutations(normal):
            order = (initial[0],) + permutation
            target = final_values.get(location)
            if target is not None and order[-1].write_value != target:
                continue
            location_orders.append(tuple(event.event_id for event in order))
        if not location_orders:
            return
        orders.append(location_orders)
    for selected in itertools.product(*orders):
        co: Relation = set()
        for order in selected:
            co.update((left, right) for index, left in enumerate(order) for right in order[index + 1 :])
        yield co


def _check_execution(
    events: Sequence[MemoryEvent],
    static: _StaticRelations,
    rf: Relation,
    co: Relation,
) -> tuple[Execution, str | None, tuple[str, ...] | None]:
    event_map = static.event_map
    fr = _from_read(rf, co)
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
    ppo_rules = _ppo_relations(events, static, rf, rfi)
    ppo = set().union(*ppo_rules.values()) if ppo_rules else set()
    execution = Execution(set(rf), set(co), fr, set(static.po), set(static.po_loc), ppo, ppo_rules)

    coherence = co | rf | fr | static.po_loc
    cycle = _find_cycle(coherence)
    if cycle:
        return execution, "Coherence", cycle
    model = co | rfe | fr | ppo
    cycle = _find_cycle(model)
    if cycle:
        return execution, "Model", cycle
    if not _atomicity_ok(events, rf, co):
        # Report the equivalent cat relation when possible.
        atomic_pairs = _compose(fre, coe)
        cycle = tuple(next(iter(atomic_pairs))) if atomic_pairs else None
        return execution, "Atomic", cycle
    return execution, None, None


def _from_read(rf: Relation, co: Relation) -> Relation:
    successors: dict[str, set[str]] = defaultdict(set)
    for before, after in co:
        successors[before].add(after)
    return {
        (read, later)
        for source, read in rf
        for later in successors.get(source, ())
        if later != read
    }


def _ppo_relations(
    events: Sequence[MemoryEvent],
    static: _StaticRelations,
    rf: Relation,
    rfi: Relation,
) -> dict[str, Relation]:
    reads = {event.event_id for event in events if event.read and not event.initial}
    writes = {event.event_id for event in events if event.write and not event.initial}
    memory = reads | writes
    amo = {event.event_id for event in events if event.amo}
    aq = {event.event_id for event in events if event.aq}
    rl = {event.event_id for event in events if event.rl}
    rcsc = {event.event_id for event in events if event.rcsc}

    r1 = {(left, right) for left, right in static.po_loc if left in memory and right in writes}
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
    source_for_read = {read: write for write, read in rf}
    rsw = {
        (left, right)
        for left in reads for right in reads
        if left != right and source_for_read.get(left) == source_for_read.get(right)
    }
    r2 = po_loc_no_w - rsw
    r3 = {(left, right) for left, right in rfi if left in amo and right in reads}
    r4 = set(static.fence)
    r5 = {(left, right) for left, right in static.po if left in aq and right in memory}
    r6 = {(left, right) for left, right in static.po if left in memory and right in rl}
    r7 = {(left, right) for left, right in static.po if left in rcsc and right in rcsc}
    # r8 is the internal R->W half of an RMW.  MemoryEvent collapses those two
    # halves into one event, so its ordering effect is represented by the
    # immediate-co atomicity check rather than a self edge.
    r8: Relation = set()
    r9 = {(left, right) for left, right in static.addr if left in memory and right in memory}
    r10 = {(left, right) for left, right in static.data if left in memory and right in writes}
    r11 = {(left, right) for left, right in static.ctrl if left in memory and right in writes}
    dep_to_write = {(left, right) for left, right in static.addr | static.data if left in memory and right in writes}
    r12 = _compose(dep_to_write, {(left, right) for left, right in rfi if right in reads})
    addr_to_memory = {(left, right) for left, right in static.addr if left in memory and right in memory}
    r13 = _compose(addr_to_memory, {(left, right) for left, right in static.po if right in writes})
    return {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "r5": r5,
        "r6": r6,
        "r7": r7,
        "r8": r8,
        "r9": r9,
        "r10": r10,
        "r11": r11,
        "r12": r12,
        "r13": r13,
    }


def _atomicity_ok(events: Sequence[MemoryEvent], rf: Relation, co: Relation) -> bool:
    source_for_read = {read: write for write, read in rf}
    by_location: dict[str, list[str]] = defaultdict(list)
    for event in events:
        if event.write:
            by_location[event.location].append(event.event_id)
    for event in events:
        if not event.amo:
            continue
        source = source_for_read.get(event.event_id)
        if source is None:
            return False
        order = sorted(
            by_location[event.location],
            key=lambda candidate: sum((other, candidate) in co for other in by_location[event.location]),
        )
        try:
            if order.index(event.event_id) != order.index(source) + 1:
                return False
        except ValueError:
            return False
    return True


def _compose(left: Relation, right: Relation) -> Relation:
    by_start: dict[str, set[str]] = defaultdict(set)
    for middle, target in right:
        by_start[middle].add(target)
    return {(source, target) for source, middle in left for target in by_start.get(middle, ())}


def _find_cycle(relation: Relation) -> tuple[str, ...] | None:
    graph: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for source, target in relation:
        graph[source].add(target)
        nodes.update((source, target))
    state: dict[str, int] = {}
    stack: list[str] = []
    position: dict[str, int] = {}

    def visit(node: str) -> tuple[str, ...] | None:
        state[node] = 1
        position[node] = len(stack)
        stack.append(node)
        for target in sorted(graph.get(node, ())):
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
