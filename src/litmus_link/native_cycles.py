from __future__ import annotations

"""Exhaustive finite-domain cycle generation for native scalar litmus tests."""

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator, Sequence

from .native_edges import DIFFERENT, EXTERNAL, LOCAL, SAME, NativeEdge


@dataclass(frozen=True)
class NativeCycle:
    edges: tuple[NativeEdge, ...]
    family: str = ""
    annotations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.edges) < 2:
            raise ValueError("a litmus cycle requires at least two edges")
        if self.annotations and len(self.annotations) != len(self.edges):
            raise ValueError("cycle annotations must contain one mode per event")
        unknown = set(self.annotations) - {"P", "Aq", "Rl", "AR"}
        if unknown:
            raise ValueError(f"unknown event annotations: {', '.join(sorted(unknown))}")

    @property
    def labels(self) -> tuple[str, ...]:
        if not self.annotations:
            return tuple(edge.label for edge in self.edges)
        out = []
        for index, edge in enumerate(self.edges):
            source = self.annotations[index]
            target = self.annotations[(index + 1) % len(self.edges)]
            suffix = "" if source == "P" and target == "P" else source + target
            out.append(edge.label + suffix)
        return tuple(out)

    @property
    def size(self) -> int:
        return len(self.edges)

    @property
    def nprocs(self) -> int:
        return len(_process_components(self.edges))

    @property
    def canonical_key(self) -> tuple[str, ...]:
        labels = self.labels
        return min(labels[index:] + labels[:index] for index in range(len(labels)))

    def canonical(self) -> "NativeCycle":
        labels = self.labels
        index = min(range(len(labels)), key=lambda offset: labels[offset:] + labels[:offset])
        annotations = self.annotations[index:] + self.annotations[:index] if self.annotations else ()
        return NativeCycle(self.edges[index:] + self.edges[:index], self.family, annotations)

    def to_json(self) -> dict:
        return {
            "family": self.family,
            "size": self.size,
            "nprocs": self.nprocs,
            "cycle": " ".join(self.labels),
            "annotations": list(self.annotations or ("P",) * len(self.edges)),
            "canonical_key": list(self.canonical_key),
            "edges": [edge.to_json() for edge in self.edges],
        }


@dataclass(frozen=True)
class CycleDecision:
    accepted: bool
    reason: str


@dataclass(frozen=True)
class EnumerationReport:
    candidates: int
    accepted: int
    duplicate: int
    excluded: dict[str, int]

    def to_json(self) -> dict:
        return {
            "candidates": self.candidates,
            "accepted": self.accepted,
            "duplicate": self.duplicate,
            "excluded": dict(sorted(self.excluded.items())),
        }


def validate_cycle(
    edges: Sequence[NativeEdge],
    *,
    max_procs: int | None = None,
    exact_procs: bool = False,
    max_accesses_per_proc: int | None = None,
) -> CycleDecision:
    if len(edges) < 2:
        return CycleDecision(False, "too_short")
    for index, edge in enumerate(edges):
        following = edges[(index + 1) % len(edges)]
        if edge.dst != following.src:
            return CycleDecision(False, "direction_mismatch")
    external_count = sum(edge.scope == EXTERNAL for edge in edges)
    if external_count == 1:
        return CycleDecision(False, "single_external_edge")
    if external_count == 0:
        return CycleDecision(False, "no_external_communication")
    if not any(edge.scope == LOCAL for edge in edges):
        return CycleDecision(False, "no_local_ordering_edge")
    if not _constraints_satisfiable(edges, scope=True):
        return CycleDecision(False, "hart_constraint_conflict")
    if not _constraints_satisfiable(edges, scope=False):
        return CycleDecision(False, "location_constraint_conflict")
    components = _process_components(edges)
    nprocs = len(components)
    if max_procs is not None:
        if exact_procs and nprocs != max_procs:
            return CycleDecision(False, "hart_count_not_exact")
        if not exact_procs and nprocs > max_procs:
            return CycleDecision(False, "too_many_harts")
    if max_accesses_per_proc is not None and any(len(component) > max_accesses_per_proc for component in components):
        return CycleDecision(False, "too_many_accesses_per_hart")
    return CycleDecision(True, "accepted")


def enumerate_template_cycles(
    family: str,
    axes: Sequence[NativeEdge | Sequence[NativeEdge]],
    *,
    max_procs: int | None = None,
    exact_procs: bool = False,
    max_accesses_per_proc: int | None = None,
) -> tuple[list[NativeCycle], EnumerationReport]:
    choices = [tuple(value) if isinstance(value, (list, tuple)) else (value,) for value in axes]
    cycles: list[NativeCycle] = []
    seen: set[tuple[str, ...]] = set()
    excluded: Counter[str] = Counter()
    candidates = 0
    duplicate = 0
    for selected in product(*choices):
        candidates += 1
        decision = validate_cycle(
            selected,
            max_procs=max_procs,
            exact_procs=exact_procs,
            max_accesses_per_proc=max_accesses_per_proc,
        )
        if not decision.accepted:
            excluded[decision.reason] += 1
            continue
        cycle = NativeCycle(tuple(selected), family).canonical()
        key = cycle.canonical_key
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        cycles.append(cycle)
    cycles.sort(key=lambda cycle: cycle.canonical_key)
    return cycles, EnumerationReport(candidates, len(cycles), duplicate, dict(excluded))


def enumerate_relation_cycles(
    edges: Sequence[NativeEdge],
    *,
    min_size: int = 2,
    max_size: int = 4,
    max_procs: int | None = 2,
    exact_procs: bool = False,
    max_accesses_per_proc: int | None = None,
    require_labels: Iterable[str] = (),
) -> Iterator[NativeCycle]:
    """Enumerate every canonical cycle in the configured finite edge domain.

    The DFS prunes direction-incompatible prefixes.  It intentionally has no
    hidden output cap; callers may stop consuming the iterator, while an audit
    caller can exhaust it to prove the complete configured domain was covered.
    """
    if min_size < 2 or max_size < min_size:
        raise ValueError("cycle size range must satisfy 2 <= min_size <= max_size")
    required = set(require_labels)
    unknown = required - {edge.label for edge in edges}
    if unknown:
        raise ValueError(f"required edges are outside the domain: {', '.join(sorted(unknown))}")
    by_src = {
        direction: tuple(edge for edge in edges if edge.src == direction)
        for direction in {edge.src for edge in edges}
    }
    seen: set[tuple[str, ...]] = set()

    def extend(prefix: tuple[NativeEdge, ...], target_size: int) -> Iterator[NativeCycle]:
        if len(prefix) == target_size:
            if prefix[-1].dst != prefix[0].src:
                return
            if required and not required.issubset({edge.label for edge in prefix}):
                return
            decision = validate_cycle(
                prefix,
                max_procs=max_procs,
                exact_procs=exact_procs,
                max_accesses_per_proc=max_accesses_per_proc,
            )
            if not decision.accepted:
                return
            cycle = NativeCycle(prefix).canonical()
            if cycle.canonical_key in seen:
                return
            seen.add(cycle.canonical_key)
            yield cycle
            return
        next_direction = prefix[-1].dst
        for edge in by_src.get(next_direction, ()):
            yield from extend(prefix + (edge,), target_size)

    for size in range(min_size, max_size + 1):
        for first in edges:
            yield from extend((first,), size)


def vertex_directions(edges: Sequence[NativeEdge]) -> tuple[str, ...]:
    decision = validate_cycle(edges)
    if not decision.accepted:
        raise ValueError(f"cannot derive vertices from invalid cycle: {decision.reason}")
    return tuple(edge.src for edge in edges)


def process_ids(edges: Sequence[NativeEdge]) -> tuple[int, ...]:
    components = _components(edges, equal_when=lambda edge: edge.scope == LOCAL)
    ordered = sorted(components, key=min)
    mapping = {vertex: proc for proc, component in enumerate(ordered) for vertex in component}
    return tuple(mapping[index] for index in range(len(edges)))


def location_ids(edges: Sequence[NativeEdge]) -> tuple[int, ...]:
    components = _components(edges, equal_when=lambda edge: edge.location == SAME)
    ordered = sorted(components, key=min)
    mapping = {vertex: location for location, component in enumerate(ordered) for vertex in component}
    return tuple(mapping[index] for index in range(len(edges)))


def _process_components(edges: Sequence[NativeEdge]) -> list[set[int]]:
    return _components(edges, equal_when=lambda edge: edge.scope == LOCAL)


def _constraints_satisfiable(edges: Sequence[NativeEdge], *, scope: bool) -> bool:
    equal_when = (lambda edge: edge.scope == LOCAL) if scope else (lambda edge: edge.location == SAME)
    different_when = (lambda edge: edge.scope == EXTERNAL) if scope else (lambda edge: edge.location == DIFFERENT)
    components = _components(edges, equal_when=equal_when)
    owner = {vertex: index for index, component in enumerate(components) for vertex in component}
    for index, edge in enumerate(edges):
        if different_when(edge) and owner[index] == owner[(index + 1) % len(edges)]:
            return False
    return True


def _components(edges: Sequence[NativeEdge], *, equal_when) -> list[set[int]]:
    parent = list(range(len(edges)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, edge in enumerate(edges):
        if equal_when(edge):
            union(index, (index + 1) % len(edges))
    groups: dict[int, set[int]] = {}
    for vertex in range(len(edges)):
        groups.setdefault(find(vertex), set()).add(vertex)
    return list(groups.values())
