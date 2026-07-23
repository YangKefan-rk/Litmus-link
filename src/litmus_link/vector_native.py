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
import multiprocessing
import os
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from itertools import islice, permutations, product
from math import prod as _product
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .amo import AMO_OPERATIONS, AMO_ORDERINGS, AmoSpec
from .fusion_layout import (
    EndpointFootprint,
    FUSION_OVERLAP_LAYOUTS,
    FusionAddressLayout,
    FusionLayoutError,
    synthesize_address_layout,
)
from .fusion_values import FusionValuePlan, synthesize_fusion_values
from .litmus_ir import (
    LitmusCaseIR,
    LitmusEvent,
    MemoryAccess,
    _rebase_vector,
    _scalar_memory_operands,
    _vector_instruction,
    _vector_metadata,
    _vector_setup,
    _vector_store_broadcast_events,
)
from .models import Combination, Decision, GENERATED, GeneratedCase
from .naming import vector_native_case_identity
from .native_cycles import NativeCycle, location_ids, vertex_directions
from .native_edges import READ, WRITE
from .native_scalar import (
    DEFAULT_NATIVE_MECHANISMS,
    lower_native_cycle,
    native_template_cycles,
)
from .profiles import (
    NANHU_VECTOR_ATTRIBUTES,
    WHOLE_REGISTER_VECTOR_OPS,
    VECTOR_INDEX_EEWS,
    VECTOR_LENGTHS,
    VECTOR_LMULS,
    VECTOR_MASKS,
    VECTOR_NFIELDS,
    VECTOR_OPS,
    VECTOR_TAILS,
    VECTOR_WHOLE_NREGS,
    VECTOR_WIDTHS,
    vector_footprint_kind,
    vector_memory_config_legal,
)
from .renderer import render_ir
from .solver import solve_generated_case


ProgressCallback = Callable[[int, int, str], None]

VECTOR_ALIGNMENTS = ("aligned",)

SCALAR_WIDTHS = ("b", "h", "w", "d")
SCALAR_WIDTH_BYTES = {"b": 1, "h": 2, "w": 4, "d": 8}
AMO_WIDTHS = ("w", "d")
AMO_WIDTH_BYTES = {"w": 4, "d": 8}
ENDPOINT_CATEGORIES = ("scalar", "amo", "vector")
ENDPOINT_COMPOSITIONS = (
    "vector_only",
    "vector_scalar",
    "vector_amo",
    "vector_scalar_amo",
)
_COMPOSITION_CATEGORIES = {
    "vector_only": frozenset({"vector"}),
    "vector_scalar": frozenset({"vector", "scalar"}),
    "vector_amo": frozenset({"vector", "amo"}),
    "vector_scalar_amo": frozenset({"vector", "scalar", "amo"}),
}

VECTOR_SAMPLE_BALANCED = "balanced"
VECTOR_SAMPLE_DOMAIN_WEIGHTED = "domain_weighted"
VECTOR_SAMPLE_MODES = (
    VECTOR_SAMPLE_BALANCED,
    VECTOR_SAMPLE_DOMAIN_WEIGHTED,
)

VECTOR_GENERATE_ALL = "all"
VECTOR_GENERATION_MODES = (*VECTOR_SAMPLE_MODES, VECTOR_GENERATE_ALL)

VECTOR_SAMPLING_LABELS = {
    VECTOR_SAMPLE_BALANCED: "balanced-skeleton-coverage-random-without-replacement",
    VECTOR_SAMPLE_DOMAIN_WEIGHTED: "domain-weighted-random-without-replacement",
    VECTOR_GENERATE_ALL: "exhaustive-deterministic-enumeration",
}

VECTOR_VERIFICATION_EFFORTS = ("interactive", "balanced", "thorough")
VECTOR_VERIFICATION_LIMITS: Mapping[str, Mapping[str, Any]] = {
    "interactive": {
        "max_candidates": 50_000,
        "timeout_seconds": 5.0,
        "max_search_steps": 1_000_000,
        "max_memory_events": 256,
        "external_max_projections": 2,
        "external_timeout": 3,
        "external_case_limit": 4,
    },
    "balanced": {
        "max_candidates": 100_000,
        "timeout_seconds": 10.0,
        "max_search_steps": 2_000_000,
        "max_memory_events": 256,
        "external_max_projections": 16,
        "external_timeout": 10,
        "external_case_limit": 32,
    },
    "thorough": {
        "max_candidates": 1_000_000,
        "timeout_seconds": 30.0,
        "max_search_steps": 10_000_000,
        "max_memory_events": None,
        "external_max_projections": 64,
        "external_timeout": 30,
        "external_case_limit": None,
    },
}


def _amo_mask_satisfiable(cycle: NativeCycle, amo_mask: int) -> bool:
    """Return whether AMO read facets admit a coherent writer order.

    A relation-cycle ``R`` vertex normally has no write facet.  Selecting AMO
    for that vertex adds a write to coherence and constrains its read facet to
    observe either the initial value or the cycle's explicit ``rf`` source.
    The check is structural: fusion layouts place every same-location endpoint
    around a common naturally aligned byte, so an intervening writer would
    necessarily invalidate the required AMO source.
    """

    directions, locations, relations = _cycle_structure_key(cycle)
    return _amo_structure_satisfiable(directions, locations, relations, amo_mask)


@lru_cache(maxsize=65_536)
def _cycle_structure_key(
    cycle: NativeCycle,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    """Collapse label-only variants that have identical memory structure."""

    return (
        tuple(edge.src for edge in cycle.edges),
        location_ids(cycle.edges),
        tuple(
            edge.relation if edge.relation in {"rf", "fr", "co"} else "local"
            for edge in cycle.edges
        ),
    )


def _cached_fusion_layout(
    structure: tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]],
    categories: tuple[str, ...],
    widths: tuple[int, ...],
    layout: str,
) -> FusionAddressLayout:
    result, reason, detail = _fusion_layout_result(
        structure,
        categories,
        widths,
        layout,
    )
    if result is None:
        raise FusionLayoutError(reason, detail)
    return result


@lru_cache(maxsize=65_536)
def _fusion_layout_result(
    structure: tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]],
    categories: tuple[str, ...],
    widths: tuple[int, ...],
    layout: str,
) -> tuple[FusionAddressLayout | None, str, str]:
    directions, locations, relations = structure
    if len(categories) != len(directions) or len(widths) != len(directions):
        return (
            None,
            "excluded_unsatisfiable_value_layout",
            "endpoint categories and widths do not match the relation cycle",
        )
    amo_mask = sum(
        1 << vertex
        for vertex, category in enumerate(categories)
        if category == "amo"
    )
    if not _amo_structure_satisfiable(
        directions,
        locations,
        relations,
        amo_mask,
    ):
        return (
            None,
            "excluded_unsatisfiable_value_layout",
            "AMO read facets cannot realize the cycle's rf/fr/co witness",
        )
    groups = tuple(
        tuple(
            vertex
            for vertex, actual in enumerate(locations)
            if actual == location
        )
        for location in sorted(set(locations))
    )
    footprints = tuple(
        EndpointFootprint(vertex, width, frozenset(range(width)))
        for vertex, width in enumerate(widths)
    )
    try:
        return synthesize_address_layout(footprints, groups, layout), "", ""
    except FusionLayoutError as exc:
        return None, exc.reason, str(exc)


@lru_cache(maxsize=None)
def _amo_structure_satisfiable(
    directions: tuple[str, ...],
    locations: tuple[int, ...],
    relations: tuple[str, ...],
    amo_mask: int,
) -> bool:
    if amo_mask < 0 or amo_mask >> len(directions):
        return False

    rf_sources = {
        (vertex + 1) % len(relations): vertex
        for vertex, relation in enumerate(relations)
        if relation == "rf"
    }
    writers = {
        vertex
        for vertex, direction in enumerate(directions)
        if direction == WRITE or amo_mask & (1 << vertex)
    }
    constraints: set[tuple[int, int]] = set()
    for vertex, relation in enumerate(relations):
        target = (vertex + 1) % len(relations)
        if relation == "co":
            constraints.add((vertex, target))
        elif relation == "fr":
            source = rf_sources.get(vertex)
            if source is not None:
                constraints.add((source, target))
            if amo_mask & (1 << vertex):
                constraints.add((vertex, target))
        elif relation == "rf" and amo_mask & (1 << target):
            constraints.add((vertex, target))

    for location in sorted(set(locations)):
        location_writers = tuple(
            vertex
            for vertex in sorted(writers)
            if locations[vertex] == location
        )
        location_constraints = tuple(
            (before, after)
            for before, after in constraints
            if locations[before] == location and locations[after] == location
        )
        read_amos = tuple(
            vertex
            for vertex in location_writers
            if directions[vertex] == READ and amo_mask & (1 << vertex)
        )

        found = False
        for order in permutations(location_writers):
            positions = {vertex: index for index, vertex in enumerate(order)}
            if any(
                positions[before] >= positions[after]
                for before, after in location_constraints
            ):
                continue
            valid = True
            for vertex in read_amos:
                source = rf_sources.get(vertex)
                position = positions[vertex]
                if source is None:
                    valid = position == 0
                else:
                    valid = (
                        source in positions
                        and position > 0
                        and order[position - 1] == source
                    )
                if not valid:
                    break
            if valid:
                found = True
                break
        if not found:
            return False
    return True


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

    @property
    def width_bytes(self) -> int:
        values = dict(self.params or {})
        if self.category == "vector":
            return int(str(values.get("sew", "e32")).removeprefix("e")) // 8
        if self.category == "amo":
            return int(values.get("amo_width_bytes", 4))
        return int(values.get("width_bytes", 4))

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
    overlap_layout: str = "same_start"

    def __post_init__(self) -> None:
        if self.alignment not in VECTOR_ALIGNMENTS:
            raise ValueError(f"unknown Vector assignment alignment: {self.alignment}")
        if self.overlap_layout not in FUSION_OVERLAP_LAYOUTS:
            raise ValueError(
                f"unknown Vector assignment overlap layout: {self.overlap_layout}"
            )
        directions = tuple(edge.src for edge in self.cycle.edges)
        if len(self.choices) != len(directions):
            raise ValueError("Vector assignment must provide one endpoint choice per cycle vertex")
        if any(choice.direction != direction for choice, direction in zip(self.choices, directions)):
            raise ValueError("Vector endpoint choice direction does not match its cycle vertex")
        if not any(choice.is_vector for choice in self.choices):
            raise ValueError("Vector assignment must contain at least one Vector memory endpoint")
        if any(choice.width_bytes < 1 for choice in self.choices):
            raise ValueError("endpoint widths must be positive")

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.cycle.family,
            self.cycle.canonical_key,
            self.alignment,
            self.overlap_layout,
            *(choice.choice_id for choice in self.choices),
        )


@dataclass(frozen=True)
class VectorNativeDomain:
    cycles: tuple[NativeCycle, ...]
    read_choices: tuple[EndpointChoice, ...]
    write_choices: tuple[EndpointChoice, ...]
    relation_audit: Mapping[str, Any]
    alignments: tuple[str, ...]
    overlap_layouts: tuple[str, ...]
    compositions: tuple[str, ...]
    endpoint_audit: Mapping[str, Any]
    request_exclusions: Mapping[str, int]

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

        categories = _endpoint_categories(payload)
        vector_choices, vector_audit = (
            _vector_choices_with_audit(payload)
            if "vector" in categories
            else ((), _empty_endpoint_audit())
        )
        scalar_choices = _scalar_choices(payload) if "scalar" in categories else ()
        amo_choices, amo_audit = (
            _amo_choices_with_audit(payload)
            if "amo" in categories
            else ((), _empty_endpoint_audit())
        )
        requested_alignments = _selected(payload, "alignments", ("aligned",))
        if not requested_alignments:
            raise ValueError("select at least one Vector alignment")
        known_misaligned = {
            "misalign_same16",
            "misalign_cross16",
            "misalign_cross64",
        }
        unknown_alignments = (
            set(requested_alignments) - set(VECTOR_ALIGNMENTS) - known_misaligned
        )
        if unknown_alignments:
            raise ValueError(f"unknown Vector alignment(s): {', '.join(sorted(unknown_alignments))}")
        alignments = tuple(
            alignment
            for alignment in requested_alignments
            if alignment in VECTOR_ALIGNMENTS
        )
        request_exclusions, supported_scope_requested = _fusion_request_audit(
            payload,
            categories,
            requested_alignments,
        )
        if not supported_scope_requested:
            alignments = ()
        overlap_layouts = _selected(payload, "overlap_layouts", ("same_start",))
        if not overlap_layouts:
            raise ValueError("select at least one fusion overlap layout")
        unknown_layouts = set(overlap_layouts) - set(FUSION_OVERLAP_LAYOUTS)
        if unknown_layouts:
            raise ValueError(
                f"unknown fusion overlap layout(s): {', '.join(sorted(unknown_layouts))}"
            )
        compositions = _selected(
            payload,
            "endpoint_compositions",
            tuple(
                name
                for name, required in _COMPOSITION_CATEGORIES.items()
                if required <= set(categories)
            ),
        )
        unknown_compositions = set(compositions) - set(ENDPOINT_COMPOSITIONS)
        if unknown_compositions:
            raise ValueError(
                f"unknown endpoint composition(s): {', '.join(sorted(unknown_compositions))}"
            )
        if not compositions:
            raise ValueError("select at least one endpoint composition")
        if any(not _COMPOSITION_CATEGORIES[name] <= set(categories) for name in compositions):
            raise ValueError("endpoint composition requires a disabled endpoint category")

        choices = tuple((*scalar_choices, *amo_choices, *vector_choices))
        read_choices = tuple(choice for choice in choices if choice.direction == READ)
        write_choices = tuple(choice for choice in choices if choice.direction == WRITE)
        impossible_compositions = _impossible_endpoint_compositions(
            compositions,
            read_choices,
            write_choices,
            cycles,
        )
        if impossible_compositions:
            request_exclusions = {
                **dict(request_exclusions),
                "excluded_unsatisfiable_endpoint_composition": len(
                    impossible_compositions
                ),
            }
        return cls(
            tuple(cycles),
            read_choices,
            write_choices,
            relation_audit,
            alignments,
            overlap_layouts,
            compositions,
            {
                "scalar": {
                    "raw_configurations": len(scalar_choices),
                    "generated_endpoint_choices": len(scalar_choices),
                    "excluded": {},
                },
                "amo": amo_audit,
                "vector": vector_audit,
            },
            request_exclusions,
        )

    def choices_for(self, direction: str) -> tuple[EndpointChoice, ...]:
        return self.read_choices if direction == READ else self.write_choices

    def count_for_cycle(self, cycle: NativeCycle) -> int:
        return self._cycle_count_breakdown(cycle)["generated"]

    @cached_property
    def _breakdowns_by_structure(
        self,
    ) -> Mapping[
        tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]],
        Mapping[str, Any],
    ]:
        representatives: dict[
            tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]],
            NativeCycle,
        ] = {}
        for cycle in self.cycles:
            representatives.setdefault(_cycle_structure_key(cycle), cycle)
        return {
            key: self._calculate_cycle_count_breakdown(cycle)
            for key, cycle in representatives.items()
        }

    @cached_property
    def _width_multiplicities(
        self,
    ) -> Mapping[str, Mapping[str, Mapping[int, int]]]:
        by_direction: dict[str, dict[str, Counter[int]]] = {}
        for direction in (READ, WRITE):
            by_category: dict[str, Counter[int]] = {}
            for choice in self.choices_for(direction):
                by_category.setdefault(choice.category, Counter())[choice.width_bytes] += 1
            by_direction[direction] = by_category
        return by_direction

    def _cycle_count_breakdown(self, cycle: NativeCycle) -> Mapping[str, Any]:
        return self._breakdowns_by_structure[_cycle_structure_key(cycle)]

    def _calculate_cycle_count_breakdown(self, cycle: NativeCycle) -> dict[str, Any]:
        if not self.alignments:
            return {"raw": 0, "generated": 0, "excluded": {}}
        directions = vertex_directions(cycle.edges)
        locations = location_ids(cycle.edges)
        category_domains = [
            tuple(self._width_multiplicities[direction])
            for direction in directions
        ]
        allowed_compositions = {
            _COMPOSITION_CATEGORIES[name] for name in self.compositions
        }
        location_groups = [
            tuple(
                vertex
                for vertex, actual_location in enumerate(locations)
                if actual_location == location
            )
            for location in sorted(set(locations))
        ]

        raw = 0
        generated = 0
        excluded: Counter[str] = Counter()
        for categories in product(*category_domains):
            if frozenset(categories) not in allowed_compositions:
                continue
            width_counts = [
                self._width_multiplicities[direction][category]
                for direction, category in zip(directions, categories)
            ]
            endpoint_count = 1
            for counts in width_counts:
                endpoint_count *= sum(counts.values())

            # Naturally aligned power-of-two accesses can realize a nontrivial
            # low/center/high containment layout iff at least one abstract
            # location contains two different endpoint widths.  Count its
            # complement analytically: every endpoint in each location picks
            # the same width.
            all_locations_uniform = 1
            for vertices in location_groups:
                uniform_at_location = sum(
                    _product(width_counts[vertex].get(width, 0) for vertex in vertices)
                    for width in (1, 2, 4, 8)
                )
                all_locations_uniform *= uniform_at_location
            mixed_width_count = endpoint_count - all_locations_uniform

            amo_mask = sum(
                1 << vertex
                for vertex, category in enumerate(categories)
                if category == "amo"
            )
            structure_ok = _amo_mask_satisfiable(cycle, amo_mask)
            for layout in self.overlap_layouts:
                weighted = endpoint_count * len(self.alignments)
                raw += weighted
                if not structure_ok or layout == "disjoint_control":
                    excluded["excluded_unsatisfiable_value_layout"] += weighted
                    continue
                eligible = (
                    endpoint_count
                    if layout == "same_start"
                    else mixed_width_count
                ) * len(self.alignments)
                generated += eligible
                excluded["excluded_unsatisfiable_value_layout"] += weighted - eligible
        return {
            "raw": raw,
            "generated": generated,
            "excluded": {
                reason: count for reason, count in excluded.items() if count
            },
        }

    @cached_property
    def total_cases(self) -> int:
        return sum(self.count_for_cycle(cycle) for cycle in self.cycles)

    def random_assignments(
        self,
        limit: int,
        seed: int,
        strategy: str = VECTOR_SAMPLE_BALANCED,
    ) -> list[VectorAssignment]:
        if strategy == VECTOR_SAMPLE_BALANCED:
            return self._balanced_assignments(limit, seed)
        if strategy == VECTOR_SAMPLE_DOMAIN_WEIGHTED:
            return self._domain_weighted_assignments(limit, seed)
        raise ValueError(
            f"unknown Vector sampling strategy: {strategy}; "
            f"expected one of {', '.join(VECTOR_SAMPLE_MODES)}"
        )

    def _balanced_assignments(self, limit: int, seed: int) -> list[VectorAssignment]:
        if limit < 1 or self.total_cases < 1:
            return []
        target = min(limit, self.total_cases)
        family_cycles: dict[str, tuple[NativeCycle, ...]] = {
            family: tuple(cycle for cycle in self.cycles if cycle.family == family)
            for family in dict.fromkeys(cycle.family for cycle in self.cycles)
        }
        family_totals = {
            family: sum(self.count_for_cycle(cycle) for cycle in cycles)
            for family, cycles in family_cycles.items()
        }
        family_order = list(family_cycles)
        random.Random(seed).shuffle(family_order)
        quotas = _balanced_quotas(family_order, family_totals, target)

        sampled: dict[str, list[VectorAssignment]] = {}
        for family in family_order:
            quota = quotas[family]
            if quota < 1:
                sampled[family] = []
                continue
            family_seed = _derived_seed(seed, family)
            sampled[family] = self._domain_weighted_assignments_from_cycles(
                family_cycles[family],
                quota,
                family_seed,
                family_totals[family],
            )

        # Interleave families so the GUI never shows a long block from one
        # skeleton even though each family is sampled independently.
        assignments: list[VectorAssignment] = []
        max_family_size = max((len(values) for values in sampled.values()), default=0)
        for index in range(max_family_size):
            for family in family_order:
                values = sampled[family]
                if index < len(values):
                    assignments.append(values[index])
        return assignments

    def _domain_weighted_assignments(
        self, limit: int, seed: int
    ) -> list[VectorAssignment]:
        return self._domain_weighted_assignments_from_cycles(
            self.cycles,
            limit,
            seed,
            self.total_cases,
        )

    def _domain_weighted_assignments_from_cycles(
        self,
        cycles: Sequence[NativeCycle],
        limit: int,
        seed: int,
        total: int | None = None,
    ) -> list[VectorAssignment]:
        if limit < 1:
            return []
        selected_cycles = tuple(cycles)
        if total is None:
            total = sum(self.count_for_cycle(cycle) for cycle in selected_cycles)
        if total < 1:
            return []
        target = min(limit, total)
        weights = [self.count_for_cycle(cycle) for cycle in selected_cycles]
        cumulative: list[int] = []
        running = 0
        for weight in weights:
            running += weight
            cumulative.append(running)

        rng = random.Random(seed)
        selected: dict[tuple[Any, ...], VectorAssignment] = {}
        def random_assignment(cycle: NativeCycle) -> VectorAssignment:
            directions = vertex_directions(cycle.edges)
            while True:
                choices = tuple(
                    rng.choice(self.choices_for(direction)) for direction in directions
                )
                alignment = rng.choice(self.alignments)
                overlap_layout = rng.choice(self.overlap_layouts)
                if not self._composition_allowed(choices):
                    continue
                try:
                    self._layout_for(cycle, choices, overlap_layout)
                except FusionLayoutError:
                    continue
                return VectorAssignment(
                    cycle,
                    choices,
                    alignment,
                    overlap_layout,
                )

        attempts = 0
        max_attempts = max(target * 40, 1000)
        while len(selected) < target and attempts < max_attempts:
            attempts += 1
            cycle_index = bisect.bisect_right(cumulative, rng.randrange(total))
            assignment = random_assignment(selected_cycles[cycle_index])
            selected.setdefault(assignment.key, assignment)

        if len(selected) < target:
            for assignment in self._assignments_for_cycles(selected_cycles):
                selected.setdefault(assignment.key, assignment)
                if len(selected) >= target:
                    break
        return list(selected.values())

    def _coverage_weighted_assignments(
        self, limit: int, seed: int
    ) -> list[VectorAssignment]:
        # Family balancing is handled by the caller. Within one family, sample
        # the exact generated domain without replacement by rejection over its
        # finite endpoint product; invalid layout states are never returned.
        return self._domain_weighted_assignments(limit, seed)

    def _composition_allowed(self, choices: Sequence[EndpointChoice]) -> bool:
        categories = frozenset(choice.category for choice in choices)
        return any(
            categories == _COMPOSITION_CATEGORIES[name]
            for name in self.compositions
        )

    def _layout_for(
        self,
        cycle: NativeCycle,
        choices: Sequence[EndpointChoice],
        layout: str,
    ) -> FusionAddressLayout:
        return _cached_fusion_layout(
            _cycle_structure_key(cycle),
            tuple(choice.category for choice in choices),
            tuple(choice.width_bytes for choice in choices),
            layout,
        )

    def assignments(self) -> Iterator[VectorAssignment]:
        yield from self._assignments_for_cycles(self.cycles)

    def _assignments_for_cycles(
        self,
        cycles: Sequence[NativeCycle],
    ) -> Iterator[VectorAssignment]:
        for cycle in cycles:
            directions = vertex_directions(cycle.edges)
            for alignment in self.alignments:
                domains = [self.choices_for(direction) for direction in directions]
                for overlap_layout in self.overlap_layouts:
                    for choices in product(*domains):
                        if not self._composition_allowed(choices):
                            continue
                        try:
                            self._layout_for(cycle, choices, overlap_layout)
                        except FusionLayoutError:
                            continue
                        yield VectorAssignment(
                            cycle, tuple(choices), alignment, overlap_layout
                        )

    def audit(self) -> dict[str, Any]:
        family_cycles: dict[str, int] = {}
        family_cases: dict[str, int] = {}
        raw = 0
        excluded: Counter[str] = Counter()
        for cycle in self.cycles:
            breakdown = self._cycle_count_breakdown(cycle)
            family_cycles[cycle.family] = family_cycles.get(cycle.family, 0) + 1
            family_cases[cycle.family] = (
                family_cases.get(cycle.family, 0) + breakdown["generated"]
            )
            raw += breakdown["raw"]
            excluded.update(breakdown["excluded"])
        return {
            "schema": "litmus-link.vector-native-audit.v2",
            "relation_cycles": len(self.cycles),
            "read_endpoint_choices": len(self.read_choices),
            "write_endpoint_choices": len(self.write_choices),
            "alignments": list(self.alignments),
            "overlap_layouts": list(self.overlap_layouts),
            "endpoint_compositions": list(self.compositions),
            "formal_scope": {
                "pbmt": 0,
                "attribute": "cacheable",
                "pma_atomic": True,
                "natural_alignment": True,
            },
            "raw_combinations": raw,
            "total_cases": self.total_cases,
            "generated": self.total_cases,
            "excluded": dict(sorted(excluded.items())),
            "excluded_illegal": sum(
                count
                for reason, count in self.request_exclusions.items()
                if reason.startswith("excluded_illegal")
            ) + sum(
                count
                for audit in self.endpoint_audit.values()
                for reason, count in dict(audit.get("excluded", {})).items()
                if reason.startswith("excluded_illegal")
            ),
            "excluded_unsupported": sum(
                count
                for reason, count in self.request_exclusions.items()
                if reason.startswith("excluded_unsupported")
            ) + sum(
                count
                for audit in self.endpoint_audit.values()
                for reason, count in dict(audit.get("excluded", {})).items()
                if reason.startswith("excluded_unsupported")
            ),
            "endpoint_domain": dict(self.endpoint_audit),
            "request_exclusions": dict(sorted(self.request_exclusions.items())),
            "hand_required": 0,
            "missing": 0,
            "family_relation_cycles": dict(sorted(family_cycles.items())),
            "family_cases": dict(sorted(family_cases.items())),
            "relation_audit": dict(self.relation_audit),
        }


def _derived_seed(seed: int, family: str) -> int:
    family_value = sum(
        (index + 1) * ord(character) for index, character in enumerate(family)
    )
    return (seed * 1_000_003 + family_value) & 0x7FFF_FFFF_FFFF_FFFF


def _impossible_endpoint_compositions(
    compositions: Sequence[str],
    read_choices: Sequence[EndpointChoice],
    write_choices: Sequence[EndpointChoice],
    cycles: Sequence[NativeCycle],
) -> tuple[str, ...]:
    categories = {
        READ: tuple(dict.fromkeys(choice.category for choice in read_choices)),
        WRITE: tuple(dict.fromkeys(choice.category for choice in write_choices)),
    }
    direction_shapes = {
        vertex_directions(cycle.edges)
        for cycle in cycles
    }
    impossible = []
    for name in compositions:
        required = _COMPOSITION_CATEGORIES[name]
        possible = any(
            all(categories[direction] for direction in shape)
            and any(
                frozenset(assignment) == required
                for assignment in product(
                    *(categories[direction] for direction in shape)
                )
            )
            for shape in direction_shapes
        )
        if not possible:
            impossible.append(name)
    return tuple(impossible)


def _balanced_quotas(
    family_order: Sequence[str],
    family_totals: Mapping[str, int],
    target: int,
) -> dict[str, int]:
    quotas = {family: 0 for family in family_order}
    active = [family for family in family_order if family_totals.get(family, 0) > 0]
    remaining = target
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        requested = max(share, 1)
        progressed = 0
        next_active: list[str] = []
        for index, family in enumerate(active):
            room = family_totals[family] - quotas[family]
            take = min(room, requested + int(share > 0 and index < extra))
            quotas[family] += take
            remaining -= take
            progressed += take
            if quotas[family] < family_totals[family]:
                next_active.append(family)
            if remaining == 0:
                break
        if progressed == 0:
            break
        active = next_active
    return quotas


def _lower_scalar_endpoint(
    event: LitmusEvent,
    choice: EndpointChoice,
    value: Any,
    offset: int,
) -> tuple[LitmusEvent, list[LitmusEvent]]:
    size = choice.width_bytes
    load = {1: "lbu", 2: "lhu", 4: "lwu", 8: "ld"}
    store = {1: "sb", 2: "sh", 4: "sw", 8: "sd"}
    base_register, data_register = _scalar_memory_operands(event)
    mnemonic = load[size] if event.kind == "load" else store[size]
    observed = (
        value.read_register_value
        if event.kind == "load"
        else value.write_value
    )
    return (
        replace(
            event,
            instruction=f"{mnemonic} {data_register},{offset}({base_register})",
            value=f"0x{int(observed or 0):x}",
            role="scalar-fusion-cycle-event",
            memory_access=MemoryAccess.create(event.location, offset, size),
        ),
        [],
    )


def _lower_amo_endpoint(
    event: LitmusEvent,
    choice: EndpointChoice,
    value: Any,
    offset: int,
    sequence: Sequence[LitmusEvent],
) -> tuple[LitmusEvent, list[LitmusEvent]]:
    params = dict(choice.params or {})
    spec = AmoSpec(
        str(params["amo_op"]),
        int(params["amo_width_bytes"]),
        str(params["amo_ordering"]),
    )
    base_register, destination = _scalar_memory_operands(event)
    temporary_count = int(event.kind == "load") + int(offset != 0)
    temporary_registers = iter(_free_temp_registers(sequence, temporary_count))
    operand_register = (
        next(temporary_registers) if event.kind == "load" else destination
    )
    address_register = base_register
    setup = (
        [
            LitmusEvent(
                f"{event.event_id}_amo_operand",
                event.hart,
                "setup",
                f"li {operand_register},{int(value.amo_operand or 0)}",
                role="amo-operand",
            )
        ]
        if event.kind == "load"
        else []
    )
    if offset:
        address_register = next(temporary_registers)
        setup.append(
            LitmusEvent(
                f"{event.event_id}_amo_base",
                event.hart,
                "setup",
                f"addi {address_register},{base_register},{offset}",
                role="amo-fusion-base",
            )
        )
    if event.kind == "store":
        destination = "x0"
    return (
        replace(
            event,
            kind="amo",
            instruction=(
                f"{spec.mnemonic} {destination},{operand_register},"
                f"({address_register})"
            ),
            value=f"0x{int(value.read_register_value or 0):x}",
            read_value=f"0x{int(value.amo_old or 0):x}",
            write_value=f"0x{int(value.amo_new or 0):x}",
            amo_op=spec.operation,
            amo_operand=f"0x{int(value.amo_operand or 0):x}",
            amo_width_bytes=spec.width_bytes,
            amo_ordering=spec.ordering,
            role="amo-fusion-cycle-event",
            memory_access=MemoryAccess.create(
                event.location,
                offset,
                spec.width_bytes,
                transaction_kind="amo_rmw",
            ),
        ),
        setup,
    )


def _fusion_init_lines(
    old_lines: Sequence[str],
    case: LitmusCaseIR,
    choices: Sequence[EndpointChoice],
    plan: FusionValuePlan,
) -> list[str]:
    location_names = set(plan.initial_bytes)
    declarations = [
        (
            f"uint8_t {name}[{len(values)}]={{"
            + ",".join(f"0x{value:02x}" for value in values)
            + "};"
        )
        for name, values in sorted(plan.initial_bytes.items())
    ]
    register_values: dict[tuple[int, str], int] = {}
    for vertex, choice in enumerate(choices):
        source = case.event_map()[f"v{vertex}"]
        if source.kind != "store":
            continue
        endpoint = plan.endpoints[vertex]
        register_values[(source.hart, source.register)] = int(
            endpoint.amo_operand
            if choice.category == "amo"
            else endpoint.write_value
            or 0
        )
    address_init: list[str] = []
    register_pattern = re.compile(
        r"\s*(\d+):(x\d+)\s*=\s*(-?(?:0x[0-9a-fA-F]+|\d+))\s*;\s*"
    )
    for line in old_lines:
        if any(
            re.fullmatch(rf"\s*{re.escape(name)}\s*=.*", line)
            for name in location_names
        ):
            continue
        match = register_pattern.fullmatch(line)
        key = (int(match.group(1)), match.group(2)) if match else None
        if key in register_values:
            address_init.append(
                f"{key[0]}:{key[1]}=0x{register_values[key]:x};"
            )
        else:
            address_init.append(line)
    return [*declarations, *address_init]


def _fusion_exists(case: LitmusCaseIR, plan: FusionValuePlan) -> str:
    source_events = case.event_map()
    terms: list[str] = []
    for vertex, value in sorted(plan.endpoints.items()):
        event = source_events[f"v{vertex}"]
        if event.kind != "load":
            continue
        terms.append(
            f"{event.hart}:{event.register}=0x{int(value.read_register_value or 0):x}"
        )
    for name, offsets in sorted(plan.observed_final_bytes.items()):
        image = plan.final_bytes[name]
        terms.extend(
            f"{name}[{offset}]=0x{image[offset]:02x}"
            for offset in offsets
        )
    if not terms:
        raise FusionLayoutError(
            "excluded_unsatisfiable_value_layout",
            "fusion case has no observable register or final-memory value",
        )
    return "(" + " /\\ ".join(terms) + ")"


def _composition_name(choices: Sequence[EndpointChoice]) -> str:
    categories = frozenset(choice.category for choice in choices)
    for name, required in _COMPOSITION_CATEGORIES.items():
        if categories == required:
            return name
    return "+".join(sorted(categories))


def lower_vector_assignment(assignment: VectorAssignment) -> GeneratedCase:
    scalar = lower_native_cycle(assignment.cycle)
    case = scalar.case_ir
    locations = location_ids(assignment.cycle.edges)
    location_names = {
        location: case.event_map()[f"v{vertex}"].location
        for vertex, location in enumerate(locations)
    }
    address_layout = _cached_fusion_layout(
        _cycle_structure_key(assignment.cycle),
        tuple(choice.category for choice in assignment.choices),
        tuple(choice.width_bytes for choice in assignment.choices),
        assignment.overlap_layout,
    )
    value_plan = synthesize_fusion_values(
        assignment.cycle,
        assignment.choices,
        address_layout,
        location_names,
    )

    harts: list[list[LitmusEvent]] = []
    init_lines = _fusion_init_lines(
        case.init_lines, case, assignment.choices, value_plan
    )
    vector_metadata: dict[str, dict[str, Any]] = {}

    for hart_id, sequence in enumerate(case.harts):
        expanded: list[LitmusEvent] = []
        for event in sequence:
            vertex = _event_vertex(event.event_id)
            if vertex is None:
                expanded.append(event)
                continue
            choice = assignment.choices[vertex]
            value = value_plan.endpoints[vertex]
            offset = address_layout.offsets[vertex]
            if choice.category == "scalar":
                scalar_event, setup = _lower_scalar_endpoint(
                    event, choice, value, offset
                )
                expanded.extend(setup)
                expanded.append(scalar_event)
                continue
            if choice.category == "amo":
                amo_event, setup = _lower_amo_endpoint(
                    event, choice, value, offset, sequence
                )
                expanded.extend(setup)
                expanded.append(amo_event)
                continue

            combination = _choice_combination(assignment.cycle.family, event, choice)
            base_register, data_register = _scalar_memory_operands(event)
            setup, extra_init = _vector_setup(combination, hart_id, f"{event.event_id}_vector")
            large_avl = str((choice.params or {}).get("vl")) in {"vl32", "vl64"}
            strided = "strided" in choice.vector_form
            scale_register = strided or choice.vector_form.startswith(
                "segment_indexed_"
            )
            temporary_registers = iter(
                _free_temp_registers(
                    sequence,
                    1 + int(large_avl) + int(scale_register) + int(offset != 0),
                )
            )
            config_register = next(temporary_registers)
            register_map = {"x10": config_register}
            if large_avl:
                register_map["x11"] = next(temporary_registers)
            if scale_register:
                register_map["x20"] = next(temporary_registers)
            setup = [
                replace(setup_event, instruction=_replace_registers(setup_event.instruction, register_map))
                for setup_event in setup
            ]
            extra_init = [_replace_registers(line, register_map) for line in extra_init]
            expanded.extend(setup)
            vector_base = base_register
            element_bytes = choice.width_bytes
            if offset:
                vector_base = next(temporary_registers)
                expanded.append(
                    LitmusEvent(
                        f"{event.event_id}_offset_base",
                        hart_id,
                        "setup",
                        f"addi {vector_base},{base_register},{offset}",
                        role="vector-fusion-base",
                    )
                )
            instruction = _replace_registers(
                _rebase_vector(_vector_instruction(combination), vector_base),
                register_map,
            )
            vector_access = MemoryAccess.create(
                event.location,
                offset,
                element_bytes,
                "aligned_atomic",
            )
            if event.kind == "store":
                expanded.extend(
                    replace(
                        broadcast,
                        instruction=_replace_registers(
                            broadcast.instruction,
                            register_map,
                        ),
                    )
                    for broadcast in _vector_store_broadcast_events(
                        combination,
                        hart_id,
                        event.event_id,
                        data_register,
                    )
                )
                expanded.append(
                    replace(
                        event,
                        instruction=instruction,
                        register="v8",
                        value=f"0x{int(value.write_value or 0):x}",
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
                        value=f"0x{int(value.read_memory_value or 0):x}",
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
                init_lines.extend(extra_init)
            metadata = dict(_vector_metadata(combination))
            footprint = vector_footprint_kind(
                choice.vector_form,
                str((choice.params or {}).get("sew", "e32")),
                str((choice.params or {}).get("lmul", "m1")),
                str((choice.params or {}).get("mask", "unmasked")),
                str((choice.params or {}).get("vl", "vl1")),
                base_offset=offset,
                nf=(choice.params or {}).get("nf"),
                whole_nreg=(choice.params or {}).get("whole_nreg"),
            )
            metadata.update(
                {
                    "base_offset_bytes": offset,
                    "alignment": "aligned",
                    "atomicity_model": "aligned_atomic",
                    "footprint": footprint,
                }
            )
            vector_metadata[event.event_id] = metadata
        harts.append(expanded)

    exists = _fusion_exists(case, value_plan)
    cycle_labels = tuple(edge.label for edge in assignment.cycle.edges)
    cycle_text = " ".join(cycle_labels)
    endpoint_choices = [choice.to_json() for choice in assignment.choices]
    identity = vector_native_case_identity(
        assignment.cycle.family or "Cycle",
        assignment.cycle.to_json(),
        cycle_labels,
        endpoint_choices,
        assignment.alignment,
        assignment.overlap_layout,
    )
    name = str(identity["machine_name"])
    relations = [
        replace(
            relation,
            label=cycle_labels[index],
            src_facet=(
                "read"
                if assignment.choices[index].category == "amo"
                and assignment.cycle.edges[index].src == READ
                else "write"
                if assignment.choices[index].category == "amo"
                else ""
            ),
            dst_facet=(
                "read"
                if assignment.choices[(index + 1) % len(assignment.choices)].category
                == "amo"
                and assignment.cycle.edges[index].dst == READ
                else "write"
                if assignment.choices[(index + 1) % len(assignment.choices)].category
                == "amo"
                else ""
            ),
        )
        for index, relation in enumerate(case.relations)
    ]
    transformed = replace(
        case,
        harts=harts,
        init_lines=init_lines,
        relations=relations,
        exists=exists,
    )
    formal, formal_reason = _formal_scope(transformed, assignment)
    metadata = dict(case.metadata)
    metadata.update(
        {
            "vectors": vector_metadata,
            "endpoint_choices": endpoint_choices,
            "source_cycle": assignment.cycle.to_json(),
            "file_identity": identity,
            "memory_layout": {
                "alignment": assignment.alignment,
                "pbmt": 0,
                "attribute": "cacheable",
                "pma_atomic": True,
                "overlap_layout": assignment.overlap_layout,
                "address_layout": address_layout.to_json(),
            },
            "value_plan": value_plan.to_json(),
            "formal_scope": formal_reason,
        }
    )
    case = replace(
        transformed,
        name=name,
        display_name=str(identity["display_name"]),
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
            "endpoint_choices": [choice.choice_id for choice in assignment.choices],
            "endpoint_composition": _composition_name(assignment.choices),
            "overlap_layout": assignment.overlap_layout,
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


_PARALLEL_SOLVER_THRESHOLD = 64
_DEFAULT_PARALLEL_SOLVER_WORKERS = 16
_MAX_PARALLEL_SOLVER_WORKERS = 64


def _embedded_solver_worker(
    job: tuple[GeneratedCase, Mapping[str, Any]],
) -> dict[str, Any]:
    case, limits = job
    return solve_generated_case(
        case,
        vector_external_check=False,
        vector_solver_limits=limits,
    ).to_json()


def _embedded_preview_solver_worker(
    job: tuple[GeneratedCase, Mapping[str, Any]],
) -> dict[str, Any]:
    """Solve one case without returning duplicated expansion data to the GUI.

    Process-pool results are pickled before they reach the parent process.  A
    normal solver result contains both case_ir's architectural events and a
    second, fully expanded Vector/event representation.  Returning the compact
    form here avoids materializing that duplicate payload in the GUI process.
    """

    return _compact_preview_solver(_embedded_solver_worker(job))


def _compact_preview_external(external: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in external.items()
        if key not in {"raw", "raw_output", "stdout", "stderr"}
    }
    projection = compact.get("projection")
    if isinstance(projection, Mapping):
        compact["projection"] = {
            key: value
            for key, value in projection.items()
            if key not in {"raw", "raw_output", "stdout", "stderr", "litmus"}
        }
    return compact


def _compact_preview_solver(solver: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the formal result and witness needed by the case inspector.

    ``case_ir`` already records parent instructions, endpoint values and
    Vector configuration.  Preview rows therefore do not need another copy of
    every expanded Vector element or every embedded MemoryEvent.  The compact
    execution witness is retained so byte-level rf/co/fr and PPO remain
    inspectable.  Full generation still uses the normal serializer and writes
    complete ``.solver.json`` files.
    """

    compact = {
        key: value
        for key, value in solver.items()
        if key not in {"raw_output", "edges", "vector"}
    }
    compact["raw_output"] = ""
    compact["edges"] = []
    vector = solver.get("vector")
    if not isinstance(vector, Mapping):
        return compact

    compact_vector = {
        key: value
        for key, value in vector.items()
        if key not in {"vector_ir", "embedded", "external"}
    }
    embedded = vector.get("embedded")
    if isinstance(embedded, Mapping):
        compact_vector["embedded"] = {
            key: value
            for key, value in embedded.items()
            if key != "events"
        }
    else:
        compact_vector["embedded"] = embedded
    external = vector.get("external")
    compact_vector["external"] = (
        _compact_preview_external(external)
        if isinstance(external, Mapping)
        else external
    )
    compact_vector["preview_compact"] = True
    compact["vector"] = compact_vector
    return compact


def _parallel_solver_workers(
    case_count: int,
    requested_workers: object | None = None,
) -> int:
    if case_count < _PARALLEL_SOLVER_THRESHOLD:
        return 1
    if requested_workers is None:
        requested = _DEFAULT_PARALLEL_SOLVER_WORKERS
    else:
        try:
            requested = int(str(requested_workers))
        except (TypeError, ValueError) as exc:
            raise ValueError("solver_workers must be an integer") from exc
        if requested < 1:
            raise ValueError("solver_workers must be positive")
    configured_cap = os.environ.get("LITMUS_LINK_SOLVER_WORKERS")
    if configured_cap is not None:
        try:
            cap = int(configured_cap)
        except ValueError as exc:
            raise ValueError("LITMUS_LINK_SOLVER_WORKERS must be an integer") from exc
        if cap < 1:
            raise ValueError("LITMUS_LINK_SOLVER_WORKERS must be positive")
        requested = min(requested, cap)
    try:
        available = max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return min(
        requested,
        available,
        _MAX_PARALLEL_SOLVER_WORKERS,
        case_count,
    )


def sample_vector_cases(
    payload: Mapping[str, Any],
    *,
    compute_verdicts: bool,
    compact_solver_results: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GeneratedCase], dict[str, Any]]:
    if progress_callback is not None:
        progress_callback(0, 0, "Building the selected relation and endpoint domain")
    domain = VectorNativeDomain.from_payload(payload)
    solver_backend = _vector_solver_backend(payload)
    effort, solver_limits, external_case_limit = _vector_verification_settings(payload)
    limit = int(payload.get("sample_limit", 1000))
    seed = int(payload.get("random_seed", 1))
    sampling = str(payload.get("preview_sampling", VECTOR_SAMPLE_BALANCED))
    if sampling not in VECTOR_SAMPLE_MODES:
        raise ValueError(
            f"unknown Vector preview sampling mode: {sampling}; "
            f"expected one of {', '.join(VECTOR_SAMPLE_MODES)}"
        )
    if progress_callback is not None:
        progress_callback(
            0,
            0,
            f"Selecting up to {limit:,} reproducible cases from {domain.total_cases:,} legal combinations",
        )
    assignments = domain.random_assignments(limit, seed, sampling)
    if progress_callback is not None:
        progress_callback(
            0,
            max(len(assignments), 1),
            (
                f"Selected {len(assignments):,} cases; starting outcome verification"
                if compute_verdicts
                else f"Selected {len(assignments):,} cases; building preview records"
            ),
        )
    cases: list[GeneratedCase] = []
    total = max(len(assignments), 1)
    solver_statuses: Counter[str] = Counter()
    external_statuses: Counter[str] = Counter()
    external_attempts = 0
    parallel_workers = (
        _parallel_solver_workers(
            len(assignments),
            payload.get("solver_workers"),
        )
        if compute_verdicts and solver_backend == "embedded"
        else 1
    )
    lowered: list[GeneratedCase] = []
    for index, assignment in enumerate(assignments, start=1):
        lowered.append(lower_vector_assignment(assignment))
        if parallel_workers > 1 and progress_callback is not None:
            progress_callback(
                index,
                total * 2,
                f"Prepared {index:,}/{len(assignments):,} cases for {parallel_workers} solver workers",
            )

    parallel_solvers: list[dict[str, Any] | None] | None = None
    if parallel_workers > 1:
        parallel_solvers = [None] * len(lowered)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=parallel_workers,
            mp_context=context,
        ) as executor:
            worker = (
                _embedded_preview_solver_worker
                if compact_solver_results
                else _embedded_solver_worker
            )
            futures = {
                executor.submit(
                    worker,
                    (case, solver_limits),
                ): index
                for index, case in enumerate(lowered)
            }
            completed = 0
            for future in as_completed(futures):
                case_index = futures[future]
                solver = future.result()
                parallel_solvers[case_index] = solver
                completed += 1
                status = str(solver.get("status", "unknown") or "unknown")
                external_status = _solver_external_status(solver)
                solver_statuses[status] += 1
                external_statuses[external_status] += 1
                if progress_callback is not None:
                    progress_callback(
                        total + completed,
                        total * 2,
                        _verification_progress(
                            "Verified",
                            completed,
                            len(lowered),
                            solver_statuses,
                            external_statuses,
                        ),
                    )

    for index, case in enumerate(lowered, start=1):
        if compute_verdicts:
            if parallel_solvers is not None:
                selected_solver = parallel_solvers[index - 1]
                if selected_solver is None:
                    raise RuntimeError("parallel Vector solver returned no result")
                solver = selected_solver
            else:
                request_external = solver_backend == "crosscheck" and (
                    external_case_limit is None
                    or external_attempts < external_case_limit
                )
                solver = solve_generated_case(
                    case,
                    vector_external_check=request_external,
                    vector_solver_limits=solver_limits,
                ).to_json()
                if compact_solver_results:
                    solver = _compact_preview_solver(solver)
                external_status = _solver_external_status(solver)
                if request_external and external_status != "not_run":
                    external_attempts += 1
                elif (
                    solver_backend == "crosscheck"
                    and not request_external
                    and solver.get("status") == "verified"
                ):
                    solver = _mark_external_batch_skipped(
                        solver,
                        external_case_limit,
                    )
        else:
            solver = _unchecked_solver()
        status = str(solver.get("status", "unknown") or "unknown")
        external_status = _solver_external_status(solver)
        if parallel_solvers is None:
            solver_statuses[status] += 1
            external_statuses[external_status] += 1
        cases.append(replace(case, solver=solver))
        if progress_callback is not None and parallel_solvers is None:
            progress_callback(
                index,
                total,
                _verification_progress(
                    "Verified" if compute_verdicts else "Sampled",
                    index,
                    len(assignments),
                    solver_statuses,
                    external_statuses,
                ),
            )
    audit = domain.audit()
    audit.update(
        {
            "sample_seed": seed,
            "sample_requested": limit,
            "sampled_cases": len(cases),
            "sampling_mode": sampling,
            "sampling": VECTOR_SAMPLING_LABELS[sampling],
            "solver_backend": solver_backend,
            "solver_workers": parallel_workers,
            "verification_effort": effort,
            "verification_limits": {
                **solver_limits,
                "external_case_limit": external_case_limit,
            },
            "solver_status": dict(sorted(solver_statuses.items())),
            "external_status": dict(sorted(external_statuses.items())),
        }
    )
    return cases, audit


def generate_vector_cases(
    payload: Mapping[str, Any],
    out_dir: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback(0, 0, "Building the selected relation and endpoint domain")
    domain = VectorNativeDomain.from_payload(payload)
    generation_mode = str(payload.get("generation_mode", VECTOR_SAMPLE_BALANCED))
    if generation_mode not in VECTOR_GENERATION_MODES:
        raise ValueError(
            f"unknown Vector generation mode: {generation_mode}; "
            f"expected one of {', '.join(VECTOR_GENERATION_MODES)}"
        )
    seed = int(payload.get("random_seed", 1))
    if generation_mode == VECTOR_GENERATE_ALL:
        requested: int | None = None
        assignments: Iterable[VectorAssignment] = domain.assignments()
        target = domain.total_cases
    else:
        requested = int(payload.get("generate_limit", 10000))
        if requested < 1:
            raise ValueError("generate_limit must be positive for sampled generation")
        if progress_callback is not None:
            progress_callback(
                0,
                0,
                f"Selecting up to {requested:,} reproducible cases from {domain.total_cases:,} legal combinations",
            )
        sampled = domain.random_assignments(requested, seed, generation_mode)
        assignments = sampled
        target = len(sampled)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = bool(payload.get("compute_verdicts", True))
    solver_backend = _vector_solver_backend(payload)
    effort, solver_limits, external_case_limit = _vector_verification_settings(payload)
    generated_count = 0
    solver_statuses: dict[str, int] = {}
    solver_verdicts: dict[str, int] = {}
    external_statuses: dict[str, int] = {}
    external_attempts = 0
    seen_file_identities: dict[str, Mapping[str, Any]] = {}
    atfile_tmp = out_dir / "@all.tmp"
    parallel_workers = (
        _parallel_solver_workers(
            target,
            payload.get("solver_workers"),
        )
        if judge and solver_backend == "embedded"
        else 1
    )
    executor: ProcessPoolExecutor | None = None
    try:
        if parallel_workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=parallel_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        with atfile_tmp.open("w", encoding="utf-8") as atfile:
            assignment_iterator = iter(assignments)
            batch_size = max(parallel_workers * 8, 1) if executor else 1
            while True:
                batch = list(islice(assignment_iterator, batch_size))
                if not batch:
                    break
                batch_cases = [lower_vector_assignment(assignment) for assignment in batch]
                if executor is not None:
                    batch_solvers = list(
                        executor.map(
                            _embedded_solver_worker,
                            ((case, solver_limits) for case in batch_cases),
                            chunksize=1,
                        )
                    )
                else:
                    batch_solvers = []
                    for case in batch_cases:
                        if judge:
                            request_external = solver_backend == "crosscheck" and (
                                external_case_limit is None
                                or external_attempts < external_case_limit
                            )
                            solver = solve_generated_case(
                                case,
                                vector_external_check=request_external,
                                vector_solver_limits=solver_limits,
                            ).to_json()
                            external = _solver_external_status(solver)
                            if request_external and external != "not_run":
                                external_attempts += 1
                            elif (
                                solver_backend == "crosscheck"
                                and not request_external
                                and solver.get("status") == "verified"
                            ):
                                solver = _mark_external_batch_skipped(
                                    solver,
                                    external_case_limit,
                                )
                        else:
                            solver = _unchecked_solver()
                        batch_solvers.append(solver)

                for raw_case, solver in zip(batch_cases, batch_solvers):
                    case = replace(raw_case, solver=solver)
                    _claim_vector_file_identity(out_dir, case, seen_file_identities)
                    litmus_path = out_dir / case.file_name
                    litmus_path.write_text(case.litmus, encoding="utf-8")
                    (out_dir / f"{case.name}.meta.json").write_text(
                        json.dumps(case.meta(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    (out_dir / f"{case.name}.solver.json").write_text(
                        json.dumps(solver, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    atfile.write(litmus_path.name + "\n")
                    generated_count += 1
                    status = str(solver.get("status", "unknown"))
                    solver_statuses[status] = solver_statuses.get(status, 0) + 1
                    verdict = str(solver.get("verdict", "unknown"))
                    solver_verdicts[verdict] = solver_verdicts.get(verdict, 0) + 1
                    external = str(solver.get("cross_check", "not_run") or "not_run")
                    external_statuses[external] = external_statuses.get(external, 0) + 1
                    if progress_callback is not None:
                        progress_callback(
                            generated_count,
                            max(target, 1),
                            _verification_progress(
                                "Generated",
                                generated_count,
                                target,
                                solver_statuses,
                                external_statuses,
                                file_name=case.file_name,
                            ),
                        )
    except Exception:
        atfile_tmp.unlink(missing_ok=True)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    atfile_tmp.replace(out_dir / "@all")
    audit = domain.audit()
    report = {
        "schema": "litmus-link.vector-native-generation.v1",
        "profile": "vector-native",
        "available_litmus": domain.total_cases,
        "generated_litmus": generated_count,
        "generation_mode": generation_mode,
        "generation_limit": requested,
        "generation_limited": generated_count < domain.total_cases,
        "random_seed": seed if generation_mode != VECTOR_GENERATE_ALL else None,
        "sampling": VECTOR_SAMPLING_LABELS[generation_mode],
        "file_name_scheme": "LLV-<family>-<sha256>.litmus",
        "solver": solver_statuses,
        "solver_verdict": solver_verdicts,
        "external_status": external_statuses,
        "solver_backend": solver_backend,
        "solver_workers": parallel_workers,
        "verification_effort": effort,
        "verification_limits": {
            **solver_limits,
            "external_case_limit": external_case_limit,
        },
        "output": str(out_dir),
        "atfile": str(out_dir / "@all"),
        "audit": audit,
    }
    (out_dir / "audit-report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "generation-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _claim_vector_file_identity(
    out_dir: Path,
    case: GeneratedCase,
    seen: dict[str, Mapping[str, Any]],
) -> None:
    if case.case_ir is None:
        raise ValueError("Vector case is missing case IR for file identity validation")
    identity = case.case_ir.metadata.get("file_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"Vector case {case.name} is missing canonical file identity")
    canonical = identity.get("canonical")
    if not isinstance(canonical, Mapping):
        raise ValueError(f"Vector case {case.name} has an invalid canonical file identity")
    expected_file_name = str(identity.get("file_name", ""))
    if expected_file_name != case.file_name:
        raise ValueError(
            f"Vector case {case.name} file identity names {expected_file_name!r}, "
            f"expected {case.file_name!r}"
        )
    if case.name in seen:
        raise ValueError(
            f"duplicate Vector file identity {case.name}; generation would overwrite a case"
        )
    seen[case.name] = canonical

    litmus_path = out_dir / case.file_name
    meta_path = out_dir / f"{case.name}.meta.json"
    solver_path = out_dir / f"{case.name}.solver.json"
    existing = [path for path in (litmus_path, meta_path, solver_path) if path.exists()]
    if not existing:
        return
    if not litmus_path.exists() or not meta_path.exists():
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite incomplete Vector artifacts for {case.name}: {names}"
        )
    try:
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(
            f"refusing to overwrite unreadable Vector metadata {meta_path}"
        ) from exc
    old_identity = (
        ((existing_meta.get("case_ir") or {}).get("metadata") or {}).get(
            "file_identity"
        )
    )
    old_canonical = old_identity.get("canonical") if isinstance(old_identity, Mapping) else None
    if old_canonical != canonical:
        raise FileExistsError(
            f"Vector file ID collision at {litmus_path}; existing case has a different identity"
        )


def _endpoint_categories(payload: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = payload.get("endpoint_categories")
    if explicit is not None:
        selected = _selected(payload, "endpoint_categories", ENDPOINT_CATEGORIES)
    else:
        legacy = _selected(payload, "endpoint_modes", ())
        selected_values = {"vector"}
        if "P" in legacy or not legacy:
            selected_values.add("scalar")
        if any(mode in {"AMO", "Aq", "Rl", "AR"} for mode in legacy) or not legacy:
            selected_values.add("amo")
        selected = tuple(
            category for category in ENDPOINT_CATEGORIES if category in selected_values
        )
    unknown = set(selected) - set(ENDPOINT_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown endpoint category: {', '.join(sorted(unknown))}")
    if "vector" not in selected:
        raise ValueError("Vector workflow requires the vector endpoint category")
    return selected


def _fusion_request_audit(
    payload: Mapping[str, Any],
    categories: Sequence[str],
    requested_alignments: Sequence[str],
) -> tuple[dict[str, int], bool]:
    """Classify requests outside the currently implemented fusion scope."""

    excluded: Counter[str] = Counter()
    supported_scope = True

    misaligned = sum(
        alignment != "aligned" for alignment in requested_alignments
    )
    if misaligned:
        reason = (
            "excluded_illegal_misaligned_amo_request"
            if "amo" in categories
            else "excluded_unsupported_misaligned_fusion_request"
        )
        excluded[reason] += misaligned
    if "aligned" not in requested_alignments:
        supported_scope = False

    raw_attributes = payload.get("attributes", payload.get("attribute"))
    if raw_attributes is not None:
        attributes = _request_values(raw_attributes, "attributes")
        if not attributes:
            raise ValueError("select at least one fusion memory attribute")
        known = {
            "cacheable",
            "pbmt_nc",
            "pbmt_io",
            "nc_alias",
            "cacheable_nc_alias",
            "nc",
            "io",
        }
        unknown = set(attributes) - known
        if unknown:
            raise ValueError(
                f"unknown fusion memory attribute(s): {', '.join(sorted(unknown))}"
            )
        unsupported = sum(attribute != "cacheable" for attribute in attributes)
        if unsupported:
            excluded["excluded_unsupported_pbmt_nc_io_request"] += unsupported
        if "cacheable" not in attributes:
            supported_scope = False

    if "pbmt" in payload:
        pbmt_values = _request_values(payload.get("pbmt"), "pbmt")
        if not pbmt_values:
            raise ValueError("select at least one PBMT value")
        parsed_pbmt: list[int] = []
        for value in pbmt_values:
            try:
                parsed = int(value, 0)
            except ValueError as exc:
                raise ValueError(f"PBMT value must be 0, 1, 2, or 3: {value!r}") from exc
            if parsed not in {0, 1, 2, 3}:
                raise ValueError(f"PBMT value must be 0, 1, 2, or 3: {parsed}")
            parsed_pbmt.append(parsed)
        excluded["excluded_unsupported_pbmt_nc_io_request"] += sum(
            value in {1, 2} for value in parsed_pbmt
        )
        excluded["excluded_illegal_pbmt_reserved_request"] += sum(
            value == 3 for value in parsed_pbmt
        )
        if 0 not in parsed_pbmt:
            supported_scope = False

    if payload.get("pma_atomic") is False:
        excluded["excluded_unsupported_pma_nonatomic_request"] += 1
        # The current aligned-fusion implementation requires PMA atomic=true.
        # Treat an explicit ``false`` request like the
        # unsupported members of the attribute/PBMT axes: account for it in
        # the audit, then keep the supported current target instead of clearing
        # an otherwise valid aligned domain.

    return (
        {reason: count for reason, count in sorted(excluded.items()) if count},
        supported_scope,
    )


def _request_values(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return (str(value),)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a value or list")
    return tuple(dict.fromkeys(str(item) for item in value))


def _scalar_choices(payload: Mapping[str, Any]) -> tuple[EndpointChoice, ...]:
    widths = _selected(payload, "scalar_widths", SCALAR_WIDTHS)
    if not widths:
        raise ValueError("select at least one scalar width")
    unknown = set(widths) - set(SCALAR_WIDTHS)
    if unknown:
        raise ValueError(f"unknown scalar width(s): {', '.join(sorted(unknown))}")
    return tuple(
        EndpointChoice(
            f"scalar:{width}:{direction}",
            "scalar",
            direction,
            "P",
            params={
                "width": width,
                "width_bytes": str(SCALAR_WIDTH_BYTES[width]),
            },
        )
        for direction in (READ, WRITE)
        for width in widths
    )


def _empty_endpoint_audit() -> dict[str, Any]:
    return {
        "raw_configurations": 0,
        "generated_endpoint_choices": 0,
        "excluded": {},
    }


def _amo_choices_with_audit(
    payload: Mapping[str, Any],
) -> tuple[tuple[EndpointChoice, ...], dict[str, Any]]:
    operations = _selected(payload, "amo_ops", AMO_OPERATIONS)
    widths = _selected(payload, "amo_widths", AMO_WIDTHS)
    orderings = _selected(payload, "amo_orderings", AMO_ORDERINGS)
    if not operations:
        raise ValueError("select at least one AMO opcode")
    if not widths:
        raise ValueError("select at least one AMO width")
    if not orderings:
        raise ValueError("select at least one AMO ordering")
    unknown_ops = set(operations) - set(AMO_OPERATIONS)
    unknown_widths = set(widths) - {"b", "h", *AMO_WIDTHS}
    unknown_orderings = set(orderings) - set(AMO_ORDERINGS)
    if unknown_ops:
        raise ValueError(f"unknown Nanhu AMO operation(s): {', '.join(sorted(unknown_ops))}")
    if unknown_widths:
        raise ValueError(f"unknown AMO width(s): {', '.join(sorted(unknown_widths))}")
    if unknown_orderings:
        raise ValueError(f"unknown AMO ordering(s): {', '.join(sorted(unknown_orderings))}")
    choices: list[EndpointChoice] = []
    excluded: Counter[str] = Counter()
    raw = 0
    for direction in (READ, WRITE):
        for operation, width, ordering in product(operations, widths, orderings):
            raw += 1
            if width not in AMO_WIDTHS:
                excluded["excluded_illegal_nanhu_amo_width"] += 1
                continue
            choices.append(
                EndpointChoice(
                    f"amo:{operation}:{width}:{ordering}:{direction}",
                    "amo",
                    direction,
                    {
                        "relaxed": "AMO",
                        "aq": "Aq",
                        "rl": "Rl",
                        "aqrl": "AR",
                    }[ordering],
                    params={
                        "amo_op": operation,
                        "amo_width": width,
                        "amo_width_bytes": str(AMO_WIDTH_BYTES[width]),
                        "amo_ordering": ordering,
                    },
                )
            )
    return tuple(choices), {
        "raw_configurations": raw,
        "generated_endpoint_choices": len(choices),
        "excluded": dict(sorted(excluded.items())),
    }


def _vector_choices_with_audit(
    payload: Mapping[str, Any],
) -> tuple[tuple[EndpointChoice, ...], dict[str, Any]]:
    forms = _selected(payload, "forms", VECTOR_OPS)
    sews = _selected(payload, "sew", VECTOR_WIDTHS)
    lmuls = _selected(payload, "lmul", VECTOR_LMULS)
    index_eews = _selected(payload, "index_eew", VECTOR_INDEX_EEWS)
    nfields = _selected(payload, "nf", VECTOR_NFIELDS)
    whole_nregs = _selected(payload, "whole_nreg", VECTOR_WHOLE_NREGS)
    masks = _selected(payload, "mask", VECTOR_MASKS)
    tails = _selected(payload, "tail", VECTOR_TAILS)
    vls = _selected(payload, "vl", VECTOR_LENGTHS)
    regular_forms = tuple(
        form for form in forms if form not in WHOLE_REGISTER_VECTOR_OPS
    )
    required_axes = {"Vector form": forms, "SEW": sews}
    if regular_forms:
        required_axes.update(
            {
                "LMUL": lmuls,
                "mask mode": masks,
                "tail policy": tails,
                "Vector length": vls,
            }
        )
    for label, selected in required_axes.items():
        if not selected:
            raise ValueError(f"select at least one {label}")
    if any("indexed" in form for form in forms) and not index_eews:
        raise ValueError("select at least one indexed offset EEW")
    if any(form.startswith("segment_") for form in forms) and not nfields:
        raise ValueError("select at least one Segment NFIELDS value")
    if any(form in WHOLE_REGISTER_VECTOR_OPS for form in forms) and not whole_nregs:
        raise ValueError("select at least one whole-register NREG value")
    validators = (
        ("Vector form", forms, VECTOR_OPS),
        ("SEW", sews, VECTOR_WIDTHS),
        ("LMUL", lmuls, VECTOR_LMULS),
        ("index EEW", index_eews, VECTOR_INDEX_EEWS),
        ("Segment NFIELDS", nfields, VECTOR_NFIELDS),
        ("whole-register NREG", whole_nregs, VECTOR_WHOLE_NREGS),
        ("mask mode", masks, VECTOR_MASKS),
        ("tail policy", tails, VECTOR_TAILS),
        ("Vector length", vls, VECTOR_LENGTHS),
    )
    for label, selected, known in validators:
        unknown = set(selected) - set(known)
        if unknown:
            raise ValueError(f"unknown {label}(s): {', '.join(sorted(unknown))}")
    out: list[EndpointChoice] = []
    excluded: Counter[str] = Counter()
    raw = 0
    for form, sew, lmul, mask, tail, vl in product(
        regular_forms, sews, lmuls, masks, tails, vls
    ):
        selected_indexes: Sequence[str | None] = index_eews if "indexed" in form else (None,)
        selected_nfields: Sequence[str | None] = (
            nfields if form.startswith("segment_") else (None,)
        )
        for index_eew, nf in product(selected_indexes, selected_nfields):
            raw += 1
            if not vector_memory_config_legal(
                form, sew, lmul, mask, vl, index_eew, nf
            ):
                excluded["excluded_illegal_vector_config"] += 1
                continue
            footprint = vector_footprint_kind(
                form, sew, lmul, mask, vl, nf=nf
            )
            if footprint not in {"same_line", "cross_line"}:
                excluded["excluded_unsupported_cross_page"] += 1
                continue
            params = {
                "sew": sew,
                "lmul": lmul,
                "mask": mask,
                "tail": tail,
                "vl": vl,
                "footprint": footprint,
            }
            if index_eew is not None:
                params["index_eew"] = index_eew
            if nf is not None:
                params["nf"] = nf
            direction = WRITE if form.endswith("store") else READ
            choice_id = "vector:" + ":".join(
                [
                    form,
                    sew,
                    lmul,
                    str(index_eew or "-"),
                    str(nf or "-"),
                    mask,
                    tail,
                    vl,
                ]
            )
            out.append(EndpointChoice(choice_id, "vector", direction, "P", form, params))
    for form in (item for item in forms if item in WHOLE_REGISTER_VECTOR_OPS):
        selected_sews = sews if form == "whole_register_load" else ("e8",)
        for sew, whole_nreg in product(selected_sews, whole_nregs):
            raw += 1
            if not vector_memory_config_legal(
                form,
                sew,
                "m1",
                "unmasked",
                "vl1",
                whole_nreg=whole_nreg,
            ):
                excluded["excluded_illegal_vector_config"] += 1
                continue
            footprint = vector_footprint_kind(
                form,
                sew,
                "m1",
                "unmasked",
                "vl1",
                whole_nreg=whole_nreg,
            )
            if footprint not in {"same_line", "cross_line"}:
                excluded["excluded_unsupported_cross_page"] += 1
                continue
            params = {
                "sew": sew,
                "whole_nreg": whole_nreg,
                "footprint": footprint,
            }
            direction = WRITE if form.endswith("store") else READ
            choice_id = f"vector:{form}:{sew}:{whole_nreg}"
            out.append(
                EndpointChoice(
                    choice_id,
                    "vector",
                    direction,
                    "P",
                    form,
                    params,
                )
            )
    return tuple(out), {
        "raw_configurations": raw,
        "generated_endpoint_choices": len(out),
        "excluded": dict(sorted(excluded.items())),
    }


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


def _formal_scope(case: LitmusCaseIR, assignment: VectorAssignment) -> tuple[bool, str]:
    if assignment.alignment != "aligned":
        return (
            False,
            "Misaligned Vector/scalar/AMO fusion is outside the Nanhu aligned formal domain.",
        )
    for event in case.events():
        if event.kind not in {"load", "store", "amo"} or not event.location:
            continue
        if event.memory_access is None or not event.memory_access.natural_aligned:
            return False, f"Memory event {event.event_id} is not naturally aligned."
        if event.kind == "amo" and event.memory_access.size_bytes not in {4, 8}:
            return False, f"AMO event {event.event_id} is not a Nanhu W/D AMO."
    metadata = case.metadata.get("vectors")
    if isinstance(metadata, Mapping):
        for event_id, raw in metadata.items():
            if isinstance(raw, Mapping) and raw.get("footprint") == "cross_page":
                return False, f"Vector event {event_id} crosses the formal 4 KiB page."
    return (
        True,
        "Nanhu aligned fusion: PBMT=0, cacheable main memory, PMA atomic=true; "
        "mixed-size byte overlap and W/D AMO transactions are modeled by the Vector-aware RVWMO solver.",
    )


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


def _vector_solver_backend(payload: Mapping[str, Any]) -> str:
    backend = str(payload.get("solver_backend", "embedded"))
    if backend not in {"embedded", "crosscheck"}:
        raise ValueError(
            "Vector solver_backend must be 'embedded' or 'crosscheck'"
        )
    return backend


def _vector_verification_settings(
    payload: Mapping[str, Any],
) -> tuple[str, dict[str, Any], int | None]:
    # Library/CLI callers retain the historical deep-search behavior unless
    # they explicitly select a batch effort. The Qt GUI sends "interactive".
    effort = str(payload.get("verification_effort", "thorough"))
    if effort not in VECTOR_VERIFICATION_EFFORTS:
        raise ValueError(
            f"unknown Vector verification effort: {effort}; expected one of "
            + ", ".join(VECTOR_VERIFICATION_EFFORTS)
        )
    selected = dict(VECTOR_VERIFICATION_LIMITS[effort])
    external_case_limit = selected.pop("external_case_limit")
    return effort, selected, external_case_limit


def _solver_external_status(solver: Mapping[str, Any]) -> str:
    vector = solver.get("vector")
    if isinstance(vector, Mapping):
        external = vector.get("external")
        if isinstance(external, Mapping):
            return str(external.get("status", "external_unsupported") or "external_unsupported")
    return str(solver.get("cross_check", "not_run") or "not_run")


def _mark_external_batch_skipped(
    solver: Mapping[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    out = dict(solver)
    out["cross_check"] = "batch_limit_skipped"
    vector = out.get("vector")
    if isinstance(vector, Mapping):
        vector_payload = dict(vector)
        vector_payload["external"] = {
            "schema": "litmus-link.vector-herd-reference.v1",
            "status": "batch_limit_skipped",
            "verdict": "unknown",
            "allowed": None,
            "reason": (
                "The selected verification effort limits herd7 projection "
                f"cross-checks to {limit} cases in this batch."
            ),
            "results": [],
        }
        out["vector"] = vector_payload
    return out


def _verification_progress(
    verb: str,
    current: int,
    total: int,
    solver_statuses: Mapping[str, int],
    external_statuses: Mapping[str, int],
    *,
    file_name: str = "",
) -> str:
    counts = ", ".join(
        f"{status}={count}"
        for status, count in sorted(solver_statuses.items())
        if count
    ) or "pending"
    external_checked = sum(
        count
        for status, count in external_statuses.items()
        if status not in {"not_run", "batch_limit_skipped"}
    )
    external_skipped = int(external_statuses.get("batch_limit_skipped", 0))
    external = f"herd={external_checked}"
    if external_skipped:
        external += f", herd-skipped={external_skipped}"
    suffix = f" | {file_name}" if file_name else ""
    return (
        f"{verb} {current:,}/{total:,} | {counts} | {external}{suffix}"
    )
