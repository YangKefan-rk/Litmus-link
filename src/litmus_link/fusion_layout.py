from __future__ import annotations

"""Naturally aligned address layouts for scalar/AMO/Vector fusion cases."""

from dataclasses import dataclass
from typing import Mapping, Sequence

from .profiles import vector_effective_vl


FUSION_OVERLAP_LAYOUTS = (
    "same_start",
    "contained",
    "low_partial",
    "high_partial",
    "disjoint_control",
)


class FusionLayoutError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(detail)


@dataclass(frozen=True)
class EndpointFootprint:
    vertex: int
    alignment_bytes: int
    relative_bytes: frozenset[int]

    def __post_init__(self) -> None:
        if self.alignment_bytes not in {1, 2, 4, 8}:
            raise ValueError(f"invalid endpoint alignment: {self.alignment_bytes}")
        if not self.relative_bytes or min(self.relative_bytes) < 0:
            raise ValueError("endpoint footprint must contain non-negative bytes")

    @property
    def span_bytes(self) -> int:
        return max(self.relative_bytes) + 1

    def placed(self, base_offset: int) -> frozenset[int]:
        if base_offset < 0 or base_offset % self.alignment_bytes:
            raise ValueError(
                f"vertex {self.vertex} base {base_offset} is not naturally aligned to "
                f"{self.alignment_bytes} bytes"
            )
        return frozenset(base_offset + byte for byte in self.relative_bytes)


@dataclass(frozen=True)
class FusionAddressLayout:
    shape: str
    offsets: Mapping[int, int]
    footprints: Mapping[int, frozenset[int]]
    strict_overlap: bool

    def __post_init__(self) -> None:
        if self.shape not in FUSION_OVERLAP_LAYOUTS:
            raise ValueError(f"unknown fusion overlap layout: {self.shape}")

    def to_json(self) -> dict:
        return {
            "shape": self.shape,
            "offsets": {str(key): value for key, value in sorted(self.offsets.items())},
            "footprints": {
                str(key): sorted(value) for key, value in sorted(self.footprints.items())
            },
            "strict_overlap": self.strict_overlap,
            "naturally_aligned": True,
        }


def endpoint_footprint(vertex: int, choice: object) -> EndpointFootprint:
    category = str(getattr(choice, "category"))
    params = dict(getattr(choice, "params", None) or {})
    if category == "scalar":
        size = int(params.get("width_bytes", 4))
        return EndpointFootprint(vertex, size, frozenset(range(size)))
    if category == "amo":
        size = int(params.get("amo_width_bytes", params.get("width_bytes", 4)))
        if size not in {4, 8}:
            raise FusionLayoutError(
                "excluded_illegal_misaligned_amo",
                f"Nanhu AMO vertex {vertex} has unsupported width {size} bytes",
            )
        return EndpointFootprint(vertex, size, frozenset(range(size)))
    if category != "vector":
        raise FusionLayoutError(
            "excluded_unsupported_endpoint",
            f"unknown endpoint category at vertex {vertex}: {category}",
        )

    sew = str(params.get("sew", "e32"))
    lmul = str(params.get("lmul", "m1"))
    vl = str(params.get("vl", "vl1"))
    mask = str(params.get("mask", "unmasked"))
    form = str(getattr(choice, "vector_form", ""))
    effective_vl = vector_effective_vl(sew, lmul, vl)
    if effective_vl is None:
        raise FusionLayoutError(
            "excluded_illegal_vector_config",
            f"illegal VLEN/SEW/LMUL/VL combination at vertex {vertex}",
        )
    element_bytes = int(sew.removeprefix("e")) // 8
    stride = element_bytes * 2 if form.startswith("strided_") else element_bytes
    active = [
        index
        for index in range(effective_vl)
        if mask == "unmasked" or index % 2 == 0
    ]
    if not active:
        raise FusionLayoutError(
            "excluded_illegal_vector_config",
            f"Vector endpoint {vertex} has no active elements",
        )
    # The relation-cycle endpoint is element 0. Remaining active elements are
    # retained by the Vector parent as additional transactions, but do not
    # change which byte relation realizes the named cycle edge.
    return EndpointFootprint(vertex, element_bytes, frozenset(range(element_bytes)))


def synthesize_address_layout(
    footprints: Sequence[EndpointFootprint],
    location_groups: Sequence[Sequence[int]],
    shape: str,
) -> FusionAddressLayout:
    if shape not in FUSION_OVERLAP_LAYOUTS:
        raise FusionLayoutError(
            "excluded_unsupported_layout", f"unknown fusion overlap layout: {shape}"
        )
    by_vertex = {footprint.vertex: footprint for footprint in footprints}
    offsets: dict[int, int] = {}
    strict_overlap = False

    if shape == "disjoint_control":
        raise FusionLayoutError(
            "excluded_unsatisfiable_value_layout",
            "disjoint endpoints cannot realize the rf/fr/co communication edges of the selected relation cycle",
        )

    for group in location_groups:
        members = [by_vertex[vertex] for vertex in group]
        if not members:
            continue
        max_span = max(member.span_bytes for member in members)
        distinct_spans = len({member.span_bytes for member in members}) > 1
        strict_overlap = strict_overlap or distinct_spans

        anchor = {
            "same_start": 0,
            "low_partial": 0,
            "contained": max_span // 2,
            "high_partial": max_span - 1,
        }[shape]
        for member in members:
            # For naturally aligned power-of-two accesses, overlap implies
            # nesting: two different widths cannot form a strict non-contained
            # partial overlap. Place every endpoint's aligned interval around a
            # common low/center/high anchor so all cycle relations retain at
            # least one shared byte.
            offset = _align_down(anchor, member.alignment_bytes)
            offsets[member.vertex] = offset

        placed = {
            member.vertex: member.placed(offsets[member.vertex])
            for member in members
        }
        if len(members) > 1 and not _connected_overlap(tuple(placed.values())):
            raise FusionLayoutError(
                "excluded_unsatisfiable_value_layout",
                f"{shape} does not preserve byte overlap for location group {list(group)}",
            )

    if shape != "same_start" and not strict_overlap:
        raise FusionLayoutError(
            "excluded_unsatisfiable_value_layout",
            f"{shape} requires at least two different endpoint footprint spans",
        )

    placed_all = {
        footprint.vertex: footprint.placed(offsets.get(footprint.vertex, 0))
        for footprint in footprints
    }
    if max((max(value) for value in placed_all.values()), default=0) >= 4096:
        raise FusionLayoutError(
            "excluded_unsupported_cross_page",
            "fusion footprint reaches a second 4 KiB page",
        )
    return FusionAddressLayout(shape, offsets, placed_all, strict_overlap)


def _connected_overlap(footprints: Sequence[frozenset[int]]) -> bool:
    unseen = set(range(len(footprints)))
    stack = [unseen.pop()]
    visited: set[int] = set()
    while stack:
        index = stack.pop()
        if index in visited:
            continue
        visited.add(index)
        neighbors = {
            other
            for other in unseen
            if footprints[index] & footprints[other]
        }
        unseen -= neighbors
        stack.extend(neighbors)
    return not unseen


def _align_down(value: int, alignment: int) -> int:
    return max(value, 0) // alignment * alignment

