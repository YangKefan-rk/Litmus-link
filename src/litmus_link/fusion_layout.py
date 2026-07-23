from __future__ import annotations

"""Address layouts for scalar/AMO/Vector fusion cases.

Scalar and AMO endpoints remain naturally aligned.  Vector endpoints may use
one of the explicit no-MAG misalignment placements below; the placement names
describe the boundary crossed by relation endpoint element zero.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Mapping, Sequence

from .profiles import WHOLE_REGISTER_VECTOR_OPS, vector_effective_vl, vector_whole_nregs


FUSION_OVERLAP_LAYOUTS = (
    "same_start",
    "contained",
    "low_partial",
    "high_partial",
    "disjoint_control",
)

VECTOR_ALIGNMENT_MODES = (
    "aligned",
    "misalign_same16",
    "misalign_cross16",
    "misalign_cross64",
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
    placement: str = "aligned"

    def __post_init__(self) -> None:
        if self.alignment_bytes not in {1, 2, 4, 8}:
            raise ValueError(f"invalid endpoint alignment: {self.alignment_bytes}")
        if not self.relative_bytes or min(self.relative_bytes) < 0:
            raise ValueError("endpoint footprint must contain non-negative bytes")
        if self.placement not in VECTOR_ALIGNMENT_MODES:
            raise ValueError(f"unknown endpoint placement: {self.placement}")
        if self.placement != "aligned" and self.alignment_bytes == 1:
            raise ValueError("an 8-bit memory element cannot be misaligned")

    @property
    def span_bytes(self) -> int:
        return max(self.relative_bytes) + 1

    def placed(self, base_offset: int) -> frozenset[int]:
        if base_offset < 0 or not self.accepts(base_offset):
            raise ValueError(
                f"vertex {self.vertex} base {base_offset} does not satisfy "
                f"{self.placement} for {self.alignment_bytes} bytes"
            )
        return frozenset(base_offset + byte for byte in self.relative_bytes)

    def accepts(self, base_offset: int) -> bool:
        if base_offset < 0:
            return False
        naturally_aligned = base_offset % self.alignment_bytes == 0
        if self.placement == "aligned":
            return naturally_aligned
        if naturally_aligned:
            return False
        boundary = _access_boundary(base_offset, self.span_bytes)
        return boundary == {
            "misalign_same16": "same16",
            "misalign_cross16": "cross16",
            "misalign_cross64": "cross64",
        }[self.placement]


@dataclass(frozen=True)
class FusionAddressLayout:
    shape: str
    offsets: Mapping[int, int]
    footprints: Mapping[int, frozenset[int]]
    strict_overlap: bool
    alignment_modes: Mapping[int, str] = field(default_factory=dict)

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
            "alignment_modes": {
                str(key): value for key, value in sorted(self.alignment_modes.items())
            },
            "naturally_aligned": all(
                value == "aligned" for value in self.alignment_modes.values()
            ),
        }


def endpoint_footprint(
    vertex: int,
    choice: object,
    vector_alignment: str = "aligned",
) -> EndpointFootprint:
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
    if form in WHOLE_REGISTER_VECTOR_OPS:
        nregs = vector_whole_nregs(form, params.get("whole_nreg"))
        if nregs is None:
            raise FusionLayoutError(
                "excluded_illegal_vector_config",
                f"illegal whole-register NREG at vertex {vertex}",
            )
        element_bytes = int(sew.removeprefix("e")) // 8
        return EndpointFootprint(
            vertex,
            element_bytes,
            frozenset(range(element_bytes)),
            vector_alignment,
        )
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
    return EndpointFootprint(
        vertex,
        element_bytes,
        frozenset(range(element_bytes)),
        vector_alignment,
    )


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
    shape_realized = shape == "same_start"

    if shape == "disjoint_control":
        raise FusionLayoutError(
            "excluded_unsatisfiable_value_layout",
            "disjoint endpoints cannot realize the rf/fr/co communication edges of the selected relation cycle",
        )

    if any(footprint.placement != "aligned" for footprint in footprints):
        return _synthesize_misaligned_layout(footprints, location_groups, shape)

    for group in location_groups:
        members = tuple(by_vertex[vertex] for vertex in group)
        if not members:
            continue
        selected, placed, group_matches = _select_group_layout(members, shape)
        offsets.update(
            (member.vertex, offset)
            for member, offset in zip(members, selected)
        )
        strict_overlap = strict_overlap or len(set(placed.values())) > 1
        shape_realized = shape_realized or group_matches

    if shape != "same_start" and (not strict_overlap or not shape_realized):
        raise FusionLayoutError(
            "excluded_unsatisfiable_value_layout",
            f"{shape} requires at least one location group with that concrete geometry",
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
    return FusionAddressLayout(
        shape,
        offsets,
        placed_all,
        strict_overlap,
        {footprint.vertex: footprint.placement for footprint in footprints},
    )


def _synthesize_misaligned_layout(
    footprints: Sequence[EndpointFootprint],
    location_groups: Sequence[Sequence[int]],
    shape: str,
) -> FusionAddressLayout:
    by_vertex = {footprint.vertex: footprint for footprint in footprints}
    offsets: dict[int, int] = {}
    placed_all: dict[int, frozenset[int]] = {}
    strict_overlap = False
    shape_realized = shape == "same_start"

    for group in location_groups:
        members = tuple(by_vertex[vertex] for vertex in group)
        if not members:
            continue
        selected_offsets, selected_footprints, group_matches = _select_group_layout(
            members, shape
        )
        shape_realized = shape_realized or group_matches
        offsets.update(
            (member.vertex, offset)
            for member, offset in zip(members, selected_offsets)
        )
        placed_all.update(selected_footprints)
        strict_overlap = strict_overlap or len(set(selected_footprints.values())) > 1

    for footprint in footprints:
        if footprint.vertex not in offsets:
            candidates = _placement_candidates(
                footprint, _misaligned_anchor((footprint,))
            )
            if not candidates:
                raise FusionLayoutError(
                    "excluded_unsatisfiable_misaligned_layout",
                    f"no placement exists for vertex {footprint.vertex}",
                )
            offsets[footprint.vertex] = candidates[0]
            placed_all[footprint.vertex] = footprint.placed(candidates[0])

    if shape != "same_start" and not shape_realized:
        raise FusionLayoutError(
            "excluded_unsatisfiable_misaligned_layout",
            f"{shape} requires at least one location group with that concrete geometry",
        )

    if max((max(value) for value in placed_all.values()), default=0) >= 4096:
        raise FusionLayoutError(
            "excluded_unsupported_cross_page",
            "fusion footprint reaches a second 4 KiB page",
        )
    return FusionAddressLayout(
        shape,
        offsets,
        placed_all,
        strict_overlap,
        {footprint.vertex: footprint.placement for footprint in footprints},
    )


def _select_group_layout(
    members: Sequence[EndpointFootprint],
    shape: str,
) -> tuple[tuple[int, ...], dict[int, frozenset[int]], bool]:
    """Select one concrete group placement and report whether it realizes shape.

    The counting path calls this same helper, so audit cardinality cannot drift
    from the layouts emitted by ``synthesize_address_layout``.
    """

    if any(member.placement != "aligned" for member in members):
        return _select_misaligned_group_layout(members, shape)

    max_span = max(member.span_bytes for member in members)
    anchor = {
        "same_start": 0,
        "low_partial": 0,
        "contained": max_span // 2,
        "high_partial": max_span - 1,
    }[shape]
    selected = tuple(_align_down(anchor, member.alignment_bytes) for member in members)
    placed = {
        member.vertex: member.placed(offset)
        for member, offset in zip(members, selected)
    }
    if len(members) > 1 and not _connected_overlap(tuple(placed.values())):
        raise FusionLayoutError(
            "excluded_unsatisfiable_value_layout",
            f"{shape} does not preserve byte overlap",
        )
    realized = _overlap_shape_matches(members, placed, shape)
    if shape == "same_start" and not realized:
        raise FusionLayoutError(
            "excluded_unsatisfiable_value_layout",
            "same_start cannot be realized by naturally aligned footprints",
        )
    return selected, placed, realized


def _select_misaligned_group_layout(
    members: Sequence[EndpointFootprint],
    shape: str,
) -> tuple[tuple[int, ...], dict[int, frozenset[int]], bool]:
    anchor = _misaligned_anchor(members)
    candidates = {
        member.vertex: _placement_candidates(member, anchor)
        for member in members
    }
    if any(not values for values in candidates.values()):
        raise FusionLayoutError(
            "excluded_unsatisfiable_misaligned_layout",
            f"no {shape} placement satisfies the requested alignment boundary",
        )

    if shape == "same_start":
        common = set.intersection(
            *(set(candidates[member.vertex]) for member in members)
        )
        if not common:
            raise FusionLayoutError(
                "excluded_unsatisfiable_misaligned_layout",
                "same_start cannot preserve a common byte",
            )
        offset = min(
            common,
            key=lambda value: _misaligned_selection_key(
                shape,
                tuple(value for _member in members),
                anchor,
            ),
        )
        selected = tuple(offset for _member in members)
        placed = {member.vertex: member.placed(offset) for member in members}
        return selected, placed, True

    domains = tuple(candidates[member.vertex] for member in members)
    for selected in product(*domains):
        placed = {
            member.vertex: member.placed(offset)
            for member, offset in zip(members, selected)
        }
        if _overlap_shape_matches(members, placed, shape):
            return tuple(selected), placed, True

    # A case-level layout needs only one location group to realize the named
    # shape. Other groups use a neutral same-start/contained placement.
    for selected in product(*domains):
        placed = {
            member.vertex: member.placed(offset)
            for member, offset in zip(members, selected)
        }
        if _overlap_shape_matches(
            members, placed, "same_start"
        ) or _overlap_shape_matches(members, placed, "contained"):
            return tuple(selected), placed, False

    raise FusionLayoutError(
        "excluded_unsatisfiable_misaligned_layout",
        f"{shape} cannot preserve a common byte",
    )


def _misaligned_anchor(members: Sequence[EndpointFootprint]) -> int:
    placements = {member.placement for member in members}
    if "misalign_cross64" in placements:
        return 63
    if "misalign_cross16" in placements:
        return 15
    return 7


def _placement_candidates(
    footprint: EndpointFootprint,
    anchor: int,
) -> tuple[int, ...]:
    focus = {anchor, anchor + 1}
    candidates = [
        offset
        for offset in range(max(anchor - 16, 0), anchor + 17)
        if footprint.accepts(offset)
        and footprint.placed(offset).intersection(focus)
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda offset: (
                abs((2 * offset + footprint.span_bytes - 1) - 2 * anchor),
                offset,
            ),
        )[:8]
    )


def _overlap_shape_matches(
    members: Sequence[EndpointFootprint],
    placed: Mapping[int, frozenset[int]],
    shape: str,
) -> bool:
    """Check the concrete geometry represented by an overlap label.

    ``same_start`` and ``contained`` are deliberately distinct.  A partial
    layout requires a strict, non-contained overlap and is oriented relative
    to the first misaligned Vector endpoint (or the lowest-numbered endpoint
    when all accesses are aligned).  This prevents low/high labels from being
    aliases that differ only in metadata.
    """

    values = tuple(placed[member.vertex] for member in members)
    if not values:
        return False
    starts = tuple(min(value) for value in values)
    if shape == "same_start":
        return len(set(starts)) == 1
    if len(values) < 2 or not set.intersection(*(set(value) for value in values)):
        return False

    pair_relations = []
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if not left & right:
                continue
            if left == right:
                pair_relations.append("equal")
            elif left < right or right < left:
                pair_relations.append("contained")
            else:
                pair_relations.append("partial")

    if shape == "contained":
        return (
            len(set(starts)) > 1
            and "contained" in pair_relations
            and "partial" not in pair_relations
        )
    if shape not in {"low_partial", "high_partial"}:
        return False
    if "partial" not in pair_relations:
        return False

    reference_index = next(
        (
            index
            for index, member in enumerate(members)
            if member.placement != "aligned"
        ),
        0,
    )
    reference = values[reference_index]
    common = frozenset.intersection(*values)
    touches_low = min(common) == min(reference)
    touches_high = max(common) == max(reference)
    if shape == "low_partial":
        return touches_low and not touches_high
    return touches_high and not touches_low


def _misaligned_selection_key(
    shape: str,
    offsets: tuple[int, ...],
    anchor: int,
) -> tuple[int, ...]:
    distance = sum(abs(offset - anchor) for offset in offsets)
    if shape == "high_partial":
        return (distance, *(-offset for offset in offsets))
    return (distance, *offsets)


def _access_boundary(offset: int, size: int) -> str:
    end = offset + size - 1
    if offset // 64 != end // 64:
        return "cross64"
    if offset // 16 != end // 16:
        return "cross16"
    return "same16"


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
