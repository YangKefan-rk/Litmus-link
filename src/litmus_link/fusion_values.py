from __future__ import annotations

"""Value/provenance synthesis for aligned Vector/scalar/AMO cycles."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Mapping, Sequence

from .amo import amo_read_result, apply_amo
from .fusion_layout import FusionAddressLayout, FusionLayoutError
from .native_cycles import NativeCycle, location_ids, vertex_directions
from .native_edges import READ, WRITE
from .profiles import (
    NANHU_VLEN_BITS,
    WHOLE_REGISTER_VECTOR_OPS,
    vector_effective_vl,
    vector_nfields,
    vector_whole_nregs,
)


@dataclass(frozen=True)
class EndpointValue:
    vertex: int
    direction: str
    category: str
    width_bytes: int
    read_memory_value: int | None = None
    read_register_value: int | None = None
    write_value: int | None = None
    amo_operand: int | None = None
    amo_old: int | None = None
    amo_new: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "vertex": self.vertex,
            "direction": self.direction,
            "category": self.category,
            "width_bytes": self.width_bytes,
            "read_memory_value": self.read_memory_value,
            "read_register_value": self.read_register_value,
            "write_value": self.write_value,
            "amo_operand": self.amo_operand,
            "amo_old": self.amo_old,
            "amo_new": self.amo_new,
        }


@dataclass(frozen=True)
class FusionValuePlan:
    endpoints: Mapping[int, EndpointValue]
    initial_bytes: Mapping[str, tuple[int, ...]]
    final_bytes: Mapping[str, Mapping[int, int]]
    co_orders: Mapping[str, tuple[int, ...]]
    rf_sources: Mapping[int, int]
    object_sizes: Mapping[str, int]
    observed_final_bytes: Mapping[str, tuple[int, ...]]

    def to_json(self) -> dict[str, Any]:
        return {
            "endpoints": {
                str(key): value.to_json()
                for key, value in sorted(self.endpoints.items())
            },
            "initial_bytes": {
                key: list(value) for key, value in sorted(self.initial_bytes.items())
            },
            "final_bytes": {
                key: {str(offset): value for offset, value in sorted(image.items())}
                for key, image in sorted(self.final_bytes.items())
            },
            "co_orders": {
                key: list(value) for key, value in sorted(self.co_orders.items())
            },
            "rf_sources": {
                str(key): value for key, value in sorted(self.rf_sources.items())
            },
            "object_sizes": dict(sorted(self.object_sizes.items())),
            "observed_final_bytes": {
                key: list(value)
                for key, value in sorted(self.observed_final_bytes.items())
            },
        }


def synthesize_fusion_values(
    cycle: NativeCycle,
    choices: Sequence[object],
    layout: FusionAddressLayout,
    location_names: Mapping[int, str],
) -> FusionValuePlan:
    directions = vertex_directions(cycle.edges)
    locations = location_ids(cycle.edges)
    if len(choices) != len(directions):
        raise FusionLayoutError(
            "excluded_unsatisfiable_value_layout",
            "endpoint choices do not match the relation-cycle size",
        )

    rf_sources = {
        (vertex + 1) % len(cycle.edges): vertex
        for vertex, edge in enumerate(cycle.edges)
        if edge.relation == "rf"
    }
    endpoint_footprints = {
        vertex: tuple(
            bytes_
            for _element, bytes_ in _write_footprints(
                choices[vertex], directions[vertex], layout.offsets[vertex]
            )
        )
        for vertex in range(len(choices))
    }
    access_footprints = {
        vertex: frozenset(layout.footprints[vertex])
        for vertex in range(len(choices))
    }
    _validate_cycle_overlap(cycle, access_footprints)

    writers = {
        vertex
        for vertex, (choice, direction) in enumerate(zip(choices, directions))
        if direction == WRITE or str(getattr(choice, "category")) == "amo"
    }
    constraints: set[tuple[int, int]] = set()
    for vertex, edge in enumerate(cycle.edges):
        target = (vertex + 1) % len(cycle.edges)
        if edge.relation == "co":
            constraints.add((vertex, target))
        elif edge.relation == "fr":
            source = rf_sources.get(vertex)
            if source is not None:
                constraints.add((source, target))
            if str(getattr(choices[vertex], "category")) == "amo":
                constraints.add((vertex, target))
        elif edge.relation == "rf" and str(getattr(choices[target], "category")) == "amo":
            constraints.add((vertex, target))

    groups: dict[int, list[int]] = defaultdict(list)
    for vertex, location in enumerate(locations):
        groups[location].append(vertex)

    selected_orders: dict[int, tuple[int, ...]] = {}
    endpoint_values: dict[int, EndpointValue] = {}
    initial_images: dict[int, list[int]] = {}
    final_images: dict[int, dict[int, int]] = {}
    write_bytes_by_vertex: dict[int, dict[int, int]] = {}

    for location, vertices in sorted(groups.items()):
        base = location_names[location]
        full_footprints = {
            vertex: _all_memory_footprint(
                choices[vertex], layout.offsets[vertex]
            )
            for vertex in vertices
        }
        max_byte = max(
            (max(bytes_) for bytes_ in full_footprints.values() if bytes_),
            default=0,
        )
        object_size = _object_size(max_byte + 1)
        initial = [_initial_byte(location, offset) for offset in range(object_size)]
        initial_images[location] = initial
        location_writers = tuple(vertex for vertex in vertices if vertex in writers)
        location_constraints = {
            (left, right)
            for left, right in constraints
            if left in location_writers and right in location_writers
        }
        candidates = _writer_orders(location_writers, location_constraints)
        selected: tuple[
            tuple[int, ...],
            dict[int, EndpointValue],
            dict[int, int],
            dict[int, dict[int, int]],
        ] | None = None
        for order in candidates:
            evaluated = _evaluate_writer_order(
                order,
                vertices,
                choices,
                directions,
                layout,
                initial,
                rf_sources,
            )
            if evaluated is not None:
                values, image, write_values = evaluated
                selected = (order, values, image, write_values)
                break
        if selected is None:
            raise FusionLayoutError(
                "excluded_unsatisfiable_value_layout",
                f"no coherent AMO/value witness for location {base}",
            )
        order, values, image, write_values = selected
        selected_orders[location] = order
        endpoint_values.update(values)
        final_images[location] = image
        write_bytes_by_vertex.update(write_values)

    # Plain and Vector reads are evaluated at a coherence cut immediately
    # after their designated rf source, or before all writes when they read init.
    for vertex, (choice, direction) in enumerate(zip(choices, directions)):
        category = str(getattr(choice, "category"))
        if direction != READ or category == "amo":
            continue
        location = locations[vertex]
        order = selected_orders[location]
        initial = initial_images[location]
        footprint = sorted(access_footprints[vertex])
        source = rf_sources.get(vertex)
        if source is None:
            bytes_ = [initial[offset] for offset in footprint]
        else:
            if source not in order:
                raise FusionLayoutError(
                    "excluded_unsatisfiable_value_layout",
                    f"rf source v{source} is not a writer for read v{vertex}",
                )
            cut = order.index(source)
            bytes_ = [
                _latest_value_at_cut(
                    offset,
                    order[: cut + 1],
                    write_bytes_by_vertex,
                    initial[offset],
                )
                for offset in footprint
            ]
        memory_value = _bytes_value(bytes_)
        width = _choice_width(choice)
        endpoint_values[vertex] = EndpointValue(
            vertex,
            direction,
            category,
            width,
            read_memory_value=memory_value,
            read_register_value=(
                _sign_extend(memory_value, width * 8)
                if category == "vector"
                else memory_value
            ),
        )

    initial_by_name = {
        location_names[location]: tuple(image)
        for location, image in initial_images.items()
    }
    final_by_name = {
        location_names[location]: image
        for location, image in final_images.items()
    }
    orders_by_name = {
        location_names[location]: order
        for location, order in selected_orders.items()
    }
    sizes_by_name = {
        location_names[location]: len(image)
        for location, image in initial_images.items()
    }
    observed_by_name = {
        location_names[location]: tuple(
            sorted(
                {
                    offset
                    for vertex in groups[location]
                    if vertex in writers
                    for offset in _all_memory_footprint(
                        choices[vertex], layout.offsets[vertex]
                    )
                }
            )
        )
        for location in groups
    }
    return FusionValuePlan(
        endpoint_values,
        initial_by_name,
        final_by_name,
        orders_by_name,
        rf_sources,
        sizes_by_name,
        observed_by_name,
    )


def _evaluate_writer_order(
    order: Sequence[int],
    location_vertices: Sequence[int],
    choices: Sequence[object],
    directions: Sequence[str],
    layout: FusionAddressLayout,
    initial: Sequence[int],
    rf_sources: Mapping[int, int],
) -> tuple[
    dict[int, EndpointValue],
    dict[int, int],
    dict[int, dict[int, int]],
] | None:
    image = {offset: value for offset, value in enumerate(initial)}
    provenance: dict[int, int | None] = {offset: None for offset in image}
    endpoint_values: dict[int, EndpointValue] = {}
    write_values: dict[int, dict[int, int]] = {}

    for vertex in order:
        choice = choices[vertex]
        category = str(getattr(choice, "category"))
        direction = directions[vertex]
        width = _choice_width(choice)
        base_offset = layout.offsets[vertex]
        if category == "amo":
            footprint = tuple(range(base_offset, base_offset + width))
            source = rf_sources.get(vertex) if direction == READ else None
            if direction == READ:
                if source is None:
                    if any(provenance[offset] is not None for offset in footprint):
                        return None
                else:
                    shared = set(footprint) & set(write_values.get(source, {}))
                    if not shared or any(provenance[offset] != source for offset in shared):
                        return None
            old = _bytes_value(image[offset] for offset in footprint)
            operation = str((getattr(choice, "params", None) or {}).get("amo_op", "swap"))
            operand = _amo_operand(operation, width, old, vertex)
            new = apply_amo(operation, width, old, operand)
            bytes_written = {
                offset: (new >> (8 * index)) & 0xFF
                for index, offset in enumerate(footprint)
            }
            write_values[vertex] = bytes_written
            image.update(bytes_written)
            provenance.update({offset: vertex for offset in bytes_written})
            endpoint_values[vertex] = EndpointValue(
                vertex,
                direction,
                category,
                width,
                read_memory_value=old,
                read_register_value=amo_read_result(width, old),
                write_value=new,
                amo_operand=operand,
                amo_old=old,
                amo_new=new,
            )
            continue

        if direction != WRITE:
            continue
        byte_tag = _write_tag(vertex)
        value = _repeat_byte(byte_tag, width)
        writes: dict[int, int] = {}
        for _element, footprint in _write_footprints(choice, direction, base_offset):
            for index, offset in enumerate(sorted(footprint)):
                writes[offset] = (value >> (8 * index)) & 0xFF
        write_values[vertex] = writes
        image.update(writes)
        provenance.update({offset: vertex for offset in writes})
        endpoint_values[vertex] = EndpointValue(
            vertex,
            direction,
            category,
            width,
            write_value=value,
        )

    return endpoint_values, image, write_values


def _writer_orders(
    writers: Sequence[int], constraints: set[tuple[int, int]]
) -> Sequence[tuple[int, ...]]:
    if len(writers) > 8:
        raise FusionLayoutError(
            "excluded_unsupported_writer_count",
            "fusion value synthesis supports at most eight writers per location",
        )
    valid = [
        order
        for order in permutations(sorted(writers))
        if all(order.index(left) < order.index(right) for left, right in constraints)
    ]
    return valid


def _validate_cycle_overlap(
    cycle: NativeCycle, footprints: Mapping[int, frozenset[int]]
) -> None:
    for vertex, edge in enumerate(cycle.edges):
        if edge.location != "same":
            continue
        target = (vertex + 1) % len(cycle.edges)
        if not footprints[vertex] & footprints[target]:
            raise FusionLayoutError(
                "excluded_unsatisfiable_value_layout",
                f"edge {edge.label} has no byte overlap after layout synthesis",
            )


def _write_footprints(
    choice: object, direction: str, base_offset: int
) -> tuple[tuple[int, frozenset[int]], ...]:
    category = str(getattr(choice, "category"))
    width = _choice_width(choice)
    if category != "vector":
        if direction != WRITE and category != "amo":
            return ()
        return ((0, frozenset(range(base_offset, base_offset + width))),)
    if direction != WRITE:
        return ()
    shape = _vector_memory_shape(choice)
    if shape is None:
        return ()
    effective_vl, nf, stride, mask = shape
    return tuple(
        (
            index * nf + field,
            frozenset(
                range(
                    base_offset + index * stride + field * width,
                    base_offset + index * stride + (field + 1) * width,
                )
            ),
        )
        for index in range(effective_vl)
        if mask == "unmasked" or index % 2 == 0
        for field in range(nf)
    )


def _all_memory_footprint(choice: object, base_offset: int) -> frozenset[int]:
    direction = str(getattr(choice, "direction"))
    writes = _write_footprints(choice, direction, base_offset)
    if writes:
        return frozenset().union(*(bytes_ for _element, bytes_ in writes))
    category = str(getattr(choice, "category"))
    width = _choice_width(choice)
    if category != "vector":
        return frozenset(range(base_offset, base_offset + width))
    shape = _vector_memory_shape(choice)
    if shape is None:
        return frozenset()
    effective_vl, nf, stride, mask = shape
    return frozenset(
        base_offset + index * stride + field * width + byte
        for index in range(effective_vl)
        if mask == "unmasked" or index % 2 == 0
        for field in range(nf)
        for byte in range(width)
    )


def _vector_memory_shape(
    choice: object,
) -> tuple[int, int, int, str] | None:
    params = dict(getattr(choice, "params", None) or {})
    form = str(getattr(choice, "vector_form", ""))
    width = _choice_width(choice)
    if form in WHOLE_REGISTER_VECTOR_OPS:
        nregs = vector_whole_nregs(form, params.get("whole_nreg"))
        if nregs is None:
            return None
        effective_vl = nregs * NANHU_VLEN_BITS // (width * 8)
        return effective_vl, 1, width, "unmasked"

    effective_vl = vector_effective_vl(
        str(params.get("sew", "e32")),
        str(params.get("lmul", "m1")),
        str(params.get("vl", "vl1")),
    )
    nf = vector_nfields(form, params.get("nf"))
    if effective_vl is None or nf is None:
        return None
    if form.startswith("segment_unit_") or form.startswith("segment_indexed_"):
        stride = width * nf
    elif form.startswith("segment_strided_"):
        stride = width * nf * 2
    else:
        stride = width * 2 if form.startswith("strided_") else width
    return effective_vl, nf, stride, str(params.get("mask", "unmasked"))


def _choice_width(choice: object) -> int:
    params = dict(getattr(choice, "params", None) or {})
    category = str(getattr(choice, "category"))
    if category == "vector":
        return int(str(params.get("sew", "e32")).removeprefix("e")) // 8
    if category == "amo":
        return int(params.get("amo_width_bytes", 4))
    return int(params.get("width_bytes", 4))


def _latest_value_at_cut(
    offset: int,
    order: Sequence[int],
    writes: Mapping[int, Mapping[int, int]],
    initial: int,
) -> int:
    for vertex in reversed(order):
        if offset in writes.get(vertex, {}):
            return writes[vertex][offset]
    return initial


def _amo_operand(operation: str, width: int, old: int, vertex: int) -> int:
    bits = width * 8
    mask = (1 << bits) - 1
    tag = _repeat_byte(_write_tag(vertex), width)
    candidates = {
        "swap": [tag, tag ^ mask],
        "add": [tag, 1, mask],
        "xor": [tag or 1, mask],
        "and": [0xF0F0F0F0F0F0F0F0 & mask, 0x0F0F0F0F0F0F0F0F & mask],
        "or": [0x0F0F0F0F0F0F0F0F & mask, 0xF0F0F0F0F0F0F0F0 & mask],
        "min": [(1 << (bits - 1)) | (vertex + 1), 0],
        "max": [(1 << (bits - 1)) - 1 - vertex, 1],
        "minu": [1, vertex + 1],
        "maxu": [mask - vertex, mask],
    }.get(operation, [tag])
    for operand in candidates:
        if apply_amo(operation, width, old, operand) != old:
            return operand & mask
    return candidates[0] & mask


def _initial_byte(location: int, offset: int) -> int:
    if offset >= 16:
        return 0
    return 0x40 + ((location * 17 + offset) % 0x30)


def _write_tag(vertex: int) -> int:
    return 0x11 + (vertex * 0x11) % 0xCC


def _repeat_byte(value: int, width: int) -> int:
    return sum((value & 0xFF) << (8 * index) for index in range(width))


def _bytes_value(values: Sequence[int] | Any) -> int:
    return sum((int(value) & 0xFF) << (8 * index) for index, value in enumerate(values))


def _sign_extend(value: int, bits: int, xlen: int = 64) -> int:
    value &= (1 << bits) - 1
    if bits < xlen and value & (1 << (bits - 1)):
        value |= ((1 << (xlen - bits)) - 1) << bits
    return value


def _object_size(required: int) -> int:
    size = 16
    while size < required:
        size *= 2
    if size > 4096:
        raise FusionLayoutError(
            "excluded_unsupported_cross_page",
            f"fusion memory object requires {size} bytes",
        )
    return size
