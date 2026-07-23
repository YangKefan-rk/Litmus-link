from __future__ import annotations

from dataclasses import dataclass

import pytest

from litmus_link.amo import AMO_OPERATIONS, apply_amo
from litmus_link.fusion_layout import endpoint_footprint, synthesize_address_layout
from litmus_link.fusion_values import synthesize_fusion_values
from litmus_link.native_cycles import location_ids, vertex_directions
from litmus_link.native_scalar import native_template_cycles


@dataclass(frozen=True)
class _Choice:
    category: str
    direction: str
    params: dict[str, object]
    vector_form: str = ""


def _plan(operation: str = "add"):
    cycles, _audit = native_template_cycles(["MP"], ["po"])
    cycle = cycles[0]
    directions = vertex_directions(cycle.edges)
    choices = tuple(
        _Choice(
            "vector",
            direction,
            {
                "sew": "e16",
                "lmul": "m1",
                "vl": "vl2",
                "mask": "unmasked",
            },
            "unit_load" if direction == "R" else "unit_store",
        )
        if vertex == 0
        else _Choice(
            "amo",
            direction,
            {
                "amo_width_bytes": 8,
                "amo_op": operation,
                "amo_ordering": "aqrl",
            },
        )
        if vertex == 1
        else _Choice("scalar", direction, {"width_bytes": 4})
        for vertex, direction in enumerate(directions)
    )
    locations = location_ids(cycle.edges)
    groups = [
        tuple(vertex for vertex, actual in enumerate(locations) if actual == location)
        for location in sorted(set(locations))
    ]
    layout = synthesize_address_layout(
        [endpoint_footprint(vertex, choice) for vertex, choice in enumerate(choices)],
        groups,
        "contained",
    )
    names = {location: "xyz"[location] for location in set(locations)}
    return cycle, choices, layout, synthesize_fusion_values(
        cycle, choices, layout, names
    )


@pytest.mark.parametrize("operation", AMO_OPERATIONS)
def test_fusion_values_calculate_every_nanhu_amo(operation: str) -> None:
    _cycle, _choices, _layout, plan = _plan(operation)
    amo = plan.endpoints[1]
    assert amo.amo_old is not None
    assert amo.amo_operand is not None
    assert amo.amo_new == apply_amo(operation, 8, amo.amo_old, amo.amo_operand)
    assert amo.write_value == amo.amo_new


def test_fusion_value_plan_records_byte_provenance_and_final_image() -> None:
    cycle, choices, layout, plan = _plan()
    assert plan.rf_sources
    assert plan.co_orders
    assert all(len(image) >= 16 for image in plan.initial_bytes.values())
    assert all(plan.final_bytes[name] for name in plan.initial_bytes)
    assert all(
        layout.offsets[vertex]
        % (
            int(choices[vertex].params.get("amo_width_bytes", 4))
            if choices[vertex].category == "amo"
            else int(str(choices[vertex].params.get("sew", "e32"))[1:]) // 8
            if choices[vertex].category == "vector"
            else int(choices[vertex].params.get("width_bytes", 4))
        )
        == 0
        for vertex in range(cycle.size)
    )
