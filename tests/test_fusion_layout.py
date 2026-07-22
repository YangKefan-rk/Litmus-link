from __future__ import annotations

import pytest

from litmus_link.fusion_layout import (
    FUSION_OVERLAP_LAYOUTS,
    EndpointFootprint,
    FusionLayoutError,
    synthesize_address_layout,
)


@pytest.mark.parametrize("shape", FUSION_OVERLAP_LAYOUTS[:-1])
def test_fusion_layouts_keep_every_transaction_naturally_aligned(shape: str) -> None:
    footprints = [
        EndpointFootprint(0, 8, frozenset(range(8))),
        EndpointFootprint(1, 4, frozenset(range(4))),
        EndpointFootprint(2, 2, frozenset(range(2))),
    ]
    layout = synthesize_address_layout(footprints, [(0, 1, 2)], shape)
    assert layout.shape == shape
    for footprint in footprints:
        assert layout.offsets[footprint.vertex] % footprint.alignment_bytes == 0
    assert layout.footprints[0] & layout.footprints[1]
    assert layout.footprints[1] & layout.footprints[2]


def test_high_partial_matches_nanhu_mixed_width_example() -> None:
    layout = synthesize_address_layout(
        [
            EndpointFootprint(0, 8, frozenset(range(8))),
            EndpointFootprint(1, 4, frozenset(range(4))),
            EndpointFootprint(2, 2, frozenset(range(2))),
        ],
        [(0, 1, 2)],
        "high_partial",
    )
    assert layout.offsets == {0: 0, 1: 4, 2: 6}


def test_non_same_start_layout_rejects_equal_footprints() -> None:
    with pytest.raises(FusionLayoutError) as error:
        synthesize_address_layout(
            [
                EndpointFootprint(0, 4, frozenset(range(4))),
                EndpointFootprint(1, 4, frozenset(range(4))),
            ],
            [(0, 1)],
            "contained",
        )
    assert error.value.reason == "excluded_unsatisfiable_value_layout"


def test_disjoint_control_is_excluded_from_relation_cycle_lowering() -> None:
    with pytest.raises(FusionLayoutError) as error:
        synthesize_address_layout(
            [
                EndpointFootprint(0, 8, frozenset(range(8))),
                EndpointFootprint(1, 4, frozenset(range(4))),
            ],
            [(0, 1)],
            "disjoint_control",
        )
    assert error.value.reason == "excluded_unsatisfiable_value_layout"
