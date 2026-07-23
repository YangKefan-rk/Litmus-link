from __future__ import annotations

import pytest

from litmus_link.fusion_layout import (
    FUSION_OVERLAP_LAYOUTS,
    EndpointFootprint,
    FusionLayoutError,
    synthesize_address_layout,
)


@pytest.mark.parametrize("shape", ("same_start", "contained"))
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


def test_contained_matches_nanhu_mixed_width_example() -> None:
    layout = synthesize_address_layout(
        [
            EndpointFootprint(0, 8, frozenset(range(8))),
            EndpointFootprint(1, 4, frozenset(range(4))),
            EndpointFootprint(2, 2, frozenset(range(2))),
        ],
        [(0, 1, 2)],
        "contained",
    )
    assert layout.offsets == {0: 0, 1: 4, 2: 4}


@pytest.mark.parametrize("shape", ("low_partial", "high_partial"))
def test_naturally_aligned_power_of_two_accesses_cannot_partially_overlap(
    shape: str,
) -> None:
    with pytest.raises(FusionLayoutError) as error:
        synthesize_address_layout(
            [
                EndpointFootprint(0, 8, frozenset(range(8))),
                EndpointFootprint(1, 4, frozenset(range(4))),
            ],
            [(0, 1)],
            shape,
        )
    assert error.value.reason == "excluded_unsatisfiable_value_layout"


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


@pytest.mark.parametrize(
    ("placement", "expected_offset", "crosses_16", "crosses_64"),
    [
        ("misalign_same16", 7, False, False),
        ("misalign_cross16", 15, True, False),
        ("misalign_cross64", 63, True, True),
    ],
)
def test_vector_misalignment_placements_have_real_boundary_offsets(
    placement: str,
    expected_offset: int,
    crosses_16: bool,
    crosses_64: bool,
) -> None:
    layout = synthesize_address_layout(
        [
            EndpointFootprint(0, 4, frozenset(range(4)), placement),
            EndpointFootprint(1, 1, frozenset({0})),
        ],
        [(0, 1)],
        "same_start",
    )
    offset = layout.offsets[0]
    end = offset + 3
    assert offset == layout.offsets[1] == expected_offset
    assert offset % 4 != 0
    assert (offset // 16 != end // 16) is crosses_16
    assert (offset // 64 != end // 64) is crosses_64
    assert layout.alignment_modes == {0: placement, 1: "aligned"}
    assert layout.to_json()["naturally_aligned"] is False


def test_e8_vector_element_has_no_misaligned_placement() -> None:
    with pytest.raises(ValueError, match="8-bit memory element cannot be misaligned"):
        EndpointFootprint(
            0,
            1,
            frozenset({0}),
            "misalign_cross64",
        )


@pytest.mark.parametrize(
    "shape",
    ["low_partial", "high_partial"],
)
def test_misaligned_partial_layout_is_oriented_to_vector_endpoint(
    shape: str,
) -> None:
    layout = synthesize_address_layout(
        [
            EndpointFootprint(0, 4, frozenset(range(4)), "misalign_same16"),
            EndpointFootprint(1, 4, frozenset(range(4))),
        ],
        [(0, 1)],
        shape,
    )
    vector = layout.footprints[0]
    common = vector & layout.footprints[1]
    assert (min(common) == min(vector)) is (shape == "low_partial")
    assert (max(common) == max(vector)) is (shape == "high_partial")
