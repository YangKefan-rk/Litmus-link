from __future__ import annotations

import pytest

from litmus_link.litmus_ir import MemoryAccess
from litmus_link.memory_layout import all_accesses_overlap, expand_memory_layouts


def test_memory_access_computes_real_byte_ranges_and_boundaries() -> None:
    same = MemoryAccess.create("x", 1, 4)
    cross16 = MemoryAccess.create("x", 12, 8)
    cross64 = MemoryAccess.create("x", 63, 2)
    assert same.covered_bytes == (1, 2, 3, 4)
    assert same.boundary == "same16"
    assert same.atomicity_model == "byte_level_no_mag"
    assert cross16.boundary == "cross16_same_line"
    assert cross64.boundary == "cross64"


def test_no_mag_model_rejects_naturally_aligned_access() -> None:
    with pytest.raises(ValueError, match="reserved for misaligned"):
        MemoryAccess("x", 0, 4, (0, 1, 2, 3), True, "same16", "byte_level_no_mag")


def test_layout_expansion_covers_width_boundary_and_mixed_axes() -> None:
    layouts = expand_memory_layouts(
        ("aligned", "misaligned", "mixed"),
        widths=(2, 8),
        boundaries=("same16", "cross64"),
    )
    assert len(layouts) == 1 + 2 * 2 + 2
    assert layouts[0].id == "aligned-w32"
    assert all(layout.to_json()["mag_bytes"] is None for layout in layouts)


@pytest.mark.parametrize("boundary", ["same16", "cross16", "cross64"])
def test_mixed_layout_accesses_are_misaligned_and_share_bytes(boundary: str) -> None:
    layout = next(
        item
        for item in expand_memory_layouts(("mixed",), boundaries=(boundary,))
        if item.boundary == boundary
    )
    accesses = [layout.access_for("x", index) for index in range(6)]
    assert {access.size_bytes for access in accesses} == {2, 4, 8}
    assert all(not access.natural_aligned for access in accesses)
    assert all(access.atomicity_model == "byte_level_no_mag" for access in accesses)
    assert all_accesses_overlap(accesses)
