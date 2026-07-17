from __future__ import annotations

import json

import pytest

from litmus_link.litmus_ir import MemoryAccess
from litmus_link.memory_layout import all_accesses_overlap, expand_memory_layouts
from litmus_link.native_cycles import NativeCycle
from litmus_link.native_edges import edge_by_label
from litmus_link.native_scalar import lower_native_cycle
from litmus_link.native_scalar import generate_native_templates
from litmus_link.toolchain import ToolchainError, herd_judge, tools_available


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


def _mp_cycle() -> NativeCycle:
    return NativeCycle(
        tuple(edge_by_label(label) for label in ("Rfe", "PodRR", "Fre", "PodWW")),
        "MP",
    )


def test_native_lowering_emits_real_cross_line_misaligned_assembly() -> None:
    layout = next(
        item
        for item in expand_memory_layouts(("misaligned",), widths=(8,), boundaries=("cross64",))
    )
    case = lower_native_cycle(_mp_cycle(), memory_layout=layout)
    assert "uint8_t x[128];" in case.litmus
    assert "sd x5,60(" in case.litmus
    assert "ld x5,60(" in case.litmus
    assert "x[63]=0x01" in case.litmus
    accesses = [
        event.memory_access
        for event in case.case_ir.events()
        if event.kind in {"load", "store"}
    ]
    assert all(access is not None and access.boundary == "cross64" for access in accesses)


@pytest.mark.skipif(not tools_available(), reason="herd7/riscv.cat is not installed")
def test_real_misaligned_litmus_is_accepted_by_herd_mixed_unaligned() -> None:
    layout = next(
        item
        for item in expand_memory_layouts(("mixed",), boundaries=("cross16",))
    )
    case = lower_native_cycle(_mp_cycle(), memory_layout=layout)
    try:
        verdict = herd_judge(case.litmus, variants=("mixed", "unaligned"))
    except ToolchainError as exc:
        if "Mixed mode not implemented for architecture RISCV" in str(exc):
            pytest.skip("installed herd7 does not implement RISC-V mixed-size semantics")
        raise
    assert verdict.outcome in {"observable", "forbidden"}


def test_native_generation_expands_layout_domain_and_excludes_amo_misalignment(tmp_path) -> None:
    layouts = expand_memory_layouts(
        ("aligned", "misaligned", "mixed"),
        widths=(2, 8),
        boundaries=("same16", "cross64"),
    )
    report = generate_native_templates(
        out_dir=tmp_path,
        presets=("MP",),
        mechanisms=("po",),
        include_same=False,
        annotations=("P", "Aq"),
        limit=20,
        judge=False,
        memory_layouts=layouts,
    )
    audit = report["audit"]
    assert report["available_litmus"] > audit["annotated_cycles"]
    assert audit["excluded_misaligned_atomic_annotations"] > 0
    assert len(audit["memory_layouts"]) == len(layouts)
    metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in tmp_path.glob("*.meta.json")
    ]
    assert any(item["native"]["memory_layout"]["mode"] == "misaligned" for item in metadata)
    assert all(
        set(item["native"]["annotations"]) == {"P"}
        for item in metadata
        if item["native"]["memory_layout"]["mode"] != "aligned"
    )
