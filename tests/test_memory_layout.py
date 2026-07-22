from __future__ import annotations

import json

import pytest

from litmus_link.litmus_ir import LitmusCaseIR, LitmusEvent, MemoryAccess
from litmus_link.memory_layout import all_accesses_overlap, expand_memory_layouts
from litmus_link.native_cycles import NativeCycle
from litmus_link.native_edges import edge_by_label
from litmus_link.native_scalar import generate_native_templates, lower_native_cycle, native_template_audit
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


def test_transaction_ir_enforces_vector_and_amo_invariants() -> None:
    vector = MemoryAccess.create(
        "x",
        4,
        4,
        transaction_kind="vector_element",
        parent_instruction="v0",
        element_index=1,
    )
    assert vector.covered_bytes == (4, 5, 6, 7)
    assert vector.to_json()["transaction_kind"] == "vector_element"
    with pytest.raises(ValueError, match="parent_instruction"):
        MemoryAccess.create(
            "x", 0, 4, transaction_kind="vector_element", element_index=0
        )
    with pytest.raises(ValueError, match="naturally aligned"):
        MemoryAccess.create(
            "x", 1, 4, transaction_kind="amo_rmw"
        )


def test_case_ir_v1_remains_readable_and_new_ir_defaults_to_v2() -> None:
    legacy = {
        "schema": "litmus-link.case-ir.v1",
        "name": "legacy",
        "harts": [[{
            "event_id": "v0",
            "hart": 0,
            "kind": "load",
            "instruction": "lw x5,0(x8)",
            "location": "x",
            "value": "0",
        }]],
        "relations": [],
    }
    restored = LitmusCaseIR.from_json(legacy)
    assert restored.schema == "litmus-link.case-ir.v1"
    assert restored.events()[0].read_value == ""

    new_event = LitmusEvent(
        "v0",
        0,
        "amo",
        "amoadd.w.aq x5,x6,(x8)",
        location="x",
        memory_access=MemoryAccess.create(
            "x", 0, 4, transaction_kind="amo_rmw"
        ),
        read_value="1",
        write_value="3",
        amo_op="add",
        amo_operand="2",
        amo_width_bytes=4,
        amo_ordering="aq",
    )
    assert new_event.to_json()["write_value"] == "3"


def test_layout_expansion_covers_width_boundary_and_mixed_axes() -> None:
    layouts = expand_memory_layouts(
        ("aligned", "misaligned", "mixed"),
        widths=(2, 8),
        boundaries=("same16", "cross64"),
    )
    assert len(layouts) == 1 + 2 * 2 + (2**2 - 2) * 2
    assert layouts[0].id == "aligned-w32"
    assert all(layout.to_json()["mag_bytes"] is None for layout in layouts)


def test_atomic_layouts_cover_fixed_width_and_aligned_overlap_shapes() -> None:
    layouts = expand_memory_layouts(
        ("atomic", "atomic_mixed"),
        widths=(2, 4, 8),
        atomic_overlaps=("same_start", "partial_overlap"),
        event_count=4,
    )
    assert [layout.id for layout in layouts[:3]] == ["atomic-w16", "atomic-w32", "atomic-w64"]
    assert len(layouts) == 3 + (3**4 - 3) * 2
    assert {layout.overlap for layout in layouts[3:]} == {"same_start", "partial_overlap"}
    assert all(len(layout.width_pattern) == 4 for layout in layouts[3:])
    assert all(len(set(layout.width_pattern)) >= 2 for layout in layouts[3:])
    for layout in layouts:
        accesses = [layout.access_for("x", index) for index in range(4)]
        assert all(access.natural_aligned for access in accesses)
        if layout.mode == "atomic_mixed":
            assert all(access.atomicity_model == "mixed_size_atomic" for access in accesses)
            assert len({access.size_bytes for access in accesses}) >= 2
            assert all_accesses_overlap(accesses)


@pytest.mark.parametrize("boundary", ["same16", "cross16", "cross64"])
def test_mixed_layout_accesses_are_misaligned_and_share_bytes(boundary: str) -> None:
    layout = next(
        item for item in expand_memory_layouts(
            ("mixed",), boundaries=(boundary,), event_count=3
        ) if item.width_pattern == (2, 4, 8)
    )
    accesses = [layout.access_for("x", index) for index in range(3)]
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


def test_mixed_native_lowering_varies_widths_within_each_location() -> None:
    layout = next(
        item for item in expand_memory_layouts(
            ("mixed",), boundaries=("same16",), event_count=4
        ) if item.width_pattern == (2, 4, 8, 2)
    )
    case = lower_native_cycle(_mp_cycle(), memory_layout=layout)
    by_event = {}
    for event in case.case_ir.events():
        if event.role == "cycle-event" and event.memory_access is not None:
            by_event[event.event_id] = event.memory_access
    assert [by_event[f"v{index}"].size_bytes for index in range(4)] == [2, 4, 8, 2]
    assert all_accesses_overlap(by_event.values())


def test_atomic_mixed_lowering_is_aligned_and_partial_overlap() -> None:
    layout = next(
        item
        for item in expand_memory_layouts(
            ("atomic_mixed",),
            widths=(2, 4, 8),
            atomic_overlaps=("partial_overlap",),
            event_count=4,
        )
        if item.width_pattern == (2, 4, 2, 8)
    )
    cycle = NativeCycle(_mp_cycle().edges, "MP", ("AMO",) * 4)
    case = lower_native_cycle(cycle, memory_layout=layout)
    events = [event for event in case.case_ir.events() if event.role == "cycle-event"]
    accesses = [event.memory_access for event in events]
    assert all(access is not None and access.natural_aligned for access in accesses)
    assert all(access is not None and access.atomicity_model == "mixed_size_atomic" for access in accesses)
    assert any("amoor.h" in event.instruction or "amoswap.h" in event.instruction for event in events)
    assert any(access is not None and access.offset_bytes == 4 for access in accesses)
    assert case.case_ir.expected_outcome == "manual_oracle_required"


def test_atomic_mixed_widths_are_assigned_per_cycle_event() -> None:
    layout = next(
        item
        for item in expand_memory_layouts(
            ("atomic_mixed",),
            widths=(2, 4, 8),
            atomic_overlaps=("same_start",),
            event_count=4,
        )
        if item.width_pattern == (2, 4, 2, 8)
    )
    case = lower_native_cycle(
        NativeCycle(_mp_cycle().edges, "MP", ("AMO",) * 4),
        memory_layout=layout,
    )
    by_event = {
        event.event_id: event.memory_access.size_bytes
        for event in case.case_ir.events()
        if event.role == "cycle-event" and event.memory_access is not None
    }
    assert by_event == {"v0": 2, "v1": 4, "v2": 2, "v3": 8}


def test_audit_uses_concrete_event_level_layout_count() -> None:
    layouts = expand_memory_layouts(
        ("atomic", "atomic_mixed"),
        widths=(2, 4, 8),
        atomic_overlaps=("same_start", "partial_overlap"),
    )
    audit = native_template_audit(
        ["MP"],
        ("po",),
        include_same=False,
        annotations=("AMO",),
        memory_layouts=layouts,
    )
    assert audit["accepted"] == 159
    assert audit["memory_layout_counts_by_cycle_size"]["4"] == {
        "cycle_events": 4,
        "concrete_layouts": 159,
        "mode_atomic": 3,
        "mode_atomic_mixed": 156,
    }


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
