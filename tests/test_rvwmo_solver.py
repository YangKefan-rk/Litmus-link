from __future__ import annotations

from dataclasses import replace

import pytest

from litmus_link.amo import AMO_OPERATIONS, AMO_ORDERINGS, AmoSpec, apply_amo
from litmus_link.litmus_ir import LitmusCaseIR, LitmusEvent, MemoryAccess
from litmus_link.memory_layout import MemoryLayoutConfig, expand_memory_layouts
from litmus_link.native_cycles import NativeCycle
from litmus_link.native_diy import DEFAULT_DIY_RELAX, DEFAULT_DIY_SAFE, DiyConfig, enumerate_diy_cycles
from litmus_link.native_edges import edge_by_label
from litmus_link.native_scalar import lower_native_cycle
from litmus_link.rvwmo_solver import solve_rvwmo
from litmus_link.toolchain import herd_judge, tools_available


def _case(labels: list[str]):
    cycle = NativeCycle(tuple(edge_by_label(label) for label in labels), "TEST")
    return lower_native_cycle(cycle)


def _amo_case(
    operation: str,
    width_bytes: int,
    ordering: str,
    *,
    old: int,
    operand: int,
) -> LitmusCaseIR:
    spec = AmoSpec(operation, width_bytes, ordering)
    new = apply_amo(operation, width_bytes, old, operand)
    return LitmusCaseIR(
        name="AMO",
        display_name="AMO",
        combination_name="AMO",
        skeleton="AMO",
        variant="amo-value",
        cycle="Rmw",
        init_lines=[f"x=0x{old:x};", "0:x6=x;", f"0:x7=0x{operand:x};"],
        harts=[
            [
                LitmusEvent(
                    "a0",
                    0,
                    "amo",
                    f"{spec.mnemonic} x5,x7,(x6)",
                    "x",
                    "x5",
                    read_value=f"0x{old:x}",
                    write_value=f"0x{new:x}",
                    amo_op=operation,
                    amo_operand=f"0x{operand:x}",
                    amo_width_bytes=width_bytes,
                    amo_ordering=ordering,
                    memory_access=MemoryAccess.create(
                        "x", 0, width_bytes, transaction_kind="amo_rmw"
                    ),
                )
            ]
        ],
        relations=[],
        exists=f"(0:x5=0x{old:x} /\\ x=0x{new:x})",
        expected_outcome="solver_required",
        model="rvwmo",
    )


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["Rfe", "PodRR", "Fre", "PodWW"], "observable"),
        (["Rfe", "Fence.rw.rwdRR", "Fre", "Fence.rw.rwdWW"], "forbidden"),
        (["Rfe", "DpCtrldR", "Fre", "Fence.rw.rwdWW"], "observable"),
    ],
)
def test_embedded_rvwmo_mp_verdicts(labels: list[str], expected: str) -> None:
    verdict = solve_rvwmo(_case(labels).case_ir)
    assert verdict.status == "verified"
    assert verdict.verdict == expected
    assert verdict.allowed is (expected == "observable")


def test_embedded_forbidden_requires_exhaustive_search() -> None:
    case = _case(["Fence.r.rsWR", "Fre", "Fence.r.rsWW", "Wse"])
    complete = solve_rvwmo(case.case_ir, max_candidates=10)
    assert complete.verdict == "forbidden"
    assert complete.candidates == 2
    limited = solve_rvwmo(case.case_ir, max_candidates=1)
    assert limited.status == "inconclusive"
    assert limited.allowed is None


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["Rfe", "PodRR", "Fre", "PodWW"], "observable"),
        (["Rfe", "Fence.rw.rwdRR", "Fre", "Fence.rw.rwdWW"], "forbidden"),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        MemoryLayoutConfig("misaligned", 2, "same16"),
        MemoryLayoutConfig("misaligned", 8, "cross64"),
        MemoryLayoutConfig("mixed", 4, "cross16"),
    ],
)
def test_embedded_byte_level_no_mag_preserves_rvwmo_ordering(
    labels: list[str], expected: str, layout: MemoryLayoutConfig
) -> None:
    cycle = NativeCycle(tuple(edge_by_label(label) for label in labels), "MP")
    case = lower_native_cycle(cycle, memory_layout=layout)
    verdict = solve_rvwmo(case.case_ir, max_candidates=1_000_000)
    assert verdict.status == "verified"
    assert verdict.verdict == expected
    assert all(
        event.atomicity_model in {"byte_level_no_mag", "initial"}
        for event in verdict.events
    )
    payload = verdict.to_json()
    assert payload["model"] == "riscv.cat+byte_level_no_mag"
    assert payload["model_extensions"] == ["mixed-size", "unaligned", "byte-level-no-mag"]


def test_byte_level_no_mag_allows_a_torn_read_without_sync() -> None:
    case = lower_native_cycle(
        _case(["Rfe", "PodRR", "Fre", "PodWW"]).cycle,
        memory_layout=MemoryLayoutConfig("misaligned", 2, "same16"),
    )
    harts = []
    changed = False
    for hart in case.case_ir.harts:
        events = []
        for event in hart:
            if event.event_id == "v1":
                # The intended source writes 0x0101.  Read its low byte while
                # the high byte still reads from the initial write.
                event = replace(event, value="0x1")
                changed = True
            events.append(event)
        harts.append(events)
    assert changed
    torn = replace(case.case_ir, harts=harts, exists=case.case_ir.exists.replace("x5=0x101", "x5=0x1"))
    verdict = solve_rvwmo(torn)
    assert verdict.status == "verified"
    assert verdict.verdict == "observable"
    source_by_read = {
        read: write
        for write, read in (verdict.execution.rf if verdict.execution else set())
    }
    assert source_by_read["v1.b0"] == "v0.b0"
    assert source_by_read["v1.b1"].startswith("init:")


def test_aligned_mixed_size_plain_transactions_receive_formal_verdict() -> None:
    layout = next(
        item
        for item in expand_memory_layouts(
            ("atomic_mixed",),
            atomic_overlaps=("partial_overlap",),
            event_count=4,
        )
    )
    case = lower_native_cycle(_case(["Rfe", "PodRR", "Fre", "PodWW"]).cycle, memory_layout=layout)
    verdict = solve_rvwmo(case.case_ir)
    assert verdict.status == "verified"
    assert verdict.verdict == "observable"
    assert verdict.execution is not None
    assert verdict.execution.rf_bytes
    assert all(
        len(event.footprint) == event.access_size
        for event in verdict.events
        if not event.initial
    )
    assert all(
        not event.event_id.endswith(tuple(f".b{index}" for index in range(8)))
        for event in verdict.events
        if not event.initial
    )


def test_aligned_partial_overlap_is_computed_by_byte_footprint() -> None:
    layout = next(
        item
        for item in expand_memory_layouts(
            ("atomic_mixed",),
            widths=(2, 4, 8),
            atomic_overlaps=("partial_overlap",),
            event_count=4,
        )
        if item.width_pattern == (2, 4, 8, 2)
    )
    case = lower_native_cycle(
        _case(["Rfe", "PodRR", "Fre", "PodWW"]).cycle,
        memory_layout=layout,
    )
    verdict = solve_rvwmo(case.case_ir)
    assert verdict.status == "verified"
    assert verdict.execution is not None
    rf_bytes = verdict.execution.rf_bytes
    assert ("v0", "v1", "x[6]") in rf_bytes
    assert ("v0", "v1", "x[7]") in rf_bytes
    assert ("init:x[4]", "v1", "x[4]") in rf_bytes
    assert ("init:x[5]", "v1", "x[5]") in rf_bytes


def test_byte_array_initializer_populates_embedded_initial_writes() -> None:
    case = _case(["Rfe", "PodRR", "Fre", "PodWW"]).case_ir
    load = LitmusEvent(
        "r0",
        0,
        "load",
        "lw x5,0(x6)",
        "x",
        "x5",
        value="0x44332211",
        memory_access=MemoryAccess.create("x", 0, 4),
    )
    array_case = replace(
        case,
        init_lines=["uint8_t x[8]={0x11,0x22,0x33,0x44,0,0,0,0};", "0:x6=x;"],
        harts=[[load]],
        relations=[],
        exists="(0:x5=0x44332211)",
    )
    verdict = solve_rvwmo(array_case)
    assert verdict.status == "verified"
    assert verdict.verdict == "observable"
    assert verdict.execution is not None
    assert {
        (f"init:x[{offset}]", "r0", f"x[{offset}]")
        for offset in range(4)
    } <= verdict.execution.rf_bytes


def test_memory_access_round_trips_through_case_ir_json() -> None:
    case = lower_native_cycle(
        _case(["Rfe", "PodRR", "Fre", "PodWW"]).cycle,
        memory_layout=MemoryLayoutConfig("misaligned", 8, "cross64"),
    ).case_ir
    restored = LitmusCaseIR.from_json(case.to_json())
    assert restored.to_json() == case.to_json()
    assert restored.events()[0].memory_access is not None


@pytest.mark.parametrize("operation", AMO_OPERATIONS)
@pytest.mark.parametrize("width_bytes", [4, 8])
@pytest.mark.parametrize("ordering", AMO_ORDERINGS)
def test_nanhu_amo_old_operand_new_values_are_solved(
    operation: str, width_bytes: int, ordering: str
) -> None:
    mask = (1 << (width_bytes * 8)) - 1
    old = (mask - 0x10203) & mask
    operand = 0x102030405 & mask
    verdict = solve_rvwmo(
        _amo_case(operation, width_bytes, ordering, old=old, operand=operand)
    )
    assert verdict.status == "verified"
    assert verdict.verdict == "observable"
    amo = next(event for event in verdict.events if event.event_id == "a0")
    assert amo.read_value == old
    assert amo.amo_operand == operand
    assert amo.write_value == apply_amo(operation, width_bytes, old, operand)
    assert amo.aq is (ordering in {"aq", "aqrl"})
    assert amo.rl is (ordering in {"rl", "aqrl"})


def test_amo_d_reconstructs_old_value_from_partial_overlap_sources() -> None:
    old = 0x1122334455667788
    high = 0xAABBCCDD
    reconstructed = 0xAABBCCDD55667788
    case = _amo_case("or", 8, "relaxed", old=reconstructed, operand=0)
    store = LitmusEvent(
        "w0",
        0,
        "store",
        "sw x9,4(x6)",
        "x",
        "x9",
        value=f"0x{high:x}",
        memory_access=MemoryAccess.create("x", 4, 4),
    )
    amo = replace(case.harts[0][0], hart=1)
    case = replace(
        case,
        init_lines=[f"x=0x{old:x};", "0:x6=x;", "1:x6=x;", f"0:x9=0x{high:x};"],
        harts=[[store], [amo]],
        exists=f"(1:x5=0x{reconstructed:x} /\\ x=0x{reconstructed:x})",
    )
    verdict = solve_rvwmo(case)
    assert verdict.status == "verified"
    assert verdict.verdict == "observable"
    assert verdict.execution is not None
    assert all(
        (f"init:x[{byte}]", "a0", f"x[{byte}]") in verdict.execution.rf_bytes
        for byte in range(4)
    )
    assert all(
        ("w0", "a0", f"x[{byte}]") in verdict.execution.rf_bytes
        for byte in range(4, 8)
    )


def test_aligned_load_cannot_observe_a_torn_amo_value() -> None:
    case = _amo_case("swap", 8, "relaxed", old=0, operand=0xFFFFFFFFFFFFFFFF)
    amo = case.harts[0][0]
    load = LitmusEvent(
        "r0",
        1,
        "load",
        "ld x10,0(x6)",
        "x",
        "x10",
        value="0xffffffff",
        memory_access=MemoryAccess.create("x", 0, 8),
    )
    torn = replace(
        case,
        init_lines=["x=0;", "0:x6=x;", "1:x6=x;", "0:x7=-1;"],
        harts=[[amo], [load]],
        exists="(0:x5=0 /\\ 1:x10=0xffffffff /\\ x=0xffffffffffffffff)",
    )
    verdict = solve_rvwmo(torn)
    assert verdict.status == "verified"
    assert verdict.verdict == "forbidden"


def test_misaligned_amo_is_rejected_from_nanhu_formal_scope() -> None:
    with pytest.raises(ValueError, match="amo_rmw transactions must be naturally aligned"):
        MemoryAccess.create(
            "x", 2, 4, "byte_level_no_mag", transaction_kind="amo_rmw"
        )


@pytest.mark.skipif(not tools_available(), reason="herd7/riscv.cat is not installed")
@pytest.mark.parametrize(
    "labels",
    [
        ["Rfe", "PodRR", "Fre", "PodWW"],
        ["Rfe", "Fence.rw.rwdRR", "Fre", "Fence.rw.rwdWW"],
        ["Rfe", "DpCtrldR", "Fre", "Fence.rw.rwdWW"],
        ["PodRW", "Rfe", "PodRW", "Rfe"],
        ["PodWR", "Fre", "PodWR", "Fre"],
    ],
)
def test_embedded_rvwmo_matches_herd7(labels: list[str]) -> None:
    case = _case(labels)
    embedded = solve_rvwmo(case.case_ir)
    external = herd_judge(case.litmus)
    assert embedded.verdict == external.outcome
    assert embedded.allowed == external.allowed


@pytest.mark.skipif(not tools_available(), reason="herd7/riscv.cat is not installed")
def test_embedded_rvwmo_matches_herd7_for_complete_total_mode_domain() -> None:
    cycles, _audit = enumerate_diy_cycles(
        DiyConfig(
            safe=DEFAULT_DIY_SAFE,
            relax=DEFAULT_DIY_RELAX,
            mode="total",
        )
    )
    assert len(cycles) == 59
    for cycle in cycles:
        case = lower_native_cycle(cycle)
        embedded = solve_rvwmo(case.case_ir)
        external = herd_judge(case.litmus)
        assert embedded.status == "verified", cycle.labels
        assert embedded.allowed == external.allowed, cycle.labels
