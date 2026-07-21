from __future__ import annotations

from dataclasses import replace

import pytest

from litmus_link.litmus_ir import LitmusCaseIR
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


def test_mixed_size_atomic_is_not_claimed_verified_by_embedded_solver() -> None:
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
    assert verdict.status == "not_applicable"
    assert "mixed-size atomic" in verdict.reason


def test_memory_access_round_trips_through_case_ir_json() -> None:
    case = lower_native_cycle(
        _case(["Rfe", "PodRR", "Fre", "PodWW"]).cycle,
        memory_layout=MemoryLayoutConfig("misaligned", 8, "cross64"),
    ).case_ir
    restored = LitmusCaseIR.from_json(case.to_json())
    assert restored.to_json() == case.to_json()
    assert restored.events()[0].memory_access is not None


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
