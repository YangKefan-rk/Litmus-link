from __future__ import annotations

import pytest

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
