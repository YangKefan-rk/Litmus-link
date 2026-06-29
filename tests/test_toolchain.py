import pytest

from toolchain import tools_available, diycross_generate, herd_judge, _parse_herd


requires_tools = pytest.mark.skipif(
    not tools_available(), reason="herdtools7 (diycross7/herd7/riscv.cat) not installed"
)


# --- Pure parser tests: no herd7 binary needed, so they run unconditionally. ---

def test_parse_herd_never_is_forbidden() -> None:
    v = _parse_herd("Observation T Never 0 5\n")
    assert v.outcome == "forbidden" and v.allowed is False


def test_parse_herd_sometimes_is_observable() -> None:
    v = _parse_herd("Observation T Sometimes 2 5\n")
    assert v.outcome == "observable" and v.allowed is True


def test_parse_herd_no_observation_is_unknown() -> None:
    v = _parse_herd("nothing recognizable here")
    assert v.outcome == "unknown" and v.allowed is None


def test_parse_herd_contradictory_counts_are_unknown() -> None:
    # Defensive consistency check (solver Fix I): the Observation word and the
    # witness counts must agree (Never <=> positive==0). A contradiction must
    # degrade to unknown rather than emit a possibly-wrong verdict.
    never_with_witness = _parse_herd("Observation T Never 3 5\n")
    assert never_with_witness.outcome == "unknown" and never_with_witness.allowed is None
    sometimes_without_witness = _parse_herd("Observation T Sometimes 0 5\n")
    assert sometimes_without_witness.outcome == "unknown" and sometimes_without_witness.allowed is None


@requires_tools
def test_diycross_generates_cartesian_product() -> None:
    # Crossing a 2-alternative reader edge with a 2-alternative writer edge must
    # yield the full 2x2 product, plus diycross's base case.
    tests = diycross_generate("MP", [
        "Rfe",
        "PodRR,Fence.rw.rwdRR",
        "Fre",
        "PodWW,Fence.rw.rwdWW",
    ])
    names = {t.name for t in tests}
    assert len(tests) >= 4, f"expected a cartesian product, got {names}"
    assert any("fence.rw.rw" in n for n in names)
    assert all(t.text.startswith("RISCV ") for t in tests)


@requires_tools
def test_herd_judges_outcome_observable_vs_forbidden() -> None:
    tests = {t.name: t for t in diycross_generate("MP", [
        "Rfe",
        "PodRR,Fence.rw.rwdRR,DpAddrdR",
        "Fre",
        "PodWW,Fence.rw.rwdWW",
    ])}
    # Plain MP (no ordering on either leg) -> the weak outcome is observable.
    plain = herd_judge(tests["MP"].text)
    assert plain.outcome == "observable"
    assert plain.allowed is True
    # Both legs fenced -> forbidden.
    fenced = herd_judge(tests["MP+fence.rw.rw+addr"].text)
    assert fenced.outcome == "forbidden"
    assert fenced.allowed is False


@requires_tools
def test_herd_verdict_is_outcome_property_not_test_label() -> None:
    # A control dependency on the reader R->R edge does NOT order load->load
    # under RVWMO, so the outcome stays observable even with a writer fence.
    tests = {t.name: t for t in diycross_generate("MP", [
        "Rfe", "PodRR,DpCtrldR", "Fre", "Fence.rw.rwdWW",
    ])}
    ctrl = herd_judge(tests["MP+fence.rw.rw+ctrl"].text)
    assert ctrl.outcome == "observable"
