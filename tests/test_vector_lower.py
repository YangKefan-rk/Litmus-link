import pytest

from models import Combination, Decision, GENERATED
from litmus_ir import build_litmus_ir_cases
from vector_lower import lower_vector_case, lower_vector_to_litmus
from renderer import render_cases
from solver import solve_generated_case
from toolchain import tools_available

VECTOR_FORMS = [
    ("unit_load", "vector_load"), ("unit_store", "vector_store"),
    ("strided_load", "vector_load"), ("strided_store", "vector_store"),
    ("indexed_ordered_load", "vector_load"), ("indexed_unordered_load", "vector_load"),
    ("indexed_ordered_store", "vector_store"), ("indexed_unordered_store", "vector_store"),
    ("segment_load", "vector_load"), ("segment_store", "vector_store"),
    ("fof_load", "vector_load"), ("fof_segment_load", "vector_load"),
]


def _vector_cases(vector: str, memory_event: str):
    comb = Combination("t", "vector_mem", "MP", memory_event, "cacheable", vector=vector)
    dec = Decision(status=GENERATED, reason="", rvwmo_class="rvwmo-vector",
                   expected_kind="rvwmo-vector", requires=["RV64I", "V"])
    return comb, build_litmus_ir_cases(comb, dec)


def test_lowered_text_has_no_vector_mnemonics() -> None:
    # The whole point: the lowered twin is SCALAR, so stock herd7 can parse it.
    for vector, mev in VECTOR_FORMS:
        _comb, cases = _vector_cases(vector, mev)
        text = lower_vector_to_litmus(cases[0])
        for mnem in ("vsetvli", "vle32", "vse32", "vlse", "vsse", "vlox", "vsox",
                     "vlux", "vsux", "vlseg", "vsseg", "vmv", "vid.v", "ff.v"):
            assert mnem not in text, f"{vector}: lowered text still contains {mnem}"


def test_element_count_is_bounded_and_configurable() -> None:
    # Complexity guard: lowering emits exactly n element accesses (default small).
    _comb, cases = _vector_cases("unit_store", "vector_store")
    ir = cases[0]
    for n in (1, 2, 3):
        lowered = lower_vector_case(ir, n_elems=n)
        elems = [e for hart in lowered.harts for e in hart if e.role == "vector-elem"]
        # n-1 sibling elements (element 0 keeps its original data role)
        assert len(elems) == n - 1, f"n={n}: expected {n-1} sibling elems, got {len(elems)}"


def test_cross_hart_cycle_preserved() -> None:
    # Element 0 must keep the shared location so the MP cycle still exists.
    _comb, cases = _vector_cases("unit_load", "vector_load")
    lowered = lower_vector_case(cases[0], n_elems=2)
    locations = {e.location for hart in lowered.harts for e in hart}
    assert "x" in locations and "y" in locations, "shared MP locations must survive lowering"


@pytest.mark.skipif(not tools_available(), reason="herdtools7 not present")
def test_vector_gets_real_herd_verdict_matching_cycle() -> None:
    # The deliverable: vector cases judged by REAL herd7 on the lowered twin.
    # base (no ordering) = OBSERVABLE/allowed; any FENCE variant = FORBIDDEN.
    dec = Decision(status=GENERATED, reason="", rvwmo_class="rvwmo-vector",
                   expected_kind="rvwmo-vector", requires=["RV64I", "V"])
    for vector, mev in VECTOR_FORMS:
        comb = Combination("t", "vector_mem", "MP", mev, "cacheable", vector=vector)
        for case in render_cases(comb, dec):
            r = solve_generated_case(case)
            assert r.status == "verified", f"{vector}/{case.case_ir.variant}: {r.status}"
            assert r.cross_check == "agree", f"{vector}/{case.case_ir.variant}: herd disagreed ({r.cross_check})"
            if case.case_ir.variant == "base":
                assert r.verdict == "allowed", f"{vector}/base should be observable"
            elif "fence" in case.case_ir.variant:
                assert r.verdict == "forbidden", f"{vector}/{case.case_ir.variant} should be forbidden"
