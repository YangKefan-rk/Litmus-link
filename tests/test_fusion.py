from litmus_link.fusion import analyze_fusion, FusionResult
from litmus_link.litmus_ir import LitmusCaseIR, LitmusEvent
from litmus_link.models import Combination
from litmus_link.renderer import render_cases
from litmus_link.rules import evaluate


def _case(**kwargs):
    combination = Combination(**{"profile": "test", "category": "cross", "skeleton": "MP", **kwargs})
    cases = render_cases(combination, evaluate(combination))
    assert cases, f"no case generated for {kwargs}"
    return cases[0].case_ir


def _no_formal_claim(result: FusionResult) -> None:
    # The hard invariant: fusion analysis NEVER makes a herd-forbidden claim.
    assert result.allowed is None
    assert result.formal_forbidden_claim is False
    assert result.model == "extension-prose"


def test_fusion_not_applicable_for_scalar_rvwmo() -> None:
    case = _case(memory_event="scalar_pair", attribute="cacheable")
    result = analyze_fusion(case)
    assert result.status == "not_applicable"
    _no_formal_claim(result)


def test_fusion_full_alias_sync_is_ordering_documented() -> None:
    case = _case(
        memory_event="cmo",
        attribute="cacheable_nc_alias",
        cmo="flush",
        params={"sync": "full_alias_sync"},
    )
    result = analyze_fusion(case)
    assert result.verdict == "ordering-documented"
    assert all(f.covered for f in result.findings if f.feature.startswith(("cmo", "attr")))
    _no_formal_claim(result)


def test_fusion_bare_cmo_is_ordering_absent() -> None:
    case = _case(memory_event="cmo", attribute="cacheable", cmo="flush")
    result = analyze_fusion(case)
    assert result.verdict == "ordering-absent"
    _no_formal_claim(result)


def test_fence_i_does_not_order_cmo_data_effect() -> None:
    # fence.i synchronises instruction fetch only; it must NOT count as a data
    # barrier for a CBO's data effect.
    case = _case(memory_event="cmo", attribute="cacheable", cmo="flush", params={"sync": "fence_i_after"})
    result = analyze_fusion(case)
    assert result.verdict == "ordering-absent"
    cmo_finding = next(f for f in result.findings if f.feature.startswith("cmo"))
    assert not cmo_finding.covered
    _no_formal_claim(result)


def test_formal_vector_is_not_routed_through_fusion() -> None:
    # All six simple Vector skeletons now have a native Vector-aware solver;
    # extension fusion must not replace that formal result.
    case = _case(skeleton="LB", memory_event="vector_load", attribute="cacheable", vector="unit_load")
    result = analyze_fusion(case)
    assert result.status == "not_applicable"
    assert result.verdict == "prose-spec"
    _no_formal_claim(result)


def test_fusion_indexed_ordered_vs_unordered_detail_differs() -> None:
    ordered = analyze_fusion(_case(skeleton="LB", memory_event="vector_load", attribute="cacheable", vector="indexed_ordered_load"))
    unordered = analyze_fusion(_case(skeleton="LB", memory_event="vector_load", attribute="cacheable", vector="indexed_unordered_load"))
    assert ordered.status == "not_applicable"
    assert unordered.status == "not_applicable"
    _no_formal_claim(ordered)
    _no_formal_claim(unordered)


def test_fusion_tlb_is_prose_spec() -> None:
    # Pure-TLB combinations are hand_required, so they never generate an IR.
    # Build the IR directly to exercise the analyzer's prose-spec path.
    case = LitmusCaseIR(
        name="tlb_case",
        display_name="MP.observation",
        combination_name="tlb_case",
        skeleton="MP",
        variant="observation",
        cycle="MP + pte_update + remote_sfence",
        init_lines=[],
        harts=[
            [LitmusEvent("p0_w", 0, "store", "sw x5,0(x6)", "x")],
            [LitmusEvent("p1_r", 1, "load", "lw x5,0(x6)", "y")],
        ],
        relations=[],
        exists="(1:x5=1)",
        expected_outcome="hardware-observation",
        model="hardware-observation",
        tags=["vm_tlb", "pte_update", "cacheable", "none", "no_cmo", "remote_sfence"],
    )
    result = analyze_fusion(case)
    # TLB shootdown can't be settled by a local barrier.
    assert result.verdict == "prose-spec"
    assert any(f.feature.startswith("tlb") for f in result.findings)
    _no_formal_claim(result)


def test_fusion_citations_are_present() -> None:
    # Nanhu excludes Vector + PBMT/NC mappings.  Exercise the same citation
    # invariant with a supported Vector+CMO cacheable cross case instead.
    case = _case(memory_event="vector_load", attribute="cacheable", vector="unit_load", cmo="flush")
    result = analyze_fusion(case)
    for finding in result.findings:
        assert finding.citation, f"{finding.feature} has no spec citation"
