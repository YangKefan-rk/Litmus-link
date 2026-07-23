from __future__ import annotations

from dataclasses import replace

import pytest

from litmus_link.litmus_ir import LitmusCaseIR
from litmus_link.models import Combination, EXCLUDED_UNSUPPORTED, GENERATED
from litmus_link.profiles import FORMAL_VECTOR_SKELETONS, VECTOR_ENDPOINTS
from litmus_link.renderer import render_cases
from litmus_link.rules import evaluate
from litmus_link.solver import solve_generated_case
from litmus_link.vector_native import VectorNativeDomain, lower_vector_assignment
from litmus_link.vector_solver import SUPPORTED_VECTOR_FORMS, expand_vector_case, solve_vector_case


FORM_EVENTS = {
    "unit_load": "vector_load",
    "unit_store": "vector_store",
    "strided_load": "vector_load",
    "strided_store": "vector_store",
    "indexed_unordered_load": "vector_load",
    "indexed_unordered_store": "vector_store",
    "indexed_ordered_load": "vector_load",
    "indexed_ordered_store": "vector_store",
}


def _case(
    form: str,
    *,
    variant: str = "base",
    sew: str = "e32",
    lmul: str = "m1",
    vl: str = "vl4",
    mask: str = "unmasked",
    index_eew: str | None = None,
):
    combination = Combination(
        "test",
        "vector_mem",
        "MP",
        FORM_EVENTS[form],
        "cacheable",
        vector=form,
        params={
            "variant": variant,
            "sew": sew,
            "lmul": lmul,
            "vl": vl,
            "mask": mask,
            "tail": "ta_ma",
            "footprint": "same_line",
            **({"index_eew": index_eew} if index_eew is not None else {}),
        },
    )
    decision = evaluate(combination)
    assert decision.status == GENERATED
    return render_cases(combination, decision)[0]


def _instruction(result):
    assert result.vector is not None
    return result.vector["vector_ir"]["instructions"][0]


@pytest.mark.parametrize("form", sorted(SUPPORTED_VECTOR_FORMS))
def test_supported_forms_use_vector_aware_solver(form: str) -> None:
    base = solve_generated_case(_case(form, variant="base"))
    fenced = solve_generated_case(_case(form, variant="fence_rw_rw"))
    split_fence = solve_generated_case(_case(form, variant="fence_w_w_r_rw"))

    assert base.status == "verified"
    assert base.verdict == "allowed"
    assert base.allowed is True
    assert fenced.verdict == "forbidden"
    assert fenced.allowed is False
    assert split_fence.verdict == "forbidden"
    assert split_fence.allowed is False
    assert base.tool == "litmus-link-vector-rvwmo"
    assert base.model == "riscv.cat+rvv-elements"
    assert base.cross_check == "not_run"
    assert base.vector["schema"] == "litmus-link.vector-solver.v1"


def test_vector_renderer_covers_every_formal_skeleton_endpoint() -> None:
    for skeleton in FORMAL_VECTOR_SKELETONS:
        for endpoint_kind, endpoints in VECTOR_ENDPOINTS[skeleton].items():
            form = "unit_store" if endpoint_kind == "store" else "unit_load"
            combination = Combination(
                "endpoint-test",
                "vector_mem",
                skeleton,
                f"vector_{endpoint_kind}",
                "cacheable",
                vector=form,
                params={
                    "sew": "e32",
                    "lmul": "m1",
                    "mask": "unmasked",
                    "tail": "ta_ma",
                    "footprint": "same_line",
                    "vl": "vl4",
                    "vector_event": endpoints[0],
                },
            )
            decision = evaluate(combination)
            assert decision.status == GENERATED, (skeleton, endpoint_kind, endpoints[0], decision.reason)
            case = render_cases(combination, decision)[0]
            result = solve_generated_case(case)
            assert result.status == "verified", (skeleton, endpoint_kind, endpoints[0], result.reason)
            assert result.tool == "litmus-link-vector-rvwmo"


@pytest.mark.parametrize(
    ("form", "offsets"),
    [
        ("unit_load", [0, 4, 8, 12]),
        ("strided_load", [0, 8, 16, 24]),
        ("indexed_unordered_load", [0, 4, 8, 12]),
        ("indexed_ordered_load", [0, 4, 8, 12]),
    ],
)
def test_vector_address_functions_use_real_byte_offsets(form: str, offsets: list[int]) -> None:
    result = solve_generated_case(_case(form))
    elements = _instruction(result)["elements"]
    assert [element["offset_bytes"] for element in elements] == offsets
    assert [element["size_bytes"] for element in elements] == [4] * len(offsets)
    assert [element["location"] for element in elements] == [
        "x" if offset == 0 else f"x[{offset}]" for offset in offsets
    ]
    embedded = result.vector["embedded"]
    active = {
        event["event_id"]: event
        for event in embedded["events"]
        if event["event_id"].startswith("p1_rx.e")
    }
    assert [active[f"p1_rx.e{index}"]["byte_offset"] for index in range(4)] == offsets
    assert all(event["access_size"] == 4 for event in active.values())
    assert all(event["atomicity_model"] == "aligned_atomic" for event in active.values())


@pytest.mark.parametrize(
    ("sew", "load_mnemonic", "store_mnemonic"),
    [("e8", "lb", "sb"), ("e16", "lh", "sh"), ("e32", "lw", "sw"), ("e64", "ld", "sd")],
)
def test_scalar_cycle_endpoint_matches_vector_element_width(
    sew: str,
    load_mnemonic: str,
    store_mnemonic: str,
) -> None:
    vector_load = _case("unit_load", sew=sew)
    vector_store = _case("unit_store", sew=sew)
    load_case_instructions = {event.event_id: event.instruction for event in vector_load.case_ir.events()}
    store_case_instructions = {event.event_id: event.instruction for event in vector_store.case_ir.events()}
    assert load_case_instructions["p0_wx"].startswith(f"{store_mnemonic} ")
    assert store_case_instructions["p1_rx"].startswith(f"{load_mnemonic} ")


def test_ordered_indexed_adds_element_ppo_but_unordered_does_not() -> None:
    ordered = solve_vector_case(_case("indexed_ordered_load").case_ir)
    unordered = solve_vector_case(_case("indexed_unordered_load").case_ir)

    assert ordered.expansion is not None
    assert unordered.expansion is not None
    assert len(ordered.expansion.ordering.preserved_order) == 6
    assert unordered.expansion.ordering.preserved_order == frozenset()
    assert ordered.embedded is not None and ordered.embedded.execution is not None
    assert unordered.embedded is not None and unordered.embedded.execution is not None
    assert len(ordered.embedded.execution.ppo_rules["vector-element-order"]) == 6
    assert unordered.embedded.execution.ppo_rules["vector-element-order"] == set()
    assert ("p1_rx.e0", "p1_rx.e1") not in unordered.embedded.execution.po


def test_parent_dependency_lifts_to_all_active_vector_elements() -> None:
    domain = VectorNativeDomain.from_payload(
        {
            "skeletons": ["LB"],
            "mechanisms": ["po", "dependency"],
            "endpoint_categories": ["vector"],
            "endpoint_compositions": ["vector_only"],
            "overlap_layouts": ["same_start"],
            "forms": ["unit_load", "unit_store"],
            "sew": ["e32"],
            "lmul": ["m1"],
            "index_eew": ["ei16"],
            "mask": ["unmasked"],
            "tail": ["ta_ma"],
            "vl": ["vl2"],
            "alignments": ["aligned"],
        }
    )
    assignment = next(
        assignment
        for assignment in domain.assignments()
        if any(label.startswith("Dp") for label in assignment.cycle.labels)
    )
    case = lower_vector_assignment(assignment).case_ir
    result = solve_vector_case(case)
    assert result.embedded is not None and result.embedded.execution is not None
    dependency = next(
        relation
        for relation in case.relations
        if relation.kind == "dependency"
    )
    expected = {
        (f"{dependency.src}.e{source}", f"{dependency.dst}.e{target}")
        for source in range(2)
        for target in range(2)
    }
    actual = set().union(
        result.embedded.execution.ppo_rules["r9"],
        result.embedded.execution.ppo_rules["r10"],
        result.embedded.execution.ppo_rules["r11"],
    )
    assert expected <= actual


def test_mask_removes_inactive_elements_from_memory_graph() -> None:
    case = _case("unit_load", mask="masked")
    assert "vmseq.vi v0,v24,0" in case.litmus
    assert "vle32.v v8,(x8),v0.t" in case.litmus
    result = solve_generated_case(case)
    instruction = _instruction(result)
    assert [element["index"] for element in instruction["elements"] if element["active"]] == [0, 2]
    assert [element["index"] for element in instruction["elements"] if not element["active"]] == [1, 3]
    embedded_ids = {
        event["event_id"]
        for event in result.vector["embedded"]["events"]
        if event["hart"] is not None
    }
    assert "p1_rx.e0" in embedded_ids and "p1_rx.e2" in embedded_ids
    assert "p1_rx.e1" not in embedded_ids and "p1_rx.e3" not in embedded_ids


def test_interactive_memory_event_limit_is_inconclusive_not_forbidden() -> None:
    case = _case("unit_load", sew="e8", lmul="m4", vl="vlmax")
    result = solve_vector_case(case.case_ir, max_memory_events=8)
    assert result.status == "inconclusive"
    assert result.verdict == "unknown"
    assert result.allowed is None
    assert result.embedded is None
    assert "transaction" in result.reason
    assert "limit is 8" in result.reason


@pytest.mark.parametrize(
    ("sew", "lmul", "vl", "vlmax", "effective"),
    [
        ("e32", "m1", "vl1", 4, 1),
        ("e32", "m1", "vl2", 4, 2),
        ("e32", "m1", "vl4", 4, 4),
        ("e32", "m1", "vlmax", 4, 4),
        ("e8", "m4", "vlmax", 64, 64),
        ("e64", "mf2", "vl4", 1, 1),
    ],
)
def test_vl_is_derived_from_nanhu_vlen_sew_lmul(
    sew: str,
    lmul: str,
    vl: str,
    vlmax: int,
    effective: int,
) -> None:
    result = solve_generated_case(_case("unit_load", sew=sew, lmul=lmul, vl=vl))
    config = result.vector["vector_ir"]["config"]
    assert config["vlen_bits"] == 128
    assert config["vlmax"] == vlmax
    assert config["effective_vl"] == effective
    assert _instruction(result)["active_element_count"] == effective


def test_element_ir_distinguishes_vl_tail_from_mask_inactivity() -> None:
    result = solve_generated_case(_case("unit_load", vl="vl1", mask="masked"))
    elements = _instruction(result)["elements"]
    assert len(elements) == 4
    assert elements[0]["within_vl"] is True
    assert elements[0]["mask_enabled"] is True
    assert elements[0]["active"] is True
    assert all(element["within_vl"] is False for element in elements[1:])
    assert all(element["active"] is False for element in elements[1:])


def test_vector_metadata_round_trips_and_drives_offline_solver() -> None:
    case = _case("strided_store", sew="e16", vl="vl2", mask="masked")
    restored = LitmusCaseIR.from_json(case.case_ir.to_json())
    assert restored.to_json() == case.case_ir.to_json()
    result = solve_vector_case(restored)
    assert result.status == "verified"
    assert result.expansion is not None
    assert result.expansion.config.stride_bytes == 4
    assert result.expansion.config.tail_policy == "ta_ma"
    assert result.expansion.config.footprint == "same_line"


@pytest.mark.parametrize("index_eew", ["ei8", "ei16", "ei32", "ei64"])
def test_indexed_vector_data_sew_and_index_eew_are_independent(index_eew: str) -> None:
    case = _case("indexed_ordered_load", sew="e64", vl="vl2", index_eew=index_eew)
    result = solve_generated_case(case)
    assert result.status == "verified"
    assert result.vector is not None
    config = result.vector["vector_ir"]["config"]
    assert config["sew_bits"] == 64
    assert config["index_eew"] == index_eew
    assert f"vlox{index_eew}.v" in case.litmus


def test_register_avl_values_are_emitted_as_legal_vsetvli() -> None:
    case = _case("unit_load", sew="e8", lmul="m8", vl="vl64")
    assert "vsetvli x10,x11,e8,m8" in case.litmus
    assert "x11=64" in " ".join(case.case_ir.init_lines)
    result = solve_generated_case(case)
    assert result.status == "verified"
    assert result.vector is not None
    assert result.vector["vector_ir"]["config"]["effective_vl"] == 64


@pytest.mark.parametrize("form", ["segment_load", "segment_store", "fof_load", "fof_segment_load"])
def test_deferred_vector_forms_do_not_enter_formal_solver(form: str) -> None:
    memory_event = "vector_store" if form.endswith("store") else "vector_load"
    decision = evaluate(
        Combination("test", "vector_mem", "MP", memory_event, "cacheable", vector=form)
    )
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert decision.metadata["formal_forbidden_claim"] == "false"
    assert "rule:vector_solver_scope" in decision.notes


def test_vstart_restart_is_rejected_by_vector_solver() -> None:
    case = _case("unit_load").case_ir
    metadata = dict(case.metadata)
    metadata["vector"] = {**metadata["vector"], "vstart": 1}
    result = solve_vector_case(replace(case, metadata=metadata))
    assert result.status == "not_applicable"
    assert "vstart/restart" in result.reason


def test_mixed_size_scalar_vector_endpoint_is_formally_solved() -> None:
    case = _case("unit_load", sew="e64").case_ir
    harts = []
    for hart in case.harts:
        harts.append(
            [
                replace(event, instruction="sw x5,0(x6)")
                if event.event_id == "p0_wx"
                else event
                for event in hart
            ]
        )
    result = solve_vector_case(replace(case, harts=harts))
    assert result.status == "verified"
    assert result.embedded is not None
    assert result.embedded.execution is not None
    scalar = next(event for event in result.embedded.events if event.event_id == "p0_wx")
    vector = next(event for event in result.embedded.events if event.event_id == "p1_rx.e0")
    assert scalar.access_size == 4
    assert vector.access_size == 8
    assert scalar.byte_locations < vector.byte_locations
