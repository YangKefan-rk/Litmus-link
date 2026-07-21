from __future__ import annotations

from litmus_link.native_cycles import vertex_directions
from litmus_link.native_edges import READ, WRITE
from litmus_link.native_scalar import native_template_cycles
from litmus_link.solver import solve_generated_case
from litmus_link.vector_native import (
    EndpointChoice,
    VectorAssignment,
    VectorNativeDomain,
    lower_vector_assignment,
    sample_vector_cases,
)
from litmus_link.workflow import preview_payload


def _payload(**overrides):
    payload = {
        "mode": "vector",
        "skeletons": ["LB"],
        "mechanisms": ["po", "dependency"],
        "endpoint_modes": ["P", "AMO", "Aq", "Rl", "AR"],
        "forms": ["unit_load", "unit_store"],
        "sew": ["e32"],
        "lmul": ["m1"],
        "index_eew": ["ei16"],
        "mask": ["unmasked"],
        "tail": ["ta_ma"],
        "vl": ["vl1"],
        "alignments": ["aligned"],
        "sample_limit": 20,
        "random_seed": 11,
    }
    payload.update(overrides)
    return payload


def _vector_choice(direction: str, alignment: str = "aligned") -> EndpointChoice:
    form = "unit_load" if direction == READ else "unit_store"
    params = {
        "sew": "e32",
        "lmul": "m1",
        "mask": "unmasked",
        "tail": "ta_ma",
        "vl": "vl1",
        "footprint": "same_line",
        "alignment": alignment,
    }
    return EndpointChoice(
        f"vector:{form}:{alignment}", "vector", direction, "P", form, params
    )


def test_random_vector_preview_is_reproducible_and_not_prefix_ordered() -> None:
    first, first_audit = sample_vector_cases(_payload(), compute_verdicts=False)
    repeated, repeated_audit = sample_vector_cases(_payload(), compute_verdicts=False)
    different, _ = sample_vector_cases(
        _payload(random_seed=12), compute_verdicts=False
    )
    assert [case.name for case in first] == [case.name for case in repeated]
    assert [case.name for case in first] != [case.name for case in different]
    assert first_audit["sample_seed"] == repeated_audit["sample_seed"] == 11
    assert first_audit["total_cases"] > len(first)
    assert first_audit["sampling"].startswith("coverage-stratified-random")


def test_relation_cycle_can_contain_multiple_vectors_and_an_amo() -> None:
    cycles, _ = native_template_cycles(["LB"], ["dependency"])
    cycle = next(cycle for cycle in cycles if any("DpAddr" in label for label in cycle.labels))
    directions = vertex_directions(cycle.edges)
    choices = []
    amo_added = False
    vector_count = 0
    for index, direction in enumerate(directions):
        if index == 1:
            choices.append(EndpointChoice("scalar:AMO", "amo", direction, "AMO"))
            amo_added = True
        else:
            choices.append(_vector_choice(direction))
            vector_count += 1
    case = lower_vector_assignment(VectorAssignment(cycle, tuple(choices)))
    instructions = [event.instruction for event in case.case_ir.events()]
    assert amo_added and vector_count >= 2
    assert sum(instruction.startswith(("vle32.v", "vse32.v")) for instruction in instructions) == vector_count
    assert any(instruction.startswith(("amoor.w", "amoswap.w")) for instruction in instructions)
    assert "DpAddr" in case.name
    assert "Vle32" in case.name or "Vse32" in case.name
    assert case.case_ir.cycle == " ".join(relation.label for relation in case.case_ir.relations)
    result = solve_generated_case(case)
    assert result.status == "verified"
    assert result.verdict in {"allowed", "forbidden"}


def test_misaligned_vector_is_marked_and_never_gets_a_false_formal_verdict() -> None:
    cycles, _ = native_template_cycles(["LB"], ["po"])
    cycle = cycles[0]
    directions = vertex_directions(cycle.edges)
    choices = tuple(
        _vector_choice(direction, "misalign_cross16")
        if index == 0
        else EndpointChoice("scalar:P", "scalar", direction, "P")
        for index, direction in enumerate(directions)
    )
    case = lower_vector_assignment(
        VectorAssignment(cycle, choices, "misalign_cross16")
    )
    assert "X16" in case.name
    assert any(
        event.role == "vector-misaligned-base" and event.instruction.startswith("addi ")
        for event in case.case_ir.events()
    )
    assert case.decision.expected_kind == "hardware-observation"
    result = solve_generated_case(case)
    assert result.status == "not_applicable"
    assert result.allowed is None


def test_workflow_preview_limit_counts_final_random_cases() -> None:
    preview = preview_payload(_payload(sample_limit=17))
    assert preview["source"] == "litmus-link-native-cycle+rvv"
    assert len(preview["sample"]) == 17
    assert preview["report"]["displayed_litmus"] == 17
    assert preview["report"]["total_combinations"] > 17
    assert all(item["case_ir"]["variant"] == "vector-native-cycle" for item in preview["sample"])
    assert any(len(item["case_ir"]["metadata"]["vectors"]) > 1 for item in preview["sample"])


def test_domain_includes_all_selected_endpoint_categories() -> None:
    domain = VectorNativeDomain.from_payload(_payload())
    assert {choice.annotation for choice in domain.read_choices if not choice.is_vector} == {
        "P", "AMO", "Aq", "Rl", "AR"
    }
    assert {choice.annotation for choice in domain.write_choices if not choice.is_vector} == {
        "P", "AMO", "Aq", "Rl", "AR"
    }
    assert any(choice.is_vector for choice in domain.read_choices)
    assert any(choice.is_vector for choice in domain.write_choices)


def test_domain_count_matches_complete_small_enumeration() -> None:
    domain = VectorNativeDomain.from_payload(
        _payload(alignments=["aligned", "misalign_same16", "misalign_cross16", "misalign_cross64"])
    )
    assert domain.total_cases == sum(1 for _ in domain.assignments())


def test_random_misaligned_assignments_exclude_amo_and_e8() -> None:
    domain = VectorNativeDomain.from_payload(
        _payload(
            forms=["unit_load", "unit_store"],
            sew=["e8", "e32"],
            alignments=["aligned", "misalign_same16", "misalign_cross16", "misalign_cross64"],
        )
    )
    assignments = [
        assignment
        for seed in range(4)
        for assignment in domain.random_assignments(100, seed)
    ]
    misaligned = [assignment for assignment in assignments if assignment.alignment != "aligned"]
    assert misaligned
    assert all(not any(choice.category == "amo" for choice in assignment.choices) for assignment in misaligned)
    assert all(
        str((choice.params or {}).get("sew")) != "e8"
        for assignment in misaligned
        for choice in assignment.choices
        if choice.is_vector
    )
