from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from litmus_link.native_cycles import vertex_directions
from litmus_link.native_edges import READ, WRITE
from litmus_link.native_scalar import native_template_cycles
from litmus_link.solver import solve_generated_case
from litmus_link.validator import validate_path
from litmus_link.vector_native import (
    EndpointChoice,
    VectorAssignment,
    VectorNativeDomain,
    generate_vector_cases,
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
    assert first_audit["sampling_mode"] == "balanced"
    assert first_audit["sampling"].startswith("balanced-skeleton-coverage")


def test_balanced_sampling_allocates_equal_skeleton_quotas() -> None:
    domain = VectorNativeDomain.from_payload(
        _payload(
            skeletons=["MP", "LB", "SB", "WRC", "RWC", "R", "S"],
            mechanisms=["po"],
            endpoint_modes=["P", "AMO"],
        )
    )
    assignments = domain.random_assignments(140, 7, "balanced")
    counts = {
        family: sum(assignment.cycle.family == family for assignment in assignments)
        for family in {assignment.cycle.family for assignment in assignments}
    }
    assert counts == {
        "MP": 20,
        "LB": 20,
        "SB": 20,
        "WRC": 20,
        "RWC": 20,
        "R": 20,
        "S": 20,
    }


def test_domain_weighted_sampling_preserves_family_cardinality() -> None:
    domain = VectorNativeDomain.from_payload(
        _payload(
            skeletons=["MP", "WRC"],
            mechanisms=["po"],
            endpoint_modes=["P", "AMO"],
        )
    )
    assignments = domain.random_assignments(1000, 7, "domain_weighted")
    counts = {
        family: sum(assignment.cycle.family == family for assignment in assignments)
        for family in {assignment.cycle.family for assignment in assignments}
    }
    assert counts["WRC"] > counts["MP"] * 2


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
    assert re.fullmatch(r"LLV-LB-[0-9a-f]{64}", case.name)
    assert "DpAddr" in case.display_name
    assert "+V{" in case.display_name
    assert "VLE32" in case.display_name or "VSE32" in case.display_name
    assert case.case_ir.cycle == " ".join(
        relation.label for relation in case.case_ir.relations
    )
    assert all("Vle" not in relation.label for relation in case.case_ir.relations)
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
    assert "X16" in case.display_name
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
    assert all(re.fullmatch(r"LLV-LB-[0-9a-f]{64}", item["case_id"]) for item in preview["sample"])
    assert all(item["file_name"] == item["case_id"] + ".litmus" for item in preview["sample"])
    assert all(item["name"].startswith("LB+{") and "+V{" in item["name"] for item in preview["sample"])


def test_file_hash_covers_configuration_hidden_from_display_name() -> None:
    cycles, _ = native_template_cycles(["MP"], ["po"])
    cycle = cycles[0]
    directions = vertex_directions(cycle.edges)
    choices = tuple(
        _vector_choice(direction)
        if index == 0
        else EndpointChoice("scalar:P", "scalar", direction, "P")
        for index, direction in enumerate(directions)
    )
    first = lower_vector_assignment(VectorAssignment(cycle, choices))
    changed = list(choices)
    vector = changed[0]
    changed[0] = replace(
        vector,
        choice_id=vector.choice_id + ":m2",
        params={**dict(vector.params or {}), "lmul": "m2"},
    )
    second = lower_vector_assignment(VectorAssignment(cycle, tuple(changed)))

    assert first.display_name == second.display_name
    assert first.name != second.name
    assert first.file_name != second.file_name
    assert re.fullmatch(r"LLV-MP-[0-9a-f]{64}\.litmus", first.file_name)


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
    assert sum(domain.audit()["family_cases"].values()) == domain.total_cases


def test_exhaustive_generation_writes_every_legal_assignment(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = _payload(
        skeletons=["Co"],
        mechanisms=["po"],
        endpoint_modes=["P"],
        forms=["unit_load", "unit_store"],
        generation_mode="all",
        generate_limit=1,
        compute_verdicts=False,
    )
    domain = VectorNativeDomain.from_payload(payload)
    out = tmp_path / "vector-all"
    progress = []
    report = generate_vector_cases(
        payload,
        out,
        progress_callback=lambda current, total, _message: progress.append(
            (current, total)
        ),
    )

    assert domain.total_cases == 7
    assert report["generation_mode"] == "all"
    assert report["generation_limit"] is None
    assert report["generation_limited"] is False
    assert report["generated_litmus"] == domain.total_cases
    files = list(out.glob("*.litmus"))
    entries = (out / "@all").read_text(encoding="utf-8").splitlines()
    assert len(files) == domain.total_cases
    assert len(entries) == len(set(entries)) == domain.total_cases
    assert all(re.fullmatch(r"LLV-CoRR-[0-9a-f]{64}\.litmus", entry) for entry in entries)
    first_meta = json.loads(files[0].with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert first_meta["file_name"] == files[0].name
    assert first_meta["display_name"].startswith("CoRR+{")
    assert first_meta["case_ir"]["metadata"]["file_identity"]["sha256"] in files[0].stem
    assert len(validate_path(out / "@all")) == domain.total_cases
    assert not (out / "@all.tmp").exists()
    assert progress[-1] == (domain.total_cases, domain.total_cases)


def test_generation_refuses_to_overwrite_a_different_canonical_identity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = _payload(
        skeletons=["Co"],
        mechanisms=["po"],
        endpoint_modes=["P"],
        forms=["unit_load", "unit_store"],
        generation_mode="balanced",
        generate_limit=1,
        compute_verdicts=False,
    )
    out = tmp_path / "collision"
    generate_vector_cases(payload, out)
    entry = (out / "@all").read_text(encoding="utf-8").strip()
    litmus_path = out / entry
    original_litmus = litmus_path.read_text(encoding="utf-8")
    meta_path = litmus_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["case_ir"]["metadata"]["file_identity"]["canonical"]["alignment"] = (
        "tampered"
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="different identity"):
        generate_vector_cases(payload, out)
    assert litmus_path.read_text(encoding="utf-8") == original_litmus
    assert not (out / "@all.tmp").exists()


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
