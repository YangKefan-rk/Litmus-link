from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
import litmus_link.vector_native as vector_native

from litmus_link.amo import AMO_OPERATIONS, AMO_ORDERINGS
from litmus_link.native_cycles import vertex_directions
from litmus_link.native_edges import READ
from litmus_link.native_scalar import native_template_cycles
from litmus_link.profiles import vector_effective_vl
from litmus_link.solver import solve_generated_case
from litmus_link.vector_solver import expand_vector_case, solve_vector_case
from litmus_link.validator import validate_path
from litmus_link.vector_native import (
    EndpointChoice,
    VectorAssignment,
    VectorNativeDomain,
    _amo_mask_satisfiable,
    _compact_preview_solver,
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
        "endpoint_categories": ["scalar", "amo", "vector"],
        "endpoint_compositions": [
            "vector_only",
            "vector_scalar",
            "vector_amo",
            "vector_scalar_amo",
        ],
        "scalar_widths": ["b", "h", "w", "d"],
        "amo_ops": list(AMO_OPERATIONS),
        "amo_widths": ["w", "d"],
        "amo_orderings": list(AMO_ORDERINGS),
        "overlap_layouts": ["same_start"],
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


def _small_payload(**overrides):
    selected = {
        "endpoint_categories": ["scalar", "amo", "vector"],
        "endpoint_compositions": ["vector_scalar", "vector_amo", "vector_scalar_amo"],
        "scalar_widths": ["w"],
        "amo_ops": ["add"],
        "amo_widths": ["d"],
        "amo_orderings": ["aq"],
        "sew": ["e16"],
        "vl": ["vl2"],
        "mechanisms": ["po"],
    }
    selected.update(overrides)
    return _payload(**selected)


def _vector_choice(direction: str, sew: str = "e32") -> EndpointChoice:
    form = "unit_load" if direction == READ else "unit_store"
    return EndpointChoice(
        f"vector:{form}:{sew}",
        "vector",
        direction,
        "P",
        form,
        {
            "sew": sew,
            "lmul": "m1",
            "mask": "unmasked",
            "tail": "ta_ma",
            "vl": "vl2",
            "footprint": "same_line",
        },
    )


def _scalar_choice(direction: str, width: str = "w") -> EndpointChoice:
    sizes = {"b": 1, "h": 2, "w": 4, "d": 8}
    return EndpointChoice(
        f"scalar:{width}:{direction}",
        "scalar",
        direction,
        "P",
        params={"width": width, "width_bytes": str(sizes[width])},
    )


def _amo_choice(
    direction: str,
    operation: str = "add",
    width: str = "d",
    ordering: str = "aqrl",
) -> EndpointChoice:
    return EndpointChoice(
        f"amo:{operation}:{width}:{ordering}:{direction}",
        "amo",
        direction,
        "AR",
        params={
            "amo_op": operation,
            "amo_width": width,
            "amo_width_bytes": "4" if width == "w" else "8",
            "amo_ordering": ordering,
        },
    )


def test_random_vector_preview_is_reproducible_and_not_prefix_ordered() -> None:
    first, first_audit = sample_vector_cases(_small_payload(), compute_verdicts=False)
    repeated, repeated_audit = sample_vector_cases(_small_payload(), compute_verdicts=False)
    different, _ = sample_vector_cases(
        _small_payload(random_seed=12), compute_verdicts=False
    )
    assert [case.name for case in first] == [case.name for case in repeated]
    assert [case.name for case in first] != [case.name for case in different]
    assert first_audit["sample_seed"] == repeated_audit["sample_seed"] == 11
    assert first_audit["total_cases"] > len(first)
    assert first_audit["sampling_mode"] == "balanced"


def test_deferred_cardinality_preview_publishes_pending_rows(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(vector_native, "_MAX_EXACT_CARDINALITY_WORK", 0)
    prepared = []
    result = preview_payload(
        _small_payload(
            sample_limit=6,
            compute_verdicts=False,
            solver_backend="embedded",
        ),
        prepared_callback=prepared.append,
    )

    assert len(prepared) == 1
    pending = prepared[0]
    assert pending["verification_pending"] is True
    assert pending["cardinality_exact"] is False
    assert pending["available_litmus"] is None
    assert pending["case_upper_bound"] is not None
    assert len(pending["sample"]) == 6
    assert all(item["provisional"] is True for item in pending["sample"])
    assert all(item["solver"]["status"] == "pending" for item in pending["sample"])
    assert all(item["file_name"].endswith(".litmus") for item in pending["sample"])

    assert result["cardinality_exact"] is False
    assert result["available_litmus"] is None
    assert len(result["sample"]) == 6
    assert all("provisional" not in item for item in result["sample"])
    assert all(item["solver"]["status"] == "unchecked" for item in result["sample"])


def test_balanced_sampling_allocates_equal_skeleton_quotas() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            skeletons=["MP", "LB", "SB", "WRC", "RWC", "R", "S"]
        )
    )
    assignments = domain.random_assignments(140, 7, "balanced")
    counts = Counter(assignment.cycle.family for assignment in assignments)
    assert counts == {family: 20 for family in ["MP", "LB", "SB", "WRC", "RWC", "R", "S"]}


def test_domain_weighted_sampling_preserves_family_cardinality() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(skeletons=["MP", "WRC"])
    )
    assignments = domain.random_assignments(1000, 7, "domain_weighted")
    counts = Counter(assignment.cycle.family for assignment in assignments)
    assert counts["WRC"] > counts["MP"] * 2


def test_relation_cycle_contains_multiple_vectors_and_real_amo_metadata() -> None:
    cycles, _ = native_template_cycles(["LB"], ["dependency"])
    cycle = next(cycle for cycle in cycles if any("DpAddr" in label for label in cycle.labels))
    directions = vertex_directions(cycle.edges)
    choices = tuple(
        _amo_choice(direction, "xor", "d", "aqrl")
        if index == 1
        else _vector_choice(direction)
        for index, direction in enumerate(directions)
    )
    case = lower_vector_assignment(VectorAssignment(cycle, choices))
    instructions = [event.instruction for event in case.case_ir.events()]
    assert sum(instruction.startswith(("vle32.v", "vse32.v")) for instruction in instructions) >= 2
    assert any(instruction.startswith("amoxor.d.aqrl") for instruction in instructions)
    amo = next(event for event in case.case_ir.events() if event.kind == "amo")
    assert amo.amo_op == "xor"
    assert amo.amo_width_bytes == 8
    assert amo.amo_ordering == "aqrl"
    assert "+E{" in case.display_name
    assert "AXOR64.AQRL" in case.display_name
    assert solve_generated_case(case).status == "verified"


@pytest.mark.parametrize("composition", ["vector_scalar", "vector_amo", "vector_scalar_amo"])
def test_all_fusion_compositions_receive_formal_verdict(composition: str) -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(endpoint_compositions=[composition])
    )
    case = lower_vector_assignment(domain.random_assignments(1, 19)[0])
    assert case.case_ir.metadata["memory_layout"] == {
        **case.case_ir.metadata["memory_layout"],
        "alignment": "aligned",
        "pbmt": 0,
        "attribute": "cacheable",
        "pma_atomic": True,
    }
    verdict = solve_generated_case(case)
    assert verdict.status == "verified"
    assert verdict.verdict in {"allowed", "forbidden"}


def test_vector_crosscheck_backend_runs_external_projection() -> None:
    cases, audit = sample_vector_cases(
        _small_payload(
            scalar_widths=["w"],
            sew=["e32"],
            vl=["vl1"],
            solver_backend="crosscheck",
            sample_limit=1,
        ),
        compute_verdicts=True,
    )
    assert audit["solver_backend"] == "crosscheck"
    assert cases[0].solver["cross_check"] in {
        "agree",
        "conflict",
        "external_unsupported",
        "advisory_agree",
        "advisory_disagree",
    }


def test_vector_backend_only_requests_herd_projection_for_crosscheck(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    external_flags = []

    def fake_solver(  # type: ignore[no-untyped-def]
        _case,
        *,
        vector_external_check=False,
        vector_solver_limits=None,
    ):
        external_flags.append(vector_external_check)
        return SimpleNamespace(
            to_json=lambda: {
                "status": "verified",
                "verdict": "allowed",
                "allowed": True,
                "cross_check": "not_run",
            }
        )

    monkeypatch.setattr(vector_native, "solve_generated_case", fake_solver)
    sample_vector_cases(
        _small_payload(sample_limit=1, solver_backend="embedded"),
        compute_verdicts=True,
    )
    sample_vector_cases(
        _small_payload(sample_limit=1, solver_backend="crosscheck"),
        compute_verdicts=True,
    )
    assert external_flags == [False, True]


def test_interactive_crosscheck_is_bounded_and_progress_is_classified(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    external_flags = []
    received_limits = []

    def fake_solver(  # type: ignore[no-untyped-def]
        _case,
        *,
        vector_external_check=False,
        vector_solver_limits=None,
    ):
        external_flags.append(vector_external_check)
        received_limits.append(dict(vector_solver_limits or {}))
        cross_check = "agree" if vector_external_check else "not_run"
        external = (
            {"status": "agree", "allowed": True, "verdict": "observable"}
            if vector_external_check
            else None
        )
        return SimpleNamespace(
            to_json=lambda: {
                "status": "verified",
                "verdict": "allowed",
                "allowed": True,
                "cross_check": cross_check,
                "vector": {"external": external},
            }
        )

    monkeypatch.setattr(vector_native, "solve_generated_case", fake_solver)
    progress = []
    cases, audit = sample_vector_cases(
        _small_payload(
            sample_limit=6,
            solver_backend="crosscheck",
            verification_effort="interactive",
        ),
        compute_verdicts=True,
        progress_callback=lambda current, total, message: progress.append(
            (current, total, message)
        ),
    )

    assert len(cases) == 6
    assert external_flags == [True, True, True, True, False, False]
    assert all(limits["max_candidates"] == 100_000 for limits in received_limits)
    assert all(limits["max_memory_events"] == 1_024 for limits in received_limits)
    assert all(limits["timeout_seconds"] == 10.0 for limits in received_limits)
    assert all(limits["max_search_steps"] == 2_000_000 for limits in received_limits)
    assert audit["verification_effort"] == "interactive"
    assert audit["external_status"] == {"agree": 4, "batch_limit_skipped": 2}
    assert cases[-1].solver["cross_check"] == "batch_limit_skipped"
    assert "verified=6" in progress[-1][2]
    assert "herd=4" in progress[-1][2]
    assert "herd-skipped=2" in progress[-1][2]


def test_embedded_preview_uses_parallel_solver_workers(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LITMUS_LINK_SOLVER_WORKERS", "2")
    monkeypatch.setattr(vector_native, "_PARALLEL_SOLVER_THRESHOLD", 4)
    cases, audit = sample_vector_cases(
        _small_payload(
            sample_limit=4,
            solver_backend="embedded",
            verification_effort="interactive",
        ),
        compute_verdicts=True,
    )
    assert len(cases) == 4
    assert audit["solver_workers"] == 2
    assert sum(audit["solver_status"].values()) == 4
    assert all(case.solver["status"] in {"verified", "inconclusive"} for case in cases)


def test_preview_solver_payload_omits_duplicate_vector_expansion() -> None:
    full_cases, _audit = sample_vector_cases(
        _small_payload(sample_limit=1, solver_backend="embedded"),
        compute_verdicts=True,
    )
    full = dict(full_cases[0].solver or {})
    compact = _compact_preview_solver(full)

    assert compact["status"] == full["status"]
    assert compact["verdict"] == full["verdict"]
    assert compact["allowed"] == full["allowed"]
    assert compact["vector"]["preview_compact"] is True
    assert "vector_ir" not in compact["vector"]
    assert "events" not in compact["vector"]["embedded"]
    assert compact["vector"]["embedded"]["execution"] == full["vector"]["embedded"]["execution"]
    assert compact["raw_output"] == ""
    assert compact["edges"] == []


def test_sample_preview_requests_compact_parallel_results(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LITMUS_LINK_SOLVER_WORKERS", "2")
    monkeypatch.setattr(vector_native, "_PARALLEL_SOLVER_THRESHOLD", 2)
    cases, audit = sample_vector_cases(
        _small_payload(sample_limit=2, solver_backend="embedded"),
        compute_verdicts=True,
        compact_solver_results=True,
    )

    assert audit["solver_workers"] == 2
    assert all(case.solver["vector"]["preview_compact"] for case in cases)
    assert all("vector_ir" not in case.solver["vector"] for case in cases)
    assert all("events" not in case.solver["vector"]["embedded"] for case in cases)


def test_payload_worker_request_and_environment_cap(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(vector_native, "_PARALLEL_SOLVER_THRESHOLD", 1)
    cases, audit = sample_vector_cases(
        _small_payload(
            sample_limit=2,
            solver_backend="embedded",
            verification_effort="interactive",
            solver_workers=2,
        ),
        compute_verdicts=True,
    )
    assert len(cases) == 2
    assert audit["solver_workers"] == 2

    monkeypatch.setenv("LITMUS_LINK_SOLVER_WORKERS", "1")
    _cases, capped = sample_vector_cases(
        _small_payload(
            sample_limit=2,
            solver_backend="embedded",
            verification_effort="interactive",
            solver_workers=16,
        ),
        compute_verdicts=True,
    )
    assert capped["solver_workers"] == 1


def test_vector_search_limit_is_inconclusive_not_forbidden() -> None:
    cycles = native_template_cycles(["SB"], ["fence"])[0]
    cycle = next(
        cycle
        for cycle in cycles
        if cycle.labels
        == ("Fence.r.rsWR", "Fre", "Fence.rw.rwsWR", "Fre")
    )
    directions = vertex_directions(cycle.edges)
    choices = (
        _vector_choice(directions[0]),
        _vector_choice(directions[1]),
        _amo_choice(directions[2], "add", "d", "relaxed"),
        _vector_choice(directions[3]),
    )
    case = lower_vector_assignment(VectorAssignment(cycle, choices))
    verdict = solve_vector_case(case.case_ir, max_search_steps=1)
    assert verdict.status == "inconclusive"
    assert verdict.verdict == "unknown"
    assert verdict.allowed is None
    assert "forbidden verdict requires exhaustive search" in verdict.reason


@pytest.mark.parametrize("layout", ["same_start", "contained"])
def test_mixed_width_overlap_layouts_are_naturally_aligned_and_formal(layout: str) -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(overlap_layouts=[layout])
    )
    assignment = domain.random_assignments(1, 23)[0]
    case = lower_vector_assignment(assignment)
    accesses = [
        event.memory_access
        for event in case.case_ir.events()
        if event.memory_access is not None
    ]
    assert accesses
    assert all(access.natural_aligned for access in accesses)
    assert case.case_ir.metadata["memory_layout"]["overlap_layout"] == layout
    assert solve_generated_case(case).status == "verified"


@pytest.mark.parametrize("layout", ["low_partial", "high_partial"])
def test_aligned_partial_layouts_are_counted_as_unsatisfiable(layout: str) -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(alignments=["aligned"], overlap_layouts=[layout])
    )
    audit = domain.audit()
    assert audit["generated"] == 0
    assert audit["excluded"]["excluded_unsatisfiable_value_layout"] == audit["raw_combinations"]
    assert list(domain.assignments()) == []


@pytest.mark.parametrize("layout", ["low_partial", "high_partial"])
def test_misaligned_partial_count_matches_assignment_enumeration(layout: str) -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            endpoint_categories=["scalar", "vector"],
            endpoint_compositions=["vector_scalar"],
            scalar_widths=["w"],
            forms=["unit_load", "unit_store"],
            sew=["e32"],
            alignments=["misalign_same16"],
            overlap_layouts=[layout],
        )
    )
    assignments = list(domain.assignments())
    assert assignments
    assert domain.total_cases == len(assignments)


def test_misaligned_layout_feasibility_is_boundary_translation_invariant() -> None:
    from litmus_link.vector_native import _group_layout_status

    categories = ("vector", "scalar", "vector", "scalar")
    for widths in product((2, 4, 8), repeat=4):
        for layout in ("same_start", "contained", "low_partial", "high_partial"):
            statuses = {
                _group_layout_status(categories, widths, alignment, layout)
                for alignment in (
                    "misalign_same16",
                    "misalign_cross16",
                    "misalign_cross64",
                )
            }
            assert len(statuses) == 1


def test_disjoint_layout_is_audited_but_not_generated() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(overlap_layouts=["same_start", "disjoint_control"])
    )
    audit = domain.audit()
    assert audit["excluded"]["excluded_unsatisfiable_value_layout"] > 0
    assert audit["raw_combinations"] == (
        audit["generated"] + sum(audit["excluded"].values())
    )
    assert all(
        assignment.overlap_layout == "same_start"
        for assignment in domain.assignments()
    )


def test_workflow_preview_limit_counts_final_random_cases() -> None:
    preview = preview_payload(_small_payload(sample_limit=17))
    assert len(preview["sample"]) == 17
    assert preview["report"]["total_combinations"] > 17
    assert all(re.fullmatch(r"LLV-LB-[0-9a-f]{64}", item["case_id"]) for item in preview["sample"])
    assert all(item["name"].startswith("LB+{") and "+E{" in item["name"] for item in preview["sample"])
    groups = preview["classification_counts"]["groups"]
    assert groups["endpoint_composition"]
    assert groups["scalar_width"] == {"w": sum(groups["scalar_width"].values())}
    assert groups["amo_opcode"] == {"add": sum(groups["amo_opcode"].values())}
    assert groups["amo_width"] == {"d": sum(groups["amo_width"].values())}
    assert groups["amo_ordering"] == {"aq": sum(groups["amo_ordering"].values())}
    assert groups["sew"] == {"e16": sum(groups["sew"].values())}
    assert groups["overlap_layout"] == {"same_start": 17}
    assert groups["solver_status"] == {"verified": 17}
    assert groups["external_status"] == {"not_run": 17}


def test_file_hash_covers_endpoint_and_overlap_configuration() -> None:
    cycles, _ = native_template_cycles(["MP"], ["po"])
    cycle = cycles[0]
    directions = vertex_directions(cycle.edges)
    choices = tuple(
        _vector_choice(direction) if index == 0 else _scalar_choice(direction, "d")
        for index, direction in enumerate(directions)
    )
    first = lower_vector_assignment(VectorAssignment(cycle, choices, overlap_layout="same_start"))
    second = lower_vector_assignment(VectorAssignment(cycle, choices, overlap_layout="contained"))
    changed = list(choices)
    changed[1] = _amo_choice(directions[1], "maxu", "d", "rl")
    third = lower_vector_assignment(VectorAssignment(cycle, tuple(changed)))
    assert len({first.name, second.name, third.name}) == 3
    assert all(re.fullmatch(r"LLV-MP-[0-9a-f]{64}", case.name) for case in (first, second, third))


def test_domain_includes_all_selected_axes() -> None:
    domain = VectorNativeDomain.from_payload(_payload())
    assert {choice.category for choice in domain.read_choices} == {"scalar", "amo", "vector"}
    assert {choice.width_bytes for choice in domain.read_choices if choice.category == "scalar"} == {1, 2, 4, 8}
    assert {str((choice.params or {}).get("amo_op")) for choice in domain.read_choices if choice.category == "amo"} == set(AMO_OPERATIONS)
    assert {str((choice.params or {}).get("amo_ordering")) for choice in domain.read_choices if choice.category == "amo"} == set(AMO_ORDERINGS)
    actual_amos = {
        (
            str((choice.params or {}).get("amo_op")),
            str((choice.params or {}).get("amo_width")),
            str((choice.params or {}).get("amo_ordering")),
            choice.direction,
        )
        for choice in (*domain.read_choices, *domain.write_choices)
        if choice.category == "amo"
    }
    assert actual_amos == set(
        product(AMO_OPERATIONS, ("w", "d"), AMO_ORDERINGS, ("R", "W"))
    )


def test_all_scalar_vector_width_pairs_and_overlap_shapes_remain_aligned() -> None:
    cycle = native_template_cycles(["MP"], ["po"])[0][0]
    directions = vertex_directions(cycle.edges)
    for scalar_width, vector_sew, layout in product(
        ("b", "h", "w", "d"),
        ("e8", "e16", "e32", "e64"),
        ("same_start", "contained"),
    ):
        choices = (
            _vector_choice(directions[0], vector_sew),
            _scalar_choice(directions[1], scalar_width),
            _scalar_choice(directions[2], "d"),
            _scalar_choice(directions[3], "b"),
        )
        case = lower_vector_assignment(
            VectorAssignment(cycle, choices, overlap_layout=layout)
        )
        expansion = expand_vector_case(case.case_ir)
        assert all(
            event.memory_access is None or event.memory_access.natural_aligned
            for event in expansion.case.events()
        ), (scalar_width, vector_sew, layout)


def test_every_generated_vector_endpoint_has_aligned_active_elements() -> None:
    domain = VectorNativeDomain.from_payload(
        _payload(
            skeletons=["Co"],
            mechanisms=["po"],
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=[
                "unit_load", "unit_store", "strided_load", "strided_store",
                "indexed_unordered_load", "indexed_unordered_store",
                "indexed_ordered_load", "indexed_ordered_store",
            ],
        )
    )
    vector_choices = [
        choice
        for choice in (*domain.read_choices, *domain.write_choices)
        if choice.category == "vector"
    ]
    assert vector_choices
    for choice in vector_choices:
        params = dict(choice.params or {})
        width = choice.width_bytes
        effective_vl = vector_effective_vl(
            str(params["sew"]), str(params["lmul"]), str(params["vl"])
        )
        assert effective_vl is not None
        stride = width * 2 if choice.vector_form.startswith("strided_") else width
        active = [
            index
            for index in range(effective_vl)
            if params["mask"] == "unmasked" or index % 2 == 0
        ]
        assert active
        assert all((index * stride) % width == 0 for index in active)


def test_segment_native_cycle_lowers_all_fields_and_is_formally_solved() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            skeletons=["MP"],
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=["segment_unit_load", "segment_unit_store"],
            sew=["e32"],
            lmul=["m1"],
            nf=["nf3"],
            vl=["vl2"],
        )
    )
    case = lower_vector_assignment(next(domain.assignments()))
    expansion = expand_vector_case(case.case_ir)
    result = solve_vector_case(case.case_ir)

    assert "vlseg3e32.v" in case.litmus
    assert "vsseg3e32.v" in case.litmus
    assert all(config.nf == 3 for config in expansion.configs.values())
    assert all(instruction.active_segment_count == 2 for instruction in expansion.instructions)
    assert all(len(instruction.active_elements) == 6 for instruction in expansion.instructions)
    assert all(
        {element.field_index for element in instruction.active_elements} == {0, 1, 2}
        for instruction in expansion.instructions
    )
    assert result.status == "verified"
    assert result.verdict in {"observable", "forbidden"}


def test_segment_endpoint_audit_counts_reserved_emul_nfields_combinations() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            skeletons=["MP"],
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=["segment_unit_load", "segment_unit_store"],
            sew=["e32"],
            lmul=["m4"],
            nf=["nf2", "nf3"],
            vl=["vl1"],
        )
    )
    vector_audit = domain.audit()["endpoint_domain"]["vector"]

    assert vector_audit["raw_configurations"] == 4
    assert vector_audit["generated_endpoint_choices"] == 2
    assert vector_audit["excluded"] == {"excluded_illegal_vector_config": 2}
    assert {
        str((choice.params or {}).get("nf"))
        for choice in (*domain.read_choices, *domain.write_choices)
    } == {"nf2"}


def test_non_segment_forms_do_not_duplicate_over_the_nfields_axis() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            skeletons=["MP"],
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=["unit_load", "unit_store"],
            nf=["nf2", "nf3", "nf8"],
        )
    )

    assert len(domain.read_choices) == 1
    assert len(domain.write_choices) == 1
    assert all("nf" not in (choice.params or {}) for choice in (*domain.read_choices, *domain.write_choices))


def test_whole_register_choices_do_not_duplicate_over_irrelevant_vector_axes() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            skeletons=["MP"],
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=["whole_register_load", "whole_register_store"],
            sew=["e8", "e16", "e32", "e64"],
            lmul=["mf8", "m1", "m8"],
            nf=["nf2", "nf8"],
            whole_nreg=["nreg1", "nreg2", "nreg4", "nreg8"],
            mask=["unmasked", "masked"],
            tail=["ta_ma", "tu_mu"],
            vl=["vl1", "vlmax"],
        )
    )

    assert len(domain.read_choices) == 16
    assert len(domain.write_choices) == 4
    assert {
        str((choice.params or {})["whole_nreg"])
        for choice in (*domain.read_choices, *domain.write_choices)
    } == {"nreg1", "nreg2", "nreg4", "nreg8"}
    assert all(
        not {"lmul", "mask", "tail", "vl", "nf"}.intersection(choice.params or {})
        for choice in (*domain.read_choices, *domain.write_choices)
    )
    assert {
        str((choice.params or {})["sew"]) for choice in domain.write_choices
    } == {"e8"}


def test_whole_register_lowering_initializes_full_store_group() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            skeletons=["MP"],
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=["whole_register_load", "whole_register_store"],
            sew=["e64"],
            whole_nreg=["nreg8"],
        )
    )
    case = lower_vector_assignment(next(domain.assignments()))

    assert "vl8re64.v" in case.litmus
    assert "vs8r.v" in case.litmus
    assert "vsetvli" in case.litmus and "e8,m8,ta,ma" in case.litmus
    assert "vmv.v.x v8" in case.litmus
    vectors = case.case_ir.metadata["vectors"]
    assert all(config["whole_nreg"] == 8 for config in vectors.values())
    assert all(config["avl"] == "evl" for config in vectors.values())


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("scalar_widths", "scalar width"),
        ("amo_ops", "AMO opcode"),
        ("amo_widths", "AMO width"),
        ("amo_orderings", "AMO ordering"),
        ("forms", "Vector form"),
        ("sew", "SEW"),
        ("lmul", "LMUL"),
        ("mask", "mask mode"),
        ("tail", "tail policy"),
        ("vl", "Vector length"),
        ("alignments", "Vector alignment"),
        ("overlap_layouts", "fusion overlap layout"),
    ],
)
def test_empty_enabled_axis_is_rejected(field: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VectorNativeDomain.from_payload(_small_payload(**{field: []}))


def test_indexed_form_requires_an_index_eew() -> None:
    with pytest.raises(ValueError, match="indexed offset EEW"):
        VectorNativeDomain.from_payload(
            _small_payload(forms=["indexed_ordered_load"], index_eew=[])
        )


def test_segment_form_requires_nfields() -> None:
    with pytest.raises(ValueError, match="Segment NFIELDS"):
        VectorNativeDomain.from_payload(
            _small_payload(forms=["segment_unit_load"], nf=[])
        )


def test_impossible_vector_only_direction_is_audited() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=["unit_load"],
        )
    )
    audit = domain.audit()
    assert domain.total_cases == 0
    assert audit["request_exclusions"] == {
        "excluded_unsatisfiable_endpoint_composition": 1
    }


def test_audit_accounts_for_illegal_vector_configs_and_nanhu_amo_widths() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            amo_widths=["b", "h", "w"],
            sew=["e64"],
            lmul=["mf8", "m1"],
            vl=["vl1"],
        )
    )
    audit = domain.audit()
    assert (
        audit["endpoint_domain"]["amo"]["excluded"][
            "excluded_illegal_nanhu_amo_width"
        ]
        > 0
    )
    assert (
        audit["endpoint_domain"]["vector"]["excluded"][
            "excluded_illegal_vector_config"
        ]
        > 0
    )
    assert audit["excluded_illegal"] > 0
    assert all(choice.width_bytes == 4 for choice in domain.read_choices if choice.category == "amo")


def test_scope_audit_generates_vector_misalignment_and_excludes_amo_members() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            alignments=["aligned", "misalign_cross64"],
            attributes=["cacheable", "pbmt_nc", "pbmt_io"],
            pbmt=[0, 1, 2, 3],
            pma_atomic=False,
        )
    )
    audit = domain.audit()
    assert audit["request_exclusions"] == {
        "excluded_illegal_pbmt_reserved_request": 1,
        "excluded_unsupported_pma_nonatomic_request": 1,
        "excluded_unsupported_pbmt_nc_io_request": 4,
    }
    assert audit["formal_scope"] == {
        "pbmt": 0,
        "attribute": "cacheable",
        "pma_atomic": True,
        "vector_alignment": ["aligned", "misalign_cross64"],
        "misaligned_atomicity": "byte_level_no_mag",
        "scalar_alignment": "natural",
        "amo_alignment": "natural",
    }
    assert domain.alignments == ("aligned", "misalign_cross64")
    assert audit["excluded"]["excluded_unsupported_misaligned_amo_fusion"] > 0
    misaligned = [
        assignment
        for assignment in domain.assignments()
        if assignment.alignment == "misalign_cross64"
    ]
    assert misaligned
    assert all(
        all(choice.category != "amo" for choice in assignment.choices)
        for assignment in misaligned
    )


def test_unsupported_only_scope_request_generates_no_formal_cases() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            alignments=["misalign_cross64"],
            attributes=["pbmt_nc"],
            pbmt=[1],
        )
    )
    assert domain.total_cases == 0
    assert list(domain.assignments()) == []
    assert domain.audit()["request_exclusions"]


def test_domain_count_matches_complete_small_enumeration() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(overlap_layouts=["same_start", "contained", "high_partial"])
    )
    assert domain.total_cases == sum(1 for _ in domain.assignments())
    assert sum(domain.audit()["family_cases"].values()) == domain.total_cases


def test_misaligned_domain_count_matches_complete_small_enumeration() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            endpoint_categories=["scalar", "vector"],
            endpoint_compositions=["vector_only", "vector_scalar"],
            scalar_widths=["h", "w"],
            sew=["e16", "e32"],
            alignments=[
                "aligned",
                "misalign_same16",
                "misalign_cross16",
                "misalign_cross64",
            ],
            overlap_layouts=["same_start", "contained"],
        )
    )
    audit = domain.audit()
    assert domain.total_cases == sum(1 for _ in domain.assignments())
    assert audit["raw_combinations"] == (
        domain.total_cases + sum(audit["excluded"].values())
    )


@pytest.mark.parametrize(
    "forms",
    [
        ["unit_load", "unit_store"],
        ["strided_load", "strided_store"],
        ["indexed_unordered_load", "indexed_unordered_store"],
        ["indexed_ordered_load", "indexed_ordered_store"],
        ["segment_unit_load", "segment_unit_store"],
        ["whole_register_load", "unit_store"],
    ],
)
@pytest.mark.parametrize(
    ("alignment", "boundary"),
    [
        ("misalign_same16", "same16"),
        ("misalign_cross16", "cross16_same_line"),
        ("misalign_cross64", "cross64"),
    ],
)
def test_vector_forms_lower_and_solve_byte_level_misalignment(
    forms: list[str],
    alignment: str,
    boundary: str,
) -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=forms,
            sew=["e32"],
            whole_nreg=["nreg2"],
            nf=["nf2"],
            alignments=[alignment],
            overlap_layouts=["same_start"],
        )
    )
    assignment = next(domain.assignments())
    case = lower_vector_assignment(assignment)
    configs = case.case_ir.metadata["vectors"]
    assert configs
    assert all(config["alignment"] == alignment for config in configs.values())
    assert all(config["atomicity_model"] == "byte_level_no_mag" for config in configs.values())
    assert all(config["boundary"] == boundary for config in configs.values())
    assert all(config["base_offset_bytes"] % 4 != 0 for config in configs.values())
    assert f"/{ {'misalign_same16': 'U16', 'misalign_cross16': 'X16', 'misalign_cross64': 'X64'}[alignment]}" in case.display_name
    verdict = solve_generated_case(case)
    assert verdict.status == "verified"
    assert verdict.model == "riscv.cat+rvv-elements+byte-level-no-mag"


def test_e8_and_misaligned_amo_fusion_are_audited_not_generated() -> None:
    e8 = VectorNativeDomain.from_payload(
        _small_payload(
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            sew=["e8"],
            alignments=["misalign_cross16"],
        )
    )
    assert e8.total_cases == 0
    assert e8.audit()["excluded"]["excluded_unsatisfiable_misaligned_layout"] > 0

    amo = VectorNativeDomain.from_payload(
        _small_payload(
            endpoint_categories=["vector", "amo"],
            endpoint_compositions=["vector_amo"],
            alignments=["misalign_cross16"],
        )
    )
    assert amo.total_cases == 0
    assert amo.audit()["excluded"]["excluded_unsupported_misaligned_amo_fusion"] > 0


def test_same_location_sb_rejects_two_amos_that_both_read_initial() -> None:
    cycles, _audit = native_template_cycles(["SB"], ["po"])
    same_location = next(cycle for cycle in cycles if "PosWR" in cycle.labels)
    different_locations = next(cycle for cycle in cycles if "PodWR" in cycle.labels)
    read_vertices = tuple(
        vertex
        for vertex, direction in enumerate(vertex_directions(same_location.edges))
        if direction == READ
    )
    amo_mask = sum(1 << vertex for vertex in read_vertices)
    assert not _amo_mask_satisfiable(same_location, amo_mask)
    assert _amo_mask_satisfiable(different_locations, amo_mask)


def test_amo_heavy_domain_count_matches_assignment_enumeration() -> None:
    domain = VectorNativeDomain.from_payload(
        _small_payload(
            skeletons=["SB"],
            endpoint_categories=["amo", "vector"],
            endpoint_compositions=["vector_amo"],
            forms=["unit_load", "unit_store"],
            overlap_layouts=["same_start", "contained"],
        )
    )
    assert domain.total_cases == sum(1 for _ in domain.assignments())
    assert domain.audit()["raw_combinations"] == (
        domain.total_cases + sum(domain.audit()["excluded"].values())
    )


def test_exhaustive_generation_writes_every_legal_assignment(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = _small_payload(
        skeletons=["Co"],
        endpoint_categories=["scalar", "vector"],
        endpoint_compositions=["vector_scalar"],
        forms=["unit_load", "unit_store"],
        generation_mode="all",
        compute_verdicts=False,
    )
    domain = VectorNativeDomain.from_payload(payload)
    out = tmp_path / "vector-all"
    progress = []
    report = generate_vector_cases(
        payload,
        out,
        progress_callback=lambda current, total, _message: progress.append((current, total)),
    )
    assert report["generated_litmus"] == domain.total_cases
    assert sum(report["solver"].values()) == domain.total_cases
    assert sum(report["solver_verdict"].values()) == domain.total_cases
    assert sum(report["external_status"].values()) == domain.total_cases
    files = list(out.glob("*.litmus"))
    entries = (out / "@all").read_text(encoding="utf-8").splitlines()
    assert len(files) == len(entries) == len(set(entries)) == domain.total_cases
    assert all(re.fullmatch(r"LLV-CoRR-[0-9a-f]{64}\.litmus", entry) for entry in entries)
    assert len(validate_path(out / "@all")) == domain.total_cases
    assert progress[-1] == (domain.total_cases, domain.total_cases)


def test_sampled_generation_uses_bounded_parallel_solver(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LITMUS_LINK_SOLVER_WORKERS", "2")
    monkeypatch.setattr(vector_native, "_PARALLEL_SOLVER_THRESHOLD", 4)
    payload = _small_payload(
        generation_mode="balanced",
        generate_limit=4,
        compute_verdicts=True,
        solver_backend="embedded",
        verification_effort="interactive",
    )
    out = tmp_path / "parallel-vector"
    report = generate_vector_cases(payload, out)
    assert report["generated_litmus"] == 4
    assert report["solver_workers"] == 2
    assert len((out / "@all").read_text(encoding="utf-8").splitlines()) == 4
    assert sum(report["solver"].values()) == 4


def test_generation_refuses_identity_collision(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = _small_payload(
        skeletons=["Co"],
        endpoint_categories=["scalar", "vector"],
        endpoint_compositions=["vector_scalar"],
        generation_mode="balanced",
        generate_limit=1,
        compute_verdicts=False,
    )
    out = tmp_path / "collision"
    generate_vector_cases(payload, out)
    entry = (out / "@all").read_text(encoding="utf-8").strip()
    meta_path = (out / entry).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["case_ir"]["metadata"]["file_identity"]["canonical"]["overlap_layout"] = "tampered"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different identity"):
        generate_vector_cases(payload, out)
