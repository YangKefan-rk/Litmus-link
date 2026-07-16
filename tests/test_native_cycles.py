from __future__ import annotations

from itertools import islice
from pathlib import Path

import json
import pytest

from litmus_link.native_cycles import (
    NativeCycle,
    enumerate_relation_cycles,
    enumerate_template_cycles,
    location_ids,
    process_ids,
    validate_cycle,
)
from litmus_link.native_edges import (
    communication_edges,
    dependency_edges,
    edge_by_label,
    edges_for_shape,
    fence_edges,
    po_edges,
)
from litmus_link.native_scalar import generate_native_templates, lower_native_cycle, native_catalog, native_template_cycles
from litmus_link.toolchain import herd_judge, tools_available
from litmus_link.validator import validate_path


def test_native_edge_domain_is_generated_without_external_tools() -> None:
    assert len(communication_edges()) == 6
    assert len(po_edges()) == 8
    assert len(fence_edges()) == 96
    assert len(dependency_edges()) == 14
    rr = edges_for_shape("RR")
    ww = edges_for_shape("WW")
    assert len(rr) == 32
    assert len(ww) == 26
    assert {edge.mechanism for edge in rr} == {"po", "fence", "addr", "ctrl", "ctrl_fencei"}


def test_native_mp_template_exhausts_all_configured_local_edges() -> None:
    cycles, report = enumerate_template_cycles(
        "MP",
        [edge_by_label("Rfe"), edges_for_shape("RR"), edge_by_label("Fre"), edges_for_shape("WW")],
        max_procs=2,
        exact_procs=True,
    )
    assert report.candidates == 32 * 26
    # Mixed same/different-location rings are contradictory; both all-d and
    # all-s halves remain, so every non-conflicting configured MP is emitted.
    assert report.accepted == 416
    assert report.excluded == {"location_constraint_conflict": 416}
    assert len({cycle.canonical_key for cycle in cycles}) == 416
    assert all(cycle.nprocs == 2 for cycle in cycles)


def test_native_cycle_constraints_reject_invalid_rings() -> None:
    bad_direction = [edge_by_label("Rfe"), edge_by_label("Wse")]
    assert validate_cycle(bad_direction).reason == "direction_mismatch"

    one_external = [edge_by_label("Rfe"), edge_by_label("PosRR"), edge_by_label("Fri")]
    assert validate_cycle(one_external).reason == "single_external_edge"


def test_native_cycle_assigns_harts_and_locations_from_constraints() -> None:
    cycle = NativeCycle(
        (
            edge_by_label("Rfe"),
            edge_by_label("PodRR"),
            edge_by_label("Fre"),
            edge_by_label("PodWW"),
        ),
        "MP",
    )
    assert process_ids(cycle.edges) == (0, 1, 1, 0)
    assert location_ids(cycle.edges) == (0, 0, 1, 1)


def test_native_general_enumerator_is_canonical_and_bounded() -> None:
    domain = (
        edge_by_label("Rfe"),
        edge_by_label("Fre"),
        edge_by_label("PodRR"),
        edge_by_label("PodWW"),
    )
    cycles = list(
        enumerate_relation_cycles(
            domain,
            min_size=4,
            max_size=4,
            max_procs=2,
            exact_procs=True,
        )
    )
    assert len(cycles) == 3
    assert len({cycle.canonical_key for cycle in cycles}) == 3
    assert ("Fre", "PodWW", "Rfe", "PodRR") in {cycle.labels for cycle in cycles}
    assert all(
        cycle.canonical_key
        == min(cycle.labels[index:] + cycle.labels[:index] for index in range(4))
        for cycle in cycles
    )


def test_native_lowering_builds_self_contained_riscv_litmus() -> None:
    cycles, _audit = native_template_cycles(["MP"])
    lowered = lower_native_cycle(cycles[0])
    assert lowered.litmus.startswith(f"RISCV {lowered.name}\n")
    assert "Generator=Litmus-link-native" in lowered.litmus
    assert "Cycle=" in lowered.litmus
    assert "exists\n(" in lowered.litmus
    assert len(lowered.case_ir.harts) == 2
    assert {relation.kind for relation in lowered.case_ir.relations} >= {"rf", "fr"}


@pytest.mark.skipif(not tools_available(), reason="herdtools7 cross-check not installed")
def test_native_lowering_is_accepted_by_independent_herd_model() -> None:
    cycles, _audit = native_template_cycles(["MP"])
    verdict = herd_judge(lower_native_cycle(cycles[0]).litmus)
    assert verdict.outcome in {"observable", "forbidden"}


def test_native_generation_writes_and_validates_without_diytools(tmp_path: Path) -> None:
    report = generate_native_templates(
        out_dir=tmp_path,
        presets=["MP"],
        annotations=["P"],
        limit=3,
        judge=False,
    )
    assert report["generator"]["engine"] == "litmus-link-native"
    assert report["available_litmus"] == 416
    assert report["generated_litmus"] == 3
    assert report["generation_limited"] is True
    assert len(validate_path(tmp_path / "@all")) == 3
    metadata = json.loads(next(tmp_path.glob("*.meta.json")).read_text(encoding="utf-8"))
    assert metadata["generated_from"] == "litmus-link-native"


def test_native_annotations_expand_to_isa_legal_amo_forms() -> None:
    cycles, _audit = native_template_cycles(["MP"])
    from litmus_link.native_scalar import annotated_native_cycles

    annotated = next(
        cycle
        for cycle in annotated_native_cycles(cycles[:1], ["P", "Aq"])
        if "Aq" in cycle.annotations
    )
    lowered = lower_native_cycle(annotated)
    memory_instructions = [
        event.instruction
        for hart in lowered.case_ir.harts
        for event in hart
        if event.kind in {"load", "store", "amo"}
    ]
    assert any(instruction.startswith(("amoor.w.aq", "amoswap.w.aq")) for instruction in memory_instructions)
    assert all("lw.aq" not in instruction and "sw.aq" not in instruction for instruction in memory_instructions)


def test_native_full_template_domain_count_is_stable() -> None:
    counts = native_catalog()["template_counts_all_annotations"]
    assert counts["MP"] == 106496
    assert counts["ISA2"] == 72417280
    assert sum(counts.values()) == 74873888


def test_native_generation_does_not_spawn_diytools_when_judging_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_subprocess(*_args, **_kwargs):
        raise AssertionError("native generation attempted to spawn an external process")

    monkeypatch.setattr("subprocess.run", reject_subprocess)
    report = generate_native_templates(
        out_dir=tmp_path,
        presets=["MP"],
        mechanisms=["po"],
        annotations=["P"],
        include_same=False,
        limit=2,
        judge=False,
    )
    assert report["generated_litmus"] == 1
    assert len(validate_path(tmp_path / "@all")) == 1
