from litmus_link.models import EXCLUDED_ILLEGAL, EXCLUDED_UNSUPPORTED, GENERATED, HAND_REQUIRED, Combination
from litmus_link.rules import evaluate, RULE_DESCRIPTIONS
import re
from pathlib import Path


def test_pbmt_reserved_is_illegal() -> None:
    decision = evaluate(Combination("test", "pbmt_nc", "MP", "scalar_pair", "pbmt_reserved"))
    assert decision.status == EXCLUDED_ILLEGAL
    assert "PBMT=3" in decision.reason


def test_nonleaf_pbmt_is_illegal() -> None:
    decision = evaluate(Combination("test", "vm_tlb", "MP", "pte_update", "pbmt_nc", tlb="nonleaf_pbmt"))
    assert decision.status == EXCLUDED_ILLEGAL
    assert decision.hand_category == "exception"


def test_cbo_nonzero_offset_is_illegal() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable", cmo="flush_offset4"))
    assert decision.status == EXCLUDED_ILLEGAL
    assert "offset" in decision.reason


def test_cbo_csr_denied_is_hand_required() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable", cmo="clean_csr_denied"))
    assert decision.status == HAND_REQUIRED
    assert decision.hand_category == "exception"


def test_legacy_fake_cmo_is_illegal() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush_sync"))
    assert decision.status == EXCLUDED_ILLEGAL
    assert "not a RISC-V CMO instruction" in decision.reason


def test_cross_page_is_not_vector_instruction() -> None:
    decision = evaluate(Combination("test", "vector_mem", "MP", "vector_load", "cacheable", vector="cross_page"))
    assert decision.status == EXCLUDED_ILLEGAL
    assert "footprint parameter" in decision.reason


def test_vector_params_require_vector_axis() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable", cmo="flush", params={"sew": "e32"}))
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert "require a non-none vector" in decision.reason


def test_vector_reserved_vtype_and_cross_line_footprints_are_not_generated() -> None:
    reserved = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="unit_load",
            params={"sew": "e32", "lmul": "mf8", "mask": "unmasked", "vl": "vl1"},
        )
    )
    assert reserved.status == "excluded_illegal"
    crossing = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="strided_load",
            params={
                "sew": "e8",
                "lmul": "m4",
                "mask": "unmasked",
                "vl": "vlmax",
                "footprint": "same_line",
            },
        )
    )
    assert crossing.status == "excluded_unsupported"
    assert "cross-line" in crossing.reason


def test_vector_load_store_shape_must_match_memory_event() -> None:
    load_as_store = evaluate(Combination("test", "vector_mem", "MP", "vector_store", "cacheable", vector="unit_load"))
    store_as_load = evaluate(Combination("test", "vector_mem", "MP", "vector_load", "cacheable", vector="unit_store"))
    assert load_as_store.status == EXCLUDED_UNSUPPORTED
    assert store_as_load.status == EXCLUDED_UNSUPPORTED
    assert "vector_store" in load_as_store.reason
    assert "vector_load" in store_as_load.reason


def test_segment_nfields_and_register_groups_are_checked() -> None:
    legal = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="segment_unit_load",
            params={
                "sew": "e32",
                "lmul": "m4",
                "mask": "unmasked",
                "vl": "vl1",
                "nf": "nf2",
            },
        )
    )
    reserved = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="segment_unit_load",
            params={
                "sew": "e32",
                "lmul": "m4",
                "mask": "unmasked",
                "vl": "vl1",
                "nf": "nf3",
            },
        )
    )
    ordinary_with_nf = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="unit_load",
            params={"nf": "nf2"},
        )
    )

    assert legal.status == GENERATED
    assert reserved.status == EXCLUDED_ILLEGAL
    assert "reserved" in reserved.reason
    assert ordinary_with_nf.status == EXCLUDED_UNSUPPORTED
    assert "only applies" in ordinary_with_nf.reason


def test_whole_register_nreg_and_store_eew_are_checked() -> None:
    legal_load = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="whole_register_load",
            params={"sew": "e64", "whole_nreg": "nreg8", "footprint": "cross_line"},
        )
    )
    reserved_nreg = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="whole_register_load",
            params={"sew": "e32", "whole_nreg": "nreg3"},
        )
    )
    illegal_store_eew = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_store",
            "cacheable",
            vector="whole_register_store",
            params={"sew": "e32", "whole_nreg": "nreg1"},
        )
    )

    assert legal_load.status == GENERATED
    masked_whole = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="whole_register_load",
            params={"sew": "e32", "whole_nreg": "nreg1", "mask": "masked"},
        )
    )

    assert reserved_nreg.status == EXCLUDED_ILLEGAL
    assert illegal_store_eew.status == EXCLUDED_ILLEGAL
    assert masked_whole.status == EXCLUDED_ILLEGAL


def test_cmo_shape_must_match_memory_event() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "scalar_pair", "cacheable", cmo="flush"))
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert "memory_event=cmo" in decision.reason


def test_vm_params_require_tlb_axis() -> None:
    decision = evaluate(Combination("test", "rvwmo_base", "MP", "scalar_pair", "cacheable", params={"pte": "pa_remap"}))
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert "require a non-none TLB" in decision.reason


def test_unrendered_stress_dep_is_not_formal_coverage() -> None:
    decision = evaluate(Combination("test", "rvwmo_base", "MP", "scalar_pair", "cacheable", params={"dep": "aq"}))
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert decision.metadata["formal_forbidden_claim"] == "false"
    assert decision.metadata["unrendered_axis"] == "dep=aq"


def test_complex_vector_params_are_observation_not_formal_rvwmo() -> None:
    decision = evaluate(
        Combination(
            "test",
            "vector_mem",
            "MP",
            "vector_load",
            "cacheable",
            vector="unit_load",
            params={"sew": "e64", "footprint": "cross_page"},
        )
    )
    assert decision.status == GENERATED
    assert decision.rvwmo_class == "rvwmo-instruction-level"
    assert decision.metadata["formal_forbidden_claim"] == "false"


def test_nanhu_vector_non_cacheable_memory_is_excluded() -> None:
    for attribute in ("pbmt_nc", "pbmt_io", "nc_alias", "cacheable_nc_alias"):
        decision = evaluate(
            Combination("test", "vector_mem", "MP", "vector_load", attribute, vector="unit_load")
        )
        assert decision.status == EXCLUDED_UNSUPPORTED
        assert decision.hand_category == "vector"
        assert "cacheable" in decision.reason
        assert "rule:vector_memory_type" in decision.notes


def test_nanhu_vector_pbmt_flip_is_excluded() -> None:
    decision = evaluate(
        Combination(
            "test",
            "cross",
            "MP",
            "vector_load",
            "cacheable",
            tlb="pte_remap",
            vector="unit_load",
            params={"pte": "pbmt_flip"},
        )
    )
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert "PBMT-tagged" in decision.reason
    assert "rule:vector_memory_type" in decision.notes


def test_deferred_vector_parameters_are_not_silently_rendered() -> None:
    for params in (
        {"vl": "vl_random"},
        {"elem_order": "ordered_elements"},
        {"vstart": "1"},
        {"sew": "e128"},
    ):
        decision = evaluate(
            Combination(
                "test",
                "vector_mem",
                "MP",
                "vector_load",
                "cacheable",
                vector="unit_load",
                params=params,
            )
        )
        assert decision.status == EXCLUDED_UNSUPPORTED
        assert decision.metadata["formal_forbidden_claim"] == "false"
        assert "rule:vector_solver_scope" in decision.notes


def test_alias_flush_sync_is_generated_with_sync_metadata() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush", params={"sync": "full_alias_sync"}))
    assert decision.status == GENERATED
    assert decision.expected_kind == "prose-spec-constrained"
    assert decision.metadata["formal_forbidden_claim"] == "false"
    assert decision.metadata["alias_sync_required"].startswith("fence")


def test_full_alias_sync_requires_flush() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="clean", params={"sync": "full_alias_sync"}))
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert "requires cmo=flush" in decision.reason


def test_amo_on_nc_is_illegal_access_fault() -> None:
    # XiangShan/nanhu AtomicsUnit faults LR/AMO to NC/IO/mmio pages; A's atomic
    # support is PMA-dependent and this target's PMA does not advertise it.
    decision = evaluate(Combination("test", "amo_mem", "MP", "amo", "pbmt_nc"))
    assert decision.status == EXCLUDED_ILLEGAL
    assert decision.hand_category == "exception"
    assert "rule:pma_atomicity" in decision.notes


def test_amo_on_io_is_illegal_access_fault() -> None:
    decision = evaluate(Combination("test", "amo_mem", "MP", "amo", "pbmt_io"))
    assert decision.status == EXCLUDED_ILLEGAL
    assert "access fault" in decision.reason


def test_every_referenced_rule_has_a_description() -> None:
    # Guard against the latent bug where a rule: note has no RULE_DESCRIPTIONS
    # entry, so list_rules()/the GUI cannot describe it.
    src = Path("src/litmus_link/rules.py").read_text()
    referenced = set(re.findall(r'"rule:([a-z_0-9]+)"', src))
    missing = referenced - set(RULE_DESCRIPTIONS)
    assert not missing, f"rule keys referenced but undescribed: {sorted(missing)}"
