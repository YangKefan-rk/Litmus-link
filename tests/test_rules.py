from models import EXCLUDED_ILLEGAL, EXCLUDED_UNSUPPORTED, GENERATED, HAND_REQUIRED, Combination
from rules import evaluate


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


def test_vm_params_require_tlb_axis() -> None:
    decision = evaluate(Combination("test", "rvwmo_base", "MP", "scalar_pair", "cacheable", params={"pte": "pa_remap"}))
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert "require a non-none TLB" in decision.reason


def test_alias_flush_sync_is_generated_with_sync_metadata() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush", params={"sync": "full_alias_sync"}))
    assert decision.status == GENERATED
    assert decision.metadata["alias_sync_required"].startswith("fence")


def test_full_alias_sync_requires_flush() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="clean", params={"sync": "full_alias_sync"}))
    assert decision.status == EXCLUDED_UNSUPPORTED
    assert "requires cmo=flush" in decision.reason
