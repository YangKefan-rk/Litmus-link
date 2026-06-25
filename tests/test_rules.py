from litmus_link.models import EXCLUDED_ILLEGAL, GENERATED, HAND_REQUIRED, Combination
from litmus_link.rules import evaluate


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


def test_strided_fof_is_illegal() -> None:
    decision = evaluate(Combination("test", "vector", "MP", "vector_load", "cacheable", vector="fof_strided"))
    assert decision.status == EXCLUDED_ILLEGAL
    assert "FOF" in decision.reason


def test_alias_flush_sync_is_generated_with_sync_metadata() -> None:
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush_sync"))
    assert decision.status == GENERATED
    assert decision.metadata["alias_sync_required"].startswith("fence")
