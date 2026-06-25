from __future__ import annotations

from typing import Dict, List

from models import Combination, Decision, EXCLUDED_ILLEGAL, GENERATED, HAND_REQUIRED


RULE_DESCRIPTIONS: Dict[str, str] = {
    "pbmt_leaf_only": "PBMT bits are valid only in Sv39/Sv48/Sv57 leaf PTEs; non-leaf PBMT must be zero.",
    "pbmt_reserved": "PBMT value 3 is reserved and can only be used for negative page-fault cases.",
    "pbmte_sfence": "Changing PBMTE requires sfence.vma x0,x0 before relying on PBMT interpretation.",
    "alias_sync": "cacheable/NC alias requires fence iorw,iorw; cbo.flush; fence iorw,iorw to prevent loss of coherence/order.",
    "cbo_offset_zero": "CBO assembly offset must evaluate to zero.",
    "cbo_envcfg": "CBO execution depends on CBIE/CBCFE/CBZE privilege configuration.",
    "cbo_zero_non_atomic": "cbo.zero must not be treated as an atomic whole-block store.",
    "vector_fof_unit_only": "FOF vector loads are unit-stride only, plus legal unit-stride segment FOF loads.",
    "vector_non_idempotent_fof": "FOF into non-idempotent memory is unsafe unless restart/trimming cannot occur.",
    "vector_ordering": "Vector memory follows RVWMO at instruction level; only indexed-ordered operations order elements.",
    "remote_sfence": "sfence.vma affects the local hart; remote shootdown must be explicitly modeled.",
    "fence_i_local": "fence.i synchronizes only the executing hart's instruction fetch stream.",
    "rvwmo_scope": "FENCE.I/SFENCE.VMA/PMA/CMO interactions are prose-spec or hardware-observation tests, not herd RVWMO assertions.",
}


def list_rules() -> Dict[str, str]:
    return dict(RULE_DESCRIPTIONS)


def evaluate(combination: Combination) -> Decision:
    requires = _requires(combination)
    notes: List[str] = []

    if combination.attribute == "pbmt_reserved":
        return Decision(
            EXCLUDED_ILLEGAL,
            "PBMT=3 is reserved for future standard use and would raise a page-fault for a normal leaf PTE.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:pbmt_reserved"],
            "exception",
        )

    if combination.tlb == "nonleaf_pbmt":
        return Decision(
            EXCLUDED_ILLEGAL,
            "Non-leaf PTE PBMT bits must be zero; non-zero bits are a page-fault condition.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:pbmt_leaf_only"],
            "exception",
        )

    if combination.cmo.endswith("offset4"):
        return Decision(
            EXCLUDED_ILLEGAL,
            "CBO offset operands must evaluate to zero.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:cbo_offset_zero"],
            "exception",
        )

    if combination.cmo.endswith("csr_denied"):
        return Decision(
            HAND_REQUIRED,
            "CSR-denied CBO execution is a privilege/trap negative case and needs hand-written setup.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:cbo_envcfg"],
            "exception",
        )

    if combination.vector in {"fof_strided", "fof_indexed"}:
        return Decision(
            EXCLUDED_ILLEGAL,
            "RISC-V Vector defines unit-stride FOF loads and unit-stride segment FOF loads, not strided/indexed FOF loads.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:vector_fof_unit_only"],
            "vector",
        )

    if combination.vector in {"fof_load", "fof_segment_load"} and combination.attribute == "pbmt_io":
        return Decision(
            EXCLUDED_ILLEGAL,
            "FOF access to non-idempotent PBMT=IO memory is not generated because restart/trimming safety is not proven.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:vector_non_idempotent_fof"],
            "vector",
        )

    if combination.memory_event == "amo" and combination.attribute == "pbmt_nc":
        return Decision(
            HAND_REQUIRED,
            "AMO on PBMT=NC depends on the platform atomicity PMA and must be hand-classified.",
            "platform-specific",
            "platform-specific",
            requires,
            ["rule:pma_atomicity"],
            "pbmt_nc",
        )

    if combination.tlb != "no_tlb":
        return Decision(
            HAND_REQUIRED,
            "TLB/PTE/sfence.vma scenario requires explicit page-table and privilege setup.",
            "prose-spec",
            "hardware-observation",
            requires,
            _tlb_notes(combination),
            "vm",
        )

    if combination.attribute == "pbmt_io" and combination.cmo != "no_cmo":
        return Decision(
            HAND_REQUIRED,
            "CMO on PBMT=IO depends on supported access and I/O ordering PMAs; keep as HAND/platform case.",
            "platform-specific",
            "platform-specific",
            requires,
            ["rule:rvwmo_scope"],
            "cmo",
        )

    if combination.attribute == "pbmt_io" and combination.vector != "none":
        return Decision(
            HAND_REQUIRED,
            "Vector access to PBMT=IO needs non-idempotent trap/restart policy and platform PMA setup.",
            "platform-specific",
            "platform-specific",
            requires,
            ["rule:vector_non_idempotent_fof", "rule:rvwmo_scope"],
            "vector",
        )

    rvwmo_class = _rvwmo_class(combination)
    expected_kind = "rvwmo-herd" if rvwmo_class == "rvwmo-herd" else "hardware-observation"

    if combination.attribute == "cacheable_nc_alias":
        notes.append("rule:alias_sync")
        if combination.cmo == "flush_sync":
            notes.append("sync:fence-iorw-cbo.flush-fence-iorw")
        else:
            notes.append("coherence-risk:alias-without-full-sync")

    if combination.cmo == "zero":
        notes.append("rule:cbo_zero_non_atomic")
    if combination.cmo == "inval_as_flush":
        notes.append("inval_mode=flush")
    if combination.vector != "none":
        notes.append("rule:vector_ordering")

    return Decision(
        GENERATED,
        "Combination is legal in the Litmus-link finite generation domain.",
        rvwmo_class,
        expected_kind,
        requires,
        notes,
        None,
        _decision_metadata(combination),
    )


def _requires(combination: Combination) -> List[str]:
    requires: List[str] = ["RV64I"]
    if combination.memory_event == "amo":
        requires.append("A")
    if combination.vector != "none":
        requires.append("V")
    if combination.cmo in {"clean", "flush", "inval", "inval_as_flush", "flush_sync", "flush_offset4", "clean_csr_denied"}:
        requires.append("Zicbom")
    if combination.cmo in {"zero", "zero_csr_denied"}:
        requires.append("Zicboz")
    if combination.attribute.startswith("pbmt") or "nc_alias" in combination.attribute:
        requires.append("Svpbmt")
    if combination.tlb != "no_tlb":
        requires.extend(["S-mode", "Sv39", "sfence.vma"])
    if combination.memory_event == "ifetch" or combination.cmo == "flush_sync":
        requires.append("Zifencei")
    return sorted(set(requires))


def _rvwmo_class(combination: Combination) -> str:
    if (
        combination.attribute == "cacheable"
        and combination.tlb == "no_tlb"
        and combination.cmo == "no_cmo"
        and combination.vector == "none"
        and combination.memory_event == "scalar_pair"
    ):
        return "rvwmo-herd"
    if combination.vector != "none" and combination.cmo == "no_cmo" and combination.tlb == "no_tlb":
        return "rvwmo-instruction-level"
    if combination.cmo != "no_cmo" or combination.attribute != "cacheable":
        return "prose-spec"
    return "hardware-observation"


def _tlb_notes(combination: Combination) -> List[str]:
    notes = ["rule:remote_sfence" if combination.tlb == "remote_sfence" else "rule:pbmte_sfence"]
    if combination.memory_event == "ifetch":
        notes.append("rule:fence_i_local")
    if combination.attribute == "cacheable_nc_alias":
        notes.append("rule:alias_sync")
    return notes


def _decision_metadata(combination: Combination) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if combination.cmo == "inval_as_flush":
        metadata["inval_mode"] = "flush"
    elif combination.cmo == "inval":
        metadata["inval_mode"] = "inval"
    elif combination.cmo.endswith("csr_denied"):
        metadata["inval_mode"] = "trap"
    if combination.attribute == "pbmt_nc":
        metadata["effective_memory_type"] = "NC main memory, idempotent, RVWMO"
    if combination.attribute == "cacheable_nc_alias":
        metadata["alias_sync_required"] = "fence iorw,iorw; cbo.flush; fence iorw,iorw"
    return metadata
