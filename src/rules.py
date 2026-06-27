from __future__ import annotations

from typing import Dict, List

from models import Combination, Decision, EXCLUDED_ILLEGAL, EXCLUDED_UNSUPPORTED, GENERATED, HAND_REQUIRED
from profiles import ATTRIBUTES, CMO_OPS, CMO_SYNC_SEQUENCES, SKELETONS, TLB_OPS, VECTOR_OPS


MEMORY_EVENTS = {"scalar_pair", "vector_load", "vector_store", "cmo", "pte_update", "ifetch", "amo"}
LEGACY_FAKE_CMOS = {"flush_sync", "inval_as_flush"}
NEGATIVE_CMOS = {"flush_offset4", "clean_csr_denied", "zero_csr_denied"}
NEGATIVE_ATTRIBUTES = {"pbmt_reserved"}
NEGATIVE_TLBS = {"nonleaf_pbmt"}
ILLEGAL_VECTOR_FORMS = {"fof_strided", "fof_indexed"}
VECTOR_LOAD_OPS = {operation for operation in VECTOR_OPS if not operation.endswith("store")}
VECTOR_STORE_OPS = {operation for operation in VECTOR_OPS if operation.endswith("store")}
VECTOR_PARAMS = {"sew", "lmul", "mask", "tail", "vl", "elem_order"}
VM_PARAMS = {"vm", "shootdown", "pte"}


RULE_DESCRIPTIONS: Dict[str, str] = {
    "pbmt_leaf_only": "PBMT bits are valid only in Sv39/Sv48/Sv57 leaf PTEs; non-leaf PBMT must be zero.",
    "pbmt_reserved": "PBMT value 3 is reserved and can only be used for negative page-fault cases.",
    "pbmte_sfence": "Changing PBMTE requires sfence.vma x0,x0 before relying on PBMT interpretation.",
    "alias_sync": "cacheable/NC alias requires fence iorw,iorw; cbo.flush; fence iorw,iorw to prevent loss of coherence/order.",
    "cbo_offset_zero": "CBO assembly offset must evaluate to zero.",
    "cbo_envcfg": "CBO execution depends on CBIE/CBCFE/CBZE privilege configuration.",
    "cbo_zero_non_atomic": "cbo.zero must not be treated as an atomic whole-block store.",
    "vector_fof_unit_only": "FOF vector loads are unit-stride only, plus legal unit-stride segment FOF loads.",
    "vector_event_shape": "Vector load/store instruction forms must match the generated memory-event shape.",
    "vector_non_idempotent_fof": "FOF into non-idempotent memory is unsafe unless restart/trimming cannot occur.",
    "vector_ordering": "Vector memory follows RVWMO at instruction level; only indexed-ordered operations order elements.",
    "cmo_event_shape": "CMO operations must be emitted as CMO, ifetch, or explicit Vector+CMO cross observation shapes.",
    "remote_sfence": "sfence.vma affects the local hart; remote shootdown must be explicitly modeled.",
    "fence_i_local": "fence.i synchronizes only the executing hart's instruction fetch stream.",
    "rvwmo_scope": "FENCE.I/SFENCE.VMA/PMA/CMO interactions are prose-spec or hardware-observation tests, not herd RVWMO assertions.",
}


def list_rules() -> Dict[str, str]:
    return dict(RULE_DESCRIPTIONS)


def evaluate(combination: Combination) -> Decision:
    requires = _requires(combination)
    notes: List[str] = []

    invalid = _axis_validity(combination, requires)
    if invalid is not None:
        return invalid

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

    if combination.vector in ILLEGAL_VECTOR_FORMS:
        return Decision(
            EXCLUDED_ILLEGAL,
            "RISC-V Vector defines unit-stride FOF loads and unit-stride segment FOF loads, not strided/indexed FOF loads.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:vector_fof_unit_only"],
            "vector",
        )

    sync = str(combination.params.get("sync", "none"))
    param_decision = _param_validity(combination, requires)
    if param_decision is not None:
        return param_decision
    shape_decision = _shape_validity(combination, requires)
    if shape_decision is not None:
        return shape_decision
    if sync not in {"", *CMO_SYNC_SEQUENCES}:
        return Decision(
            EXCLUDED_UNSUPPORTED,
            f"Unknown sync sequence {sync!r}; use one of {', '.join(CMO_SYNC_SEQUENCES)}.",
            "prose-spec",
            "hardware-observation",
            requires,
            ["rule:rvwmo_scope"],
            "cmo",
        )
    if sync == "full_alias_sync" and combination.cmo != "flush":
        return Decision(
            EXCLUDED_UNSUPPORTED,
            "full_alias_sync is defined as fence iorw,iorw; cbo.flush; fence iorw,iorw and therefore requires cmo=flush.",
            "prose-spec",
            "hardware-observation",
            requires,
            ["rule:alias_sync"],
            "cmo",
        )
    if combination.memory_event == "ifetch" and combination.cmo == "no_cmo" and sync not in {"", "none", "fence_i_after"}:
        return Decision(
            EXCLUDED_UNSUPPORTED,
            "Ifetch-only synchronization currently supports no extra sync or fence_i_after; CMO fences require a CMO operation.",
            "prose-spec",
            "hardware-observation",
            requires,
            ["rule:fence_i_local", "rule:rvwmo_scope"],
            "ifetch",
        )
    if sync not in {"none", ""} and combination.cmo == "no_cmo" and combination.memory_event != "ifetch":
        return Decision(
            EXCLUDED_UNSUPPORTED,
            "CMO synchronization parameters require a CMO operation or an ifetch observation sequence.",
            "prose-spec",
            "hardware-observation",
            requires,
            ["rule:rvwmo_scope"],
            "cmo",
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

    if combination.attribute == "pbmt_io" and combination.cmo == "no_cmo" and combination.vector == "none" and combination.tlb == "no_tlb":
        return Decision(
            HAND_REQUIRED,
            "PBMT=IO/non-idempotent memory is platform-specific and must not be emitted as an ordinary main-memory RVWMO assertion.",
            "platform-specific",
            "platform-specific",
            requires,
            ["rule:rvwmo_scope"],
            "pbmt_nc",
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
    expected_kind = _expected_kind(rvwmo_class)

    if combination.attribute == "cacheable_nc_alias":
        notes.append("rule:alias_sync")
        if combination.params.get("sync") == "full_alias_sync":
            notes.append("sync:fence-iorw-cbo.flush-fence-iorw")
        else:
            notes.append("coherence-risk:alias-without-full-sync")

    if combination.cmo == "zero":
        notes.append("rule:cbo_zero_non_atomic")
    if combination.params.get("inval_mode") == "flush":
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
    if combination.cmo in {"clean", "flush", "inval"} or combination.cmo.endswith("offset4") or combination.cmo == "clean_csr_denied":
        requires.append("Zicbom")
    if combination.cmo == "zero" or combination.cmo == "zero_csr_denied":
        requires.append("Zicboz")
    if combination.attribute.startswith("pbmt") or "nc_alias" in combination.attribute:
        requires.append("Svpbmt")
    if combination.tlb != "no_tlb":
        requires.extend(["S-mode", "Sv39", "sfence.vma"])
    if combination.memory_event == "ifetch" or combination.params.get("sync") == "fence_i_after":
        requires.append("Zifencei")
    return sorted(set(requires))


def _rvwmo_class(combination: Combination) -> str:
    if _is_scalar_main_memory(combination):
        # Svpbmt defines NC as non-cacheable *main memory*, idempotent and
        # RVWMO-ordered. RVWMO's PPO rules never reference cacheability, so an
        # NC scalar test has the SAME forbidden/allowed verdict as its cacheable
        # twin -- only the justification carries one extra Svpbmt prose
        # dependency. herd7 runs the identical plain body and agrees.
        return "rvwmo-herd" if combination.attribute == "cacheable" else "rvwmo-nc"
    if combination.vector != "none" and combination.cmo == "no_cmo" and combination.tlb == "no_tlb":
        return "rvwmo-instruction-level"
    if combination.cmo != "no_cmo" or combination.attribute != "cacheable":
        return "prose-spec"
    return "hardware-observation"


def _is_scalar_main_memory(combination: Combination) -> bool:
    return (
        combination.attribute in {"cacheable", "pbmt_nc"}
        and combination.tlb == "no_tlb"
        and combination.cmo == "no_cmo"
        and combination.vector == "none"
        and combination.memory_event == "scalar_pair"
    )


def _expected_kind(rvwmo_class: str) -> str:
    if rvwmo_class in {"rvwmo-herd", "rvwmo-nc"}:
        return rvwmo_class
    if rvwmo_class == "rvwmo-instruction-level":
        return "hardware-observation"
    if rvwmo_class == "prose-spec":
        return "prose-spec-constrained"
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
    rvwmo_class = _rvwmo_class(combination)
    metadata["oracle"] = _expected_kind(rvwmo_class)
    metadata["formal_forbidden_claim"] = "true" if rvwmo_class in {"rvwmo-herd", "rvwmo-nc"} else "false"
    if combination.params.get("inval_mode") == "flush":
        metadata["inval_mode"] = "flush"
    elif combination.cmo == "inval":
        metadata["inval_mode"] = "inval"
    if combination.attribute == "pbmt_nc":
        metadata["effective_memory_type"] = "NC main memory, idempotent, RVWMO"
    if combination.attribute == "cacheable_nc_alias":
        metadata["alias_sync_required"] = "fence iorw,iorw; cbo.flush; fence iorw,iorw"
    return metadata


def _axis_validity(combination: Combination, requires: List[str]) -> Decision | None:
    checks = [
        ("skeleton", combination.skeleton, set(SKELETONS)),
        ("memory_event", combination.memory_event, MEMORY_EVENTS),
        ("attribute", combination.attribute, {"cacheable", *ATTRIBUTES, *NEGATIVE_ATTRIBUTES}),
        ("tlb", combination.tlb, {"no_tlb", *TLB_OPS, *NEGATIVE_TLBS}),
        ("cmo", combination.cmo, {"no_cmo", *CMO_OPS, *NEGATIVE_CMOS, *LEGACY_FAKE_CMOS}),
        ("vector", combination.vector, {"none", "cross_page", *VECTOR_OPS, *ILLEGAL_VECTOR_FORMS}),
    ]
    for axis, value, allowed in checks:
        if value not in allowed:
            return Decision(
                EXCLUDED_ILLEGAL,
                f"Unknown {axis} value {value!r}; it is not part of the executable Litmus-link generation domain.",
                "negative-exception",
                "negative-exception",
                requires,
                ["rule:axis_value_domain"],
                axis,
            )
    if combination.cmo in LEGACY_FAKE_CMOS:
        return Decision(
            EXCLUDED_ILLEGAL,
            f"{combination.cmo} is not a RISC-V CMO instruction. Use a real CMO op plus metadata params such as sync or inval_mode.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:cbo_instruction_domain"],
            "cmo",
        )
    if combination.vector == "cross_page":
        return Decision(
            EXCLUDED_ILLEGAL,
            "cross_page is an address-footprint parameter, not a vector instruction form.",
            "negative-exception",
            "negative-exception",
            requires,
            ["rule:vector_instruction_domain"],
            "vector",
        )
    return None


def _param_validity(combination: Combination, requires: List[str]) -> Decision | None:
    vector_params = sorted(VECTOR_PARAMS.intersection(combination.params))
    if vector_params and combination.vector == "none":
        return Decision(
            EXCLUDED_UNSUPPORTED,
            f"Vector parameters {', '.join(vector_params)} require a non-none vector operation.",
            "prose-spec",
            "hardware-observation",
            requires,
            ["rule:parameter_scope"],
            "vector",
        )
    vm_params = sorted(VM_PARAMS.intersection(combination.params))
    if vm_params and combination.tlb == "no_tlb":
        return Decision(
            EXCLUDED_UNSUPPORTED,
            f"Virtual-memory parameters {', '.join(vm_params)} require a non-none TLB/VM axis.",
            "prose-spec",
            "hardware-observation",
            requires,
            ["rule:parameter_scope"],
            "vm",
        )
    if combination.params.get("inval_mode") == "flush" and combination.cmo != "inval":
        return Decision(
            EXCLUDED_UNSUPPORTED,
            "inval_mode=flush describes platform behavior of cbo.inval and therefore requires cmo=inval.",
            "prose-spec",
            "hardware-observation",
            requires,
            ["rule:cbo_envcfg"],
            "cmo",
        )
    return None


def _shape_validity(combination: Combination, requires: List[str]) -> Decision | None:
    if combination.vector != "none":
        if combination.memory_event not in {"vector_load", "vector_store"}:
            return Decision(
                EXCLUDED_UNSUPPORTED,
                "A non-none vector operation requires memory_event=vector_load or memory_event=vector_store.",
                "prose-spec",
                "hardware-observation",
                requires,
                ["rule:vector_event_shape"],
                "vector",
            )
        if combination.vector in VECTOR_LOAD_OPS and combination.memory_event != "vector_load":
            return Decision(
                EXCLUDED_UNSUPPORTED,
                f"Vector load operation {combination.vector} cannot be emitted in a vector_store memory event.",
                "prose-spec",
                "hardware-observation",
                requires,
                ["rule:vector_event_shape"],
                "vector",
            )
        if combination.vector in VECTOR_STORE_OPS and combination.memory_event != "vector_store":
            return Decision(
                EXCLUDED_UNSUPPORTED,
                f"Vector store operation {combination.vector} cannot be emitted in a vector_load memory event.",
                "prose-spec",
                "hardware-observation",
                requires,
                ["rule:vector_event_shape"],
                "vector",
            )

    if combination.cmo != "no_cmo":
        if combination.vector == "none" and combination.memory_event not in {"cmo", "ifetch"}:
            return Decision(
                EXCLUDED_UNSUPPORTED,
                "A CMO operation without Vector must use memory_event=cmo or memory_event=ifetch.",
                "prose-spec",
                "prose-spec-constrained",
                requires,
                ["rule:cmo_event_shape"],
                "cmo",
            )
        if combination.vector != "none" and combination.memory_event not in {"vector_load", "vector_store"}:
            return Decision(
                EXCLUDED_UNSUPPORTED,
                "Vector+CMO cross scenarios must keep the vector load/store memory-event shape.",
                "prose-spec",
                "prose-spec-constrained",
                requires,
                ["rule:cmo_event_shape", "rule:vector_event_shape"],
                "cross",
            )
    return None
