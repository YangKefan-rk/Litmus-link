from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List

from models import Combination


HAND_CATEGORIES = [
    "vm",
    "pbmt_nc",
    "cmo",
    "vector",
    "ifetch",
    "cross",
    "exception",
]

SKELETONS = ["MP", "LB", "SB", "WRC", "RWC", "IRIW", "ISA2", "R", "S", "Co"]

VECTOR_OPS = [
    "unit_load",
    "unit_store",
    "strided_load",
    "strided_store",
    "indexed_ordered_load",
    "indexed_unordered_load",
    "indexed_ordered_store",
    "indexed_unordered_store",
    "segment_load",
    "segment_store",
    "fof_load",
    "fof_segment_load",
    "fof_strided",
    "fof_indexed",
]

ATTRIBUTES = [
    "cacheable",
    "pbmt_nc",
    "pbmt_io",
    "nc_alias",
    "cacheable_nc_alias",
    "pbmt_reserved",
]

CMO_OPS = [
    "clean",
    "flush",
    "inval",
    "inval_as_flush",
    "zero",
    "flush_sync",
    "flush_offset4",
    "clean_csr_denied",
    "zero_csr_denied",
]

TLB_OPS = [
    "local_sfence",
    "remote_sfence",
    "pte_remap",
    "permission_fault",
    "ad_update",
    "asid_global",
    "satp_switch",
    "nonleaf_pbmt",
]

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "smoke": "Small generated corpus used by make smoke and README examples.",
    "rvwmo_base": "Scalar main-memory RVWMO skeletons compatible with herd riscv.cat.",
    "vector_mem": "Vector memory operations crossed with PBMT/cacheability attributes.",
    "cmo_pbmt": "Zicbom/Zicboz CMO operations crossed with PBMT/cacheability attributes.",
    "vm_tlb": "RISC-V page-table, PBMT, and sfence.vma scenarios, mostly HAND-required.",
    "full-cross": "Representative CMO/PBMT/Vector/TLB cross-product audit domain.",
}


def list_profiles() -> Dict[str, str]:
    return dict(PROFILE_DESCRIPTIONS)


def axis_values() -> Dict[str, List[str]]:
    return {
        "skeleton": list(SKELETONS),
        "attribute": list(ATTRIBUTES),
        "vector": list(VECTOR_OPS),
        "cmo": list(CMO_OPS),
        "tlb": list(TLB_OPS),
        "hand": list(HAND_CATEGORIES),
    }


def profile_combinations(profile: str) -> List[Combination]:
    if profile == "smoke":
        return _smoke()
    if profile == "rvwmo_base":
        return _rvwmo_base(profile)
    if profile == "vector_mem":
        return _vector_mem(profile)
    if profile == "cmo_pbmt":
        return _cmo_pbmt(profile)
    if profile == "vm_tlb":
        return _vm_tlb(profile)
    if profile == "full-cross":
        combos: List[Combination] = []
        combos.extend(_rvwmo_base(profile))
        combos.extend(_vector_mem(profile))
        combos.extend(_cmo_pbmt(profile))
        combos.extend(_vm_tlb(profile))
        combos.extend(_cross(profile))
        return combos
    raise ValueError(f"unknown profile: {profile}")


def _smoke() -> List[Combination]:
    return [
        Combination("smoke", "rvwmo_base", "MP", "scalar_pair", "cacheable"),
        Combination("smoke", "pbmt_nc", "MP", "scalar_pair", "pbmt_nc"),
        Combination("smoke", "vector_mem", "MP", "vector_load", "cacheable", vector="unit_load"),
        Combination("smoke", "vector_mem", "LB", "vector_store", "pbmt_nc", vector="unit_store"),
        Combination("smoke", "vector_mem", "MP", "vector_load", "pbmt_nc", vector="fof_load"),
        Combination("smoke", "cmo", "MP", "cmo", "cacheable", cmo="flush"),
        Combination("smoke", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush_sync"),
        Combination("smoke", "cross", "MP", "vector_store", "cacheable_nc_alias", cmo="flush_sync", vector="cross_page"),
    ]


def _rvwmo_base(profile: str) -> List[Combination]:
    return [Combination(profile, "rvwmo_base", skeleton, "scalar_pair", "cacheable") for skeleton in SKELETONS]


def _vector_mem(profile: str) -> List[Combination]:
    combos = []
    for vector, attribute in product(VECTOR_OPS, ATTRIBUTES):
        memory_event = "vector_store" if vector.endswith("store") else "vector_load"
        combos.append(Combination(profile, "vector_mem", "MP", memory_event, attribute, vector=vector))
    return combos


def _cmo_pbmt(profile: str) -> List[Combination]:
    attrs = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias", "pbmt_reserved"]
    return [Combination(profile, "cmo", "MP", "cmo", attribute, cmo=cmo) for cmo, attribute in product(CMO_OPS, attrs)]


def _vm_tlb(profile: str) -> List[Combination]:
    attrs = ["cacheable", "pbmt_nc", "pbmt_io", "pbmt_reserved"]
    return [Combination(profile, "vm_tlb", "MP", "pte_update", attribute, tlb=tlb) for tlb, attribute in product(TLB_OPS, attrs)]


def _cross(profile: str) -> List[Combination]:
    return [
        Combination(profile, "cross", "MP", "vector_store", "cacheable", cmo="clean", vector="unit_store"),
        Combination(profile, "cross", "MP", "vector_store", "cacheable", cmo="flush", vector="unit_store"),
        Combination(profile, "cross", "MP", "vector_load", "cacheable", cmo="inval", vector="unit_load"),
        Combination(profile, "cross", "MP", "vector_load", "cacheable", cmo="zero", vector="unit_load"),
        Combination(profile, "cross", "MP", "cmo", "pbmt_nc", cmo="clean"),
        Combination(profile, "cross", "MP", "cmo", "pbmt_nc", cmo="flush"),
        Combination(profile, "cross", "MP", "cmo", "cacheable_nc_alias", cmo="flush_sync"),
        Combination(profile, "cross", "MP", "cmo", "cacheable_nc_alias", tlb="pte_remap", cmo="flush_sync"),
        Combination(profile, "cross", "MP", "vector_load", "pbmt_nc", tlb="pte_remap", vector="cross_page"),
        Combination(profile, "cross", "MP", "vector_load", "cacheable", cmo="flush", vector="cross_page"),
        Combination(profile, "cross", "MP", "vector_load", "cacheable_nc_alias", cmo="flush_sync", vector="cross_page"),
        Combination(profile, "cross", "MP", "vector_load", "cacheable_nc_alias", tlb="pte_remap", cmo="flush_sync", vector="cross_page"),
        Combination(profile, "cross", "MP", "vector_load", "pbmt_io", vector="fof_load"),
        Combination(profile, "cross", "MP", "cmo", "cacheable", tlb="permission_fault", cmo="flush"),
        Combination(profile, "cross", "MP", "ifetch", "cacheable", tlb="remote_sfence", cmo="flush_sync"),
    ]
