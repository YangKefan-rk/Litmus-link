from __future__ import annotations

from itertools import chain, product
from typing import Any, Dict, Iterable, List, Mapping

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
    "stress-large": "Practical large profile with hundreds of thousands of Vector/CMO/PBMT/TLB stress combinations.",
    "stress-all": "Large cross-product domain for exhaustive Vector/CMO/PBMT/TLB/microarchitecture stress generation.",
}

VECTOR_WIDTHS = ["e8", "e16", "e32", "e64"]
VECTOR_LMULS = ["mf2", "m1", "m2", "m4"]
VECTOR_MASKS = ["unmasked", "masked"]
VECTOR_TAILS = ["ta_ma", "ta_mu", "tu_ma", "tu_mu"]
VECTOR_FOOTPRINTS = ["same_line", "cross_line", "cross_page", "misalign", "partial_overlap"]
VECTOR_LENGTHS = ["vl1", "vl2", "vlmax", "vl_random"]
ELEMENT_ORDERS = ["single_event", "ordered_elements", "unordered_elements"]

CMO_SYNC_SEQUENCES = ["none", "pre_fence", "post_fence", "full_alias_sync", "fence_i_after"]
VM_CONTEXTS = ["bare", "sv39", "sv39_asid", "sv39_global", "satp_switch"]
SHOOTDOWN_SCOPES = ["none", "local", "remote_ipi", "remote_missing", "global"]
PTE_STATES = ["stable", "invalid_to_valid", "valid_to_invalid", "pa_remap", "permission_flip", "ad_update", "pbmt_flip"]
ALIAS_MODES = ["none", "same_pa_same_attr", "cacheable_nc", "dual_va", "synonym"]
STRESSORS = [
    "none",
    "dcache_replay",
    "miss_queue_full",
    "store_buffer_full",
    "load_queue_replay",
    "ifetch_patch",
]
LARGE_STRESSORS = ["none", "store_buffer_full", "load_queue_replay"]
STRESS_VECTOR_CONFIGS = [
    {"sew": sew, "lmul": lmul, "mask": mask, "tail": tail, "vl": vl, "elem_order": order}
    for sew, lmul, mask, tail, vl, order in product(
        VECTOR_WIDTHS,
        ["m1", "m4"],
        VECTOR_MASKS,
        ["ta_ma", "tu_mu"],
        ["vl1", "vlmax"],
        ["single_event", "unordered_elements"],
    )
]
STRESS_CROSS_VECTOR_CONFIGS = [
    {"sew": "e8", "lmul": "m1", "mask": "unmasked", "tail": "ta_ma", "vl": "vl1", "elem_order": "single_event"},
    {"sew": "e16", "lmul": "m1", "mask": "masked", "tail": "ta_mu", "vl": "vl2", "elem_order": "ordered_elements"},
    {"sew": "e32", "lmul": "m2", "mask": "unmasked", "tail": "tu_ma", "vl": "vlmax", "elem_order": "unordered_elements"},
    {"sew": "e64", "lmul": "m4", "mask": "masked", "tail": "tu_mu", "vl": "vl_random", "elem_order": "unordered_elements"},
]


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


def profile_combinations(profile: str) -> Iterable[Combination]:
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
        return chain(_rvwmo_base(profile), _vector_mem(profile), _cmo_pbmt(profile), _vm_tlb(profile), _cross(profile))
    if profile == "stress-all":
        return _stress_all(profile)
    if profile == "stress-large":
        return _stress_large(profile)
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


def _stress_all(profile: str) -> Iterable[Combination]:
    return chain(
        _stress_rvwmo(profile),
        _stress_vector(profile),
        _stress_cmo_pbmt(profile),
        _stress_vm_tlb(profile),
        _stress_vector_cmo_pbmt(profile),
        _stress_vector_tlb(profile),
        _stress_cmo_tlb(profile),
        _stress_quad_cross(profile),
    )


def _stress_large(profile: str) -> Iterable[Combination]:
    return chain(
        _stress_rvwmo(profile, stressors=LARGE_STRESSORS),
        _stress_vector(profile, stressors=LARGE_STRESSORS, configs=STRESS_CROSS_VECTOR_CONFIGS, footprints=["same_line", "cross_line", "cross_page"]),
        _stress_cmo_pbmt(profile, stressors=LARGE_STRESSORS, syncs=["none", "full_alias_sync"], aliases=["none", "cacheable_nc"], footprints=["same_line", "cross_page"]),
        _stress_vm_tlb(profile, stressors=LARGE_STRESSORS, vm_contexts=["sv39", "sv39_asid"], shootdowns=["local", "remote_ipi"], pte_states=["pa_remap", "permission_flip", "pbmt_flip"], aliases=["none", "cacheable_nc"]),
        _stress_vector_cmo_pbmt(profile, stressors=["none", "store_buffer_full"], configs=STRESS_CROSS_VECTOR_CONFIGS[:2]),
        _stress_vector_tlb(profile, stressors=["none"], configs=STRESS_CROSS_VECTOR_CONFIGS[:2]),
        _stress_cmo_tlb(profile, stressors=["none"], syncs=["none", "full_alias_sync"]),
        _stress_quad_cross(profile, configs=STRESS_CROSS_VECTOR_CONFIGS[:1]),
    )


def _stress_rvwmo(profile: str, stressors: Iterable[str] = STRESSORS) -> Iterable[Combination]:
    dependency_shapes = ["none", "addr", "data", "ctrl", "ctrl_fence", "aq", "rl", "aqrl"]
    access_widths = ["w8", "w16", "w32", "w64"]
    outcomes = ["allowed", "forbidden", "mixed_size"]
    for skeleton, dependency, width, stressor, outcome in product(SKELETONS, dependency_shapes, access_widths, stressors, outcomes):
        yield Combination(
            profile,
            "rvwmo_base",
            skeleton,
            "scalar_pair",
            "cacheable",
            params=_params(dep=dependency, width=width, stress=stressor, outcome=outcome),
        )


def _stress_vector(
    profile: str,
    stressors: Iterable[str] = STRESSORS,
    configs: Iterable[Mapping[str, str]] = STRESS_VECTOR_CONFIGS,
    footprints: Iterable[str] = VECTOR_FOOTPRINTS,
) -> Iterable[Combination]:
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias", "pbmt_reserved"]
    for skeleton, vector, attribute, config, footprint, stressor in product(
        SKELETONS,
        VECTOR_OPS,
        attributes,
        configs,
        footprints,
        stressors,
    ):
        yield Combination(
            profile,
            "vector_mem",
            skeleton,
            _vector_memory_event(vector),
            attribute,
            vector=vector,
            params=_params(**config, footprint=footprint, stress=stressor),
        )


def _stress_cmo_pbmt(
    profile: str,
    stressors: Iterable[str] = STRESSORS,
    syncs: Iterable[str] = CMO_SYNC_SEQUENCES,
    aliases: Iterable[str] = ALIAS_MODES,
    footprints: Iterable[str] = ("same_line", "cross_line", "cross_page"),
) -> Iterable[Combination]:
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "nc_alias", "cacheable_nc_alias", "pbmt_reserved"]
    for skeleton, cmo, attribute, sync, alias, footprint, stressor in product(
        SKELETONS,
        CMO_OPS,
        attributes,
        syncs,
        aliases,
        footprints,
        stressors,
    ):
        yield Combination(
            profile,
            "cmo",
            skeleton,
            "cmo",
            attribute,
            cmo=cmo,
            params=_params(sync=sync, alias=alias, footprint=footprint, stress=stressor),
        )


def _stress_vm_tlb(
    profile: str,
    stressors: Iterable[str] = STRESSORS,
    vm_contexts: Iterable[str] = VM_CONTEXTS,
    shootdowns: Iterable[str] = SHOOTDOWN_SCOPES,
    pte_states: Iterable[str] = PTE_STATES,
    aliases: Iterable[str] = ALIAS_MODES,
) -> Iterable[Combination]:
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias", "pbmt_reserved"]
    for skeleton, tlb, attribute, vm, shootdown, pte_state, alias, stressor in product(
        SKELETONS,
        TLB_OPS,
        attributes,
        vm_contexts,
        shootdowns,
        pte_states,
        aliases,
        stressors,
    ):
        yield Combination(
            profile,
            "vm_tlb",
            skeleton,
            "pte_update",
            attribute,
            tlb=tlb,
            params=_params(vm=vm, shootdown=shootdown, pte=pte_state, alias=alias, stress=stressor),
        )


def _stress_vector_cmo_pbmt(
    profile: str,
    stressors: Iterable[str] = ("none", "store_buffer_full"),
    configs: Iterable[Mapping[str, str]] = STRESS_CROSS_VECTOR_CONFIGS,
) -> Iterable[Combination]:
    vectors = ["unit_load", "unit_store", "strided_load", "strided_store", "indexed_ordered_load", "indexed_unordered_load", "segment_load", "segment_store", "fof_load"]
    cmos = ["clean", "flush", "inval", "inval_as_flush", "zero", "flush_sync"]
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias", "pbmt_reserved"]
    for skeleton, vector, cmo, attribute, config, footprint, sync, alias, stressor in product(
        SKELETONS,
        vectors,
        cmos,
        attributes,
        configs,
        ["same_line", "cross_page"],
        ["none", "full_alias_sync"],
        ["none", "cacheable_nc"],
        stressors,
    ):
        yield Combination(
            profile,
            "cross",
            skeleton,
            _vector_memory_event(vector),
            attribute,
            cmo=cmo,
            vector=vector,
            params=_params(**config, footprint=footprint, sync=sync, alias=alias, stress=stressor),
        )


def _stress_vector_tlb(
    profile: str,
    stressors: Iterable[str] = ("none", "load_queue_replay"),
    configs: Iterable[Mapping[str, str]] = STRESS_CROSS_VECTOR_CONFIGS[:3],
) -> Iterable[Combination]:
    vectors = ["unit_load", "unit_store", "strided_load", "indexed_ordered_load", "indexed_unordered_load", "fof_load"]
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias", "pbmt_reserved"]
    for skeleton, vector, tlb, attribute, config, footprint, vm, shootdown, pte_state, stressor in product(
        SKELETONS,
        vectors,
        ["local_sfence", "remote_sfence", "pte_remap", "permission_fault", "ad_update", "asid_global", "satp_switch"],
        attributes,
        configs,
        ["cross_line", "cross_page"],
        ["sv39", "sv39_asid"],
        ["local", "remote_ipi"],
        ["pa_remap", "permission_flip", "pbmt_flip"],
        stressors,
    ):
        yield Combination(
            profile,
            "cross",
            skeleton,
            _vector_memory_event(vector),
            attribute,
            tlb=tlb,
            vector=vector,
            params=_params(**config, footprint=footprint, vm=vm, shootdown=shootdown, pte=pte_state, stress=stressor),
        )


def _stress_cmo_tlb(
    profile: str,
    stressors: Iterable[str] = ("none", "dcache_replay"),
    syncs: Iterable[str] = ("none", "full_alias_sync"),
) -> Iterable[Combination]:
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias", "pbmt_reserved"]
    for skeleton, cmo, tlb, attribute, sync, vm, shootdown, pte_state, alias, stressor in product(
        SKELETONS,
        CMO_OPS,
        TLB_OPS,
        attributes,
        syncs,
        ["sv39", "sv39_asid"],
        ["local", "remote_ipi"],
        ["pa_remap", "permission_flip", "pbmt_flip"],
        ["none", "cacheable_nc"],
        stressors,
    ):
        yield Combination(
            profile,
            "cross",
            skeleton,
            "cmo",
            attribute,
            tlb=tlb,
            cmo=cmo,
            params=_params(sync=sync, vm=vm, shootdown=shootdown, pte=pte_state, alias=alias, stress=stressor),
        )


def _stress_quad_cross(profile: str, configs: Iterable[Mapping[str, str]] = STRESS_CROSS_VECTOR_CONFIGS[:2]) -> Iterable[Combination]:
    vectors = ["unit_load", "unit_store", "indexed_ordered_load", "indexed_unordered_store", "segment_load"]
    cmos = ["clean", "flush", "zero", "flush_sync"]
    tlbs = ["remote_sfence", "pte_remap", "permission_fault", "ad_update", "satp_switch"]
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias"]
    for skeleton, vector, cmo, tlb, attribute, config, footprint, sync, vm, shootdown, pte_state, alias in product(
        ["MP", "LB", "SB", "WRC", "IRIW"],
        vectors,
        cmos,
        tlbs,
        attributes,
        configs,
        ["cross_line", "cross_page"],
        ["none", "full_alias_sync"],
        ["sv39", "sv39_asid"],
        ["local", "remote_ipi"],
        ["pa_remap", "pbmt_flip"],
        ["none", "cacheable_nc"],
    ):
        yield Combination(
            profile,
            "cross",
            skeleton,
            _vector_memory_event(vector),
            attribute,
            tlb=tlb,
            cmo=cmo,
            vector=vector,
            params=_params(**config, footprint=footprint, sync=sync, vm=vm, shootdown=shootdown, pte=pte_state, alias=alias),
        )


def _vector_memory_event(vector: str) -> str:
    return "vector_store" if vector.endswith("store") else "vector_load"


def _params(**values: str) -> Mapping[str, Any]:
    return {key: value for key, value in values.items() if value not in {"none", "bare"}}
