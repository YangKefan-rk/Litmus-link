from __future__ import annotations

from fractions import Fraction
from itertools import chain, product
from typing import Any, Dict, Iterable, List, Mapping

from .models import Combination


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

FORMAL_VECTOR_SKELETONS = list(SKELETONS)
VECTOR_ENDPOINTS = {
    "MP": {
        "load": ["p1_ry", "p1_rx"],
        "store": ["p0_wx", "p0_wy"],
    },
    "LB": {
        "load": ["p0_rx", "p1_ry"],
        "store": ["p0_wy", "p1_wx"],
    },
    "SB": {
        "load": ["p0_ry", "p1_rx"],
        "store": ["p0_wx", "p1_wy"],
    },
    "WRC": {
        "load": ["p1_rx", "p2_ry", "p2_rx"],
        "store": ["p0_wx", "p1_wy"],
    },
    "RWC": {
        "load": ["p1_rx", "p1_ry", "p2_rx"],
        "store": ["p0_wx", "p2_wy"],
    },
    "IRIW": {
        "load": ["p2_rx", "p2_ry", "p3_ry", "p3_rx"],
        "store": ["p0_wx", "p1_wy"],
    },
    "ISA2": {
        "load": ["p1_ry", "p2_rz", "p2_rx"],
        "store": ["p0_wx", "p0_wy", "p1_wz"],
    },
    "R": {
        "load": ["p1_rx"],
        "store": ["p0_wx", "p0_wy", "p1_wy"],
    },
    "S": {
        "load": ["p1_rx"],
        "store": ["p0_wy", "p0_wx", "p1_wy"],
    },
    "Co": {
        "load": ["p1_rx_new", "p1_rx_old"],
        "store": ["p0_wx"],
    },
}

VECTOR_OPS = [
    "unit_load",
    "unit_store",
    "strided_load",
    "strided_store",
    "indexed_ordered_load",
    "indexed_unordered_load",
    "indexed_ordered_store",
    "indexed_unordered_store",
]

# Known RVV forms deliberately outside the first vector-aware solver scope.
# They are not exposed by profiles/GUI, but rules recognize them so old rule
# payloads receive an explicit unsupported decision instead of a fake verdict.
DEFERRED_VECTOR_OPS = [
    "segment_load",
    "segment_store",
    "fof_load",
    "fof_segment_load",
]

ATTRIBUTES = [
    "cacheable",
    "pbmt_nc",
    "pbmt_io",
    "nc_alias",
    "cacheable_nc_alias",
]

# Nanhu's supported vector-memory target domain.  PBMT/IO/NC alias mappings
# remain available to scalar, CMO, and VM profiles, but not to vector memory.
NANHU_VECTOR_ATTRIBUTES = ["cacheable"]
NANHU_VLEN_BITS = 128

CMO_OPS = [
    "clean",
    "flush",
    "inval",
    "zero",
]

TLB_OPS = [
    "local_sfence",
    "remote_sfence",
    "pte_remap",
    "permission_fault",
    "ad_update",
    "asid_global",
    "satp_switch",
]

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "smoke": "Small generated corpus used by make smoke and README examples.",
    "rvwmo_base": "Small in-process scalar IR profile for regression tests; use 'scalar cross/enumerate' for official herdtools generation.",
    "vector_mem": "Legacy one-endpoint Vector profile retained for rule compatibility; the Qt Vector workflow uses multi-endpoint relation cycles.",
    "cmo_pbmt": "Zicbom/Zicboz CMO operations crossed with PBMT/cacheability attributes.",
    "vm_tlb": "RISC-V page-table, PBMT, and sfence.vma scenarios, mostly HAND-required.",
    "full-cross": "Representative CMO/PBMT/Vector/TLB cross-product audit domain.",
    "stress-large": "Practical large profile with hundreds of thousands of Vector/CMO/PBMT/TLB stress combinations.",
    "stress-all": "Large cross-product domain for exhaustive Vector/CMO/PBMT/TLB/microarchitecture stress generation.",
}

VECTOR_WIDTHS = ["e8", "e16", "e32", "e64"]
VECTOR_INDEX_EEWS = ["ei8", "ei16", "ei32", "ei64"]
VECTOR_LMULS = ["mf8", "mf4", "mf2", "m1", "m2", "m4", "m8"]
VECTOR_LMUL_FACTORS = {
    "mf8": Fraction(1, 8),
    "mf4": Fraction(1, 4),
    "mf2": Fraction(1, 2),
    "m1": Fraction(1, 1),
    "m2": Fraction(2, 1),
    "m4": Fraction(4, 1),
    "m8": Fraction(8, 1),
}
VECTOR_MASKS = ["unmasked", "masked"]
VECTOR_TAILS = ["ta_ma", "ta_mu", "tu_ma", "tu_mu"]
VECTOR_FOOTPRINTS = ["same_line", "cross_line", "cross_page", "misalign", "partial_overlap"]
# The immediate form of vsetivli covers 1/2/4/8/16.  32 and 64 use an
# initialized AVL register and are included because they are important
# cache-line boundary cases for e8/e16/e32.  vlmax remains a separate
# architectural setting (vsetvli with rs1=x0).
VECTOR_LENGTHS = ["vl1", "vl2", "vl4", "vl8", "vl16", "vl32", "vl64", "vlmax"]


def vector_vlmax(sew: str, lmul: str) -> int | None:
    if sew not in VECTOR_WIDTHS or lmul not in VECTOR_LMUL_FACTORS:
        return None
    capacity = Fraction(NANHU_VLEN_BITS, int(sew[1:])) * VECTOR_LMUL_FACTORS[lmul]
    if capacity.denominator != 1 or capacity < 1:
        return None
    return int(capacity)


def vector_effective_vl(sew: str, lmul: str, vl: str) -> int | None:
    vlmax = vector_vlmax(sew, lmul)
    if vlmax is None:
        return None
    if vl == "vlmax":
        return vlmax
    if vl in VECTOR_LENGTHS and vl.startswith("vl") and vl[2:].isdigit():
        return min(int(vl[2:]), vlmax)
    return None


def vector_same_line_footprint(
    vector: str,
    sew: str,
    lmul: str,
    mask: str,
    vl: str,
) -> bool:
    effective_vl = vector_effective_vl(sew, lmul, vl)
    if effective_vl is None or mask not in VECTOR_MASKS:
        return False
    active = [
        index
        for index in range(effective_vl)
        if mask == "unmasked" or index % 2 == 0
    ]
    if not active:
        return False
    element_bytes = int(sew[1:]) // 8
    stride = element_bytes * 2 if vector.startswith("strided_") else element_bytes
    return max(active) * stride + element_bytes <= 64

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
    {"sew": sew, "lmul": lmul, "mask": mask, "tail": tail, "vl": vl}
    for sew, lmul, mask, tail, vl in product(
        VECTOR_WIDTHS,
        ["m1", "m4"],
        VECTOR_MASKS,
        ["ta_ma", "tu_mu"],
        ["vl1", "vlmax"],
    )
]
STRESS_CROSS_VECTOR_CONFIGS = [
    {"sew": "e8", "lmul": "m1", "mask": "unmasked", "tail": "ta_ma", "vl": "vl1"},
    {"sew": "e16", "lmul": "m1", "mask": "masked", "tail": "ta_mu", "vl": "vl2"},
    {"sew": "e32", "lmul": "m2", "mask": "unmasked", "tail": "tu_ma", "vl": "vlmax"},
    {"sew": "e64", "lmul": "m4", "mask": "masked", "tail": "tu_mu", "vl": "vl4"},
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
        Combination("smoke", "vector_mem", "LB", "vector_store", "cacheable", vector="unit_store"),
        Combination("smoke", "vector_mem", "MP", "vector_load", "cacheable", vector="indexed_ordered_load"),
        Combination("smoke", "cmo", "MP", "cmo", "cacheable", cmo="flush"),
        Combination("smoke", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush", params=_params(sync="full_alias_sync")),
        Combination("smoke", "cross", "MP", "vector_store", "cacheable", cmo="flush", vector="unit_store", params=_params(footprint="cross_page")),
    ]


def _rvwmo_base(profile: str) -> List[Combination]:
    return [Combination(profile, "rvwmo_base", skeleton, "scalar_pair", "cacheable") for skeleton in SKELETONS]


def _vector_mem(profile: str) -> List[Combination]:
    return vector_combinations(profile)


def vector_combinations(
    profile: str,
    *,
    skeletons: Iterable[str] = FORMAL_VECTOR_SKELETONS,
    vectors: Iterable[str] = VECTOR_OPS,
    widths: Iterable[str] = VECTOR_WIDTHS,
    lmuls: Iterable[str] = VECTOR_LMULS,
    masks: Iterable[str] = VECTOR_MASKS,
    tails: Iterable[str] = VECTOR_TAILS,
    lengths: Iterable[str] = VECTOR_LENGTHS,
    index_eews: Iterable[str] = VECTOR_INDEX_EEWS,
    endpoint_scope: str = "all",
) -> List[Combination]:
    """Build the canonical formal Vector domain, optionally with GUI filters."""
    if endpoint_scope not in {"all", "first"}:
        raise ValueError("endpoint_scope must be 'all' or 'first'")
    skeletons = tuple(skeletons)
    vectors = tuple(vectors)
    widths = tuple(widths)
    lmuls = tuple(lmuls)
    masks = tuple(masks)
    tails = tuple(tails)
    lengths = tuple(lengths)
    index_eews = tuple(index_eews)
    combos = []
    for skeleton, vector, sew, lmul, mask, tail, vl in product(
        skeletons,
        vectors,
        widths,
        lmuls,
        masks,
        tails,
        lengths,
    ):
        if vector_vlmax(sew, lmul) is None:
            continue
        if not vector_same_line_footprint(vector, sew, lmul, mask, vl):
            continue
        endpoint_kind = "store" if vector.endswith("store") else "load"
        memory_event = f"vector_{endpoint_kind}"
        selected_index_eews = index_eews if "indexed" in vector else [None]
        endpoints = VECTOR_ENDPOINTS[skeleton][endpoint_kind]
        if endpoint_scope == "first":
            endpoints = endpoints[:1]
        for index_eew, endpoint in product(selected_index_eews, endpoints):
            params = dict(
                sew=sew,
                lmul=lmul,
                mask=mask,
                tail=tail,
                footprint="same_line",
                vl=vl,
                vector_event=endpoint,
            )
            if index_eew is not None:
                params["index_eew"] = index_eew
            combos.append(
                Combination(
                    profile,
                    "vector_mem",
                    skeleton,
                    memory_event,
                    NANHU_VECTOR_ATTRIBUTES[0],
                    vector=vector,
                    params=_params(**params),
                )
            )
    return combos


def _cmo_pbmt(profile: str) -> List[Combination]:
    attrs = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias"]
    return [Combination(profile, "cmo", "MP", "cmo", attribute, cmo=cmo) for cmo, attribute in product(CMO_OPS, attrs)]


def _vm_tlb(profile: str) -> List[Combination]:
    attrs = ["cacheable", "pbmt_nc", "pbmt_io"]
    return [Combination(profile, "vm_tlb", "MP", "pte_update", attribute, tlb=tlb) for tlb, attribute in product(TLB_OPS, attrs)]


def _cross(profile: str) -> List[Combination]:
    return [
        Combination(profile, "cross", "MP", "vector_store", "cacheable", cmo="clean", vector="unit_store"),
        Combination(profile, "cross", "MP", "vector_store", "cacheable", cmo="flush", vector="unit_store"),
        Combination(profile, "cross", "MP", "vector_load", "cacheable", cmo="inval", vector="unit_load"),
        Combination(profile, "cross", "MP", "vector_load", "cacheable", cmo="zero", vector="unit_load"),
        Combination(profile, "cross", "MP", "cmo", "pbmt_nc", cmo="clean"),
        Combination(profile, "cross", "MP", "cmo", "pbmt_nc", cmo="flush"),
        Combination(profile, "cross", "MP", "cmo", "cacheable_nc_alias", cmo="flush", params=_params(sync="full_alias_sync")),
        Combination(profile, "cross", "MP", "cmo", "cacheable_nc_alias", tlb="pte_remap", cmo="flush", params=_params(sync="full_alias_sync")),
        Combination(profile, "cross", "MP", "vector_load", "cacheable", cmo="flush", vector="unit_load", params=_params(footprint="cross_page")),
        Combination(profile, "cross", "MP", "vector_load", "cacheable", tlb="pte_remap", vector="unit_load", params=_params(footprint="cross_page")),
        Combination(profile, "cross", "MP", "cmo", "cacheable", tlb="permission_fault", cmo="flush"),
        Combination(profile, "cross", "MP", "ifetch", "cacheable", tlb="remote_sfence", cmo="flush", params=_params(sync="fence_i_after")),
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
    attributes = list(NANHU_VECTOR_ATTRIBUTES)
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
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "nc_alias", "cacheable_nc_alias"]
    for skeleton, cmo, attribute, sync, alias, footprint, stressor in product(
        SKELETONS,
        CMO_OPS,
        attributes,
        syncs,
        aliases,
        footprints,
        stressors,
    ):
        if sync == "full_alias_sync" and cmo != "flush":
            continue
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
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias"]
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
    vectors = list(VECTOR_OPS)
    cmos = ["clean", "flush", "inval", "zero"]
    attributes = list(NANHU_VECTOR_ATTRIBUTES)
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
        if sync == "full_alias_sync" and cmo != "flush":
            continue
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
    vectors = list(VECTOR_OPS)
    attributes = list(NANHU_VECTOR_ATTRIBUTES)
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
    attributes = ["cacheable", "pbmt_nc", "pbmt_io", "cacheable_nc_alias"]
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
        if sync == "full_alias_sync" and cmo != "flush":
            continue
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
    vectors = ["unit_load", "unit_store", "indexed_ordered_load", "indexed_unordered_store"]
    cmos = ["clean", "flush", "zero"]
    tlbs = ["remote_sfence", "pte_remap", "permission_fault", "ad_update", "satp_switch"]
    # The vector operation itself is only generated on a cacheable mapping in
    # the Nanhu profile.  PBMT/IO coverage remains in _stress_cmo_tlb, where
    # the accesses are scalar/CMO/VM operations rather than vector memory.
    attributes = list(NANHU_VECTOR_ATTRIBUTES)
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
        if sync == "full_alias_sync" and cmo != "flush":
            continue
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
