from __future__ import annotations

from typing import Any, Dict

from models import Combination


FEATURE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "skeleton": {
        "MP": "Message Passing: one hart publishes data then a flag while another observes the flag before reading the data.",
        "LB": "Load Buffering: two harts load before storing, stressing whether both loads may observe old values.",
        "SB": "Store Buffering: two harts store before loading, stressing store-buffer visibility and propagation delay.",
        "WRC": "Write-Read Causality: an intermediate read-derived write tests transitive visibility.",
        "RWC": "Read-Write Causality: read-observed data drives a later write observed by another hart.",
        "IRIW": "Independent Reads of Independent Writes: readers may observe independent writers in different orders.",
        "ISA2": "Dependency/fence skeleton for preserved program order edges outside the common MP/LB/SB forms.",
        "R": "Read-shape relation focus for read-from/from-read/dependency variants.",
        "S": "Store-shape relation focus for propagation and same-address ordering variants.",
        "Co": "Coherence shape stressing per-location write/read order.",
    },
    "memory_event": {
        "scalar_pair": "Scalar load/store baseline RVWMO event.",
        "vector_load": "Vector load replaces or augments a scalar read event across elements, lines, or pages.",
        "vector_store": "Vector store replaces or augments a scalar write event and stresses partial visibility.",
        "cmo": "Cache-management operation participates in the observation sequence.",
        "pte_update": "Page-table update or translation-state change participates in the multi-hart interaction.",
        "ifetch": "Instruction fetch observes code bytes written or remapped by another hart.",
        "amo": "Atomic memory operation participates and depends on platform atomicity/PMA support.",
    },
    "attribute": {
        "cacheable": "Normal coherent cacheable main memory.",
        "pbmt_nc": "PBMT=NC idempotent main memory, still treated as RVWMO main memory.",
        "pbmt_io": "PBMT=IO or non-idempotent memory; platform-specific and usually HAND-required.",
        "nc_alias": "Non-cacheable mapping aliases another mapping of the same physical storage.",
        "cacheable_nc_alias": "Same PA is reachable through cacheable and NC mappings, requiring explicit synchronization.",
    },
    "vector": {
        "unit_load": "Unit-stride vector load over contiguous elements.",
        "unit_store": "Unit-stride vector store over contiguous elements.",
        "strided_load": "Strided vector load stressing non-contiguous address generation.",
        "strided_store": "Strided vector store stressing non-contiguous store merging.",
        "indexed_ordered_load": "Ordered indexed vector load with element ordering constraints.",
        "indexed_unordered_load": "Unordered indexed vector load where element order is not guaranteed.",
        "indexed_ordered_store": "Ordered indexed vector store stressing ordered scatter visibility.",
        "indexed_unordered_store": "Unordered indexed vector store stressing scatter visibility without element order.",
        "segment_load": "Segment vector load with multiple destination register groups.",
        "segment_store": "Segment vector store with interleaved multi-field writes.",
        "fof_load": "Fault-only-first unit-stride load: element0 traps, later faults trim vl.",
        "fof_segment_load": "Legal segment fault-only-first load with vl trimming behavior.",
    },
    "cmo": {
        "clean": "cbo.clean pushes dirty data toward the coherence point while keeping the line valid.",
        "flush": "cbo.flush writes back dirty data and invalidates the cache block.",
        "inval": "cbo.inval invalidates a cache block without assuming writeback unless configured otherwise.",
        "zero": "cbo.zero writes zeros but is not an atomic whole-block store.",
    },
    "tlb": {
        "local_sfence": "Local sfence.vma orders this hart's PTE writes against its later translations.",
        "remote_sfence": "Remote shootdown must be explicitly modeled; local sfence does not flush another hart.",
        "pte_remap": "Leaf PTE changes physical target, exposing stale TLB versus updated page table state.",
        "permission_fault": "PTE permission change stresses fault/no-fault ordering across harts.",
        "ad_update": "A/D bit update races with hardware page-table walks and software PTE modification.",
        "asid_global": "ASID/global mapping scenario stresses selective invalidation scope.",
        "satp_switch": "satp context switch stresses translation-root replacement and required fencing.",
    },
}

PARAM_DESCRIPTIONS: Dict[str, str] = {
    "dep": "Dependency or aq/rl ordering shape applied to the relation skeleton.",
    "width": "Scalar access width, including mixed-size overlap coverage.",
    "outcome": "Expected weak-result class for the relation variant.",
    "sew": "Vector selected element width.",
    "lmul": "Vector register grouping multiplier.",
    "mask": "Whether inactive vector elements are masked off.",
    "tail": "Vector tail/mask agnostic or undisturbed policy.",
    "footprint": "Address footprint shape such as same-line, cross-line, cross-page, misaligned, or partial overlap.",
    "vl": "Active vector length selection.",
    "elem_order": "Whether vector elements are modeled as single-event, ordered, or unordered.",
    "sync": "Fence/CMO/fence.i synchronization sequence around the operation.",
    "vm": "Virtual-memory context such as Sv39, ASID, global mapping, or satp switch.",
    "shootdown": "Local or remote TLB shootdown coverage point.",
    "pte": "PTE transition such as remap, permission flip, A/D update, or PBMT flip.",
    "alias": "VA/PA aliasing shape, including cacheable/NC aliases.",
    "stress": "Microarchitecture pressure scenario such as store-buffer, load-queue, miss-queue, replay, or ifetch patching.",
}


def feature_description_catalog() -> Dict[str, Dict[str, str]]:
    catalog = {feature: dict(values) for feature, values in sorted(FEATURE_DESCRIPTIONS.items())}
    catalog["params"] = dict(sorted(PARAM_DESCRIPTIONS.items()))
    return catalog


def describe_combination(combination: Combination) -> Dict[str, Any]:
    features = [
        _feature("skeleton", combination.skeleton),
        _feature("memory_event", combination.memory_event),
        _feature("attribute", combination.attribute),
    ]
    if combination.vector != "none":
        features.append(_feature("vector", combination.vector))
    if combination.cmo != "no_cmo":
        features.append(_feature("cmo", combination.cmo))
    if combination.tlb != "no_tlb":
        features.append(_feature("tlb", combination.tlb))
    for key, value in sorted(combination.params.items()):
        features.append(
            {
                "feature": key,
                "value": str(value),
                "test_description": PARAM_DESCRIPTIONS.get(key, f"Custom parameter {key} preserved for generator/harness interpretation."),
            }
        )
    return {"summary": _summary(combination), "features": features}


def _feature(feature: str, value: str) -> Dict[str, str]:
    description = FEATURE_DESCRIPTIONS.get(feature, {}).get(value)
    if description is None and value == "none":
        description = f"No {feature} feature is selected for this combination."
    if description is None:
        description = f"{feature}={value} is preserved in metadata for generator/harness interpretation."
    return {"feature": feature, "value": value, "test_description": description}


def _summary(combination: Combination) -> str:
    parts = [combination.skeleton]
    if combination.vector != "none":
        parts.append(combination.vector)
    if combination.cmo != "no_cmo":
        parts.append(combination.cmo)
    if combination.tlb != "no_tlb":
        parts.append(combination.tlb)
    if combination.attribute != "cacheable":
        parts.append(combination.attribute)
    if combination.params:
        params = ", ".join(f"{key}={value}" for key, value in sorted(combination.params.items()))
        parts.append(params)
    return " / ".join(parts) + " multi-hart litmus coverage point."
