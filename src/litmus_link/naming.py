from __future__ import annotations

"""Human-readable, stable names for generated Litmus tests.

The visible name describes the critical relation family and semantic
instruction modifiers.  Generation-only defaults and full axis provenance
remain in metadata instead of being repeated in every filename.
"""

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


MAX_LITMUS_NAME_LEN = 112

_VECTOR_FORMS = {
    "unit_load": "vle{sew}.v",
    "unit_store": "vse{sew}.v",
    "strided_load": "vlse{sew}.v",
    "strided_store": "vsse{sew}.v",
    "indexed_ordered_load": "vloxei{index}.v-E{sew}",
    "indexed_unordered_load": "vluxei{index}.v-E{sew}",
    "indexed_ordered_store": "vsoxei{index}.v-E{sew}",
    "indexed_unordered_store": "vsuxei{index}.v-E{sew}",
}

_VECTOR_EVENT_FORMS = {
    "unit_load": "VLE{sew}",
    "unit_store": "VSE{sew}",
    "strided_load": "VLSE{sew}",
    "strided_store": "VSSE{sew}",
    "indexed_ordered_load": "VLOXEI{index}/E{sew}",
    "indexed_unordered_load": "VLUXEI{index}/E{sew}",
    "indexed_ordered_store": "VSOXEI{index}/E{sew}",
    "indexed_unordered_store": "VSUXEI{index}/E{sew}",
}

_VECTOR_ALIGNMENT_NAMES = {
    "misalign_same16": "U16",
    "misalign_cross16": "X16",
    "misalign_cross64": "X64",
}

_CMO_NAMES = {
    "clean": "cbo.clean",
    "flush": "cbo.flush",
    "inval": "cbo.inval",
    "zero": "cbo.zero",
}

_TLB_NAMES = {
    "local_sfence": "sfence.vma.local",
    "remote_sfence": "sfence.vma.remote",
    "pte_remap": "PTE.Remap",
    "permission_fault": "PTE.Permission",
    "ad_update": "PTE.AD",
    "asid_global": "ASID.Global",
    "satp_switch": "SATP.Switch",
}

_ATTRIBUTE_NAMES = {
    "pbmt_nc": "PBMT.NC",
    "pbmt_io": "PBMT.IO",
    "nc_alias": "NC.Alias",
    "cacheable_nc_alias": "C.NC.Alias",
}

_VARIANT_NAMES = {
    "base": "",
    "fence_rw_rw": "fence.rw.rw",
    "fence_w_w_r_rw": "fence.w.w+fence.r.rw",
    "addr_dep": "addr",
    "ctrl_dep": "ctrl",
    "ctrl_fencei": "ctrlfencei",
}

_DEFAULT_VECTOR_ENDPOINTS = {
    "MP": {"vector_load": "p1_rx", "vector_store": "p0_wx"},
    "LB": {"vector_load": "p0_rx", "vector_store": "p0_wy"},
    "SB": {"vector_load": "p0_ry", "vector_store": "p0_wx"},
    "WRC": {"vector_load": "p1_rx", "vector_store": "p0_wx"},
    "RWC": {"vector_load": "p1_rx", "vector_store": "p0_wx"},
    "IRIW": {"vector_load": "p2_rx", "vector_store": "p0_wx"},
    "ISA2": {"vector_load": "p1_ry", "vector_store": "p0_wx"},
    "R": {"vector_load": "p1_rx", "vector_store": "p0_wx"},
    "S": {"vector_load": "p1_rx", "vector_store": "p0_wy"},
    "Co": {"vector_load": "p1_rx_new", "vector_store": "p0_wx"},
}

_SEMANTIC_PARAM_NAMES = {
    "sync": {
        "pre_fence": "Sync.PreFence",
        "post_fence": "Sync.PostFence",
        "full_alias_sync": "Sync.AliasFlush",
        "fence_i_after": "Sync.FenceI",
    },
    "vm": {
        "sv39": "Sv39",
        "sv39_asid": "Sv39.ASID",
        "sv39_global": "Sv39.Global",
        "satp_switch": "SATP.Switch",
    },
    "shootdown": {
        "local": "Shootdown.Local",
        "remote": "Shootdown.Remote",
        "remote_ipi": "Shootdown.RemoteIPI",
    },
    "pte": {
        "valid": "PTE.Valid",
        "invalid": "PTE.Invalid",
        "remap": "PTE.Remap",
        "permission_flip": "PTE.Permission",
        "ad_update": "PTE.AD",
        "pbmt_flip": "PTE.PBMT",
    },
}


def combination_name(combination: Any) -> str:
    """Return a semantic base name shared by all ordering variants."""
    tokens = [_token(str(combination.skeleton))]
    consumed: set[str] = set()

    vector = str(combination.vector)
    if vector != "none":
        tokens.extend(
            _vector_tokens(
                vector,
                combination.params,
                skeleton=str(combination.skeleton),
                memory_event=str(combination.memory_event),
            )
        )
        consumed.update(
            {
                "sew",
                "lmul",
                "index_eew",
                "mask",
                "tail",
                "vl",
                "vector_event",
                "footprint",
                "stride_bytes",
            }
        )

    cmo = str(combination.cmo)
    if cmo != "no_cmo":
        tokens.append(_CMO_NAMES.get(cmo, f"CMO.{_token(cmo)}"))

    tlb = str(combination.tlb)
    if tlb != "no_tlb":
        tokens.append(_TLB_NAMES.get(tlb, f"TLB.{_token(tlb)}"))

    attribute = str(combination.attribute)
    if attribute != "cacheable":
        tokens.append(_ATTRIBUTE_NAMES.get(attribute, _token(attribute)))

    explicit_ordering, ordering_key = _explicit_ordering(combination.params)
    if explicit_ordering:
        tokens.append(explicit_ordering)
    if ordering_key:
        consumed.add(ordering_key)

    for key, names in _SEMANTIC_PARAM_NAMES.items():
        if key not in combination.params:
            continue
        value = str(combination.params[key])
        label = names.get(value)
        if label:
            tokens.append(label)
            consumed.add(key)

    # These fields are already represented by the family/event tokens or are
    # generation defaults that do not change the emitted instruction stream.
    remaining = {
        str(key): value
        for key, value in combination.params.items()
        if str(key) not in consumed and not _default_param(str(key), value)
    }
    if remaining:
        tokens.append(f"Cfg.{_digest(remaining, 10)}")
    return _bounded("+".join(token for token in tokens if token))


def case_name(combination: Any, variant: str) -> str:
    base = combination_name(combination)
    _explicit_ordering_name, ordering_key = _explicit_ordering(combination.params)
    if ordering_key:
        return base
    suffix = variant_name(variant)
    return _bounded(base if not suffix else f"{base}+{suffix}")


def case_display_name(combination: Any, variant: str) -> str:
    return case_name(combination, variant)


def native_cycle_name(
    family: str,
    edges: Sequence[Any],
    labels: Sequence[str],
    annotations: Sequence[str],
) -> str:
    """Name native cycles using diy/litmus relation vocabulary."""
    root = _token(family or "Cycle")
    if family and family != "DIY":
        relation_tokens = [
            _native_local_token(edge)
            for edge in edges
            if str(edge.scope) == "local"
        ]
    else:
        relation_tokens = [_token(label) for label in labels]
    if not relation_tokens:
        relation_tokens = [_token(label) for label in labels]
    tokens = [root, *relation_tokens]
    if any(annotation != "P" for annotation in annotations):
        tokens.append("Ann." + "-".join(_token(annotation) for annotation in annotations))
    return _bounded("+".join(tokens))


def vector_native_case_identity(
    family: str,
    cycle: Mapping[str, Any],
    relation_labels: Sequence[str],
    endpoint_choices: Sequence[Mapping[str, Any]],
    alignment: str,
) -> dict[str, Any]:
    """Build separate human and machine identities for a Vector cycle case."""
    canonical = {
        "schema": "litmus-link.vector-case-identity.v1",
        "family": family,
        "cycle": dict(cycle),
        "relation_labels": list(relation_labels),
        "endpoint_choices": [dict(choice) for choice in endpoint_choices],
        "alignment": alignment,
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    machine_name = f"LLV-{_token(family or 'Cycle')}-{digest}"
    vector_events = [
        f"E{index}:{_vector_event_name(choice, alignment)}"
        for index, choice in enumerate(endpoint_choices)
        if str(choice.get("category", "")) == "vector"
    ]
    relation_ring = ">".join(str(label) for label in relation_labels)
    display_name = (
        f"{family or 'Cycle'}+{{{relation_ring}}}+V{{{','.join(vector_events)}}}"
    )
    return {
        "machine_name": machine_name,
        "file_name": f"{machine_name}.litmus",
        "display_name": display_name,
        "sha256": digest,
        "canonical": canonical,
    }


def _vector_event_name(choice: Mapping[str, Any], alignment: str) -> str:
    form = str(choice.get("vector_form", ""))
    params = choice.get("params")
    values = params if isinstance(params, Mapping) else {}
    sew = str(values.get("sew", "e32")).removeprefix("e")
    index = str(values.get("index_eew", "ei32")).removeprefix("ei")
    template = _VECTOR_EVENT_FORMS.get(form, _token(form).upper())
    name = template.format(sew=sew, index=index)
    alignment_name = _VECTOR_ALIGNMENT_NAMES.get(alignment)
    return f"{name}/{alignment_name}" if alignment_name else name


def variant_name(variant: str) -> str:
    effective = _effective_variant(variant)
    return _VARIANT_NAMES.get(effective, _token(effective))


def _vector_tokens(
    form: str,
    params: Mapping[str, Any],
    *,
    skeleton: str,
    memory_event: str,
) -> list[str]:
    sew = str(params.get("sew", "e32")).removeprefix("e")
    index = str(params.get("index_eew", "ei32")).removeprefix("ei")
    mnemonic = _VECTOR_FORMS.get(form, _token(form)).format(sew=sew, index=index)
    endpoint_value = params.get("vector_event")
    if endpoint_value is None:
        endpoint_value = _DEFAULT_VECTOR_ENDPOINTS.get(skeleton, {}).get(memory_event)
    endpoint = _endpoint(str(endpoint_value)) if endpoint_value is not None else ""
    tokens = [f"{mnemonic}-{endpoint}" if endpoint else mnemonic]
    tokens.append(str(params.get("lmul", "m1")).upper())
    tokens.append(str(params.get("vl", "vlmax")).upper())
    tail = str(params.get("tail", "ta_ma")).replace("_", ".").upper()
    tokens.append(tail)
    if str(params.get("mask", "unmasked")) == "masked":
        tokens.append("Mask")
    footprint = str(params.get("footprint", "same_line"))
    if footprint not in {"", "same_line"}:
        tokens.append(_token(footprint))
    if params.get("stride_bytes") is not None:
        tokens.append(f"Stride{_token(str(params['stride_bytes']))}")
    return tokens


def _native_local_token(edge: Any) -> str:
    relation = str(edge.relation)
    if relation == "po":
        token = "pos" if str(edge.location) == "same" else "po"
        return f"{token}.{_token(str(edge.shape))}"
    if relation == "fence":
        lowering = str(edge.lowering).replace(" ", ".").replace(",", ".")
        token = _token(lowering)
        token += "s" if str(edge.location) == "same" else ""
        return f"{token}.{_token(str(edge.shape))}"
    if relation == "dependency":
        token = {
            "addr": "addr",
            "data": "data",
            "ctrl": "ctrl",
            "ctrl_fencei": "ctrlfencei",
        }.get(str(edge.mechanism), _token(str(edge.mechanism)))
        token += "s" if str(edge.location) == "same" else ""
        return f"{token}.{_token(str(edge.shape))}"
    return _token(str(edge.label))


def _endpoint(value: str) -> str:
    match = re.fullmatch(r"p(\d+)_(r|w)([a-z][a-z0-9_]*)", value, re.IGNORECASE)
    if not match:
        return _token(value)
    proc, direction, location = match.groups()
    return f"P{proc}.{direction.upper()}{location.lower()}"


def _effective_variant(variant: str) -> str:
    if variant.startswith(("width-", "outcome-", "stress-")):
        return "base"
    if not variant.startswith("dep-"):
        return variant
    match = re.match(r"dep-(.+?)(?:_width-|_outcome-|_stress-|$)", variant)
    if not match:
        return variant
    return {
        "addr": "addr_dep",
        "ctrl": "ctrl_dep",
        "ctrl_fence": "ctrl_fencei",
        "none": "base",
        "data": "base",
        "aq": "base",
        "rl": "base",
        "aqrl": "base",
    }.get(match.group(1), match.group(1))


def _explicit_ordering(params: Mapping[str, Any]) -> tuple[str, str]:
    if "variant" in params:
        suffix = variant_name(str(params["variant"]))
        return suffix, "variant"
    if "dep" in params:
        dep = str(params["dep"])
        suffix = variant_name(f"dep-{dep}")
        if suffix:
            return suffix, "dep"
    return "", ""


def _default_param(key: str, value: Any) -> bool:
    return (key, str(value)) in {
        ("sync", "none"),
        ("stress", "none"),
        ("alias", "none"),
        ("vm", "bare"),
        ("shootdown", "none"),
        ("pte", "stable"),
    }


def _token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", ".", value).strip("._-")
    return cleaned or "x"


def _digest(value: Any, length: int) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:length]


def _bounded(name: str) -> str:
    if len(name) <= MAX_LITMUS_NAME_LEN:
        return name
    suffix = f"+ID.{_digest(name, 16)}"
    return name[: MAX_LITMUS_NAME_LEN - len(suffix)].rstrip("+._-") + suffix
