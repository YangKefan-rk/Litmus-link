from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import litmus_link.herd_reference as reference
from litmus_link.herd_reference import (
    HerdCapabilities,
    HerdCapability,
    build_scalar_projections,
    capability_for_scalar_case,
    crosscheck_vector_projection,
    probe_case_herd_capabilities,
    probe_herd_capabilities,
)
from litmus_link.rvwmo_solver import EmbeddedVerdict
from litmus_link.toolchain import HerdVerdict
from litmus_link.vector_native import (
    VectorNativeDomain,
    lower_vector_assignment,
)
from litmus_link.verification import verify_path
from litmus_link.vector_solver import expand_vector_case


def _payload(**overrides):
    payload = {
        "skeletons": ["MP"],
        "mechanisms": ["po"],
        "endpoint_categories": ["scalar", "vector"],
        "endpoint_compositions": ["vector_scalar"],
        "scalar_widths": ["w"],
        "overlap_layouts": ["same_start"],
        "forms": ["unit_load", "unit_store"],
        "sew": ["e32"],
        "lmul": ["m1"],
        "index_eew": ["ei16"],
        "mask": ["unmasked"],
        "tail": ["ta_ma"],
        "vl": ["vl1"],
        "alignments": ["aligned"],
    }
    payload.update(overrides)
    return payload


def _case(**overrides):
    domain = VectorNativeDomain.from_payload(_payload(**overrides))
    assignment = domain.random_assignments(1, 17)[0]
    return lower_vector_assignment(assignment)


def _all_supported_capabilities() -> HerdCapabilities:
    supported = HerdCapability(True, "test capability")
    amo = {
        f"{operation}.{width}.{ordering}": supported
        for operation in ("swap", "add", "xor", "and", "or", "min", "max", "minu", "maxu")
        for width in ("w", "d")
        for ordering in ("relaxed", "aq", "rl", "aqrl")
    }
    return HerdCapabilities(
        True,
        supported,
        supported,
        supported,
        amo,
        "/test/herd7",
        "test",
        "/test/riscv.cat",
    )


def _embedded(allowed: bool) -> EmbeddedVerdict:
    return EmbeddedVerdict(
        "verified",
        "observable" if allowed else "forbidden",
        allowed,
        1,
        int(allowed),
        "test",
        0.0,
        (),
    )


def _herd(allowed: bool) -> HerdVerdict:
    return HerdVerdict(
        "observable" if allowed else "forbidden",
        allowed,
        "Sometimes" if allowed else "Never",
        int(allowed),
        int(not allowed),
        1,
        "exists (...) ",
        "",
    )


def test_real_capability_probe_checks_semantics_when_herd_is_present() -> None:
    probe_herd_capabilities.cache_clear()
    capabilities = probe_herd_capabilities()
    if not capabilities.available:
        pytest.skip("local herd7 is unavailable")
    assert capabilities.scalar.supported
    assert capabilities.amo_capability("add", 4, "aqrl").supported
    if capabilities.tool_version.startswith("7.58"):
        # This local version parses some mnemonics incompletely; semantic probes
        # must reject them rather than claiming a usable differential capability.
        assert not capabilities.amo_capability("min", 4, "aqrl").supported
        assert not capabilities.amo_capability("maxu", 8, "relaxed").supported
        assert not capabilities.mixed_size.supported
    assert "/tmp/ll-herd-" not in str(capabilities.to_json())


def test_vl1_projection_contains_no_vector_instruction_and_is_exact() -> None:
    case = _case()
    expansion = expand_vector_case(case.case_ir)
    bundle = build_scalar_projections(case.case_ir, expansion)
    assert bundle.status == "ready"
    assert bundle.exact
    assert bundle.oracle_kind == "vl1-exact"
    assert len(bundle.projections) == 1
    source = bundle.projections[0].source.lower()
    assert "vset" not in source
    assert "vle" not in source
    assert "vse" not in source
    assert "uint32_t" in source
    assert bundle.projections[0].variants == ()


def test_unordered_vl2_projection_enumerates_both_element_orders() -> None:
    case = _case(
        endpoint_categories=["vector"],
        endpoint_compositions=["vector_only"],
        forms=["indexed_unordered_load", "indexed_unordered_store"],
        vl=["vl2"],
    )
    expansion = expand_vector_case(case.case_ir)
    vector_count = len(expansion.instructions)
    bundle = build_scalar_projections(
        case.case_ir,
        expansion,
        max_projections=2**vector_count,
    )
    assert bundle.status == "ready"
    assert bundle.exact
    assert bundle.oracle_kind == "unordered-permutation-exact"
    assert len(bundle.projections) == 2**vector_count
    orders = {
        tuple(tuple(order) for _parent, order in sorted(projection.element_orders.items()))
        for projection in bundle.projections
    }
    assert any((0, 1) in order for order in orders)
    assert any((1, 0) in order for order in orders)


def test_ordered_vl2_projection_is_advisory() -> None:
    case = _case(
        endpoint_categories=["vector"],
        endpoint_compositions=["vector_only"],
        forms=["indexed_ordered_load", "indexed_ordered_store"],
        vl=["vl2"],
    )
    expansion = expand_vector_case(case.case_ir)
    bundle = build_scalar_projections(case.case_ir, expansion)
    assert bundle.status == "ready"
    assert not bundle.exact
    assert bundle.oracle_kind == "ordered-fixed-advisory"
    assert len(bundle.projections) == 1


def test_projection_limit_reports_external_unsupported() -> None:
    case = _case(
        endpoint_categories=["vector"],
        endpoint_compositions=["vector_only"],
        forms=["unit_load", "unit_store"],
        vl=["vl4"],
    )
    expansion = expand_vector_case(case.case_ir)
    bundle = build_scalar_projections(case.case_ir, expansion, max_projections=8)
    assert bundle.status == "external_unsupported"
    assert bundle.requested_projections > 8


def test_crosscheck_agreement_and_conflict_are_explicit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    case = _case()
    expansion = expand_vector_case(case.case_ir)
    monkeypatch.setattr(reference, "_judge_projection", lambda *_args: _herd(True))
    agree = crosscheck_vector_projection(
        case.case_ir,
        expansion,
        _embedded(True),
        capabilities=_all_supported_capabilities(),
    )
    conflict = crosscheck_vector_projection(
        case.case_ir,
        expansion,
        _embedded(False),
        capabilities=_all_supported_capabilities(),
    )
    assert agree["status"] == "agree"
    assert conflict["status"] == "conflict"
    assert conflict["allowed"] is True


def test_ordered_advisory_disagreement_does_not_become_conflict(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    case = _case(
        endpoint_categories=["vector"],
        endpoint_compositions=["vector_only"],
        forms=["indexed_ordered_load", "indexed_ordered_store"],
        vl=["vl2"],
    )
    expansion = expand_vector_case(case.case_ir)
    monkeypatch.setattr(reference, "_judge_projection", lambda *_args: _herd(True))
    result = crosscheck_vector_projection(
        case.case_ir,
        expansion,
        _embedded(False),
        capabilities=_all_supported_capabilities(),
    )
    assert result["status"] == "advisory_disagree"


@pytest.mark.parametrize("ordered", [False, True])
def test_verify_herd7_only_verifies_exact_vector_projection(
    ordered: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    case = (
        _case(
            endpoint_categories=["vector"],
            endpoint_compositions=["vector_only"],
            forms=["indexed_ordered_load", "indexed_ordered_store"],
            vl=["vl2"],
        )
        if ordered
        else _case()
    )
    litmus = tmp_path / f"{case.name}.litmus"
    litmus.write_text(case.litmus, encoding="utf-8")
    litmus.with_suffix(".meta.json").write_text(
        json.dumps({"case_ir": case.case_ir.to_json()}), encoding="utf-8"
    )
    monkeypatch.setattr(
        reference, "probe_case_herd_capabilities", lambda *_args, **_kwargs: _all_supported_capabilities()
    )

    def judge(source: str, variants: tuple[str, ...], timeout: int) -> HerdVerdict:
        assert "vset" not in source
        assert timeout == 17
        return _herd(False)

    monkeypatch.setattr(reference, "_judge_projection", judge)
    result = verify_path(litmus, backend="herd7", timeout=17)["results"][0]["result"]
    if ordered:
        assert result["status"] in {"advisory_agree", "advisory_disagree"}
        assert result["allowed"] is None
        assert result["results"][0]["verdict"] == "forbidden"
    else:
        assert result["status"] == "verified"
        assert result["verdict"] == "forbidden"


def test_missing_mixed_capability_preserves_embedded_scope() -> None:
    case = _case(
        scalar_widths=["d"],
        sew=["e32"],
        overlap_layouts=["same_start"],
    )
    expansion = expand_vector_case(case.case_ir)
    capabilities = replace(
        _all_supported_capabilities(),
        mixed_size=HerdCapability(False, "RISC-V mixed unavailable"),
    )
    result = crosscheck_vector_projection(
        case.case_ir,
        expansion,
        _embedded(True),
        capabilities=capabilities,
    )
    assert result["status"] == "external_unsupported"
    assert result["allowed"] is None
    assert "mixed unavailable" in result["reason"]


def test_scalar_case_capability_checks_required_amo_semantics() -> None:
    case = _case(
        endpoint_categories=["amo", "vector"],
        endpoint_compositions=["vector_amo"],
        amo_ops=["maxu"],
        amo_widths=["d"],
        amo_orderings=["aqrl"],
    )
    capabilities = _all_supported_capabilities()
    denied = dict(capabilities.amo)
    denied["maxu.d.aqrl"] = HerdCapability(False, "unsigned max unsupported")
    capability = capability_for_scalar_case(
        case.case_ir,
        capabilities=replace(capabilities, amo=denied),
    )
    assert not capability.supported
    assert capability.reason == "unsigned max unsupported"


def test_interactive_capability_probe_only_checks_required_amo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    case = _case(
        endpoint_categories=["amo", "vector"],
        endpoint_compositions=["vector_amo"],
        amo_ops=["xor"],
        amo_widths=["d"],
        amo_orderings=["rl"],
    )
    probes = []

    def fake_probe(source, variants=()):  # type: ignore[no-untyped-def]
        probes.append((source, variants))
        return HerdCapability(True, "probed", variants=variants)

    herd = tmp_path / "herd7"
    model = tmp_path / "riscv.cat"
    herd.touch()
    model.touch()
    monkeypatch.setattr(reference.toolchain, "HERD", herd)
    monkeypatch.setattr(reference.toolchain, "RISCV_CAT", model)
    monkeypatch.setattr(reference.toolchain, "tool_version", lambda _path: "test")
    monkeypatch.setattr(reference, "_probe", fake_probe)
    capabilities = probe_case_herd_capabilities(
        case.case_ir,
        mixed_size=False,
    )

    assert capabilities.scalar.supported
    assert set(capabilities.amo) == {"xor.d.rl"}
    assert len(probes) == 2
    assert any("amoxor.d.rl" in source for source, _variants in probes)
    assert all(variants == () for _source, variants in probes)
