from __future__ import annotations

import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Callable, Dict, Iterable

from .amo import AMO_OPERATIONS, AMO_ORDERINGS
from .corpus_ir import corpus_to_ir
from .corpus_riscv import parse_litmus
from .diagram import DIAGRAM_RENDER_VERSION, diagram_summary, render_diagram
from .litmus_ir import LitmusCaseIR
from .models import GENERATED, Combination
from .native_scalar import (
    DEFAULT_NATIVE_MECHANISMS,
    NATIVE_ANNOTATIONS,
    generate_native_diy,
    generate_native_relations,
    generate_native_templates,
    native_catalog,
    native_template_audit,
)
from .native_diy import (
    DEFAULT_DIY_RELAX,
    DEFAULT_DIY_SAFE,
    DIY_MODES,
    DIY_OBSERVERS,
    DIY_OBSERVER_TYPES,
    DiyConfig,
)
from .memory_layout import (
    ATOMIC_OVERLAPS,
    MEMORY_LAYOUT_MODES,
    MISALIGNED_BOUNDARIES,
    MISALIGNED_WIDTHS,
    expand_memory_layouts,
)
from .profiles import (
    SKELETONS,
    VECTOR_INDEX_EEWS,
    VECTOR_LENGTHS,
    VECTOR_LMULS,
    VECTOR_MASKS,
    VECTOR_OPS,
    VECTOR_TAILS,
    VECTOR_WIDTHS,
)
from .scalar import (
    DEFAULT_MECHANISMS,
    DEFAULT_RELAX_EDGES,
    DEFAULT_SAFE_EDGES,
    generate_scalar_cross,
    generate_scalar_enumerated,
)
from .toolchain import HerdVerdict
from .vector_native import (
    AMO_WIDTHS,
    ENDPOINT_CATEGORIES,
    ENDPOINT_COMPOSITIONS,
    FUSION_OVERLAP_LAYOUTS,
    SCALAR_WIDTHS,
    VECTOR_ALIGNMENTS,
    VECTOR_GENERATION_MODES,
    VECTOR_SAMPLE_MODES,
    VectorNativeDomain,
    generate_vector_cases,
    sample_vector_cases,
)


PARAM_AXIS_VALUES: Dict[str, list[str]] = {
    "sew": list(VECTOR_WIDTHS),
    "index_eew": list(VECTOR_INDEX_EEWS),
    "lmul": list(VECTOR_LMULS),
    "mask": list(VECTOR_MASKS),
    "tail": list(VECTOR_TAILS),
    "vl": list(VECTOR_LENGTHS),
}


def options_payload() -> Dict[str, Any]:
    return {
        "axes": {
            "skeleton": list(SKELETONS),
            "vector": ["none", *VECTOR_OPS],
        },
        "param_axes": PARAM_AXIS_VALUES,
        "native_scalar": {
            **native_catalog(),
            "diy": {
                "safe": list(DEFAULT_DIY_SAFE),
                "relax": list(DEFAULT_DIY_RELAX),
                "modes": list(DIY_MODES),
                "observers": list(DIY_OBSERVERS),
                "observer_types": list(DIY_OBSERVER_TYPES),
            },
            "solver_backends": ["embedded", "herd7", "crosscheck"],
            "memory_layout": {
                "modes": list(MEMORY_LAYOUT_MODES),
                "width_bits": [value * 8 for value in MISALIGNED_WIDTHS],
                "boundaries": list(MISALIGNED_BOUNDARIES),
                "atomic_overlaps": list(ATOMIC_OVERLAPS),
                "atomicity_model": "byte_level_no_mag",
                "aligned_atomic_models": ["aligned_atomic", "mixed_size_atomic"],
                "mag_supported": False,
            },
        },
        "vector_native": {
            "mechanisms": list(DEFAULT_NATIVE_MECHANISMS),
            # Kept until the Qt control migration in the next GUI phase.  The
            # backend translates this legacy axis into independent categories.
            "endpoint_modes": list(NATIVE_ANNOTATIONS),
            "endpoint_categories": list(ENDPOINT_CATEGORIES),
            "endpoint_compositions": list(ENDPOINT_COMPOSITIONS),
            "scalar_widths": list(SCALAR_WIDTHS),
            "amo_ops": list(AMO_OPERATIONS),
            "amo_widths": list(AMO_WIDTHS),
            "amo_orderings": list(AMO_ORDERINGS),
            "overlap_layouts": list(FUSION_OVERLAP_LAYOUTS),
            "alignments": list(VECTOR_ALIGNMENTS),
            "preview_sampling_modes": list(VECTOR_SAMPLE_MODES),
            "generation_modes": list(VECTOR_GENERATION_MODES),
        },
    }


def preview_payload(
    payload: Dict[str, Any],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    mode = str(payload.get("mode", "scalar"))
    if mode == "scalar":
        return _scalar_preview_payload(payload, progress_callback=progress_callback)
    if mode == "vector":
        return _vector_native_preview_payload(payload, progress_callback=progress_callback)
    raise ValueError("GUI workflow supports only scalar and vector modes")


_DIAGRAM_DIR = Path(gettempdir()) / "litmus-link-preview-diagrams"


def _deferred_preview_diagram(
    case_ir: LitmusCaseIR,
    solver: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Describe a preview diagram without paying the PNG rendering cost."""
    fingerprint = sha256(
        json.dumps(
            {
                "diagram_renderer": DIAGRAM_RENDER_VERSION,
                "case_ir": case_ir.to_json(),
                "solver": solver or {},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    cache_dir = _DIAGRAM_DIR / fingerprint
    png_path = cache_dir / f"{case_ir.name}.diagram.png"
    summary = diagram_summary(case_ir, solver, png_path)
    summary["status"] = "ready" if png_path.exists() else "deferred"
    summary["cache_key"] = fingerprint
    return summary


def materialize_preview_diagram(item: Dict[str, Any]) -> Dict[str, Any]:
    """Render one preview diagram on demand and reuse a content-addressed cache."""
    case_ir_json = item.get("case_ir")
    if not isinstance(case_ir_json, dict):
        raise ValueError("this preview case has no case IR to draw")
    solver = item.get("solver") if isinstance(item.get("solver"), dict) else None
    case_ir = _case_ir_from_json(case_ir_json)
    descriptor = _deferred_preview_diagram(case_ir, solver)
    png_path = Path(str(descriptor["png"]))
    json_path = png_path.with_suffix("").with_suffix(".diagram.json")
    if png_path.exists() and json_path.exists():
        descriptor["status"] = "ready"
        descriptor["cached"] = True
        return descriptor

    result = render_diagram(case_ir, solver, png_path.parent)
    summary = dict(result.summary)
    summary["status"] = "ready"
    summary["cache_key"] = descriptor["cache_key"]
    summary["cached"] = False
    return summary


def _scalar_preview_payload(
    payload: Dict[str, Any],
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    sample_limit = _positive_int(payload.get("sample_limit", 1000), "sample_limit")
    with TemporaryDirectory(prefix="litmus-link-scalar-preview-") as temporary:
        preview_dir = Path(temporary)
        report = _run_scalar_generator(
            payload,
            preview_dir,
            limit=sample_limit,
            progress_callback=progress_callback,
        )
        sample = _scalar_preview_items(preview_dir, report)
    report = dict(report)
    report["displayed_litmus"] = len(sample)
    report["profile"] = "scalar"
    report["total_combinations"] = report.get("available_litmus", 0)
    report["generated"] = report.get("generated_litmus", 0)
    report["excluded_illegal"] = 0
    report["excluded_unsupported"] = 0
    report["hand_required"] = 0
    report["missing"] = 0
    return {
        "profile": "scalar",
        "source": report.get("generator", {}).get("engine", "herdtools7"),
        "report": report,
        "available_litmus": report.get("available_litmus", 0),
        "displayed_litmus": len(sample),
        "generation_limited": report.get("generation_limited", False),
        "sample": sample,
        "classification_counts": _preview_classification_counts(sample),
    }


def _vector_native_preview_payload(
    payload: Dict[str, Any],
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    compute_verdicts = bool(payload.get("compute_verdicts", True))
    cases, audit = sample_vector_cases(
        payload,
        compute_verdicts=compute_verdicts,
        progress_callback=progress_callback,
    )
    sample = [
        _preview_item(
            case.combination,
            case.decision.to_json(),
            case.name,
            case.litmus,
            case.case_ir.to_json() if case.case_ir is not None else None,
            dict(case.solver or {}),
            _deferred_preview_diagram(case.case_ir, dict(case.solver or {}))
            if case.case_ir is not None
            else None,
        )
        for case in cases
    ]
    report = {
        "schema": "litmus-link.vector-native-preview.v2",
        "profile": "vector-native",
        "total_combinations": audit["total_cases"],
        "generated": audit["total_cases"],
        "generated_litmus": audit["total_cases"],
        "displayed_litmus": len(sample),
        "raw_combinations": audit.get("raw_combinations", audit["total_cases"]),
        "excluded_illegal": audit.get("excluded_illegal", 0),
        "excluded_unsupported": audit.get("excluded_unsupported", 0),
        "excluded": audit.get("excluded", {}),
        "hand_required": 0,
        "missing": 0,
        "sampling": audit["sampling"],
        "random_seed": audit["sample_seed"],
        "relation_cycles": audit["relation_cycles"],
    }
    return {
        "profile": "vector-native",
        "source": "litmus-link-native-cycle+rvv",
        "report": report,
        "available_litmus": audit["total_cases"],
        "displayed_litmus": len(sample),
        "sample": sample,
        "classification_counts": _preview_classification_counts(sample),
        "domain_classification_counts": {
            "domain_cases": audit["total_cases"],
            "generated_cases": audit["total_cases"],
            "relation_cycles": audit["relation_cycles"],
            "read_endpoint_choices": audit["read_endpoint_choices"],
            "write_endpoint_choices": audit["write_endpoint_choices"],
            "sampling": audit["sampling"],
            "excluded": audit.get("excluded", {}),
        },
        "audit": audit,
    }


def _run_scalar_generator(
    payload: Dict[str, Any],
    out_dir: Path,
    *,
    limit: int | None = None,
    judge: bool | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    engine = str(payload.get("engine", "cross"))
    selected_limit = _optional_positive_int(payload.get("limit")) if limit is None else limit
    selected_judge = bool(payload.get("judge", True)) if judge is None else judge
    solver_backend = str(payload.get("solver_backend", "embedded"))
    timeout = _positive_int(payload.get("timeout", 180), "timeout")
    memory_layouts = _memory_layouts_from_payload(payload)
    if engine == "native_templates":
        presets = _string_list(payload.get("skeletons")) or ["MP"]
        mechanisms = _string_list(payload.get("mechanisms")) or list(DEFAULT_NATIVE_MECHANISMS)
        return generate_native_templates(
            out_dir=out_dir,
            presets=presets,
            mechanisms=mechanisms,
            include_same=bool(payload.get("include_same", True)),
            annotations=_string_list(payload.get("annotations")) or NATIVE_ANNOTATIONS,
            limit=selected_limit,
            judge=selected_judge,
            solver_backend=solver_backend,
            diagrams=bool(payload.get("diagrams", False)),
            timeout=timeout,
            memory_layouts=memory_layouts,
            progress_callback=progress_callback,
        )
    if engine == "native_cycles":
        mechanisms = _string_list(payload.get("mechanisms")) or list(DEFAULT_NATIVE_MECHANISMS)
        return generate_native_relations(
            out_dir=out_dir,
            mechanisms=["communication", *mechanisms],
            include_same=bool(payload.get("include_same", True)),
            include_internal=bool(payload.get("include_internal", True)),
            min_size=_positive_int(payload.get("min_size", 2), "min_size"),
            max_size=_positive_int(payload.get("size", 4), "size"),
            max_procs=_positive_int(payload.get("nprocs", 2), "nprocs"),
            exact_procs=bool(payload.get("exact_procs", False)),
            max_accesses_per_proc=_optional_positive_int(payload.get("max_accesses_per_proc")),
            annotations=_string_list(payload.get("annotations")) or ("P",),
            limit=selected_limit,
            judge=selected_judge,
            solver_backend=solver_backend,
            diagrams=bool(payload.get("diagrams", False)),
            timeout=timeout,
            memory_layouts=memory_layouts,
            progress_callback=progress_callback,
        )
    if engine == "native_diy":
        diy = payload.get("diy", {}) if isinstance(payload.get("diy"), dict) else {}
        return generate_native_diy(
            out_dir=out_dir,
            config=DiyConfig(
                safe=tuple(_string_list(diy.get("safe")) or DEFAULT_DIY_SAFE),
                relax=tuple(_string_list(diy.get("relax")) or DEFAULT_DIY_RELAX),
                reject=tuple(_string_list(diy.get("reject"))),
                prefixes=tuple(
                    tuple(str(token) for token in prefix)
                    for prefix in diy.get("prefixes", [])
                    if isinstance(prefix, (list, tuple))
                ),
                min_size=_positive_int(payload.get("min_size", 2), "min_size"),
                size=_positive_int(payload.get("size", 4), "size"),
                nprocs=_positive_int(payload.get("nprocs", 2), "nprocs"),
                exact_procs=bool(payload.get("exact_procs", False)),
                upto=not bool(diy.get("exact_size", False)),
                mode=str(diy.get("mode", "default")),
                mix=bool(diy.get("mix", False)),
                min_relax=int(diy.get("min_relax", 1)),
                max_relax=int(diy.get("max_relax", 1)),
                max_accesses_per_proc=_optional_positive_int(payload.get("max_accesses_per_proc")),
                include_same=bool(payload.get("include_same", False)),
                include_internal=bool(payload.get("include_internal", True)),
                observer=str(diy.get("observer", "avoid")),
                observer_type=str(diy.get("observer_type", "straight")),
                realdep=bool(diy.get("realdep", False)),
                unrollatomic=_optional_nonnegative_int(diy.get("unrollatomic")),
                moreedges=bool(diy.get("moreedges", False)),
            ),
            annotations=_string_list(payload.get("annotations")) or ("P",),
            limit=selected_limit,
            judge=selected_judge,
            solver_backend=solver_backend,
            diagrams=bool(payload.get("diagrams", False)),
            timeout=timeout,
            memory_layouts=memory_layouts,
            progress_callback=progress_callback,
        )
    if engine == "cross":
        presets = _string_list(payload.get("skeletons")) or ["MP"]
        mechanisms = _string_list(payload.get("mechanisms")) or list(DEFAULT_MECHANISMS)
        return generate_scalar_cross(
            out_dir=out_dir,
            presets=presets,
            mechanisms=mechanisms,
            limit=selected_limit,
            judge=selected_judge,
            timeout=timeout,
        )
    if engine != "enumerate":
        raise ValueError(f"unknown scalar engine: {engine}")
    return generate_scalar_enumerated(
        out_dir=out_dir,
        safe=_string_list(payload.get("safe")) or DEFAULT_SAFE_EDGES,
        relax=_string_list(payload.get("relax")) or DEFAULT_RELAX_EDGES,
        size=_positive_int(payload.get("size", 4), "size"),
        nprocs=_positive_int(payload.get("nprocs", 2), "nprocs"),
        exact=bool(payload.get("exact", False)),
        one=bool(payload.get("one", False)),
        mode=str(payload.get("enumerate_mode", "default")),
        obstype=str(payload.get("obstype", "fenced")),
        realdep=bool(payload.get("realdep", False)),
        moreedges=bool(payload.get("moreedges", False)),
        unrollatomic=_optional_positive_int(payload.get("unrollatomic")),
        limit=selected_limit,
        judge=selected_judge,
        timeout=timeout,
    )


def _memory_layouts_from_payload(payload: Dict[str, Any]):
    raw = payload.get("memory_layout")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return expand_memory_layouts(("aligned",))
    modes = _string_list(raw.get("modes"))
    if not modes:
        raise ValueError("select at least one scalar memory layout mode")
    if bool(raw.get("include_aligned", True)):
        modes = ["aligned", *modes]
    width_bits = [int(value) for value in raw.get("width_bits", ())]
    if not width_bits:
        raise ValueError("select at least one misaligned access width")
    if any(value not in {16, 32, 64} for value in width_bits):
        raise ValueError("misaligned width_bits must contain only 16, 32, or 64")
    boundaries = _string_list(raw.get("boundaries"))
    if not boundaries:
        raise ValueError("select at least one misaligned address boundary")
    atomic_overlaps = _string_list(raw.get("atomic_overlaps")) or list(ATOMIC_OVERLAPS)
    return expand_memory_layouts(
        modes,
        widths=tuple(value // 8 for value in width_bits),
        boundaries=boundaries,
        atomic_overlaps=atomic_overlaps,
    )


def _scalar_preview_items(out_dir: Path, report: Dict[str, Any]) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    generator = report.get("generator", {})
    for entry in (out_dir / "@all").read_text(encoding="utf-8").splitlines():
        if not entry.strip():
            continue
        litmus_path = out_dir / entry.strip()
        litmus = litmus_path.read_text(encoding="utf-8")
        test = parse_litmus(litmus, str(litmus_path))
        meta_path = litmus_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        solver_path = litmus_path.with_suffix(".solver.json")
        solver = json.loads(solver_path.read_text(encoding="utf-8"))
        verdict = _herd_verdict_from_scalar_solver(solver)
        ir = (
            _case_ir_from_json(meta["case_ir"])
            if meta.get("schema") == "litmus-link.native-scalar-meta.v1"
            else corpus_to_ir(test, verdict)
        )
        diagram = _deferred_preview_diagram(ir, solver)
        combination = Combination(
            profile="scalar",
            category="scalar_rvwmo",
            skeleton=ir.skeleton,
            memory_event="scalar_pair",
            attribute="cacheable",
            params={"engine": generator.get("engine", "herdtools7")},
        )
        native_generated = meta.get("generated_from") == "litmus-link-native"
        decision = {
            "status": GENERATED,
            "reason": (
                "Scalar RVWMO test exhaustively generated by Litmus-link's native relation engine."
                if native_generated
                else "Scalar RVWMO test generated by the official herdtools7 toolchain."
            ),
            "rvwmo_class": "rvwmo-herd7",
            "expected_kind": "rvwmo-herd7",
            "requires": list(meta.get("requires", ["RV64I"])),
            "notes": [f"generator:{generator.get('engine', 'unknown')}", "verdict:herd7"],
            "hand_category": "",
            "metadata": {"cycle": test.cycle},
        }
        items.append(
            _preview_item(
                combination,
                decision,
                test.unique_id,
                litmus,
                ir.to_json(),
                solver,
                diagram,
            )
        )
    return items


def _case_ir_from_json(data: Dict[str, Any]) -> LitmusCaseIR:
    return LitmusCaseIR.from_json(data)


def _herd_verdict_from_scalar_solver(solver: Dict[str, Any]) -> HerdVerdict | None:
    if solver.get("status") != "verified" or solver.get("verdict") not in {"observable", "forbidden"}:
        return None
    return HerdVerdict(
        outcome=str(solver["verdict"]),
        allowed=bool(solver.get("allowed")),
        observation=str(solver.get("observation", "")),
        positive=int(solver.get("positive", 0)),
        negative=int(solver.get("negative", 0)),
        states=int(solver.get("states", 0)),
        condition=str(solver.get("condition", "")),
        raw=str(solver.get("raw_output", "")),
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("expected a string or list of strings")


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{field} must be at least 1")
    return parsed


def _optional_positive_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return _positive_int(value, "limit")


def _optional_nonnegative_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be an integer") from exc
    if parsed < 0:
        raise ValueError("value must be non-negative")
    return parsed


def _preview_item(
    combination: Combination,
    decision: Dict[str, Any],
    name: str,
    litmus: str,
    case_ir: Dict[str, Any] | None,
    solver: Dict[str, Any] | None,
    diagram: Dict[str, Any] | None,
) -> Dict[str, Any]:
    case_id = str(case_ir.get("name", name)) if case_ir else name
    display_name = str(case_ir.get("display_name", name)) if case_ir else name
    return {
        "name": display_name,
        "case_id": case_id,
        "file_name": f"{case_id}.litmus" if litmus else "",
        "combination": combination.to_json(),
        "decision": decision,
        "litmus": litmus,
        "case_ir": case_ir,
        "solver": solver,
        "diagram": diagram,
        "analysis": _preview_analysis(combination, decision, litmus, case_ir, solver),
    }


def _preview_classification_counts(sample: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(sample)
    groups: Dict[str, Counter[str]] = {
        "status": Counter(),
        "verdict": Counter(),
        "skeleton": Counter(),
        "category": Counter(),
        "memory_layout": Counter(),
        "attribute": Counter(),
        "memory_event": Counter(),
        "vector": Counter(),
        "cmo": Counter(),
        "tlb": Counter(),
        "sew": Counter(),
        "lmul": Counter(),
        "index_eew": Counter(),
        "mask": Counter(),
        "tail": Counter(),
        "vl": Counter(),
        "endpoint_mode": Counter(),
        "endpoint_category": Counter(),
        "endpoint_composition": Counter(),
        "scalar_width": Counter(),
        "amo_opcode": Counter(),
        "amo_width": Counter(),
        "amo_ordering": Counter(),
        "overlap_layout": Counter(),
        "vector_event_form": Counter(),
        "alignment": Counter(),
        "relation_mechanism": Counter(),
    }
    for item in items:
        combination = item.get("combination", {}) or {}
        decision = item.get("decision", {}) or {}
        solver = item.get("solver", {}) or {}
        groups["status"][_count_value(decision.get("status"), "unknown")] += 1
        groups["verdict"][_preview_verdict(solver)] += 1
        for key in ("skeleton", "category", "attribute", "memory_event", "vector", "cmo", "tlb"):
            groups[key][_count_value(combination.get(key), "none")] += 1
        params = combination.get("params", {}) or {}
        for key in ("sew", "lmul", "index_eew", "mask", "tail", "vl"):
            if key in params:
                groups[key][_count_value(params.get(key), "default")] += 1
        case_ir = item.get("case_ir", {}) or {}
        metadata = case_ir.get("metadata", {}) or {}
        for choice in metadata.get("endpoint_choices", []) or []:
            if not isinstance(choice, dict):
                continue
            groups["endpoint_mode"][_count_value(choice.get("annotation"), "P")] += 1
            category = _count_value(choice.get("category"), "unknown")
            groups["endpoint_category"][category] += 1
            choice_params = choice.get("params", {}) or {}
            if category == "scalar":
                groups["scalar_width"][_count_value(choice_params.get("width"), "w")] += 1
            elif category == "amo":
                groups["amo_opcode"][_count_value(choice_params.get("amo_op"), "swap")] += 1
                groups["amo_width"][_count_value(choice_params.get("amo_width"), "w")] += 1
                groups["amo_ordering"][_count_value(choice_params.get("amo_ordering"), "relaxed")] += 1
            if choice.get("category") == "vector":
                groups["vector_event_form"][_count_value(choice.get("vector_form"), "vector")] += 1
                groups["alignment"][_count_value(choice_params.get("alignment"), "aligned")] += 1
        memory_layout = metadata.get("memory_layout", {}) or {}
        groups["overlap_layout"][_count_value(memory_layout.get("overlap_layout"), "same_start")] += 1
        choices = metadata.get("endpoint_choices", []) or []
        category_set = frozenset(
            str(choice.get("category"))
            for choice in choices
            if isinstance(choice, dict)
        )
        composition = {
            frozenset({"vector"}): "vector_only",
            frozenset({"vector", "scalar"}): "vector_scalar",
            frozenset({"vector", "amo"}): "vector_amo",
            frozenset({"vector", "scalar", "amo"}): "vector_scalar_amo",
        }.get(category_set, "+".join(sorted(category_set)) or "unknown")
        groups["endpoint_composition"][composition] += 1
        for relation in case_ir.get("relations", []) or []:
            if isinstance(relation, dict):
                groups["relation_mechanism"][_count_value(relation.get("kind"), "unknown")] += 1
        groups["memory_layout"][_preview_memory_layout(item)] += 1
    return {
        "displayed_cases": len(items),
        "groups": {
            key: dict(sorted(counter.items()))
            for key, counter in groups.items()
            if counter
        },
    }


def _preview_verdict(solver: Dict[str, Any]) -> str:
    status = str(solver.get("status", ""))
    verdict = str(solver.get("verdict", ""))
    if status == "verified" and verdict in {"allowed", "observable"}:
        return "observable"
    if status == "verified" and verdict == "forbidden":
        return "forbidden"
    return verdict or status or "unchecked"


def _preview_memory_layout(item: Dict[str, Any]) -> str:
    case_ir = item.get("case_ir", {}) or {}
    accesses = [
        event.get("memory_access")
        for hart in case_ir.get("harts", [])
        for event in hart
        if isinstance(event.get("memory_access"), dict)
    ]
    if accesses:
        atomic_mixed = [
            access for access in accesses
            if access.get("atomicity_model") == "mixed_size_atomic"
        ]
        if atomic_mixed:
            return "atomic_mixed"
        variant = str(case_ir.get("variant", "")).lower()
        if "atomic-" in variant or "atomic_" in variant:
            return "atomic"
        no_mag = [
            access for access in accesses
            if access.get("atomicity_model") == "byte_level_no_mag"
        ]
        if no_mag:
            sizes = {int(access.get("size_bytes", 0)) for access in no_mag}
            return "mixed" if len(sizes) > 1 or "mixed" in variant else "misaligned"
        return "aligned"
    combination = item.get("combination", {}) or {}
    params = combination.get("params", {}) or {}
    footprint = str(params.get("footprint", ""))
    return footprint or "unspecified"


def _count_value(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _preview_analysis(
    combination: Combination,
    decision: Dict[str, Any],
    litmus: str,
    case_ir: Dict[str, Any] | None = None,
    solver: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    exists = str(case_ir.get("exists")) if case_ir else _extract_exists(litmus)
    cycle = str(case_ir.get("cycle")) if case_ir else _cycle_text(combination, litmus)
    tokens = [relation.get("label", relation.get("kind", "")) for relation in case_ir.get("relations", [])] if case_ir else _cycle_tokens(cycle)
    forbidden = _forbidden_text(combination, decision, exists, solver)
    return {
        "cycle": cycle,
        "cycle_tokens": tokens,
        "exists": exists,
        "outcome_interpretation": forbidden,
        # Kept for compatibility with existing saved preview payloads.
        "forbidden_outcome": forbidden,
        "solver_status": solver.get("status") if solver else "not_applicable",
        "solver_verdict": solver.get("verdict") if solver else "unmodeled",
        "harts": [f"P{index}" for index, _hart in enumerate(case_ir.get("harts", []))] if case_ir else _harts_from_litmus(litmus),
        "memory_locations": _memory_locations_from_ir(case_ir) if case_ir else _memory_locations_from_litmus(litmus),
    }


def _extract_exists(litmus: str) -> str:
    match = re.search(r"\bexists\b\s*(.*)$", litmus, flags=re.DOTALL)
    return " ".join(match.group(1).split()) if match else "No exists clause in rendered preview."


def _cycle_text(combination: Combination, litmus: str) -> str:
    quoted = re.findall(r'"([^"]+)"', litmus)
    if quoted and _looks_like_cycle(quoted[0]):
        return quoted[0]
    skeleton_cycles = {
        "MP": "po/W->W -> rfe -> fre -> po/R->R",
        "LB": "po/R->W -> rfe -> po/R->W -> rfe",
        "SB": "po/W->R -> fre -> po/W->R -> fre",
        "WRC": "wse -> rfe -> po/R->W -> rfe -> fre",
        "RWC": "rfe -> po/R->R -> fre -> po/W->R -> fre",
        "IRIW": "wse -> rfe -> fre -> rfe -> fre",
        "ISA2": "fre -> po/W->W -> rfe -> po/R->W -> rfe -> po/R->R",
        "R": "fre -> po/W->W -> co -> po/W->R",
        "S": "rfe -> po/R->W -> co -> po/W->W",
        "Co": "rfe -> po-loc/R->R -> fre",
    }
    base = skeleton_cycles.get(combination.skeleton, f"{combination.skeleton} relation cycle")
    features = [value for value in [combination.vector, combination.cmo, combination.tlb, combination.attribute] if value not in {"none", "no_cmo", "no_tlb", "cacheable"}]
    return base + (" with " + ", ".join(features) if features else "")


def _looks_like_cycle(text: str) -> bool:
    tokens = {"po", "pod", "rfe", "rfi", "fre", "fri", "co", "wse", "rf", "fr", "ctrl", "addr", "data"}
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _cycle_tokens(cycle: str) -> list[str]:
    raw = re.split(r"\s*(?:->|,|\s+)\s*", cycle.strip())
    return [token for token in raw if token]


def _forbidden_text(combination: Combination, decision: Dict[str, Any], exists: str, solver: Dict[str, Any] | None = None) -> str:
    if solver:
        status = solver.get("status")
        verdict = solver.get("verdict")
        cross = solver.get("cross_check", "")
        model = str(solver.get("model", ""))
        formal_rvwmo = model == "rvwmo-herd7" or model.startswith("riscv.cat")
        tool = _solver_display_name(solver, cross)
        if status == "verified" and verdict == "forbidden":
            return f"FORBIDDEN by {tool}: the exists outcome must never be observed under the modeled RVWMO rules: {exists}"
        if status == "verified" and verdict in {"allowed", "observable"}:
            return f"OBSERVABLE by {tool}: the exists outcome is architecturally allowed under the modeled RVWMO rules: {exists}"
        if status == "conflict":
            return f"Conflict between native checker and herd7 ({solver.get('reason', '')}); native verdict {verdict} reported as primary."
        if status == "not_applicable":
            fusion = solver.get("fusion") or {}
            if fusion.get("status") == "analyzed":
                return f"Extension-prose ordering analysis ({fusion.get('verdict')}, informative -- not a herd verdict): {fusion.get('reason', '')}"
            return "No formal RVWMO forbidden assertion is emitted for this extension/prose-spec case."
        if formal_rvwmo:
            return f"Formal RVWMO verdict unavailable from {tool}: {solver.get('reason', '')}"
    outcome = str(combination.params.get("outcome", ""))
    if outcome == "forbidden":
        return f"Requested forbidden outcome, but solver verification is still required: {exists}"
    if decision.get("expected_kind") in {"rvwmo-herd", "rvwmo-nc", "rvwmo-vector"}:
        return "RVWMO decides whether the exists outcome is allowed or forbidden for this scalar main-memory case."
    return "No formal forbidden assertion is emitted; this is a hardware-observation/prose-spec outcome."


def _solver_display_name(solver: Dict[str, Any], cross_check: str) -> str:
    backend = str(solver.get("backend", ""))
    model = str(solver.get("model", ""))
    tool = str(solver.get("tool", ""))
    if backend == "crosscheck" or cross_check == "agree":
        return "embedded RVWMO and herd7/riscv.cat cross-check"
    if backend == "embedded" or tool == "litmus-link-rvwmo":
        if model == "riscv.cat+byte_level_no_mag":
            return "the embedded RVWMO byte-level no-MAG solver"
        return "the embedded RVWMO solver"
    if tool == "herd7" or model == "rvwmo-herd7":
        return "herd7 + riscv.cat"
    return tool or model or "the configured solver"


def _harts_from_litmus(litmus: str) -> list[str]:
    harts = sorted(set(re.findall(r"(?:^|[\s{;])(\d+):", litmus)))
    return [f"P{hart}" for hart in harts] or ["P0", "P1"]


def _memory_locations_from_litmus(litmus: str) -> list[str]:
    init_match = re.search(r"\{(.*?)\}", litmus, flags=re.DOTALL)
    if not init_match:
        return ["x", "y"]
    locations = sorted(set(re.findall(r"=([A-Za-z_][A-Za-z0-9_]*)(?:[;\s]|$)", init_match.group(1))))
    return [location for location in locations if not location.startswith("P")][:4] or ["x", "y"]


def _memory_locations_from_ir(case_ir: Dict[str, Any] | None) -> list[str]:
    if not case_ir:
        return ["x", "y"]
    locations: list[str] = []
    for hart in case_ir.get("harts", []):
        for event in hart:
            location = event.get("location")
            if location and location not in locations:
                locations.append(location)
    return locations or ["x", "y"]


def audit_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(str(payload.get("out") or "out/gui-audit"))
    mode = str(payload.get("mode", "scalar"))
    if mode == "scalar":
        return _scalar_audit_payload(payload, out_dir)
    if mode == "vector":
        out_dir.mkdir(parents=True, exist_ok=True)
        audit = VectorNativeDomain.from_payload(payload).audit()
        (out_dir / "audit-report.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return audit
    raise ValueError("GUI workflow supports only scalar and vector modes")


def generate_payload(
    payload: Dict[str, Any],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    out_dir = Path(str(payload.get("out") or "out/gui-generated"))
    mode = str(payload.get("mode", "scalar"))
    if mode == "scalar":
        return _run_scalar_generator(payload, out_dir, progress_callback=progress_callback)
    if mode == "vector":
        return generate_vector_cases(payload, out_dir, progress_callback=progress_callback)
    raise ValueError("GUI workflow supports only scalar and vector modes")


def _scalar_audit_payload(payload: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    engine = str(payload.get("engine", "native_templates"))
    if engine == "native_templates":
        presets = _string_list(payload.get("skeletons")) or ["MP"]
        mechanisms = _string_list(payload.get("mechanisms")) or list(DEFAULT_NATIVE_MECHANISMS)
        memory_layouts = _memory_layouts_from_payload(payload)
        native_audit = native_template_audit(
            presets,
            mechanisms,
            include_same=bool(payload.get("include_same", True)),
            annotations=_string_list(payload.get("annotations")) or NATIVE_ANNOTATIONS,
            memory_layouts=memory_layouts,
        )
        expanded = {
            "generator": {"engine": "litmus-link-native"},
            "available_litmus": native_audit["accepted"],
        }
    else:
        native_audit = None
        with TemporaryDirectory(prefix="litmus-link-scalar-audit-") as temporary:
            audit_request = dict(payload)
            audit_request["limit"] = None
            expanded = _run_scalar_generator(audit_request, Path(temporary), limit=None, judge=False)
    report = {
        "schema": "litmus-link.scalar-audit.v1",
        "profile": "scalar",
        "source": expanded.get("generator", {}).get("engine", "herdtools7"),
        "generator": expanded.get("generator", {}),
        "toolchain": expanded.get("toolchain", {}),
        "available_litmus": expanded.get("available_litmus", 0),
        "audited_litmus": expanded.get("available_litmus", 0),
        "total_combinations": expanded.get("available_litmus", 0),
        "generated": expanded.get("available_litmus", 0),
        "excluded_illegal": int(
            (native_audit or {}).get("excluded_illegal", 0)
        ),
        "excluded_unsupported": int(
            (native_audit or {}).get("excluded_misaligned_atomic_annotations", 0)
        ),
        "hand_required": 0,
        "missing": 0,
    }
    if native_audit is not None:
        report["native_audit"] = native_audit
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "cross-coverage.md").write_text(
        "# Scalar Litmus Audit\n\n"
        f"- Engine: `{report['source']}`\n"
        f"- Available litmus tests: {report['available_litmus']}\n"
        "- Missing: 0\n",
        encoding="utf-8",
    )
    return report
