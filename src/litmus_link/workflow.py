from __future__ import annotations

import json
import re
from collections import Counter
from hashlib import sha256
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Callable, Dict, Iterable, Tuple

from .corpus_ir import corpus_to_ir
from .corpus_riscv import corpus_available, judge as corpus_judge, parse_litmus, skeleton_counts, tests_for_skeleton
from .descriptions import feature_description_catalog
from .diagram import diagram_summary, render_diagram
from .generator import (
    _coverage_markdown,
    audit_summary,
    generate_combinations,
    generate_profile,
    write_audit,
    write_audit_for_combinations,
    write_one_generated_case,
)
from .litmus_ir import LitmusCaseIR, LitmusEvent, LitmusRelation, case_count
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
    ALIAS_MODES,
    ATTRIBUTES,
    CMO_OPS,
    CMO_SYNC_SEQUENCES,
    FORMAL_VECTOR_SKELETONS,
    SKELETONS,
    STRESSORS,
    TLB_OPS,
    VECTOR_FOOTPRINTS,
    VECTOR_INDEX_EEWS,
    VECTOR_LENGTHS,
    VECTOR_LMULS,
    VECTOR_MASKS,
    VECTOR_OPS,
    VECTOR_TAILS,
    VECTOR_WIDTHS,
    VM_CONTEXTS,
    PTE_STATES,
    SHOOTDOWN_SCOPES,
    list_profiles,
    profile_combinations,
    vector_combinations,
)
from .renderer import render_cases
from .rule_file import RuleFileError, load_rule_data, rule_field_values
from .rules import evaluate
from .scalar import (
    DEFAULT_MECHANISMS,
    DEFAULT_RELAX_EDGES,
    DEFAULT_SAFE_EDGES,
    generate_scalar_cross,
    generate_scalar_enumerated,
    scalar_catalog,
)
from .solver import solve_generated_case
from .toolchain import HerdVerdict
from .vector_native import (
    VECTOR_ALIGNMENTS,
    VectorNativeDomain,
    generate_vector_cases,
    sample_vector_cases,
)


PARAM_AXIS_VALUES: Dict[str, list[str]] = {
    "dep": ["addr", "data", "ctrl", "ctrl_fence", "aq", "rl", "aqrl"],
    "width": ["w8", "w16", "w32", "w64"],
    "outcome": ["allowed", "forbidden", "mixed_size"],
    "sew": list(VECTOR_WIDTHS),
    "index_eew": list(VECTOR_INDEX_EEWS),
    "lmul": list(VECTOR_LMULS),
    "mask": list(VECTOR_MASKS),
    "tail": list(VECTOR_TAILS),
    "footprint": list(VECTOR_FOOTPRINTS),
    "vl": list(VECTOR_LENGTHS),
    "sync": list(CMO_SYNC_SEQUENCES),
    "vm": list(VM_CONTEXTS),
    "shootdown": list(SHOOTDOWN_SCOPES),
    "pte": list(PTE_STATES),
    "alias": list(ALIAS_MODES),
    "stress": list(STRESSORS),
}


def options_payload() -> Dict[str, Any]:
    rule_fields = rule_field_values()
    return {
        "profiles": list_profiles(),
        "axes": {
            "skeleton": list(SKELETONS),
            "attribute": list(ATTRIBUTES),
            "vector": ["none", *VECTOR_OPS],
            "cmo": ["no_cmo", *CMO_OPS],
            "tlb": ["no_tlb", *TLB_OPS],
        },
        "rule_file_fields": rule_fields,
            "param_axes": PARAM_AXIS_VALUES,
        "features": feature_description_catalog(),
        "scalar": scalar_catalog(),
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
            "endpoint_modes": list(NATIVE_ANNOTATIONS),
            "alignments": list(VECTOR_ALIGNMENTS),
        },
    }


def preview_payload(
    payload: Dict[str, Any],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    if str(payload.get("mode", "profile")) == "scalar":
        return _scalar_preview_payload(payload, progress_callback=progress_callback)
    if str(payload.get("mode", "profile")) == "vector":
        return _vector_native_preview_payload(payload, progress_callback=progress_callback)
    name, combinations, source = _combinations_from_payload(payload)
    sample_limit = int(payload.get("sample_limit", 10))
    compute_verdicts = bool(payload.get("compute_verdicts", True))
    cached = list(combinations) if isinstance(combinations, list) else None
    use_corpus = (
        source == "gui"
        and corpus_available()
        and cached is not None
        and any(_is_scalar_corpus_combination(combination) for combination in cached)
    )

    sample: list = []
    progress_total = max(
        sample_limit
        if use_corpus or cached is None
        else min(sample_limit, len(cached)),
        1,
    )
    if progress_callback is not None:
        progress_callback(0, progress_total, f"Preparing up to {sample_limit:,} preview entries")
    if use_corpus:
        # Total-item budget: bounds herd7 runs regardless of how many skeletons
        # are checked. Corpus combinations expand into real tool-generated tests.
        budget = max(sample_limit, 0)
        for combination in (cached if cached is not None else []):
            if budget <= 0:
                break
            if _is_scalar_corpus_combination(combination):
                items = _corpus_preview_items(
                    combination,
                    min(budget, 8),
                    compute_verdicts=compute_verdicts,
                )
            else:
                items = _render_preview_items(combination, compute_verdicts=compute_verdicts)
            for item in items:
                if budget <= 0:
                    break
                sample.append(item)
                budget -= 1
                if progress_callback is not None:
                    progress_callback(
                        min(len(sample), progress_total),
                        progress_total,
                        f"Prepared {len(sample):,} preview entries",
                    )
    else:
        iterator = iter(cached if cached is not None else combinations)
        for combination_index, combination in enumerate(
            islice(iterator, max(sample_limit, 0)),
            start=1,
        ):
            sample.extend(
                _render_preview_items(
                    combination,
                    compute_verdicts=compute_verdicts,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    min(combination_index, progress_total),
                    progress_total,
                    f"Prepared {len(sample):,} cases from {combination_index:,} combinations",
                )

    summary_combinations = cached if cached is not None else _combinations_from_payload(payload)[1]
    domain_counts: Dict[str, Any] = {}
    if cached is not None and not use_corpus:
        domain_counts = _domain_classification_counts(
            cached,
            progress_callback=progress_callback,
        )
        report = _domain_audit_report(name, domain_counts, source)
    else:
        if progress_callback is not None:
            progress_callback(progress_total, progress_total, "Classifying preview domain")
        report = (
            _gui_corpus_report(name, summary_combinations, source)
            if use_corpus
            else audit_summary(name, summary_combinations, source=source)
        )
    if domain_counts:
        domain_counts["preview_displayed_cases"] = len(sample)
    return {
        "profile": name,
        "source": source,
        "report": report,
        "sample": sample,
        "classification_counts": _preview_classification_counts(sample),
        "domain_classification_counts": domain_counts,
    }


_DIAGRAM_DIR = Path(gettempdir()) / "litmus-link-preview-diagrams"


def _deferred_preview_diagram(
    case_ir: LitmusCaseIR,
    solver: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Describe a preview diagram without paying the PNG rendering cost."""
    fingerprint = sha256(
        json.dumps(
            {"case_ir": case_ir.to_json(), "solver": solver or {}},
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
        "schema": "litmus-link.vector-native-preview.v1",
        "profile": "vector-native",
        "total_combinations": audit["total_cases"],
        "generated": audit["total_cases"],
        "generated_litmus": audit["total_cases"],
        "displayed_litmus": len(sample),
        "excluded_illegal": 0,
        "excluded_unsupported": 0,
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


def _is_scalar_corpus_combination(combination: Combination) -> bool:
    """True for a pure scalar RVWMO main-memory family we can serve from the
    real corpus: no vector/CMO/TLB extension, cacheable, no body-changing params."""
    if combination.vector != "none" or combination.cmo != "no_cmo" or combination.tlb != "no_tlb":
        return False
    if combination.attribute != "cacheable":
        return False
    if combination.params:
        return False
    return skeleton_counts().get(combination.skeleton, 0) > 0


def _render_preview_items(
    combination: Combination,
    *,
    compute_verdicts: bool = True,
) -> list:
    """Original (non-corpus) preview path: evaluate -> render -> solve -> draw."""
    decision = evaluate(combination)
    rendered_cases = render_cases(combination, decision) if decision.status == GENERATED else []
    if not rendered_cases:
        return [_preview_item(combination, decision.to_json(), combination.name, "", None, None, None)]
    items = []
    for case in rendered_cases:
        solver = (
            solve_generated_case(case).to_json()
            if compute_verdicts
            else _unchecked_preview_solver()
        )
        diagram = (
            _deferred_preview_diagram(case.case_ir, solver)
            if case.case_ir is not None
            else None
        )
        items.append(
            _preview_item(
                combination,
                decision.to_json(),
                case.name,
                case.litmus,
                case.case_ir.to_json() if case.case_ir else None,
                solver,
                diagram,
            )
        )
    return items


def _safe_corpus_judge(test):
    try:
        return corpus_judge(test)
    except Exception:
        return None


def _corpus_solver_json(verdict) -> Dict[str, Any]:
    if verdict is None or verdict.outcome == "unknown":
        return {
            "schema": "litmus-link.solver.v1",
            "status": "not_applicable",
            "verdict": "unmodeled",
            "allowed": None,
            "model": "rvwmo-herd7",
            "tool": "herd7",
            "reason": "herd7 verdict unavailable for this test.",
            "cross_check": "herd7_only",
            "edges": [],
            "fusion": None,
            "observation": "",
            "raw_output": "",
            "command": [],
        }
    forbidden = verdict.outcome == "forbidden"
    return {
        "schema": "litmus-link.solver.v1",
        "status": "verified",
        "verdict": "forbidden" if forbidden else "allowed",
        "allowed": verdict.allowed,
        "model": "rvwmo-herd7",
        "tool": "herd7",
        "reason": (
            f"herd7 + riscv.cat: exists outcome {'FORBIDDEN (Never observed)' if forbidden else 'OBSERVABLE'} "
            f"[{verdict.observation} +{verdict.positive}/-{verdict.negative}]"
        ),
        "cross_check": "herd7_only",
        "edges": [],
        "fusion": None,
        "observation": verdict.observation,
        "raw_output": "",
        "command": [],
    }


def _corpus_decision_json(test) -> Dict[str, Any]:
    return {
        "status": GENERATED,
        "reason": f"Real RVWMO litmus from the {test.skeleton} family (tool-generated corpus).",
        "rvwmo_class": "rvwmo-herd7",
        "expected_kind": "rvwmo-herd7",
        "requires": ["RV64I"],
        "notes": [f"corpus:{test.family}", "verdict:herd7"],
        "hand_category": "",
        "metadata": {"corpus": "true", "cycle": test.cycle},
    }


def _corpus_preview_items(
    combination: Combination,
    limit: int,
    *,
    compute_verdicts: bool = True,
) -> list:
    items = []
    for test in tests_for_skeleton(combination.skeleton, limit=limit):
        # One malformed corpus sample (parse/IR error) must not
        # take down the whole preview list. do_POST only catches
        # ValueError/RuleFileError/FileNotFoundError, so anything else here
        # would 500 the request and lose every other valid sample. Degrade the
        # single bad row instead.
        try:
            verdict = _safe_corpus_judge(test) if compute_verdicts else None
            ir = corpus_to_ir(test, verdict)
            solver = (
                _corpus_solver_json(verdict)
                if compute_verdicts
                else _unchecked_preview_solver()
            )
            diagram = _deferred_preview_diagram(ir, solver)
            items.append(
                _preview_item(
                    combination,
                    _corpus_decision_json(test),
                    test.unique_id,
                    test.text,
                    ir.to_json(),
                    solver,
                    diagram,
                )
            )
        except Exception:
            # Skip just this malformed sample; keep every other valid one.
            continue
    return items


def _unchecked_preview_solver() -> Dict[str, Any]:
    return {
        "schema": "litmus-link.solver.v1",
        "status": "unchecked",
        "verdict": "unchecked",
        "allowed": None,
        "model": "rvwmo",
        "tool": "none",
        "reason": "Fast preview skips outcome solving; use Verify Preview to calculate the verdict.",
        "cross_check": "not_run",
        "edges": [],
        "fusion": None,
        "observation": "",
        "raw_output": "",
        "command": [],
    }


def _gui_corpus_report(name: str, combinations: Iterable[Combination], source: str | None) -> Dict[str, Any]:
    counts = {GENERATED: 0, "excluded_illegal": 0, "excluded_unsupported": 0, "hand_required": 0, "missing": 0}
    total = 0
    generated_litmus = 0
    for combination in combinations:
        total += 1
        if _is_scalar_corpus_combination(combination):
            counts[GENERATED] += 1
            generated_litmus += skeleton_counts().get(combination.skeleton, 0)
            continue
        decision = evaluate(combination)
        counts[decision.status] = counts.get(decision.status, 0) + 1
        if decision.status == GENERATED:
            generated_litmus += case_count(combination, decision)
    report = {
        "schema": "litmus-link.audit.v1",
        "profile": name,
        "total_combinations": total,
        "generated": counts.get(GENERATED, 0),
        "generated_litmus": generated_litmus,
        "excluded_illegal": counts.get("excluded_illegal", 0),
        "excluded_unsupported": counts.get("excluded_unsupported", 0),
        "hand_required": counts.get("hand_required", 0),
        "missing": counts.get("missing", 0),
    }
    if source:
        report["source"] = source
    return report


def _preview_item(
    combination: Combination,
    decision: Dict[str, Any],
    name: str,
    litmus: str,
    case_ir: Dict[str, Any] | None,
    solver: Dict[str, Any] | None,
    diagram: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "name": name,
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
            if choice.get("category") == "vector":
                groups["vector_event_form"][_count_value(choice.get("vector_form"), "vector")] += 1
                choice_params = choice.get("params", {}) or {}
                groups["alignment"][_count_value(choice_params.get("alignment"), "aligned")] += 1
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


def _domain_classification_counts(
    combinations: Iterable[Combination],
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    """Count the complete finite domain without rendering or solving cases.

    A profile preview intentionally displays only a bounded sample.  These
    counters keep the GUI honest about the full domain and weight generated
    combinations by the number of Litmus ordering variants they expand into.
    """
    groups: Dict[str, Counter[str]] = {
        "status": Counter(),
        "skeleton": Counter(),
        "category": Counter(),
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
    }
    domain_cases = 0
    generated_cases = 0
    total_combinations = len(combinations) if hasattr(combinations, "__len__") else 0
    combination_status: Counter[str] = Counter()
    last_percent = -1
    for index, combination in enumerate(combinations, start=1):
        decision = evaluate(combination)
        combination_status[decision.status] += 1
        weight = case_count(combination, decision) if decision.status == GENERATED else 1
        domain_cases += weight
        if decision.status == GENERATED:
            generated_cases += weight
        groups["status"][decision.status] += weight
        axes = combination.axes()
        for key in ("skeleton", "category", "attribute", "memory_event", "vector", "cmo", "tlb"):
            groups[key][_count_value(axes.get(key), "none")] += weight
        for key in ("sew", "lmul", "index_eew", "mask", "tail", "vl"):
            if key in combination.params:
                groups[key][_count_value(combination.params.get(key), "default")] += weight
        if progress_callback is not None and total_combinations:
            percent = int(index * 100 / total_combinations)
            if percent != last_percent:
                last_percent = percent
                progress_callback(
                    index,
                    total_combinations,
                    f"Classified {index:,}/{total_combinations:,} domain combinations",
                )
    return {
        "total_combinations": total_combinations,
        "combination_status": dict(sorted(combination_status.items())),
        "domain_cases": domain_cases,
        "generated_cases": generated_cases,
        "groups": {
            key: dict(sorted(counter.items()))
            for key, counter in groups.items()
            if counter
        },
    }


def _domain_audit_report(
    name: str,
    domain: Dict[str, Any],
    source: str | None,
) -> Dict[str, Any]:
    statuses = domain.get("combination_status", {}) or {}
    report = {
        "schema": "litmus-link.audit.v1",
        "profile": name,
        "total_combinations": int(domain.get("total_combinations", 0)),
        "generated": int(statuses.get(GENERATED, 0)),
        "generated_litmus": int(domain.get("generated_cases", 0)),
        "excluded_illegal": int(statuses.get("excluded_illegal", 0)),
        "excluded_unsupported": int(statuses.get("excluded_unsupported", 0)),
        "hand_required": int(statuses.get("hand_required", 0)),
        "missing": int(statuses.get("missing", 0)),
    }
    if source:
        report["source"] = source
    return report


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
    summary_only = bool(payload.get("summary_only", True))
    mode = str(payload.get("mode", "profile"))
    if mode == "scalar":
        return _scalar_audit_payload(payload, out_dir)
    if mode == "vector":
        out_dir.mkdir(parents=True, exist_ok=True)
        audit = VectorNativeDomain.from_payload(payload).audit()
        (out_dir / "audit-report.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return audit
    if mode == "profile":
        return write_audit(str(payload.get("profile") or "smoke"), out_dir, summary_only=summary_only)
    name, combinations, source = _combinations_from_payload(payload)
    if source == "gui" and corpus_available():
        return _gui_corpus_audit(name, list(combinations), out_dir, source, summary_only)
    return write_audit_for_combinations(name, combinations, out_dir, source=source, summary_only=summary_only)


def generate_payload(
    payload: Dict[str, Any],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    out_dir = Path(str(payload.get("out") or "out/gui-generated"))
    mode = str(payload.get("mode", "profile"))
    if mode == "scalar":
        return _run_scalar_generator(payload, out_dir, progress_callback=progress_callback)
    if mode == "vector":
        return generate_vector_cases(payload, out_dir, progress_callback=progress_callback)
    if mode == "profile":
        return generate_profile(
            str(payload.get("profile") or "smoke"),
            out_dir,
            progress_callback=progress_callback,
            diagrams=bool(payload.get("diagrams", False)),
        )
    name, combinations, source = _combinations_from_payload(payload)
    if source == "gui" and corpus_available():
        compute_verdicts = bool(payload.get("compute_verdicts", True))
        diagrams = bool(payload.get("diagrams", False))
        if diagrams:
            compute_verdicts = True
        rule = payload.get("rule") if isinstance(payload.get("rule"), dict) else {}
        limit = payload.get("generate_limit", rule.get("generate_limit", rule.get("limit")))
        return _gui_corpus_generate(
            name, list(combinations), out_dir, source,
            generation_limit=int(limit) if limit is not None else None,
            compute_verdicts=compute_verdicts,
            diagrams=diagrams,
            progress_callback=progress_callback,
        )
    return generate_combinations(
        name,
        combinations,
        out_dir,
        source=source,
        progress_callback=progress_callback,
        diagrams=bool(payload.get("diagrams", False)),
    )


# __LL_CORPUS_GENERATE__


def _corpus_meta(test, verdict_json: Dict[str, Any] | None) -> Dict[str, Any]:
    return {
        "schema": "litmus-link.corpus-meta.v1",
        "name": test.name,
        "family": test.family,
        "skeleton": test.skeleton,
        "cycle": test.cycle,
        "exists": test.exists,
        "nprocs": test.nprocs,
        "source_path": test.path,
        "model": "rvwmo-herd7",
        "verdict": verdict_json,
    }


def _write_corpus_family(
    combination,
    out_dir,
    generation_limit,
    compute_verdicts,
    diagrams,
    generated_names,
    solver_counts,
    seen_names,
    errors,
    progress_callback=None,
    progress_total=0,
) -> tuple[int, int]:
    written = 0
    rendered_diagrams = 0
    remaining = None if generation_limit is None else max(generation_limit - len(generated_names), 0)
    if remaining == 0:
        return 0, 0
    for test in tests_for_skeleton(combination.skeleton, limit=remaining):
        if test.unique_id in seen_names:
            continue
        seen_names.add(test.unique_id)
        (out_dir / f"{test.unique_id}.litmus").write_text(test.text, encoding="utf-8")
        verdict_json = None
        if compute_verdicts:
            try:
                verdict = _safe_corpus_judge(test)
                verdict_json = _corpus_solver_json(verdict)
                solver_counts[verdict_json["status"]] = solver_counts.get(verdict_json["status"], 0) + 1
                if diagrams:
                    render_diagram(corpus_to_ir(test, verdict), verdict_json, out_dir)
                    rendered_diagrams += 1
                (out_dir / f"{test.unique_id}.solver.json").write_text(
                    json.dumps(verdict_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            except Exception as exc:
                errors.append({"case": test.unique_id, "path": test.path, "error": str(exc)})
        (out_dir / f"{test.unique_id}.meta.json").write_text(
            json.dumps(_corpus_meta(test, verdict_json), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        generated_names.append(f"{test.unique_id}.litmus")
        written += 1
        if progress_callback is not None:
            progress_callback(
                min(len(generated_names), progress_total),
                progress_total,
                f"Generated {len(generated_names)}/{progress_total}: {test.unique_id}",
            )
    return written, rendered_diagrams


# __LL_CORPUS_GENERATE2__


def _gui_corpus_generate(
    name,
    combinations,
    out_dir,
    source,
    generation_limit=None,
    compute_verdicts=False,
    diagrams=False,
    progress_callback=None,
) -> Dict[str, Any]:
    """Generate GUI custom-rule output, serving scalar RVWMO families from the
    real corpus. Solver results are optional and PNG diagrams are independently
    controlled so large corpus generation can keep drawing fully on demand."""
    out_dir.mkdir(parents=True, exist_ok=True)
    potential_report = _gui_corpus_report(name, combinations, source)
    available_litmus = int(potential_report.get("generated_litmus", 0) or 0)
    generated_names: list[str] = []
    solver_counts = {"verified": 0, "conflict": 0, "not_applicable": 0}
    counts = {GENERATED: 0, "excluded_illegal": 0, "excluded_unsupported": 0, "hand_required": 0, "missing": 0}
    seen_names: set[str] = set()
    excluded: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []
    total = len(combinations)
    generated_litmus = 0
    generated_diagrams = 0
    progress_total = min(available_litmus, generation_limit) if generation_limit is not None else available_litmus
    progress_total = max(progress_total, 1)
    if progress_callback is not None:
        progress_callback(0, progress_total, f"Generating 0/{min(available_litmus, progress_total)} cases")
    for combination in combinations:
        if generation_limit is not None and generated_litmus >= generation_limit:
            break
        if _is_scalar_corpus_combination(combination):
            counts[GENERATED] += 1
            written, rendered = _write_corpus_family(
                combination, out_dir, generation_limit, compute_verdicts, diagrams,
                generated_names, solver_counts, seen_names, errors,
                progress_callback, progress_total,
            )
            generated_litmus += written
            generated_diagrams += rendered
            continue
        decision = evaluate(combination)
        counts[decision.status] = counts.get(decision.status, 0) + 1
        if decision.status == GENERATED:
            for case in render_cases(combination, decision):
                if generation_limit is not None and generated_litmus >= generation_limit:
                    break
                status, fname = write_one_generated_case(case, out_dir, diagrams=diagrams)
                if diagrams and case.case_ir is not None:
                    generated_diagrams += 1
                solver_counts[status] = solver_counts.get(status, 0) + 1
                generated_names.append(fname)
                generated_litmus += 1
                if progress_callback is not None:
                    progress_callback(
                        min(generated_litmus, progress_total),
                        progress_total,
                        f"Generated {generated_litmus}/{progress_total}: {case.name}",
                    )
        else:
            excluded.append({"combination": combination.to_json(), "decision": decision.to_json()})

    report = {
        "schema": "litmus-link.audit.v1",
        "profile": name,
        "total_combinations": total,
        "generated": counts[GENERATED],
        "generated_litmus": generated_litmus,
        "available_litmus": available_litmus,
        "excluded_illegal": counts["excluded_illegal"],
        "excluded_unsupported": counts["excluded_unsupported"],
        "hand_required": counts["hand_required"],
        "missing": counts["missing"],
        "solver": solver_counts,
        "verdict_mode": "computed" if compute_verdicts else "deferred",
        "diagram_mode": "eager" if diagrams else "on_demand",
        "generated_diagrams": generated_diagrams,
        "generation_limit": generation_limit,
        "generation_limited": generation_limit is not None and generated_litmus < available_litmus,
        "generation_errors": len(errors),
        "source": source,
    }
    (out_dir / "excluded.json").write_text(json.dumps(excluded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        (out_dir / "generation-errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "@all").write_text("\n".join(generated_names) + ("\n" if generated_names else ""), encoding="utf-8")
    (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress_callback is not None:
        progress_callback(progress_total, progress_total, "Finalized indexes and audit report")
    return report


def _gui_corpus_audit(name, combinations, out_dir, source, summary_only) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = _gui_corpus_report(name, combinations, source)
    (out_dir / "cross-coverage.md").write_text(_coverage_markdown(report), encoding="utf-8")
    (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _combinations_from_payload(payload: Dict[str, Any]) -> Tuple[str, Iterable[Combination], str | None]:
    mode = str(payload.get("mode", "profile"))
    if mode == "profile":
        profile = str(payload.get("profile") or "smoke")
        return profile, profile_combinations(profile), None
    if mode == "vector":
        return _vector_combinations_from_payload(payload)
    rule = payload.get("rule")
    if not isinstance(rule, dict):
        raise RuleFileError("custom GUI requests must include a rule object")
    rule_set = load_rule_data(rule, Path("<gui-rule>"))
    return rule_set.name, rule_set.combinations, "gui"


def _vector_combinations_from_payload(
    payload: Dict[str, Any],
) -> Tuple[str, list[Combination], str | None]:
    """Return the legality-filtered formal Vector domain selected by the GUI.

    The dedicated Vector GUI filters the canonical ``vector_mem`` profile
    instead of rebuilding a Cartesian product.  This preserves its conditional
    SEW/LMUL, same-line, form/endpoint, and indexed-EEW legality rules.
    """
    if bool(payload.get("complete", False)):
        return "vector_mem", vector_combinations("vector_mem"), "vector-gui"

    skeletons = _vector_filter_values(
        payload, "skeletons", FORMAL_VECTOR_SKELETONS
    )
    forms = _vector_filter_values(payload, "forms", VECTOR_OPS)
    sews = _vector_filter_values(payload, "sew", VECTOR_WIDTHS)
    lmuls = _vector_filter_values(payload, "lmul", VECTOR_LMULS)
    index_eews = _vector_filter_values(payload, "index_eew", VECTOR_INDEX_EEWS)
    masks = _vector_filter_values(payload, "mask", VECTOR_MASKS)
    tails = _vector_filter_values(payload, "tail", VECTOR_TAILS)
    vls = _vector_filter_values(payload, "vl", VECTOR_LENGTHS)
    endpoint_scope = str(payload.get("endpoint_scope", "all"))
    if endpoint_scope not in {"all", "first"}:
        raise ValueError("vector endpoint_scope must be 'all' or 'first'")

    selected = vector_combinations(
        str(payload.get("name") or "vector-custom"),
        skeletons=skeletons,
        vectors=forms,
        widths=sews,
        lmuls=lmuls,
        index_eews=index_eews,
        masks=masks,
        tails=tails,
        lengths=vls,
        endpoint_scope=endpoint_scope,
    )

    if not selected:
        raise ValueError("the selected Vector filters contain no legal combinations")
    return str(payload.get("name") or "vector-custom"), selected, "vector-gui"


def _vector_filter_values(
    payload: Dict[str, Any], key: str, allowed: Iterable[str]
) -> list[str]:
    domain = list(allowed)
    selected = _string_list(payload.get(key))
    if not selected:
        raise ValueError(f"select at least one Vector {key} value")
    unknown = sorted(set(selected) - set(domain))
    if unknown:
        raise ValueError(
            f"unknown Vector {key} value(s): {', '.join(unknown)}"
        )
    selected_set = set(selected)
    return [value for value in domain if value in selected_set]


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
