from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from .diagram import render_diagram
from .models import Combination, Decision, EXCLUDED_ILLEGAL, EXCLUDED_UNSUPPORTED, GENERATED, GeneratedCase, HAND_REQUIRED, MISSING, count_by_status
from .profiles import profile_combinations
from .litmus_ir import case_count
from .renderer import render_cases
from .rules import evaluate
from .solver import solve_generated_case


ProgressCallback = Callable[[int, int, str], None]


def audit_profile(profile: str) -> Tuple[List[Tuple[object, Decision]], Dict[str, object]]:
    return audit_combinations(profile, profile_combinations(profile))


def audit_combinations(profile: str, combinations: Iterable[Combination], source: str | None = None) -> Tuple[List[Tuple[Combination, Decision]], Dict[str, object]]:
    rows = []
    decisions = []
    missing = []
    generated_cases = 0
    for combination in combinations:
        decision = evaluate(combination)
        rows.append((combination, decision))
        decisions.append(decision)
        if decision.status == GENERATED:
            generated_cases += case_count(combination, decision)
        if decision.status == MISSING:
            missing.append(combination.to_json())
    counts = count_by_status(decisions)
    report = {
        "schema": "litmus-link.audit.v1",
        "profile": profile,
        "total_combinations": len(rows),
        "generated": counts.get(GENERATED, 0),
        "generated_litmus": generated_cases,
        "excluded_illegal": counts.get(EXCLUDED_ILLEGAL, 0),
        "excluded_unsupported": counts.get(EXCLUDED_UNSUPPORTED, 0),
        "hand_required": counts.get(HAND_REQUIRED, 0),
        "missing": counts.get(MISSING, 0),
    }
    if source:
        report["source"] = source
    return rows, report


def generate_profile(
    profile: str,
    out_dir: Path,
    progress_callback: ProgressCallback | None = None,
    diagrams: bool = False,
) -> Dict[str, object]:
    total_cases = None
    if progress_callback is not None:
        progress_callback(0, 0, f"Counting generated cases for profile {profile}")
        preflight = audit_summary(profile, profile_combinations(profile))
        total_cases = int(preflight.get("generated_litmus", 0))
    return generate_combinations(
        profile,
        profile_combinations(profile),
        out_dir,
        progress_callback=progress_callback,
        total_cases_hint=total_cases,
        diagrams=diagrams,
    )


def generate_combinations(
    profile: str,
    combinations: Iterable[Combination],
    out_dir: Path,
    source: str | None = None,
    progress_callback: ProgressCallback | None = None,
    total_cases_hint: int | None = None,
    diagrams: bool = False,
) -> Dict[str, object]:
    if progress_callback is not None and total_cases_hint is None:
        combinations = list(combinations)
        total_cases_hint = sum(
            case_count(combination, decision)
            for combination in combinations
            if (decision := evaluate(combination)).status == GENERATED
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_names: List[str] = []
    counts = _empty_counts()
    total = 0
    generated_cases = 0
    generated_diagrams = 0
    solver_counts = _empty_solver_counts()

    seen_names: set[str] = set()
    collisions = 0
    progress_total = max(int(total_cases_hint or 0), 1)
    if progress_callback is not None:
        progress_callback(0, progress_total, f"Generating 0/{int(total_cases_hint or 0)} cases")
    with _JsonArrayWriter(out_dir / "excluded.json") as excluded:
        for combination in combinations:
            total += 1
            decision = evaluate(combination)
            counts[decision.status] = counts.get(decision.status, 0) + 1
            if decision.status == GENERATED:
                for case in render_cases(combination, decision):
                    # Two distinct combinations whose params sanitise to the same
                    # token collapse to one filename; writing both would silently
                    # overwrite the first case's .litmus/.meta.json/.solver.json
                    # with the second's body, corrupting the verdict<->body
                    # correspondence while @all still lists both. Skip + record
                    # the collision rather than lose a test silently. (rule_file
                    # ._reject_duplicates guards its own path; this protects every
                    # other caller of generate_combinations.)
                    if case.name in seen_names:
                        collisions += 1
                        continue
                    seen_names.add(case.name)
                    status, fname = write_one_generated_case(case, out_dir, diagrams=diagrams)
                    if diagrams and case.case_ir is not None:
                        generated_diagrams += 1
                    solver_counts[status] = solver_counts.get(status, 0) + 1
                    generated_names.append(fname)
                    generated_cases += 1
                    if progress_callback is not None:
                        progress_callback(
                            min(generated_cases, progress_total),
                            progress_total,
                            f"Generated {generated_cases}/{int(total_cases_hint or generated_cases)}: {case.name}",
                        )
            else:
                excluded.write({"combination": combination.to_json(), "decision": decision.to_json()})

    report = _report(profile, total, counts, source)
    report["generated_litmus"] = generated_cases
    report["solver"] = solver_counts
    report["diagram_mode"] = "eager" if diagrams else "on_demand"
    report["generated_diagrams"] = generated_diagrams
    if collisions:
        report["name_collisions_skipped"] = collisions

    (out_dir / "@all").write_text("\n".join(generated_names) + ("\n" if generated_names else ""), encoding="utf-8")
    (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress_callback is not None:
        progress_callback(progress_total, progress_total, "Finalized indexes and audit report")
    return report


def write_audit(profile: str, out_dir: Path, summary_only: bool = False) -> Dict[str, object]:
    return write_audit_for_combinations(profile, profile_combinations(profile), out_dir, summary_only=summary_only)


def write_audit_for_combinations(profile: str, combinations: Iterable[Combination], out_dir: Path, source: str | None = None, summary_only: bool = False) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_only:
        report = audit_summary(profile, combinations, source=source)
        (out_dir / "cross-coverage.md").write_text(_coverage_markdown(report), encoding="utf-8")
        (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    counts = _empty_counts()
    total = 0
    generated_cases = 0
    with (
        _JsonArrayWriter(out_dir / "covered.json") as covered,
        _JsonArrayWriter(out_dir / "excluded-illegal.json") as excluded_illegal,
        _JsonArrayWriter(out_dir / "hand-required.json") as hand_required,
        _JsonArrayWriter(out_dir / "missing.json", empty_file_when_empty=True) as missing,
    ):
        for combination in combinations:
            total += 1
            decision = evaluate(combination)
            counts[decision.status] = counts.get(decision.status, 0) + 1
            item = {"combination": combination.to_json(), "decision": decision.to_json()}
            if decision.status == GENERATED:
                generated_cases += case_count(combination, decision)
                covered.write(item)
            elif decision.status == EXCLUDED_ILLEGAL:
                excluded_illegal.write(item)
            elif decision.status == HAND_REQUIRED:
                hand_required.write(item)
            elif decision.status == MISSING:
                missing.write(item)

    report = _report(profile, total, counts, source)
    report["generated_litmus"] = generated_cases
    (out_dir / "cross-coverage.md").write_text(_coverage_markdown(report), encoding="utf-8")
    (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def audit_summary(profile: str, combinations: Iterable[Combination], source: str | None = None) -> Dict[str, object]:
    counts = _empty_counts()
    total = 0
    generated_cases = 0
    for combination in combinations:
        total += 1
        decision = evaluate(combination)
        counts[decision.status] = counts.get(decision.status, 0) + 1
        if decision.status == GENERATED:
            generated_cases += case_count(combination, decision)
    report = _report(profile, total, counts, source)
    report["generated_litmus"] = generated_cases
    return report


def _empty_counts() -> Dict[str, int]:
    return {GENERATED: 0, EXCLUDED_ILLEGAL: 0, EXCLUDED_UNSUPPORTED: 0, HAND_REQUIRED: 0, MISSING: 0}


def _empty_solver_counts() -> Dict[str, int]:
    return {"verified": 0, "conflict": 0, "not_applicable": 0}


def _with_artifacts(case: GeneratedCase, solver: Dict[str, object], diagram: Dict[str, object] | None) -> GeneratedCase:
    from dataclasses import replace

    return replace(case, solver=solver, diagram=diagram)


def write_one_generated_case(
    case: GeneratedCase,
    out_dir: Path,
    *,
    diagrams: bool = False,
) -> tuple[str, str]:
    """Solve and write one case, drawing its diagram only when requested.

    Returns (solver_status, litmus_filename). Shared by the profile/CLI generate
    loop and the GUI corpus generate path so they stay byte-identical.
    """
    solver_result = solve_generated_case(case)
    solver_json = solver_result.to_json()
    diagram_json = None
    if diagrams and case.case_ir is not None:
        diagram_json = render_diagram(case.case_ir, solver_json, out_dir).summary
    case = _with_artifacts(case, solver_json, diagram_json)
    litmus_path = out_dir / f"{case.name}.litmus"
    litmus_path.write_text(case.litmus, encoding="utf-8")
    (out_dir / f"{case.name}.meta.json").write_text(
        json.dumps(case.meta(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / f"{case.name}.solver.json").write_text(
        json.dumps(solver_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return solver_result.status, litmus_path.name


def _report(profile: str, total: int, counts: Dict[str, int], source: str | None = None) -> Dict[str, object]:
    report: Dict[str, object] = {
        "schema": "litmus-link.audit.v1",
        "profile": profile,
        "total_combinations": total,
        "generated": counts.get(GENERATED, 0),
        "generated_litmus": counts.get(GENERATED, 0),
        "excluded_illegal": counts.get(EXCLUDED_ILLEGAL, 0),
        "excluded_unsupported": counts.get(EXCLUDED_UNSUPPORTED, 0),
        "hand_required": counts.get(HAND_REQUIRED, 0),
        "missing": counts.get(MISSING, 0),
    }
    if source:
        report["source"] = source
    return report


class _JsonArrayWriter:
    def __init__(self, path: Path, empty_file_when_empty: bool = False) -> None:
        self.path = path
        self.empty_file_when_empty = empty_file_when_empty
        self.file = None
        self.empty = True

    def __enter__(self) -> "_JsonArrayWriter":
        self.file = self.path.open("w", encoding="utf-8")
        self.file.write("[\n")
        return self

    def write(self, item: Dict[str, object]) -> None:
        if self.file is None:
            raise RuntimeError("JSON array writer is not open")
        if not self.empty:
            self.file.write(",\n")
        self.file.write(json.dumps(item, indent=2, sort_keys=True))
        self.empty = False

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.file is not None:
            if self.empty and self.empty_file_when_empty:
                self.file.close()
                self.path.write_text("", encoding="utf-8")
                return
            self.file.write("\n" if not self.empty else "")
            self.file.write("]\n")
            self.file.close()


def _coverage_markdown(report: Dict[str, object]) -> str:
    return "\n".join(
        [
            f"# Litmus-link Audit: {report['profile']}",
            "",
            f"- Total combinations: {report['total_combinations']}",
            f"- Generated: {report['generated']}",
            f"- Generated litmus files: {report.get('generated_litmus', report['generated'])}",
            f"- Excluded illegal: {report['excluded_illegal']}",
            f"- Excluded unsupported: {report['excluded_unsupported']}",
            f"- HAND-required: {report['hand_required']}",
            f"- Missing: {report['missing']}",
            "",
            "A non-zero `Missing` count fails `make audit`.",
        ]
    ) + "\n"
