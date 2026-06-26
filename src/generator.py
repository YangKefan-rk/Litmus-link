from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from models import Combination, Decision, EXCLUDED_ILLEGAL, EXCLUDED_UNSUPPORTED, GENERATED, HAND_REQUIRED, MISSING, count_by_status
from profiles import profile_combinations
from renderer import render_case
from rules import evaluate


def audit_profile(profile: str) -> Tuple[List[Tuple[object, Decision]], Dict[str, object]]:
    return audit_combinations(profile, profile_combinations(profile))


def audit_combinations(profile: str, combinations: Iterable[Combination], source: str | None = None) -> Tuple[List[Tuple[Combination, Decision]], Dict[str, object]]:
    rows = []
    decisions = []
    missing = []
    for combination in combinations:
        decision = evaluate(combination)
        rows.append((combination, decision))
        decisions.append(decision)
        if decision.status == MISSING:
            missing.append(combination.to_json())
    counts = count_by_status(decisions)
    report = {
        "schema": "litmus-link.audit.v1",
        "profile": profile,
        "total_combinations": len(rows),
        "generated": counts.get(GENERATED, 0),
        "excluded_illegal": counts.get(EXCLUDED_ILLEGAL, 0),
        "excluded_unsupported": counts.get(EXCLUDED_UNSUPPORTED, 0),
        "hand_required": counts.get(HAND_REQUIRED, 0),
        "missing": counts.get(MISSING, 0),
    }
    if source:
        report["source"] = source
    return rows, report


def generate_profile(profile: str, out_dir: Path) -> Dict[str, object]:
    return generate_combinations(profile, profile_combinations(profile), out_dir)


def generate_combinations(profile: str, combinations: Iterable[Combination], out_dir: Path, source: str | None = None) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_names: List[str] = []
    counts = _empty_counts()
    total = 0

    with _JsonArrayWriter(out_dir / "excluded.json") as excluded:
        for combination in combinations:
            total += 1
            decision = evaluate(combination)
            counts[decision.status] = counts.get(decision.status, 0) + 1
            if decision.status == GENERATED:
                case = render_case(combination, decision)
                litmus_path = out_dir / f"{combination.name}.litmus"
                meta_path = out_dir / f"{combination.name}.meta.json"
                litmus_path.write_text(case.litmus, encoding="utf-8")
                meta_path.write_text(json.dumps(case.meta(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
                generated_names.append(litmus_path.name)
            else:
                excluded.write({"combination": combination.to_json(), "decision": decision.to_json()})

    report = _report(profile, total, counts, source)

    (out_dir / "@all").write_text("\n".join(generated_names) + ("\n" if generated_names else ""), encoding="utf-8")
    (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
                covered.write(item)
            elif decision.status == EXCLUDED_ILLEGAL:
                excluded_illegal.write(item)
            elif decision.status == HAND_REQUIRED:
                hand_required.write(item)
            elif decision.status == MISSING:
                missing.write(item)

    report = _report(profile, total, counts, source)
    (out_dir / "cross-coverage.md").write_text(_coverage_markdown(report), encoding="utf-8")
    (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def audit_summary(profile: str, combinations: Iterable[Combination], source: str | None = None) -> Dict[str, object]:
    counts = _empty_counts()
    total = 0
    for combination in combinations:
        total += 1
        decision = evaluate(combination)
        counts[decision.status] = counts.get(decision.status, 0) + 1
    return _report(profile, total, counts, source)


def _empty_counts() -> Dict[str, int]:
    return {GENERATED: 0, EXCLUDED_ILLEGAL: 0, EXCLUDED_UNSUPPORTED: 0, HAND_REQUIRED: 0, MISSING: 0}


def _report(profile: str, total: int, counts: Dict[str, int], source: str | None = None) -> Dict[str, object]:
    report: Dict[str, object] = {
        "schema": "litmus-link.audit.v1",
        "profile": profile,
        "total_combinations": total,
        "generated": counts.get(GENERATED, 0),
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
            f"- Excluded illegal: {report['excluded_illegal']}",
            f"- Excluded unsupported: {report['excluded_unsupported']}",
            f"- HAND-required: {report['hand_required']}",
            f"- Missing: {report['missing']}",
            "",
            "A non-zero `Missing` count fails `make audit`.",
        ]
    ) + "\n"
