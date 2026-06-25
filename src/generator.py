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
    rows, report = audit_combinations(profile, combinations, source=source)
    generated_names: List[str] = []
    excluded: List[Dict[str, object]] = []

    for combination, decision in rows:
        if decision.status == GENERATED:
            case = render_case(combination, decision)
            litmus_path = out_dir / f"{combination.name}.litmus"
            meta_path = out_dir / f"{combination.name}.meta.json"
            litmus_path.write_text(case.litmus, encoding="utf-8")
            meta_path.write_text(json.dumps(case.meta(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            generated_names.append(litmus_path.name)
        else:
            excluded.append({"combination": combination.to_json(), "decision": decision.to_json()})

    (out_dir / "@all").write_text("\n".join(generated_names) + ("\n" if generated_names else ""), encoding="utf-8")
    (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "excluded.json").write_text(json.dumps(excluded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_audit(profile: str, out_dir: Path) -> Dict[str, object]:
    return write_audit_for_combinations(profile, profile_combinations(profile), out_dir)


def write_audit_for_combinations(profile: str, combinations: Iterable[Combination], out_dir: Path, source: str | None = None) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, report = audit_combinations(profile, combinations, source=source)
    excluded_illegal = []
    hand_required = []
    missing = []
    covered = []
    for combination, decision in rows:
        item = {"combination": combination.to_json(), "decision": decision.to_json()}
        if decision.status == GENERATED:
            covered.append(item)
        elif decision.status == EXCLUDED_ILLEGAL:
            excluded_illegal.append(item)
        elif decision.status == HAND_REQUIRED:
            hand_required.append(item)
        elif decision.status == MISSING:
            missing.append(item)

    (out_dir / "cross-coverage.md").write_text(_coverage_markdown(report), encoding="utf-8")
    (out_dir / "audit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "covered.json").write_text(json.dumps(covered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "excluded-illegal.json").write_text(json.dumps(excluded_illegal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "hand-required.json").write_text(json.dumps(hand_required, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "missing.json").write_text(json.dumps(missing, indent=2, sort_keys=True) + "\n" if missing else "", encoding="utf-8")
    return report


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
