from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cli import main
from generator import audit_profile, generate_combinations, generate_profile, write_audit
from models import EXCLUDED_ILLEGAL, GENERATED, HAND_REQUIRED, Combination
from rule_file import load_rule_file
from rules import evaluate
from upstream import import_upstream
from validator import validate_path


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_rules() -> None:
    check(evaluate(Combination("test", "pbmt_nc", "MP", "scalar_pair", "pbmt_reserved")).status == EXCLUDED_ILLEGAL, "PBMT=3 must be illegal")
    check(evaluate(Combination("test", "vm_tlb", "MP", "pte_update", "pbmt_nc", tlb="nonleaf_pbmt")).status == EXCLUDED_ILLEGAL, "non-leaf PBMT must be illegal")
    check(evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable", cmo="flush_offset4")).status == EXCLUDED_ILLEGAL, "CBO non-zero offset must be illegal")
    check(evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable", cmo="clean_csr_denied")).status == HAND_REQUIRED, "CSR-denied CBO must be HAND")
    check(evaluate(Combination("test", "vector", "MP", "vector_load", "cacheable", vector="fof_strided")).status == EXCLUDED_ILLEGAL, "strided FOF must be illegal")
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush_sync"))
    check(decision.status == GENERATED, "alias flush sync should generate")
    check("alias_sync_required" in decision.metadata, "alias sync metadata missing")


def test_generation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = generate_profile("smoke", root / "smoke")
        check(report["generated"] == 8, "smoke should generate 8 tests")
        entries = validate_path(root / "smoke" / "@all")
        check(len(entries) == 8, "validate should see 8 smoke tests")
        full = write_audit("full-cross", root / "audit")
        baseline = json.loads(Path("specs/profiles/full-cross-baseline.json").read_text())
        for key in ["profile", "total_combinations", "generated", "excluded_illegal", "excluded_unsupported", "hand_required", "missing"]:
            check(full[key] == baseline[key], f"full-cross audit changed from baseline for {key}")
        rows, vector = audit_profile("vector_mem")
        check(rows and vector["excluded_illegal"] > 0, "vector profile should include illegal exclusions")
        check(vector["hand_required"] > 0, "vector profile should include HAND cases")
        rule_file = root / "rules.json"
        rule_file.write_text(
            json.dumps(
                {
                    "name": "custom-vector-cmo",
                    "defaults": {"skeleton": "MP", "attribute": "cacheable"},
                    "axes": {"vector": ["unit_load", "unit_store"], "cmo": ["no_cmo", "flush"]},
                    "exclude": [{"vector": "unit_store", "cmo": "flush"}],
                    "limit": 10,
                }
            ),
            encoding="utf-8",
        )
        rule_set = load_rule_file(rule_file)
        check(len(rule_set.combinations) == 3, "rule file expansion count mismatch")
        custom_report = generate_combinations(rule_set.name, rule_set.combinations, root / "custom", source=str(rule_file))
        check(custom_report["total_combinations"] == 3, "custom rule report count mismatch")
        validate_path(root / "custom" / "@all")
        illegal_file = root / "illegal-rules.json"
        illegal_file.write_text(json.dumps({"name": "custom-illegal", "axes": {"vector": ["fof_strided"]}, "limit": 10}), encoding="utf-8")
        illegal_set = load_rule_file(illegal_file)
        illegal_report = generate_combinations(illegal_set.name, illegal_set.combinations, root / "illegal", source=str(illegal_file))
        check(illegal_report["generated"] == 0, "illegal rule file should not generate tests")
        check(illegal_report["excluded_illegal"] == 1, "illegal rule file should be audited as illegal")
        cross_file = root / "cross-rules.json"
        cross_file.write_text(json.dumps({"name": "custom-cross", "axes": {"vector": ["unit_load"], "cmo": ["flush"]}, "limit": 10}), encoding="utf-8")
        cross_set = load_rule_file(cross_file)
        check(cross_set.combinations[0].category == "cross", "vector+cmo rule should infer cross category")
        cross_report = generate_combinations(cross_set.name, cross_set.combinations, root / "cross", source=str(cross_file))
        check(cross_report["generated"] == 1, "cross rule file should generate one test")
        litmus = next((root / "cross").glob("*.litmus")).read_text(encoding="utf-8")
        check("vle32.v" in litmus and "cbo.flush" in litmus, "cross rule litmus should include vector and CMO operations")


def test_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke"
        check(main(["generate", "--profile", "smoke", "--out", str(out)]) == 0, "CLI generate failed")
        check(main(["validate", str(out / "@all")]) == 0, "CLI validate failed")
        check(main(["list", "rules"]) == 0, "CLI list rules failed")
        rule_file = Path(tmp) / "rules.json"
        rule_file.write_text(json.dumps({"name": "cli-custom", "axes": {"cmo": ["flush"]}}), encoding="utf-8")
        check(main(["generate", "--rule-file", str(rule_file), "--out", str(Path(tmp) / "custom")]) == 0, "CLI rule-file generate failed")
        check(main(["audit", "--rule-file", str(rule_file), "--out", str(Path(tmp) / "audit")]) == 0, "CLI rule-file audit failed")
        check(main(["generate", "--out", str(Path(tmp) / "missing-source")]) == 2, "CLI should require profile or rule file")
        check(
            main(["generate", "--profile", "smoke", "--rule-file", str(rule_file), "--out", str(Path(tmp) / "both-sources")]) == 2,
            "CLI should reject profile and rule file together",
        )


def test_upstream() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src"
        src.mkdir()
        (src / "MP.litmus").write_text("RISCV MP\n", encoding="utf-8")
        index = import_upstream(src, "riscv", root / "out")
        check(index["count"] == 1, "upstream index count mismatch")


def main_tests() -> int:
    test_rules()
    test_generation()
    test_cli()
    test_upstream()
    print("tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_tests())
