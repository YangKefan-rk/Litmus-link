from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cli import main
from generator import audit_profile, audit_summary, generate_combinations, generate_profile, write_audit
from gui import options_payload, preview_payload
from models import EXCLUDED_ILLEGAL, EXCLUDED_UNSUPPORTED, GENERATED, HAND_REQUIRED, Combination
from profiles import profile_combinations
from qt_gui import qt_binding_status
from rule_file import RuleFileError, load_rule_file
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
    decision = evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush", params={"sync": "full_alias_sync"}))
    check(decision.status == GENERATED, "alias flush sync should generate")
    check("alias_sync_required" in decision.metadata, "alias sync metadata missing")
    check(evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable_nc_alias", cmo="flush_sync")).status == EXCLUDED_ILLEGAL, "fake CMO op must be illegal")
    check(evaluate(Combination("test", "vector_mem", "MP", "vector_load", "cacheable", vector="cross_page")).status == EXCLUDED_ILLEGAL, "cross_page must be a footprint parameter")
    check(evaluate(Combination("test", "cmo", "MP", "cmo", "cacheable", cmo="flush", params={"sew": "e32"})).status == EXCLUDED_UNSUPPORTED, "vector params require vector axis")
    check(evaluate(Combination("test", "rvwmo_base", "MP", "scalar_pair", "cacheable", params={"pte": "pa_remap"})).status == EXCLUDED_UNSUPPORTED, "VM params require TLB axis")


def test_generation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = generate_profile("smoke", root / "smoke")
        check(report["generated"] == 8, "smoke should generate 8 tests")
        check(report["generated_litmus"] > report["generated"], "smoke should expand generated litmus variants")
        check(report["solver"]["verified"] + report["solver"]["solver_unavailable"] > 0, "smoke should report scalar solver status")
        entries = validate_path(root / "smoke" / "@all")
        check(len(entries) == report["generated_litmus"], "validate should see expanded smoke tests")
        first_meta = json.loads((root / "smoke" / entries[0]).with_suffix(".meta.json").read_text(encoding="utf-8"))
        check(bool(first_meta["test_description"]["summary"]), "metadata test description summary missing")
        check(bool(first_meta.get("case_ir", {}).get("relations")), "metadata case IR relations missing")
        check(first_meta.get("solver", {}).get("status") in {"verified", "solver_unavailable", "solver_error", "not_applicable"}, "metadata solver status missing")
        check(first_meta.get("diagram", {}).get("schema") == "litmus-link.diagram.v1", "metadata diagram summary missing")
        check((root / "smoke" / entries[0]).with_suffix(".diagram.png").exists(), "diagram PNG missing")
        full = write_audit("full-cross", root / "audit")
        baseline = json.loads(Path("specs/profiles/full-cross-baseline.json").read_text())
        for key in ["profile", "total_combinations", "generated", "excluded_illegal", "excluded_unsupported", "hand_required", "missing"]:
            check(full[key] == baseline[key], f"full-cross audit changed from baseline for {key}")
        rows, vector = audit_profile("vector_mem")
        check(rows and vector["excluded_illegal"] > 0, "vector profile should include illegal exclusions")
        check(vector["hand_required"] > 0, "vector profile should include HAND cases")
        stress = audit_summary("stress-large", profile_combinations("stress-large"))
        baseline = json.loads(Path("specs/profiles/stress-large-baseline.json").read_text())
        check(stress == baseline, "stress-large audit changed from baseline")
        summary_dir = root / "summary"
        summary = write_audit("stress-large", summary_dir, summary_only=True)
        check(summary["total_combinations"] == 250360, "summary-only stress-large count mismatch")
        check(not (summary_dir / "covered.json").exists(), "summary-only audit should skip detail JSON")
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
        try:
            load_rule_file(illegal_file)
        except RuleFileError as exc:
            check("invalid vector value 'fof_strided'" in str(exc), "nonexistent FOF form should be rejected")
        else:
            raise AssertionError("fof_strided should not be accepted as a vector axis")
        cross_file = root / "cross-rules.json"
        cross_file.write_text(json.dumps({"name": "custom-cross", "axes": {"vector": ["unit_load"], "cmo": ["flush"]}, "limit": 10}), encoding="utf-8")
        cross_set = load_rule_file(cross_file)
        check(cross_set.combinations[0].category == "cross", "vector+cmo rule should infer cross category")
        cross_report = generate_combinations(cross_set.name, cross_set.combinations, root / "cross", source=str(cross_file))
        check(cross_report["generated"] == 1, "cross rule file should generate one test")
        litmus = next((root / "cross").glob("*.litmus")).read_text(encoding="utf-8")
        check("vle32.v" in litmus and "cbo.flush" in litmus, "cross rule litmus should include vector and CMO operations")
        param_file = root / "param-rules.json"
        param_file.write_text(
            json.dumps(
                {
                    "name": "custom-param-axis",
                    "axes": {"vector": ["unit_load"], "attribute": ["cacheable", "pbmt_nc"]},
                    "param_axes": {"sew": ["e32", "e64"], "footprint": ["same_line", "cross_page"]},
                    "param_defaults": {"stress": "load_queue_replay"},
                    "limit": 20,
                }
            ),
            encoding="utf-8",
        )
        param_set = load_rule_file(param_file)
        check(len(param_set.combinations) == 8, "param_axes expansion count mismatch")
        check({combination.params["sew"] for combination in param_set.combinations} == {"e32", "e64"}, "param_axes sew values missing")
        long_name = Combination(
            "test",
            "cross",
            "IRIW",
            "vector_load",
            "cacheable_nc_alias",
            cmo="flush",
            vector="indexed_unordered_load",
            params={
                "alias": "cacheable_nc",
                "elem_order": "ordered_elements",
                "footprint": "cross_page",
                "lmul": "m1",
                "mask": "masked",
                "sew": "e16",
                "stress": "store_buffer_full",
                "sync": "full_alias_sync",
                "tail": "ta_mu",
                "vl": "vl2",
            },
        ).name
        check(len(long_name) <= 180 and "params_" in long_name, "long parameterized name should be hashed")


def test_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke"
        check(main(["generate", "--profile", "smoke", "--out", str(out)]) == 0, "CLI generate failed")
        check(main(["validate", str(out / "@all")]) == 0, "CLI validate failed")
        check(main(["list", "rules"]) == 0, "CLI list rules failed")
        check(main(["list", "features"]) == 0, "CLI list features failed")
        rule_file = Path(tmp) / "rules.json"
        rule_file.write_text(json.dumps({"name": "cli-custom", "axes": {"cmo": ["flush"]}}), encoding="utf-8")
        check(main(["generate", "--rule-file", str(rule_file), "--out", str(Path(tmp) / "custom")]) == 0, "CLI rule-file generate failed")
        check(main(["audit", "--rule-file", str(rule_file), "--out", str(Path(tmp) / "audit")]) == 0, "CLI rule-file audit failed")
        check(main(["audit", "--profile", "stress-large", "--summary-only", "--out", str(Path(tmp) / "audit-large")]) == 0, "CLI summary-only audit failed")
        options = options_payload()
        check("stress-large" in options["profiles"], "GUI options should include stress-large")
        preview = preview_payload({"mode": "rule", "rule": {"name": "gui-test", "axes": {"vector": ["unit_load"]}, "param_axes": {"sew": ["e32"]}, "limit": 10}})
        check(preview["report"]["total_combinations"] == 1, "GUI preview count mismatch")
        check(preview["sample"][0]["litmus"].startswith("RISCV "), "GUI preview should include rendered litmus")
        check(bool(preview["sample"][0]["analysis"]["cycle"]), "GUI preview should include cycle analysis")
        check("solver" in preview["sample"][0], "GUI preview should include solver result")
        check("diagram" in preview["sample"][0], "GUI preview should include diagram result")
        mp_preview = preview_payload({"mode": "rule", "rule": {"name": "mp-cacheable", "axes": {"skeleton": ["MP"], "attribute": ["cacheable"]}, "limit": 10}, "sample_limit": 1})
        check(mp_preview["report"]["generated_litmus"] >= 6, "MP cacheable should expand to multiple litmus variants")
        check(len([item for item in mp_preview["sample"] if item.get("litmus")]) >= 6, "MP preview should include expanded variants")
        check(main(["qt-gui", "--check"]) == 0, "Qt GUI check should not require Qt")
        check("PyQt6" in qt_binding_status(), "Qt GUI status should include PyQt6")
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
