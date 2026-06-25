from __future__ import annotations

import json
import tempfile
from pathlib import Path

from litmus_link.cli import main
from litmus_link.generator import audit_profile, generate_profile, write_audit
from litmus_link.models import EXCLUDED_ILLEGAL, GENERATED, HAND_REQUIRED, Combination
from litmus_link.rules import evaluate
from litmus_link.upstream import import_upstream
from litmus_link.validator import validate_path


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


def test_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke"
        check(main(["generate", "--profile", "smoke", "--out", str(out)]) == 0, "CLI generate failed")
        check(main(["validate", str(out / "@all")]) == 0, "CLI validate failed")
        check(main(["list", "rules"]) == 0, "CLI list rules failed")


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
