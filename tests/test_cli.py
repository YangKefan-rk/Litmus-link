from pathlib import Path
import json
import subprocess
import sys

from cli import main
from gui import options_payload, preview_payload


def test_cli_generate_and_validate(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "smoke"
    assert main(["generate", "--profile", "smoke", "--out", str(out)]) == 0
    assert main(["validate", str(out / "@all")]) == 0
    captured = capsys.readouterr()
    assert "validated 8 litmus files" in captured.out


def test_cli_list_rules(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["list", "rules"]) == 0
    assert "pbmt_leaf_only" in capsys.readouterr().out


def test_cli_list_features(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["list", "features"]) == 0
    out = capsys.readouterr().out
    assert "vector" in out
    assert "pbmt_nc" in out


def test_cli_rule_file_generate_and_audit(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    rule_file = tmp_path / "rules.json"
    rule_file.write_text(json.dumps({"name": "cli-custom", "axes": {"cmo": ["flush"]}}), encoding="utf-8")
    out = tmp_path / "custom"
    assert main(["generate", "--rule-file", str(rule_file), "--out", str(out)]) == 0
    assert (out / "LL_cmo_MP_cmo_cacheable_no_tlb_flush_none.litmus").exists()
    assert main(["audit", "--rule-file", str(rule_file), "--out", str(tmp_path / "audit")]) == 0
    assert "cli-custom" in capsys.readouterr().out


def test_cli_summary_only_audit(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    assert main(["audit", "--profile", "stress-large", "--summary-only", "--out", str(out)]) == 0
    assert (out / "audit-report.json").exists()
    assert not (out / "covered.json").exists()


def test_gui_options_and_preview() -> None:
    options = options_payload()
    assert "stress-large" in options["profiles"]
    assert "sew" in options["param_axes"]
    preview = preview_payload(
        {
            "mode": "rule",
            "rule": {
                "name": "gui-test",
                "axes": {"vector": ["unit_load"], "attribute": ["cacheable"]},
                "param_axes": {"sew": ["e32"], "footprint": ["same_line"]},
                "limit": 10,
            },
        }
    )
    assert preview["report"]["total_combinations"] == 1
    assert preview["sample"][0]["combination"]["params"]["sew"] == "e32"


def test_cli_requires_exactly_one_generation_source(tmp_path: Path) -> None:
    assert main(["generate", "--out", str(tmp_path / "out")]) == 2
    assert main(["generate", "--profile", "smoke", "--rule-file", str(tmp_path / "rules.json"), "--out", str(tmp_path / "out")]) == 2


def test_python_m_cli_entrypoint_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cli", "list", "profiles"],
        check=False,
        capture_output=True,
        env={"PYTHONPATH": "src"},
        text=True,
    )
    assert result.returncode == 0
    assert "smoke" in result.stdout
