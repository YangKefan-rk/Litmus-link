from pathlib import Path
import json
import pytest
import subprocess
import sys

from litmus_link.cli import main
from litmus_link.workflow import generate_payload, options_payload, preview_payload
from litmus_link.qt_gui import _summary_text, qt_binding_status
from litmus_link.toolchain import tools_available
from litmus_link.validator import validate_path


def test_cli_generate_and_validate(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "smoke"
    assert main(["generate", "--profile", "smoke", "--out", str(out)]) == 0
    assert main(["validate", str(out / "@all")]) == 0
    captured = capsys.readouterr()
    assert "validated 22 litmus files" in captured.out


def test_cli_list_rules(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["list", "rules"]) == 0
    assert "pbmt_leaf_only" in capsys.readouterr().out


def test_cli_list_features(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["list", "features"]) == 0
    out = capsys.readouterr().out
    assert "vector" in out
    assert "pbmt_nc" in out


def test_cli_scalar_catalog_and_tools(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["scalar", "catalog"]) == 0
    assert '"MP"' in capsys.readouterr().out
    expected = 0 if tools_available() else 1
    assert main(["scalar", "tools"]) == expected
    assert '"diycross7"' in capsys.readouterr().out


def test_cli_native_catalog_and_generation(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["native", "catalog"]) == 0
    catalog_output = capsys.readouterr().out
    assert '"MP": 416' in catalog_output
    out = tmp_path / "native-mp"
    assert main([
        "native", "templates", "--skeleton", "MP", "--limit", "3",
        "--no-judge", "--out", str(out),
    ]) == 0
    report_output = capsys.readouterr().out
    assert '"available_litmus": 106496' in report_output
    assert len(validate_path(out / "@all")) == 3


def test_cli_native_diy_and_verify(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "native-diy"
    assert main([
        "native", "diy", "--limit", "3", "--solver-backend", "embedded",
        "--out", str(out),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["available_litmus"] == 30
    assert report["verdicts"] == {"verified": 3}
    assert main(["verify", str(out / "@all"), "--backend", "embedded"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["counts"] == {"verified": 3}


def test_cli_native_generates_real_no_mag_misaligned_cases(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "native-misaligned"
    assert main([
        "native", "templates", "--skeleton", "MP", "--mechanism", "po",
        "--different-location-only", "--annotation", "P",
        "--memory-layout", "misaligned", "--misalign-width", "64",
        "--misalign-boundary", "cross64", "--limit", "1",
        "--solver-backend", "embedded", "--out", str(out),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["available_litmus"] == 1
    assert report["verdicts"] == {"verified": 1}
    litmus = next(out.glob("*.litmus")).read_text(encoding="utf-8")
    assert "sd " in litmus and ",60(" in litmus
    metadata = json.loads(next(out.glob("*.meta.json")).read_text(encoding="utf-8"))
    assert metadata["native"]["memory_layout"]["atomicity_model"] == "byte_level_no_mag"
    assert metadata["native"]["memory_layout"]["mag_bytes"] is None


def test_cli_native_diy_prefix(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "native-prefix"
    assert main([
        "native", "diy", "--prefix", "PodWW Rfe", "--limit", "2",
        "--no-judge", "--out", str(out),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["available_litmus"] == 107
    assert report["audit"]["expanded_prefixes"]


def test_cli_native_diy_min_relax_implies_mix(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "native-mix"
    assert main([
        "native", "diy", "--min-relax", "2", "--no-judge", "--out", str(out),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["available_litmus"] == 3
    assert report["audit"]["config"]["mix"] is True


def test_cli_verify_fails_when_embedded_backend_cannot_model_file(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    litmus = tmp_path / "external.litmus"
    litmus.write_text("RISCV External\n{}\n P0;\n nop;\nexists (1=1)\n", encoding="utf-8")
    assert main(["verify", str(litmus), "--backend", "embedded"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["counts"] == {"not_applicable": 1}


@pytest.mark.skipif(not tools_available(), reason="herdtools7 toolchain not installed")
def test_cli_scalar_cross_round_trip(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "scalar"
    assert main([
        "scalar", "cross", "--skeleton", "MP", "--mechanism", "po",
        "--no-judge", "--out", str(out),
    ]) == 0
    assert main(["validate", str(out / "@all")]) == 0
    assert "validated 1 litmus files" in capsys.readouterr().out


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


@pytest.mark.skipif(not tools_available(), reason="herdtools7 toolchain not installed")
def test_gui_scalar_mp_preview_uses_full_official_family() -> None:
    preview = preview_payload(
        {
            "mode": "scalar",
            "engine": "cross",
            "skeletons": ["MP"],
            "mechanisms": ["po", "fence", "dependency"],
            "sample_limit": 1000,
            "judge": True,
        }
    )
    assert preview["available_litmus"] == 18
    assert preview["displayed_litmus"] == 18
    assert len(preview["sample"]) == 18
    assert all(not item["name"].startswith("LL_custom_") for item in preview["sample"])
    assert all(item["solver"]["status"] == "verified" for item in preview["sample"])
    assert all(Path(item["diagram"]["png"]).exists() for item in preview["sample"])


@pytest.mark.skipif(not tools_available(), reason="herdtools7 toolchain not installed")
def test_gui_scalar_generate_honors_file_limit(tmp_path: Path) -> None:
    out = tmp_path / "gui-scalar"
    report = generate_payload(
        {
            "mode": "scalar",
            "engine": "cross",
            "skeletons": ["MP"],
            "mechanisms": ["po", "fence", "dependency"],
            "limit": 3,
            "judge": True,
            "out": str(out),
        }
    )
    assert report["available_litmus"] == 18
    assert report["generated_litmus"] == 3
    assert report["generation_limited"] is True
    assert report["verdicts"] == {"verified": 3}
    assert len(validate_path(out / "@all")) == 3


def test_gui_native_scalar_preview_is_exhaustive_for_configured_mp_domain() -> None:
    preview = preview_payload(
        {
            "mode": "scalar",
            "engine": "native_templates",
            "skeletons": ["MP"],
            "mechanisms": ["po", "fence", "dependency"],
            "include_same": True,
            "sample_limit": 3,
            "judge": False,
        }
    )
    assert preview["available_litmus"] == 106496
    assert preview["displayed_litmus"] == 3
    assert all(item["name"].startswith("NATIVE_MP_") for item in preview["sample"])
    assert all(item["decision"]["reason"].startswith("Scalar RVWMO test exhaustively") for item in preview["sample"])
    assert all(len(item["case_ir"]["relations"]) == 4 for item in preview["sample"])


def test_gui_native_diy_preview_uses_embedded_verification() -> None:
    preview = preview_payload(
        {
            "mode": "scalar",
            "engine": "native_diy",
            "sample_limit": 4,
            "judge": True,
            "solver_backend": "embedded",
            "annotations": ["P"],
            "diy": {},
        }
    )
    assert preview["available_litmus"] == 30
    assert preview["displayed_litmus"] == 4
    assert all(item["solver"]["backend"] == "embedded" for item in preview["sample"])
    assert all(item["solver"]["status"] == "verified" for item in preview["sample"])


def test_gui_native_preview_expands_misaligned_layout_configuration() -> None:
    preview = preview_payload(
        {
            "mode": "scalar",
            "engine": "native_templates",
            "skeletons": ["MP"],
            "mechanisms": ["po"],
            "include_same": False,
            "annotations": ["P"],
            "sample_limit": 10,
            "judge": True,
            "solver_backend": "embedded",
            "memory_layout": {
                "enabled": True,
                "include_aligned": False,
                "modes": ["misaligned", "mixed"],
                "width_bits": [16, 64],
                "boundaries": ["same16", "cross64"],
            },
        }
    )
    assert preview["available_litmus"] == 6
    assert preview["displayed_litmus"] == 6
    classifications = preview["classification_counts"]
    assert classifications["displayed_cases"] == 6
    assert classifications["groups"]["status"] == {"generated": 6}
    assert classifications["groups"]["skeleton"] == {"MP": 6}
    assert classifications["groups"]["memory_layout"] == {"misaligned": 4, "mixed": 2}
    assert all(item["solver"]["status"] == "verified" for item in preview["sample"])
    assert any("mixed-size" in item["litmus"] or "mixed_" in item["name"] for item in preview["sample"])
    assert any(",60(" in item["litmus"] for item in preview["sample"])
    for item in preview["sample"]:
        interpretation = item["analysis"]["outcome_interpretation"]
        assert "hardware-observation/prose-spec" not in interpretation
        expected = "FORBIDDEN" if item["solver"]["verdict"] == "forbidden" else "OBSERVABLE"
        assert expected in interpretation
        assert "byte-level no-MAG solver" in interpretation


def test_gui_options_expose_only_no_mag_atomicity() -> None:
    memory = options_payload()["native_scalar"]["memory_layout"]
    assert memory["atomicity_model"] == "byte_level_no_mag"
    assert memory["mag_supported"] is False


def test_gui_native_diy_preview_accepts_fixed_prefixes() -> None:
    preview = preview_payload(
        {
            "mode": "scalar",
            "engine": "native_diy",
            "sample_limit": 2,
            "judge": False,
            "annotations": ["P"],
            "diy": {"prefixes": [["PodWW", "Rfe"]]},
        }
    )
    assert preview["available_litmus"] == 107
    assert preview["displayed_litmus"] == 2


def test_qt_gui_check(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["qt-gui", "--check"]) == 0
    assert "PyQt6" in capsys.readouterr().out
    assert "PySide6" in qt_binding_status()


def test_qt_summary_text_highlights_generated_artifacts() -> None:
    summary = _summary_text(
        "Generate Files",
        {
            "profile": "qt-custom",
            "total_combinations": 4,
            "generated": 3,
            "generated_litmus": 7,
            "excluded_illegal": 1,
            "excluded_unsupported": 0,
            "hand_required": 0,
            "missing": 0,
            "solver": {"verified": 5, "conflict": 0, "not_applicable": 2},
        },
        "out/qt-custom",
    )
    assert "Generate Files complete" in summary
    assert "generated combinations: 3" in summary
    assert "litmus files: 7" in summary
    assert "solver results: 7" in summary
    assert "out/qt-custom/@all" in summary
    assert "out/qt-custom/audit-report.json" in summary


def test_gui_generate_corpus_computes_solver_and_diagram_by_default(tmp_path: Path) -> None:
    from litmus_link.corpus_riscv import corpus_available

    if not corpus_available():
        return
    out = tmp_path / "gui-mp"
    report = generate_payload(
        {
            "mode": "rule",
            "rule": {"name": "mp-cacheable", "axes": {"skeleton": ["MP"], "attribute": ["cacheable"]}, "limit": 10},
            "out": str(out),
            "generate_limit": 2,
        }
    )
    assert report["verdict_mode"] == "computed"
    assert report["generated_litmus"] == 2
    assert len(list(out.glob("*.litmus"))) == 2
    assert len(list(out.glob("*.solver.json"))) == 2
    assert len(list(out.glob("*.diagram.png"))) == 2


def test_gui_generate_uses_rule_limit_as_total_litmus_cap(tmp_path: Path) -> None:
    from litmus_link.corpus_riscv import corpus_available

    if not corpus_available():
        return
    out = tmp_path / "gui-mp-rule-limit"
    report = generate_payload(
        {
            "mode": "rule",
            "rule": {"name": "mp-cacheable", "axes": {"skeleton": ["MP"], "attribute": ["cacheable"]}, "limit": 3},
            "out": str(out),
        }
    )
    assert report["generated_litmus"] == 3
    assert report["available_litmus"] > 3
    assert report["generation_limit"] == 3
    assert report["generation_limited"] is True
    assert len(list(out.glob("*.litmus"))) == 3
    assert len(list(out.glob("*.solver.json"))) == 3
    assert len(list(out.glob("*.diagram.png"))) == 3


def test_cli_requires_exactly_one_generation_source(tmp_path: Path) -> None:
    assert main(["generate", "--out", str(tmp_path / "out")]) == 2
    assert main(["generate", "--profile", "smoke", "--rule-file", str(tmp_path / "rules.json"), "--out", str(tmp_path / "out")]) == 2


def test_cli_asm_check_returns_nonzero_on_assembler_failure(tmp_path: Path) -> None:
    litmus = tmp_path / "T.litmus"
    litmus.write_text("RISCV T\n{\n}\n P0 ;\n definitely.not.an.op ;\nexists\n(0:x1=0)\n", encoding="utf-8")
    atfile = tmp_path / "@all"
    atfile.write_text("T.litmus\n", encoding="utf-8")
    fake_gcc = tmp_path / "fake-gcc"
    fake_gcc.write_text("#!/bin/sh\necho assembler nope >&2\nexit 1\n", encoding="utf-8")
    fake_gcc.chmod(0o755)
    assert main(["asm-check", str(atfile), "--gcc", str(fake_gcc)]) == 1


def test_python_m_cli_entrypoint_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "litmus_link", "list", "profiles"],
        check=False,
        capture_output=True,
        env={"PYTHONPATH": "src"},
        text=True,
    )
    assert result.returncode == 0
    assert "smoke" in result.stdout
