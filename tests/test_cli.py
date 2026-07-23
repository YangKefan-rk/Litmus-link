from pathlib import Path
import json
import pytest
import subprocess
import sys

from litmus_link.cli import main
from litmus_link.workflow import audit_payload, generate_payload, options_payload, preview_payload
from litmus_link.qt_gui import _summary_text, qt_binding_status
from litmus_link.toolchain import tools_available
from litmus_link.validator import validate_path


def test_cli_generate_and_validate(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "smoke"
    assert main(["generate", "--profile", "smoke", "--out", str(out)]) == 0
    assert main(["validate", str(out / "@all")]) == 0
    captured = capsys.readouterr()
    assert "validated 24 litmus files" in captured.out


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
    assert '"available_litmus": 260000' in report_output
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
    assert (out / "MP+cbo.flush.litmus").exists()
    assert main(["audit", "--rule-file", str(rule_file), "--out", str(tmp_path / "audit")]) == 0
    assert "cli-custom" in capsys.readouterr().out


def test_cli_summary_only_audit(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    assert main(["audit", "--profile", "stress-large", "--summary-only", "--out", str(out)]) == 0
    assert (out / "audit-report.json").exists()
    assert not (out / "covered.json").exists()


def test_gui_options_expose_only_scalar_and_vector_workflows() -> None:
    options = options_payload()
    assert set(options) == {"axes", "param_axes", "native_scalar", "vector_native"}
    assert set(options["axes"]) == {"skeleton", "vector"}
    assert set(options["param_axes"]) == {"sew", "lmul", "index_eew", "mask", "tail", "vl"}
    assert set(options["axes"]["vector"]) == {
        "none",
        "unit_load",
        "unit_store",
        "strided_load",
        "strided_store",
        "indexed_unordered_load",
        "indexed_unordered_store",
        "indexed_ordered_load",
        "indexed_ordered_store",
    }
    assert "elem_order" not in options["param_axes"]
    assert options["param_axes"]["vl"] == ["vl1", "vl2", "vl4", "vl8", "vl16", "vl32", "vl64", "vlmax"]
    assert options["param_axes"]["index_eew"] == ["ei8", "ei16", "ei32", "ei64"]
    with pytest.raises(ValueError, match="only scalar and vector"):
        preview_payload({"mode": "profile", "profile": "smoke"})
    with pytest.raises(ValueError, match="only scalar and vector"):
        preview_payload({"mode": "rule", "rule": {"name": "obsolete"}})
    with pytest.raises(ValueError, match="only scalar and vector"):
        audit_payload({"mode": "profile", "profile": "smoke"})
    with pytest.raises(ValueError, match="only scalar and vector"):
        generate_payload({"mode": "rule", "rule": {"name": "obsolete"}})


def test_dedicated_vector_mode_uses_relation_cycles_and_random_final_cases(tmp_path: Path) -> None:
    payload = {
        "mode": "vector",
        "name": "vector-gui-focused",
        "complete": False,
        "skeletons": ["MP"],
        "mechanisms": ["po", "dependency"],
        "endpoint_modes": ["P", "AMO"],
        "forms": ["unit_load", "unit_store"],
        "sew": ["e32"],
        "lmul": ["m1"],
        "index_eew": ["ei8"],
        "mask": ["unmasked"],
        "tail": ["ta_ma"],
        "vl": ["vl1"],
        "alignments": ["aligned"],
        "sample_limit": 5,
        "generate_limit": 3,
        "random_seed": 9,
        "out": str(tmp_path / "vector-gui"),
    }
    preview = preview_payload(payload)
    assert preview["source"] == "litmus-link-native-cycle+rvv"
    assert preview["report"]["total_combinations"] > 5
    assert preview["report"]["generated_litmus"] == preview["report"]["total_combinations"]
    assert len(preview["sample"]) == 5
    assert {item["case_ir"]["variant"] for item in preview["sample"]} == {
        "vector-native-cycle"
    }
    assert all("Rfe" in item["name"] or "Fre" in item["name"] for item in preview["sample"])

    generated = generate_payload(payload)
    assert generated["generated_litmus"] == 3
    assert sum(generated["solver"].values()) == 3
    assert generated["generation_mode"] == "balanced"
    assert generated["sampling"] == "balanced-skeleton-coverage-random-without-replacement"


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
    assert all(item["diagram"]["status"] in {"deferred", "ready"} for item in preview["sample"])


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
    assert preview["available_litmus"] == 260000
    assert preview["displayed_litmus"] == 3
    assert all(item["name"].startswith("MP+") for item in preview["sample"])
    assert all("NATIVE" not in item["name"] for item in preview["sample"])
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


def test_gui_native_cycle_preview_filters_impossible_coherence_cycles() -> None:
    preview = preview_payload(
        {
            "mode": "scalar",
            "engine": "native_cycles",
            "mechanisms": ["po"],
            "annotations": ["P"],
            "include_same": True,
            "include_internal": True,
            "min_size": 2,
            "size": 3,
            "nprocs": 2,
            "max_accesses_per_proc": 4,
            "sample_limit": 20,
            "judge": True,
            "solver_backend": "embedded",
        }
    )
    # Fre Wsi Rfe used to reach lowering first and abort the whole GUI preview:
    # its rf/fr/co constraints require opposite write orders on one location.
    assert preview["available_litmus"] == 5
    assert preview["displayed_litmus"] == 5
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
    assert preview["available_litmus"] == 32
    assert preview["displayed_litmus"] == 10
    classifications = preview["classification_counts"]
    assert classifications["displayed_cases"] == 10
    assert classifications["groups"]["status"] == {"generated": 10}
    assert classifications["groups"]["skeleton"] == {"MP": 10}
    assert classifications["groups"]["memory_layout"] == {"misaligned": 4, "mixed": 6}
    assert all(item["solver"]["status"] == "verified" for item in preview["sample"])
    assert any("mixed-size" in item["litmus"] or "mixed_" in item["name"] for item in preview["sample"])
    assert any(",60(" in item["litmus"] for item in preview["sample"])
    for item in preview["sample"]:
        interpretation = item["analysis"]["outcome_interpretation"]
        assert "hardware-observation/prose-spec" not in interpretation
        expected = "FORBIDDEN" if item["solver"]["verdict"] == "forbidden" else "OBSERVABLE"
        assert expected in interpretation
        assert "byte-level no-MAG solver" in interpretation


def test_gui_native_preview_expands_aligned_atomic_layouts() -> None:
    preview = preview_payload(
        {
            "mode": "scalar",
            "engine": "native_templates",
            "skeletons": ["MP"],
            "mechanisms": ["po"],
            "include_same": False,
            "annotations": ["AMO"],
            "sample_limit": 10,
            "judge": True,
            "solver_backend": "embedded",
            "memory_layout": {
                "enabled": True,
                "include_aligned": False,
                "modes": ["atomic", "atomic_mixed"],
                "width_bits": [16, 32, 64],
                "boundaries": ["same16"],
                "atomic_overlaps": ["same_start", "partial_overlap"],
            },
        }
    )
    # MP with one selected relation cycle has three uniform widths plus every
    # four-event non-uniform width assignment (3^4 - 3), for both overlap
    # shapes.  The preview row cap remains independent from the audit count.
    assert preview["available_litmus"] == 159
    assert preview["displayed_litmus"] == 10
    assert any("amoor.h" in item["litmus"] for item in preview["sample"])
    assert any("amoor.d" in item["litmus"] for item in preview["sample"])
    assert any(
        "amoor.h" in item["litmus"] and "Zabha" in item["decision"]["requires"]
        for item in preview["sample"]
    )
    assert any(
        item["solver"]["status"] == "not_applicable"
        and item["case_ir"]["expected_outcome"] == "manual_oracle_required"
        for item in preview["sample"]
    )


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
            "solver_workers": 16,
            "solver": {"verified": 5, "conflict": 0, "not_applicable": 2},
        },
        "out/qt-custom",
    )
    assert "Generate Files complete" in summary
    assert "generated combinations: 3" in summary
    assert "litmus files: 7" in summary
    assert "solver results: 7" in summary
    assert "Solver processes used: 16" in summary
    assert "out/qt-custom/@all" in summary
    assert "out/qt-custom/audit-report.json" in summary


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
