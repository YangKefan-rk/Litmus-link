from __future__ import annotations

import json
from pathlib import Path

import pytest

from litmus_link.native_diy import DEFAULT_DIY_RELAX, DEFAULT_DIY_SAFE, DiyConfig
from litmus_link.native_scalar import generate_native_diy, generate_native_templates
from litmus_link.generator import generate_combinations, generate_profile
from litmus_link.memory_layout import MemoryLayoutConfig
from litmus_link.models import Combination
from litmus_link.toolchain import HerdVerdict, tools_available
from litmus_link.verification import VerificationError, verify_path


def _corpus(path: Path) -> Path:
    generate_native_diy(
        out_dir=path,
        config=DiyConfig(safe=DEFAULT_DIY_SAFE, relax=DEFAULT_DIY_RELAX),
        annotations=("P",),
        limit=3,
        judge=False,
    )
    return path / "@all"


def test_verify_path_embedded_can_update_solver_artifacts(tmp_path: Path) -> None:
    atfile = _corpus(tmp_path)
    report = verify_path(atfile, backend="embedded", write=True)
    assert report["tests"] == 3
    assert report["counts"] == {"verified": 3}
    for entry in atfile.read_text(encoding="utf-8").splitlines():
        solver = json.loads((tmp_path / entry).with_suffix(".solver.json").read_text(encoding="utf-8"))
        assert solver["backend"] == "embedded"
        assert solver["verdict"] in {"observable", "forbidden"}


def test_verify_path_embedded_does_not_claim_unmodeled_upstream_file(tmp_path: Path) -> None:
    litmus = tmp_path / "upstream.litmus"
    litmus.write_text("RISCV Upstream\n{}\n P0;\n nop;\nexists (1=1)\n", encoding="utf-8")
    report = verify_path(litmus, backend="embedded")
    assert report["counts"] == {"not_applicable": 1}


def test_verify_path_checks_source_against_case_ir(tmp_path: Path) -> None:
    atfile = _corpus(tmp_path)
    litmus = tmp_path / atfile.read_text(encoding="utf-8").splitlines()[0]
    source = litmus.read_text(encoding="utf-8")
    litmus.write_text("(* test comment *)\n" + source, encoding="utf-8")
    assert verify_path(litmus)["counts"] == {"verified": 1}

    body, marker, _ = source.partition("\nexists\n")
    assert marker
    litmus.write_text(body + marker + "(0:x5=999)\n", encoding="utf-8")
    with pytest.raises(VerificationError, match="litmus body differs from case_ir metadata"):
        verify_path(litmus, write=True)


def test_verify_path_leaves_generated_observation_cases_unmodeled(tmp_path: Path) -> None:
    generate_profile("smoke", tmp_path)
    observations = []
    for path in tmp_path.glob("*.litmus"):
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if meta["case_ir"]["variant"] == "observation":
            observations.append(path)
    assert len(observations) == 3
    for path in observations:
        assert verify_path(path)["counts"] == {"not_applicable": 1}


def test_verify_path_herd7_uses_misaligned_scalar_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate_native_templates(
        out_dir=tmp_path,
        presets=("MP",),
        mechanisms=("po",),
        annotations=("P",),
        memory_layouts=(MemoryLayoutConfig("misaligned"),),
        limit=1,
        judge=False,
    )

    def judge(source: str, *, timeout: int, variants: tuple[str, ...]) -> HerdVerdict:
        assert "RISCV MP" in source
        assert timeout == 13
        assert variants == ("mixed", "unaligned")
        return HerdVerdict("observable", True, "Sometimes", 1, 1, 2, "", "")

    monkeypatch.setattr("litmus_link.verification.herd_judge", judge)
    report = verify_path(tmp_path, backend="herd7", timeout=13)
    assert report["counts"] == {"verified": 1}


def test_verify_path_dispatches_vector_metadata_to_vector_solver(tmp_path: Path) -> None:
    combination = Combination(
        "verify-vector",
        "vector_mem",
        "MP",
        "vector_load",
        "cacheable",
        vector="indexed_ordered_load",
        params={
            "variant": "fence_rw_rw",
            "sew": "e32",
            "lmul": "m1",
            "mask": "masked",
            "tail": "ta_ma",
            "vl": "vl4",
            "footprint": "same_line",
        },
    )
    report = generate_combinations("verify-vector", [combination], tmp_path)
    assert report["generated_litmus"] == 1
    litmus = next(tmp_path.glob("*.litmus"))

    embedded = verify_path(litmus, backend="embedded")
    result = embedded["results"][0]["result"]
    assert result["status"] == "verified"
    assert result["verdict"] == "forbidden"
    assert result["backend"] == "vector-aware-embedded"
    assert result["vector_ir"]["config"]["effective_vl"] == 4

    external = verify_path(litmus, backend="herd7")
    assert external["counts"] == {"external_unsupported": 1}
    crosscheck = verify_path(litmus, backend="crosscheck")
    cross_result = crosscheck["results"][0]["result"]
    assert cross_result["status"] == "verified"
    assert cross_result["cross_check"] == "external_unsupported"


@pytest.mark.skipif(not tools_available(), reason="herd7/riscv.cat is not installed")
def test_verify_path_crosschecks_herd7(tmp_path: Path) -> None:
    report = verify_path(_corpus(tmp_path), backend="crosscheck")
    assert report["counts"] == {"verified": 3}
    assert all(item["result"]["reason"] == "Embedded RVWMO and herd7 agree." for item in report["results"])
