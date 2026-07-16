from __future__ import annotations

import json
from pathlib import Path

import pytest

from litmus_link.native_diy import DEFAULT_DIY_RELAX, DEFAULT_DIY_SAFE, DiyConfig
from litmus_link.native_scalar import generate_native_diy
from litmus_link.toolchain import tools_available
from litmus_link.verification import verify_path


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


@pytest.mark.skipif(not tools_available(), reason="herd7/riscv.cat is not installed")
def test_verify_path_crosschecks_herd7(tmp_path: Path) -> None:
    report = verify_path(_corpus(tmp_path), backend="crosscheck")
    assert report["counts"] == {"verified": 3}
    assert all(item["result"]["reason"] == "Embedded RVWMO and herd7 agree." for item in report["results"])
