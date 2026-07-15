import json
from pathlib import Path

import pytest

from litmus_link.scalar import (
    DEFAULT_MECHANISMS,
    SCALAR_PRESETS,
    ScalarGenerationError,
    cross_edge_args,
    generate_scalar_cross,
    generate_scalar_enumerated,
    parse_custom_cycle,
    scalar_catalog,
)
from litmus_link.toolchain import tools_available
from litmus_link.validator import validate_path


requires_tools = pytest.mark.skipif(not tools_available(), reason="herdtools7 toolchain not installed")


def test_scalar_catalog_exposes_basic_skeletons_and_mechanisms() -> None:
    catalog = scalar_catalog()
    assert {"MP", "LB", "SB", "WRC", "RWC", "IRIW", "ISA2"}.issubset(catalog["presets"])
    assert set(catalog["mechanisms"]) == {"po", "fence", "dependency"}
    assert catalog["enumerate_defaults"]["size"] == 4
    assert catalog["enumerate_defaults"]["nprocs"] == 2


def test_mp_cross_axes_expand_by_access_shape() -> None:
    axes = cross_edge_args(SCALAR_PRESETS["MP"], DEFAULT_MECHANISMS)
    assert axes[0] == "Rfe"
    assert axes[2] == "Fre"
    assert set(axes[1].split(",")) == {
        "PodRR",
        "Fence.r.rwdRR",
        "Fence.rw.rwdRR",
        "DpAddrdR",
        "DpCtrldR",
        "DpCtrlFenceIdR",
    }
    assert set(axes[3].split(",")) == {"PodWW", "Fence.w.wdWW", "Fence.rw.wdWW"}


def test_dependency_only_rejects_write_to_read_shape() -> None:
    with pytest.raises(ScalarGenerationError, match="no WR edge"):
        cross_edge_args(SCALAR_PRESETS["SB"], ["dependency"])


def test_custom_cycle_parser_preserves_alternative_axes() -> None:
    assert parse_custom_cycle("Rfe; PodRR, Fence.rw.rwdRR ;Fre;PodWW") == (
        "Rfe",
        "PodRR,Fence.rw.rwdRR",
        "Fre",
        "PodWW",
    )
    with pytest.raises(ScalarGenerationError):
        parse_custom_cycle("Rfe")


def test_custom_cycle_rejects_unsafe_name(tmp_path: Path) -> None:
    with pytest.raises(ScalarGenerationError, match="unsafe generated test name"):
        generate_scalar_cross(
            out_dir=tmp_path,
            presets=[],
            custom_name="../escape",
            custom_cycle=("Rfe", "PodRR", "Fre", "PodWW"),
            judge=False,
        )


@requires_tools
def test_cross_generates_and_herd_judges_mp_family(tmp_path: Path) -> None:
    report = generate_scalar_cross(out_dir=tmp_path, presets=["MP"], judge=True, timeout=30)
    assert report["available_litmus"] == 18
    assert report["generated_litmus"] == 18
    assert report["verdicts"] == {"verified": 18}
    assert len(validate_path(tmp_path / "@all")) == 18

    verdicts = {
        json.loads(path.read_text(encoding="utf-8"))["verdict"]
        for path in tmp_path.glob("*.solver.json")
    }
    assert verdicts == {"observable", "forbidden"}
    meta = json.loads((tmp_path / "MP.meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == "litmus-link.scalar-meta.v1"
    assert meta["generator"]["engine"] == "diycross7"
    assert meta["cycle"] == "Rfe PodRR Fre PodWW"


@requires_tools
def test_diy_enumerates_basic_scalar_domain(tmp_path: Path) -> None:
    report = generate_scalar_enumerated(out_dir=tmp_path, judge=False, timeout=30)
    assert report["available_litmus"] >= 20
    assert report["generated_litmus"] == report["available_litmus"]
    assert report["verdicts"] == {"unchecked": report["generated_litmus"]}
    assert len(validate_path(tmp_path)) == report["generated_litmus"]
    assert any(path.name.startswith("MP+") for path in tmp_path.glob("*.litmus"))
    assert (tmp_path / "LB.litmus").exists()
    assert (tmp_path / "SB.litmus").exists()


@requires_tools
def test_cross_limit_is_reported_and_indexed(tmp_path: Path) -> None:
    report = generate_scalar_cross(
        out_dir=tmp_path,
        presets=["MP"],
        mechanisms=["po", "fence"],
        limit=2,
        judge=False,
        timeout=30,
    )
    assert report["available_litmus"] == 9
    assert report["generated_litmus"] == 2
    assert report["generation_limited"] is True
    assert len(validate_path(tmp_path / "@all")) == 2
