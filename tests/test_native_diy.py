from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from litmus_link.native_diy import (
    DEFAULT_DIY_RELAX,
    DEFAULT_DIY_SAFE,
    DIY_MODES,
    DIY_OBSERVERS,
    DiyConfig,
    enumerate_diy_cycles,
    expand_relaxations,
)
from litmus_link.native_scalar import generate_native_diy
from litmus_link.toolchain import DIY


def _canonical(labels: list[str]) -> tuple[str, ...]:
    normalized = tuple(label.replace("Coe", "Wse").replace("Coi", "Wsi") for label in labels)
    return min(normalized[index:] + normalized[:index] for index in range(len(normalized)))


def test_native_diy_default_domain_is_stable() -> None:
    cycles, audit = enumerate_diy_cycles(
        DiyConfig(safe=DEFAULT_DIY_SAFE, relax=DEFAULT_DIY_RELAX)
    )
    assert len(cycles) == 30
    assert audit["accepted"] == 30
    assert audit["excluded"]["no_relaxation"] > 0
    assert len({cycle.canonical_key for cycle in cycles}) == len(cycles)


def test_native_diy_policy_mode_counts_are_stable_offline() -> None:
    expected = {
        "default": 30,
        "sc": 30,
        "thin": 0,
        "uni": 58,
        "critical": 30,
        "free": 59,
        "ppo": 58,
        "transitive": 30,
        "total": 59,
        "mixedcheck": 30,
    }
    actual = {
        mode: len(enumerate_diy_cycles(
            DiyConfig(safe=DEFAULT_DIY_SAFE, relax=DEFAULT_DIY_RELAX, mode=mode)
        )[0])
        for mode in DIY_MODES
    }
    assert actual == expected


def test_native_diy_expands_wildcards_sequences_and_cumulativity() -> None:
    wildcards = expand_relaxations(
        ["Pod**"], include_same=False, include_internal=True
    )
    assert {item.label for item in wildcards} == {"PodRR", "PodRW", "PodWR", "PodWW"}
    sequences = expand_relaxations(
        ["[Rfe,PodRR]"], include_same=False, include_internal=True
    )
    assert [edge.label for edge in sequences[0].edges] == ["Rfe", "PodRR"]
    cumulative = expand_relaxations(
        ["ACFence.rw.rwdRR"], include_same=False, include_internal=True
    )
    assert [edge.label for edge in cumulative[0].edges] == ["Rfe", "Fence.rw.rwdRR"]


def test_native_diy_expands_official_edge_shorthands() -> None:
    fence_i = expand_relaxations(
        ["Fence.i"], include_same=False, include_internal=True
    )
    assert {item.label for item in fence_i} == {
        "Fence.idRR", "Fence.idRW", "Fence.idWR", "Fence.idWW",
    }
    fence_i_wildcard = expand_relaxations(
        ["Fence.i***"], include_same=False, include_internal=True
    )
    assert {item.label for item in fence_i_wildcard} == {
        "Fence.idRR", "Fence.idRW", "Fence.idWR", "Fence.idWW",
    }
    assert all(item.first.lowering == "fence.i" for item in fence_i_wildcard)
    fenced_rr = expand_relaxations(
        ["FencedRR"], include_same=False, include_internal=True
    )
    assert len(fenced_rr) == 12
    assert all(item.first.location == "different" and item.first.shape == "RR" for item in fenced_rr)
    address = expand_relaxations(
        ["DpAddr"], include_same=False, include_internal=True
    )
    assert {item.label for item in address} == {"DpAddrdR", "DpAddrdW"}
    control = expand_relaxations(
        ["Ctrl**"], include_same=False, include_internal=True
    )
    assert {item.label for item in control} == {"DpCtrldR", "DpCtrldW"}


@pytest.mark.parametrize(
    "config",
    [
        DiyConfig(safe=DEFAULT_DIY_SAFE, relax=DEFAULT_DIY_RELAX, moreedges=True),
        DiyConfig(safe=DEFAULT_DIY_SAFE, relax=DEFAULT_DIY_RELAX, unrollatomic=2),
    ],
)
def test_native_diy_rejects_unimplemented_atom_expansions(config: DiyConfig) -> None:
    with pytest.raises(ValueError, match="not implemented"):
        enumerate_diy_cycles(config)


@pytest.mark.skipif(not DIY.exists(), reason="official diy7 is not installed")
def test_native_diy_default_cycles_match_official_diy7() -> None:
    command = [
        str(DIY),
        "-arch", "RISCV",
        "-safe", " ".join(DEFAULT_DIY_SAFE),
        "-relax", " ".join(DEFAULT_DIY_RELAX),
        "-size", "4",
        "-nprocs", "2",
        "-cycleonly", "true",
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    official = {
        _canonical(line.split(":", 1)[1].split())
        for line in process.stdout.splitlines()
        if ":" in line
    }
    native, _audit = enumerate_diy_cycles(
        DiyConfig(safe=DEFAULT_DIY_SAFE, relax=DEFAULT_DIY_RELAX)
    )
    assert {cycle.canonical_key for cycle in native} == official


@pytest.mark.skipif(not DIY.exists(), reason="official diy7 is not installed")
def test_native_diy_prefix_cycles_match_official_diy7() -> None:
    command = [
        str(DIY),
        "-arch", "RISCV",
        "-safe", " ".join(DEFAULT_DIY_SAFE),
        "-relax", " ".join(DEFAULT_DIY_RELAX),
        "-size", "4",
        "-nprocs", "2",
        "-prefix", "PodWW Rfe",
        "-cycleonly", "true",
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    official = {
        _canonical(line.split(":", 1)[1].split())
        for line in process.stdout.splitlines()
        if ":" in line
    }
    native, audit = enumerate_diy_cycles(
        DiyConfig(
            safe=DEFAULT_DIY_SAFE,
            relax=DEFAULT_DIY_RELAX,
            prefixes=(("PodWW", "Rfe"),),
        )
    )
    assert len(official) == 107
    assert {cycle.canonical_key for cycle in native} == official
    assert audit["expanded_prefixes"]


@pytest.mark.skipif(not DIY.exists(), reason="official diy7 is not installed")
def test_native_diy_reject_sequence_matches_official_diy7() -> None:
    reject = "[PodWW,Rfe]"
    command = [
        str(DIY),
        "-arch", "RISCV",
        "-safe", " ".join(DEFAULT_DIY_SAFE),
        "-relax", " ".join(DEFAULT_DIY_RELAX),
        "-rejectlist", reject,
        "-size", "4",
        "-nprocs", "2",
        "-cycleonly", "true",
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    official = {
        _canonical(line.split(":", 1)[1].split())
        for line in process.stdout.splitlines()
        if ":" in line
    }
    native, _audit = enumerate_diy_cycles(
        DiyConfig(
            safe=DEFAULT_DIY_SAFE,
            relax=DEFAULT_DIY_RELAX,
            reject=(reject,),
        )
    )
    assert {cycle.canonical_key for cycle in native} == official


@pytest.mark.skipif(not DIY.exists(), reason="official diy7 is not installed")
def test_native_diy_policy_modes_match_official_cycle_sets() -> None:
    for mode in DIY_MODES:
        official_mode = "mixed" if mode == "mixedcheck" else mode
        command = [
            str(DIY),
            "-arch", "RISCV",
            "-safe", " ".join(DEFAULT_DIY_SAFE),
            "-relax", " ".join(DEFAULT_DIY_RELAX),
            "-size", "4",
            "-nprocs", "2",
            "-mode", official_mode,
            "-cycleonly", "true",
        ]
        process = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
        official = {
            _canonical(line.split(":", 1)[1].split())
            for line in process.stdout.splitlines()
            if ":" in line
        }
        native, _audit = enumerate_diy_cycles(
            DiyConfig(safe=DEFAULT_DIY_SAFE, relax=DEFAULT_DIY_RELAX, mode=mode)
        )
        assert {cycle.canonical_key for cycle in native} == official, mode


@pytest.mark.skipif(not DIY.exists(), reason="official diy7 is not installed")
@pytest.mark.parametrize(
    ("official_args", "config_overrides"),
    [
        (["-exact"], {"upto": False}),
        (["-eprocs"], {"exact_procs": True}),
        (["-ins", "2"], {"max_accesses_per_proc": 2}),
        (["-mix", "true"], {"mix": True, "max_relax": 100}),
        (["-minrelax", "2"], {"mix": True, "min_relax": 2, "max_relax": 100}),
        (["-mix", "true", "-maxrelax", "2"], {"mix": True, "max_relax": 2}),
        *[(["-obs", observer], {"observer": observer}) for observer in DIY_OBSERVERS],
    ],
)
def test_native_diy_option_cycles_match_official_diy7(
    official_args: list[str], config_overrides: dict,
) -> None:
    command = [
        str(DIY),
        "-arch", "RISCV",
        "-safe", " ".join(DEFAULT_DIY_SAFE),
        "-relax", " ".join(DEFAULT_DIY_RELAX),
        "-size", "4",
        "-nprocs", "2",
        "-cycleonly", "true",
        *official_args,
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    official = {
        _canonical(line.split(":", 1)[1].split())
        for line in process.stdout.splitlines()
        if ":" in line
    }
    native, _audit = enumerate_diy_cycles(
        DiyConfig(
            safe=DEFAULT_DIY_SAFE,
            relax=DEFAULT_DIY_RELAX,
            **config_overrides,
        )
    )
    assert {cycle.canonical_key for cycle in native} == official


@pytest.mark.skipif(not DIY.exists(), reason="official diy7 is not installed")
@pytest.mark.parametrize(
    "macro",
    ["allRR", "someRR", "allRW", "someRW", "allWR", "someWR", "allWW", "someWW"],
)
def test_native_diy_relaxation_macros_match_official_diy7(macro: str) -> None:
    safe = ("Rfe", "Fre", "Wse", macro)
    command = [
        str(DIY),
        "-arch", "RISCV",
        "-safe", " ".join(safe),
        "-relax", " ".join(DEFAULT_DIY_RELAX),
        "-size", "4",
        "-nprocs", "2",
        "-cycleonly", "true",
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    official = {
        _canonical(line.split(":", 1)[1].split())
        for line in process.stdout.splitlines()
        if ":" in line
    }
    native, _audit = enumerate_diy_cycles(
        DiyConfig(safe=safe, relax=DEFAULT_DIY_RELAX)
    )
    assert {cycle.canonical_key for cycle in native} == official


@pytest.mark.skipif(not DIY.exists(), reason="official diy7 is not installed")
@pytest.mark.parametrize(
    "macro",
    ["ACFence.rw.rwdRR", "BCFence.rw.rwdWW"],
)
def test_native_diy_cumulative_macros_match_official_diy7(macro: str) -> None:
    safe = ("Fre", "Wse", macro)
    command = [
        str(DIY),
        "-arch", "RISCV",
        "-safe", " ".join(safe),
        "-relax", " ".join(DEFAULT_DIY_RELAX),
        "-size", "4",
        "-nprocs", "3",
        "-cycleonly", "true",
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    official = {
        _canonical(line.split(":", 1)[1].split())
        for line in process.stdout.splitlines()
        if ":" in line
    }
    native, _audit = enumerate_diy_cycles(
        DiyConfig(safe=safe, relax=DEFAULT_DIY_RELAX, nprocs=3)
    )
    assert {cycle.canonical_key for cycle in native} == official


def test_native_diy_real_dependencies_and_embedded_solver(tmp_path: Path) -> None:
    report = generate_native_diy(
        out_dir=tmp_path,
        config=DiyConfig(
            safe=DEFAULT_DIY_SAFE,
            relax=("PodWW",),
            realdep=True,
        ),
        annotations=("P",),
        limit=10,
        judge=True,
        solver_backend="embedded",
    )
    assert report["solver_backend"] == "embedded"
    assert report["verdicts"] == {"verified": report["generated_litmus"]}
    assert any("andi " in path.read_text(encoding="utf-8") for path in tmp_path.glob("*.litmus"))
