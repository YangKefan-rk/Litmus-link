from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .corpus_riscv import parse_litmus
from .toolchain import (
    RISCV_CAT,
    GeneratedLitmus,
    ToolchainError,
    diy_generate,
    diycross_generate,
    herd_judge,
    toolchain_info,
)


class ScalarGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScalarPreset:
    name: str
    description: str
    cycle: tuple[str, ...]
    oneloc: bool = False

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "cycle": list(self.cycle),
            "oneloc": self.oneloc,
        }


# Communication edges are literal herdtools edges. RR/RW/WR/WW are local-edge
# slots expanded from the selected mechanism groups below.
SCALAR_PRESETS: dict[str, ScalarPreset] = {
    "MP": ScalarPreset("MP", "Message Passing", ("Rfe", "RR", "Fre", "WW")),
    "LB": ScalarPreset("LB", "Load Buffering", ("Rfe", "RW", "Rfe", "RW")),
    "SB": ScalarPreset("SB", "Store Buffering", ("Fre", "WR", "Fre", "WR")),
    "WRC": ScalarPreset("WRC", "Write-Read Causality", ("Rfe", "RW", "Rfe", "RR", "Fre")),
    "RWC": ScalarPreset("RWC", "Read-Write Causality", ("Rfe", "RR", "Fre", "WR", "Fre")),
    "IRIW": ScalarPreset("IRIW", "Independent Reads of Independent Writes", ("Rfe", "RR", "Fre", "Rfe", "RR", "Fre")),
    "ISA2": ScalarPreset("ISA2", "Three-hart causality shape", ("Fre", "WW", "Rfe", "RW", "Rfe", "RR")),
    "R": ScalarPreset("R", "Read/coherence shape", ("Fre", "WW", "Wse", "WR")),
    "S": ScalarPreset("S", "Store/coherence shape", ("Rfe", "RW", "Wse", "WW")),
    "CoRR": ScalarPreset("CoRR", "Single-location read-read coherence shape", ("Rfe", "PosRR", "Fre"), oneloc=True),
}


MECHANISM_EDGES: dict[str, dict[str, tuple[str, ...]]] = {
    "po": {
        "RR": ("PodRR",),
        "RW": ("PodRW",),
        "WR": ("PodWR",),
        "WW": ("PodWW",),
    },
    "fence": {
        "RR": ("Fence.r.rwdRR", "Fence.rw.rwdRR"),
        "RW": ("Fence.rw.wdRW", "Fence.rw.rwdRW"),
        "WR": ("Fence.w.rdWR", "Fence.rw.rwdWR"),
        "WW": ("Fence.w.wdWW", "Fence.rw.wdWW"),
    },
    "dependency": {
        "RR": ("DpAddrdR", "DpCtrldR", "DpCtrlFenceIdR"),
        "RW": ("DpAddrdW", "DpDatadW", "DpCtrldW", "DpCtrlFenceIdW"),
        "WR": (),
        "WW": (),
    },
}

DEFAULT_MECHANISMS = ("po", "fence", "dependency")

# This is the current-herdtools executable subset of litmus-tests-riscv's
# historical safe.conf. Old WsBack/WsLeave tokens are deliberately excluded:
# herdtools7 7.58 rejects them even though older generated corpora mention them.
DEFAULT_SAFE_EDGES = (
    "Rfe",
    "Fre",
    "Wse",
    "Fence.rw.rwd**",
    "Fence.rw.wd*W",
    "Fence.w.wdWW",
    "DpAddrdR",
    "DpAddrdW",
    "DpDatadW",
    "DpCtrldR",
    "DpCtrldW",
)
DEFAULT_RELAX_EDGES = ("PodRR", "PodRW", "PodWR", "PodWW")


def scalar_catalog() -> dict:
    return {
        "presets": {name: preset.to_json() for name, preset in SCALAR_PRESETS.items()},
        "mechanisms": {
            mechanism: {shape: list(edges) for shape, edges in shapes.items()}
            for mechanism, shapes in MECHANISM_EDGES.items()
        },
        "enumerate_defaults": {
            "safe": list(DEFAULT_SAFE_EDGES),
            "relax": list(DEFAULT_RELAX_EDGES),
            "size": 4,
            "nprocs": 2,
        },
    }


def cross_edge_args(preset: ScalarPreset, mechanisms: Sequence[str]) -> list[str]:
    selected = _validate_mechanisms(mechanisms)
    arguments: list[str] = []
    for token in preset.cycle:
        if token not in {"RR", "RW", "WR", "WW"}:
            arguments.append(token)
            continue
        alternatives = _deduplicate(
            edge
            for mechanism in selected
            for edge in MECHANISM_EDGES[mechanism][token]
        )
        if not alternatives:
            raise ScalarGenerationError(
                f"no {token} edge is available for mechanisms: {', '.join(selected)}"
            )
        arguments.append(",".join(alternatives))
    return arguments


def parse_custom_cycle(value: str) -> tuple[str, ...]:
    """Parse ``edge;alternative,alternative;edge`` CLI syntax."""
    axes = tuple(
        ",".join(edge.strip() for edge in axis.split(","))
        for raw_axis in value.split(";")
        if (axis := raw_axis.strip())
    )
    if len(axes) < 2:
        raise ScalarGenerationError("custom cycle must contain at least two ';'-separated edge positions")
    for axis in axes:
        if any(not edge.strip() for edge in axis.split(",")):
            raise ScalarGenerationError(f"custom cycle contains an empty edge alternative: {axis!r}")
    return axes


def generate_scalar_cross(
    *,
    out_dir: Path,
    presets: Sequence[str],
    mechanisms: Sequence[str] = DEFAULT_MECHANISMS,
    limit: int | None = None,
    judge: bool = True,
    timeout: int = 180,
    custom_name: str | None = None,
    custom_cycle: Sequence[str] | None = None,
) -> dict:
    if limit is not None and limit < 1:
        raise ScalarGenerationError("limit must be at least 1")
    selected = _validate_mechanisms(mechanisms)
    runs: list[dict] = []
    generated: list[GeneratedLitmus] = []

    if custom_cycle is not None:
        if not custom_name:
            raise ScalarGenerationError("custom cycle generation requires custom_name")
        _validate_generated_name(custom_name)
        cycle = tuple(custom_cycle)
        tests = diycross_generate(custom_name, cycle, timeout=timeout)
        runs.append({"name": custom_name, "cycle_axes": list(cycle), "generated": len(tests)})
        generated.extend(tests)
    else:
        if not presets:
            raise ScalarGenerationError("select at least one scalar preset")
        for name in presets:
            try:
                preset = SCALAR_PRESETS[name]
            except KeyError as exc:
                raise ScalarGenerationError(f"unknown scalar preset: {name}") from exc
            edge_args = cross_edge_args(preset, selected)
            extra = ["-oneloc"] if preset.oneloc else None
            tests = diycross_generate(name, edge_args, extra_args=extra, timeout=timeout)
            if not tests:
                raise ScalarGenerationError(f"diycross7 generated no tests for preset {name}")
            runs.append({
                "name": name,
                "description": preset.description,
                "cycle": list(preset.cycle),
                "cycle_axes": edge_args,
                "generated": len(tests),
            })
            generated.extend(tests)

    unique = _unique_tests(generated)
    available = len(unique)
    selected_tests = unique[:limit] if limit is not None else unique
    return _write_scalar_corpus(
        selected_tests,
        out_dir,
        judge=judge,
        timeout=timeout,
        generator={
            "engine": "diycross7",
            "mechanisms": [] if custom_cycle is not None else list(selected),
            "custom_cycle": custom_cycle is not None,
            "runs": runs,
        },
        available=available,
        limit=limit,
    )


def generate_scalar_enumerated(
    *,
    out_dir: Path,
    safe: Sequence[str] = DEFAULT_SAFE_EDGES,
    relax: Sequence[str] = DEFAULT_RELAX_EDGES,
    size: int = 4,
    nprocs: int = 2,
    exact: bool = False,
    one: bool = False,
    mode: str = "default",
    obstype: str = "fenced",
    realdep: bool = False,
    moreedges: bool = False,
    unrollatomic: int | None = None,
    limit: int | None = None,
    judge: bool = True,
    timeout: int = 180,
) -> dict:
    if limit is not None and limit < 1:
        raise ScalarGenerationError("limit must be at least 1")
    tests = diy_generate(
        safe=safe,
        relax=relax,
        size=size,
        nprocs=nprocs,
        exact=exact,
        one=one,
        mode=mode,
        obstype=obstype,
        realdep=realdep,
        moreedges=moreedges,
        unrollatomic=unrollatomic,
        timeout=timeout,
    )
    if not tests:
        raise ScalarGenerationError("diy7 generated no tests for the selected domain")
    unique = _unique_tests(tests)
    available = len(unique)
    selected_tests = unique[:limit] if limit is not None else unique
    return _write_scalar_corpus(
        selected_tests,
        out_dir,
        judge=judge,
        timeout=timeout,
        generator={
            "engine": "diy7",
            "safe": list(safe),
            "relax": list(relax),
            "size": size,
            "nprocs": nprocs,
            "exact": exact,
            "one": one,
            "mode": mode,
            "obstype": obstype,
            "realdep": realdep,
            "moreedges": moreedges,
            "unrollatomic": unrollatomic,
        },
        available=available,
        limit=limit,
    )


def _write_scalar_corpus(
    tests: Sequence[GeneratedLitmus],
    out_dir: Path,
    *,
    judge: bool,
    timeout: int,
    generator: dict,
    available: int,
    limit: int | None,
) -> dict:
    _prepare_output(out_dir)
    verdict_counts: Counter[str] = Counter()
    filenames: list[str] = []
    tool_info = toolchain_info()

    for generated in tests:
        _validate_generated_name(generated.name)
        parsed = parse_litmus(generated.text, f"{generated.name}.litmus")
        if parsed.name != generated.name:
            raise ScalarGenerationError(
                f"generated filename/header mismatch: {generated.name!r} != {parsed.name!r}"
            )
        if not parsed.harts or not parsed.exists:
            raise ScalarGenerationError(f"generated test is structurally incomplete: {generated.name}")

        litmus_path = out_dir / f"{generated.name}.litmus"
        litmus_path.write_text(generated.text, encoding="utf-8")
        solver = _judge_payload(generated, judge=judge, timeout=timeout)
        verdict_counts[solver["status"]] += 1
        solver_path = out_dir / f"{generated.name}.solver.json"
        solver_path.write_text(json.dumps(solver, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        meta = {
            "schema": "litmus-link.scalar-meta.v1",
            "name": generated.name,
            "architecture": "RISCV",
            "requires": _required_extensions(generated.text),
            "cycle": parsed.cycle,
            "exists": parsed.exists,
            "nprocs": parsed.nprocs,
            "harts": [list(hart) for hart in parsed.harts],
            "generator": generator,
            "toolchain": tool_info,
            "solver": solver,
            "generated_from": "herdtools7",
        }
        (out_dir / f"{generated.name}.meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        filenames.append(f"{generated.name}.litmus")

    (out_dir / "@all").write_text("\n".join(filenames) + "\n", encoding="utf-8")
    report = {
        "schema": "litmus-link.scalar-generation.v1",
        "architecture": "RISCV",
        "generator": generator,
        "toolchain": tool_info,
        "available_litmus": available,
        "generated_litmus": len(filenames),
        "generation_limit": limit,
        "generation_limited": len(filenames) < available,
        "judge": judge,
        "verdicts": dict(sorted(verdict_counts.items())),
        "output": str(out_dir),
        "atfile": str(out_dir / "@all"),
    }
    (out_dir / "generation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _judge_payload(test: GeneratedLitmus, *, judge: bool, timeout: int) -> dict:
    if not judge:
        return {
            "schema": "litmus-link.scalar-solver.v1",
            "status": "unchecked",
            "tool": "herd7",
            "model": "riscv.cat",
            "allowed": None,
            "verdict": "unchecked",
            "reason": "herd7 judging disabled by the user",
        }
    try:
        verdict = herd_judge(test.text, timeout=timeout)
    except ToolchainError as exc:
        raise ScalarGenerationError(f"herd7 failed for {test.name}: {exc}") from exc
    status = "verified" if verdict.outcome in {"observable", "forbidden"} else "unknown"
    return {
        "schema": "litmus-link.scalar-solver.v1",
        "status": status,
        "tool": "herd7",
        "model": "riscv.cat",
        "model_path": str(RISCV_CAT),
        "allowed": verdict.allowed,
        "verdict": verdict.outcome,
        "observation": verdict.observation,
        "positive": verdict.positive,
        "negative": verdict.negative,
        "states": verdict.states,
        "condition": verdict.condition,
        "raw_output": verdict.raw,
    }


def _prepare_output(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atfile = out_dir / "@all"
    if atfile.exists():
        for entry in atfile.read_text(encoding="utf-8").splitlines():
            litmus = out_dir / entry.strip()
            if not entry.strip() or litmus.parent != out_dir or litmus.suffix != ".litmus":
                continue
            for suffix in (".litmus", ".meta.json", ".solver.json"):
                litmus.with_suffix(suffix).unlink(missing_ok=True)
    atfile.unlink(missing_ok=True)
    (out_dir / "generation-report.json").unlink(missing_ok=True)


def _required_extensions(text: str) -> list[str]:
    requires = ["RV64I"]
    if re.search(r"\b(?:amo\w*|lr\.[wd]|sc\.[wd])", text):
        requires.append("A")
    if re.search(r"\bfence\.i\b", text):
        requires.append("Zifencei")
    return requires


def _validate_generated_name(name: str) -> None:
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ScalarGenerationError(f"unsafe generated test name: {name!r}")


def _validate_mechanisms(mechanisms: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(mechanisms or DEFAULT_MECHANISMS)
    unknown = [name for name in selected if name not in MECHANISM_EDGES]
    if unknown:
        raise ScalarGenerationError(f"unknown scalar mechanisms: {', '.join(unknown)}")
    return tuple(_deduplicate(selected))


def _unique_tests(tests: Iterable[GeneratedLitmus]) -> list[GeneratedLitmus]:
    unique: dict[str, GeneratedLitmus] = {}
    for test in tests:
        previous = unique.get(test.name)
        if previous is not None and previous.text != test.text:
            raise ScalarGenerationError(f"different generated tests share the name {test.name!r}")
        unique[test.name] = test
    return [unique[name] for name in sorted(unique)]


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
