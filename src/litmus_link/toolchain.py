from __future__ import annotations

"""Wrappers around the external herdtools7 reference toolchain.

Litmus-link's native generator and embedded RVWMO solver do not call these
wrappers.  They remain available as independent reference paths:

* ``diycross7`` enumerates a litmus family as the cartesian product of per-edge
  ordering mechanisms (this is what makes "check more axes -> get more tests"
  true: the tool itself does the cross-product).
* ``herd7`` + ``riscv.cat`` decides, for each supplied test, whether its
  ``exists`` outcome is architecturally *observable* (Allowed/Sometimes) or
  *forbidden* (Never). The verdict is a property of the OUTCOME, not the test.

The native tests use these tools for differential checks when installed.
Binary/lib locations are configurable via the HERDTOOLS_BIN / HERDTOOLS_LIB
environment variables; defaults point at the local Nanhu-V5.1 build.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

_DEFAULT_BIN = "/nfs/home/yangkefan/Nanhu-V5.1/herdtools7/_build/install/default/bin"
_DEFAULT_LIB = "/nfs/home/yangkefan/Nanhu-V5.1/herdtools7/herd/libdir"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_THIRD_PARTY_ROOT = _PROJECT_ROOT / "third_party" / "herdtools7"


def _resolve_executable(name: str) -> Path:
    configured = os.environ.get("HERDTOOLS_BIN")
    candidates = []
    if configured:
        candidates.append(Path(configured) / name)
    candidates.append(_THIRD_PARTY_ROOT / "_build" / "install" / "default" / "bin" / name)
    on_path = shutil.which(name)
    if on_path:
        candidates.append(Path(on_path))
    candidates.append(Path(_DEFAULT_BIN) / name)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _resolve_libdir() -> Path:
    configured = os.environ.get("HERDTOOLS_LIB")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend([_THIRD_PARTY_ROOT / "herd" / "libdir", Path(_DEFAULT_LIB)])
    return next((candidate for candidate in candidates if (candidate / "riscv.cat").exists()), candidates[0])


DIY = _resolve_executable("diy7")
DIYCROSS = _resolve_executable("diycross7")
HERD = _resolve_executable("herd7")
HERDTOOLS_BIN = DIY.parent
HERDTOOLS_LIB = _resolve_libdir()
RISCV_CAT = HERDTOOLS_LIB / "riscv.cat"


class ToolchainError(RuntimeError):
    pass


def tools_available() -> bool:
    """True iff the scalar generation and judging toolchain is present."""
    return DIY.exists() and DIYCROSS.exists() and HERD.exists() and RISCV_CAT.exists()


def missing_tools() -> list[str]:
    out = []
    if not DIY.exists():
        out.append(f"diy7 ({DIY})")
    if not DIYCROSS.exists():
        out.append(f"diycross7 ({DIYCROSS})")
    if not HERD.exists():
        out.append(f"herd7 ({HERD})")
    if not RISCV_CAT.exists():
        out.append(f"riscv.cat ({RISCV_CAT})")
    return out


@lru_cache(maxsize=None)
def tool_version(executable: str) -> str:
    path = Path(executable)
    if not path.exists():
        return "unavailable"
    try:
        proc = subprocess.run(
            [str(path), "-version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    output = (proc.stdout or proc.stderr).strip()
    return output.splitlines()[0] if output else "unknown"


def toolchain_info() -> dict:
    tools = {"diy7": DIY, "diycross7": DIYCROSS, "herd7": HERD}
    return {
        "available": tools_available(),
        "missing": missing_tools(),
        "tools": {
            name: {"path": str(path), "version": tool_version(str(path))}
            for name, path in tools.items()
        },
        "model": {"path": str(RISCV_CAT), "available": RISCV_CAT.exists()},
        "libdir": str(HERDTOOLS_LIB),
    }


@dataclass(frozen=True)
class GeneratedLitmus:
    name: str
    text: str


def diycross_generate(
    name: str,
    edge_args: Sequence[str],
    *,
    arch: str = "RISCV",
    extra_args: Sequence[str] | None = None,
    timeout: int = 180,
) -> list[GeneratedLitmus]:
    """Run diycross7 and return the generated tests.

    ``edge_args`` are the positional cross-product arguments, e.g.
    ``["Rfe", "PodRR,Fence.rw.rwdRR,DpAddrdR", "Fre", "PodWW,Fence.rw.rwdWW"]``.
    diycross7 emits one test per element of the cartesian product of the
    comma-separated alternatives at each position.
    """
    if not DIYCROSS.exists():
        raise ToolchainError(f"diycross7 not found at {DIYCROSS}")
    workdir = Path(tempfile.mkdtemp(prefix="ll-diycross-"))
    cmd = [str(DIYCROSS), "-arch", arch, "-name", name]
    if extra_args:
        cmd += list(extra_args)
    cmd += list(edge_args)
    try:
        proc = subprocess.run(
            cmd, cwd=str(workdir), capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            raise ToolchainError(
                f"diycross7 failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        tests: list[GeneratedLitmus] = []
        for path in sorted(workdir.glob("*.litmus")):
            tests.append(GeneratedLitmus(path.stem, path.read_text()))
        return tests
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(f"diycross7 timed out after {timeout}s") from exc
    finally:
        _rmtree(workdir)


def diy_generate(
    *,
    safe: Sequence[str],
    relax: Sequence[str],
    arch: str = "RISCV",
    size: int = 4,
    nprocs: int = 2,
    exact: bool = False,
    one: bool = False,
    mode: str = "default",
    obstype: str = "fenced",
    realdep: bool = False,
    moreedges: bool = False,
    unrollatomic: int | None = None,
    extra_args: Sequence[str] | None = None,
    timeout: int = 180,
) -> list[GeneratedLitmus]:
    """Enumerate scalar cycles with the official ``diy7`` generator.

    ``safe`` and ``relax`` use herdtools edge syntax. Each sequence is joined
    into one comma-separated relaxation list, matching a normal diy7 config.
    The generated test text is returned unchanged.
    """
    if not DIY.exists():
        raise ToolchainError(f"diy7 not found at {DIY}")
    if size < 2:
        raise ToolchainError("diy7 cycle size must be at least 2")
    if nprocs < 1:
        raise ToolchainError("diy7 nprocs must be at least 1")
    if not safe:
        raise ToolchainError("diy7 safe edge list cannot be empty")
    if not relax:
        raise ToolchainError("diy7 relax edge list cannot be empty")

    workdir = Path(tempfile.mkdtemp(prefix="ll-diy-"))
    cmd = [
        str(DIY),
        "-arch", arch,
        "-num", "false",
        "-safe", ",".join(safe),
        "-relax", ",".join(relax),
        "-size", str(size),
        "-nprocs", str(nprocs),
        "-mode", mode,
        "-obstype", obstype,
    ]
    if exact:
        cmd.append("-exact")
    if one:
        cmd.append("-one")
    if realdep:
        cmd.append("-realdep")
    if moreedges:
        cmd.append("-moreedges")
    if unrollatomic is not None:
        cmd += ["-unrollatomic", str(unrollatomic)]
    if extra_args:
        cmd += list(extra_args)

    try:
        proc = subprocess.run(
            cmd, cwd=str(workdir), capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            raise ToolchainError(
                f"diy7 failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        tests: list[GeneratedLitmus] = []
        seen: set[str] = set()
        for path in sorted(workdir.rglob("*.litmus")):
            if path.stem in seen:
                raise ToolchainError(f"diy7 generated duplicate test name: {path.stem}")
            seen.add(path.stem)
            tests.append(GeneratedLitmus(path.stem, path.read_text(encoding="utf-8")))
        return tests
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(f"diy7 timed out after {timeout}s") from exc
    finally:
        _rmtree(workdir)


@dataclass(frozen=True)
class HerdVerdict:
    outcome: str            # "observable" | "forbidden" | "unknown"
    allowed: bool | None    # True if the exists outcome can be observed
    observation: str        # raw "Never"/"Sometimes"/"Always"
    positive: int
    negative: int
    states: int
    condition: str          # the "exists (...)" clause herd echoed back
    raw: str

    def to_json(self) -> dict:
        return {
            "schema": "litmus-link.herd-verdict.v1",
            "outcome": self.outcome,
            "allowed": self.allowed,
            "observation": self.observation,
            "positive": self.positive,
            "negative": self.negative,
            "states": self.states,
            "condition": self.condition,
        }


_OBS_RE = re.compile(r"^Observation\s+(\S+)\s+(Never|Sometimes|Always)\s+(\d+)\s+(\d+)", re.M)
_STATES_RE = re.compile(r"^States\s+(\d+)", re.M)
_COND_RE = re.compile(r"^Condition\s+(exists.*)$", re.M)


def _strip_nondeterministic(raw: str) -> str:
    """herd7 emits a wall-clock ``Time <name> <seconds>`` line that is pure
    jitter. Storing it in committed-corpus raw_output makes every regeneration
    produce spurious git diffs (0.00 -> 0.01), so you cannot tell a real output
    change from timing noise. Strip it; every other line (States/Observation/
    Witnesses/Condition/Hash) is deterministic and kept."""
    return "".join(
        line for line in raw.splitlines(keepends=True) if not line.startswith("Time ")
    )


def herd_judge(
    litmus_text: str,
    *,
    timeout: int = 120,
    variants: Sequence[str] = (),
) -> HerdVerdict:
    """Run herd7 on one litmus test and parse its per-outcome verdict."""
    if not HERD.exists():
        raise ToolchainError(f"herd7 not found at {HERD}")
    if not RISCV_CAT.exists():
        raise ToolchainError(f"riscv.cat not found at {RISCV_CAT}")
    workdir = Path(tempfile.mkdtemp(prefix="ll-herd-"))
    test_path = workdir / "t.litmus"
    test_path.write_text(litmus_text)
    cmd = [str(HERD), "-I", str(HERDTOOLS_LIB), "-model", str(RISCV_CAT)]
    selected_variants = tuple(dict.fromkeys(str(value) for value in variants if str(value)))
    if selected_variants:
        cmd += ["-variant", ",".join(selected_variants)]
    cmd.append(str(test_path))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        raw = _strip_nondeterministic(proc.stdout)
        if not raw.strip() and (proc.returncode != 0 or proc.stderr.strip()):
            raise ToolchainError(f"herd7 failed: {proc.stderr.strip() or f'rc={proc.returncode}'}")
        return _parse_herd(raw)
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(f"herd7 timed out after {timeout}s") from exc
    finally:
        _rmtree(workdir)


def _parse_herd(raw: str) -> HerdVerdict:
    obs = _OBS_RE.search(raw)
    states = _STATES_RE.search(raw)
    cond = _COND_RE.search(raw)
    states_n = int(states.group(1)) if states else 0
    cond_s = cond.group(1).strip() if cond else ""
    if not obs:
        return HerdVerdict("unknown", None, "", 0, 0, states_n, cond_s, raw)
    observation = obs.group(2)
    positive = int(obs.group(3))
    negative = int(obs.group(4))
    # Never  -> the exists outcome is forbidden (0 positive witnesses)
    # Sometimes / Always -> observable (>=1 positive witness)
    # Defensive consistency check: the Observation word and the witness counts
    # must agree (Never <=> positive==0). herd is self-consistent in practice,
    # but if they ever contradict, return unknown rather than trust one half --
    # a missing verdict is safer than a wrong one for a verification tool.
    if observation == "Never":
        if positive != 0:
            return HerdVerdict("unknown", None, observation, positive, negative, states_n, cond_s, raw)
        outcome, allowed = "forbidden", False
    else:
        if positive == 0:
            return HerdVerdict("unknown", None, observation, positive, negative, states_n, cond_s, raw)
        outcome, allowed = "observable", True
    return HerdVerdict(outcome, allowed, observation, positive, negative, states_n, cond_s, raw)


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
