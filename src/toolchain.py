from __future__ import annotations

"""Wrappers around the real herdtools7 toolchain.

Per the project owner's decision, litmus generation and verdicts are produced by
the authoritative tools, never by hand-rolled logic:

* ``diycross7`` enumerates a litmus family as the cartesian product of per-edge
  ordering mechanisms (this is what makes "check more axes -> get more tests"
  true: the tool itself does the cross-product).
* ``herd7`` + ``riscv.cat`` decides, for each generated test, whether its
  ``exists`` outcome is architecturally *observable* (Allowed/Sometimes) or
  *forbidden* (Never). The verdict is a property of the OUTCOME, not the test.

Binary/lib locations are configurable via the HERDTOOLS_BIN / HERDTOOLS_LIB
environment variables; defaults point at the local Nanhu-V5.1 build.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_DEFAULT_BIN = "/nfs/home/yangkefan/Nanhu-V5.1/herdtools7/_build/install/default/bin"
_DEFAULT_LIB = "/nfs/home/yangkefan/Nanhu-V5.1/herdtools7/herd/libdir"

HERDTOOLS_BIN = Path(os.environ.get("HERDTOOLS_BIN", _DEFAULT_BIN))
HERDTOOLS_LIB = Path(os.environ.get("HERDTOOLS_LIB", _DEFAULT_LIB))

DIYCROSS = HERDTOOLS_BIN / "diycross7"
DIY = HERDTOOLS_BIN / "diy7"
HERD = HERDTOOLS_BIN / "herd7"
RISCV_CAT = HERDTOOLS_LIB / "riscv.cat"


class ToolchainError(RuntimeError):
    pass


def tools_available() -> bool:
    """True iff diycross7, herd7 and riscv.cat are all present."""
    return DIYCROSS.exists() and HERD.exists() and RISCV_CAT.exists()


def missing_tools() -> list[str]:
    out = []
    if not DIYCROSS.exists():
        out.append(f"diycross7 ({DIYCROSS})")
    if not HERD.exists():
        out.append(f"herd7 ({HERD})")
    if not RISCV_CAT.exists():
        out.append(f"riscv.cat ({RISCV_CAT})")
    return out


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


def herd_judge(litmus_text: str, *, timeout: int = 120) -> HerdVerdict:
    """Run herd7 on one litmus test and parse its per-outcome verdict."""
    if not HERD.exists():
        raise ToolchainError(f"herd7 not found at {HERD}")
    if not RISCV_CAT.exists():
        raise ToolchainError(f"riscv.cat not found at {RISCV_CAT}")
    workdir = Path(tempfile.mkdtemp(prefix="ll-herd-"))
    test_path = workdir / "t.litmus"
    test_path.write_text(litmus_text)
    cmd = [str(HERD), "-I", str(HERDTOOLS_LIB), "-model", str(RISCV_CAT), str(test_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        raw = proc.stdout
        if proc.returncode != 0 and not raw.strip():
            raise ToolchainError(f"herd7 failed: {proc.stderr.strip()}")
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
    if observation == "Never":
        outcome, allowed = "forbidden", False
    else:
        outcome, allowed = "observable", True
    return HerdVerdict(outcome, allowed, observation, positive, negative, states_n, cond_s, raw)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
