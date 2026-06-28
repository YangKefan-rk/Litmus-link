from __future__ import annotations

"""Harvest + judge the real RISC-V RVWMO litmus corpus.

The scalar RVWMO families (MP/LB/SB/WRC/RWC/IRIW/ISA2/R/S/Co/...) are not
re-derived by hand: they are the real output of diy7/diycross7, already on disk
under litmus-tests-riscv. Per the owner's decision, generation and verdicts come
from the tools -- this module indexes the tool-generated corpus and judges each
test's `exists` outcome with the real herd7 (see toolchain.py).

Corpus root is configurable via the LITMUS_RISCV_ROOT env var.
"""

import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from toolchain import HerdVerdict, herd_judge

_DEFAULT_ROOT = "/nfs/home/yangkefan/Nanhu-V5.1/litmus-tests-riscv/tests/non-mixed-size"
CORPUS_ROOT = Path(os.environ.get("LITMUS_RISCV_ROOT", _DEFAULT_ROOT))

# GUI skeleton axis values (the canonical litmus families).
SKELETONS = ["MP", "LB", "SB", "WRC", "RWC", "IRIW", "ISA2", "R", "S", "Co"]


def family_to_skeleton(prefix: str) -> str:
    """Map a file's family prefix (token before the first '+') to a GUI skeleton."""
    p = prefix.upper()
    if p.startswith("MP"):
        return "MP"
    if p.startswith("SB") or ".SB" in p:
        return "SB"
    if p.startswith("LB") or ".LB" in p or p.endswith("LB"):
        return "LB"
    if p.startswith("WRC"):
        return "WRC"
    if p.startswith("RWC"):
        return "RWC"
    if "IRIW" in p or "IRWIW" in p or "IRRWIW" in p:
        return "IRIW"
    if p.startswith("ISA2"):
        return "ISA2"
    if p.startswith("CO"):
        return "Co"
    if prefix == "R":
        return "R"
    if prefix == "S":
        return "S"
    return "OTHER"


@dataclass(frozen=True)
class CorpusTest:
    name: str
    family: str
    skeleton: str
    path: str
    init_lines: tuple
    harts: tuple          # per-hart tuple of instruction strings
    exists: str
    cycle: str
    nprocs: int
    text: str
    suite: str = ""       # category dir relative to the corpus root (e.g. "SAFE")
    uid: str = ""         # suite-qualified unique id; distinguishes same-named tests

    @property
    def unique_id(self) -> str:
        return self.uid or self.name


_CYCLE_RE = re.compile(r"^Cycle=(.*)$", re.M)


def parse_litmus(text: str, path: str = "", suite: str = "", uid: str = "") -> CorpusTest:
    """Parse a herdtools .litmus file into a CorpusTest (structure only)."""
    lines = text.splitlines()
    name = ""
    for line in lines:
        s = line.strip()
        if s.startswith("RISCV "):
            name = s[len("RISCV "):].strip()
            break
    family = re.split(r"[+]", name)[0] if name else ""
    cyc = _CYCLE_RE.search(text)
    cycle = cyc.group(1).strip() if cyc else ""

    # init block between the first '{' and its matching '}'
    init_lines: list[str] = []
    try:
        open_i = next(i for i, l in enumerate(lines) if l.strip().startswith("{"))
        close_i = next(i for i in range(open_i, len(lines)) if "}" in lines[i])
        blob = "\n".join(lines[open_i:close_i + 1])
        blob = blob[blob.index("{") + 1: blob.rindex("}")]
        init_lines = [seg.strip() for seg in blob.replace("\n", " ").split(";") if seg.strip()]
    except (StopIteration, ValueError):
        close_i = 0

    # thread table: rows after the init block until exists/forall/final
    harts_cols: list[list[str]] = []
    exists = ""
    i = close_i + 1
    collecting_cond = False
    cond_parts: list[str] = []
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        i += 1
        if not s:
            continue
        low = s.lower()
        if collecting_cond:
            cond_parts.append(s)
            continue
        if low.startswith(("exists", "forall", "~exists", "observed")):
            m = re.search(r"(exists|forall|~exists)\s*(.*)", s, re.I)
            cond_parts.append(m.group(2) if m and m.group(2) else "")
            collecting_cond = True
            continue
        if low.startswith(("locations", "filter", "with")):
            continue
        if "|" in s or s.endswith(";"):
            cells = [c.strip() for c in s.rstrip(";").split("|")]
            for col, cell in enumerate(cells):
                while len(harts_cols) <= col:
                    harts_cols.append([])
                if cell:
                    harts_cols[col].append(cell)
    # drop the "P0 | P1" header row from each column
    harts = []
    for col in harts_cols:
        if col and re.fullmatch(r"P\d+", col[0]):
            col = col[1:]
        harts.append(tuple(col))
    harts = tuple(h for h in harts if h)
    exists = " ".join(p for p in cond_parts if p).strip()

    return CorpusTest(
        name=name,
        family=family,
        skeleton=family_to_skeleton(family),
        path=path,
        init_lines=tuple(init_lines),
        harts=harts,
        exists=exists,
        cycle=cycle,
        nprocs=len(harts),
        text=text,
        suite=suite,
        uid=uid or name,
    )


@lru_cache(maxsize=1)
def _index() -> dict:
    """Index the corpus once: skeleton -> sorted list of (uid, suite, path).

    Tests are keyed by a suite-qualified unique id, NOT by cycle-name: the same
    herdtools cycle name (e.g. "MP") appears in several suites (BASIC_2_THREAD,
    SF_THESIS/BASIC, ...) as GENUINELY DIFFERENT tests (e.g. sw/lw vs sd/ld width
    variants). Keying by name alone silently dropped 21 of 23 such MP tests on
    generate and showed duplicates in preview. We keep every distinct file and
    only suite-qualify the uid where a bare name collides.
    """
    by_skeleton: dict[str, list[tuple]] = {}
    if not CORPUS_ROOT.exists():
        return by_skeleton
    raw: dict[str, list[tuple]] = {}  # skeleton -> [(name, suite, path)]
    for dirpath, _dirs, files in os.walk(CORPUS_ROOT):
        for f in files:
            if not f.endswith(".litmus"):
                continue
            name = f[:-7]
            prefix = re.split(r"[+]", name)[0]
            skel = family_to_skeleton(prefix)
            suite = os.path.relpath(dirpath, CORPUS_ROOT)
            raw.setdefault(skel, []).append((name, suite, os.path.join(dirpath, f)))
    for skel, entries in raw.items():
        name_counts = Counter(name for name, _suite, _path in entries)
        out: list[tuple] = []
        for name, suite, path in entries:
            if name_counts[name] > 1:
                uid = f"{name}__{suite.replace(os.sep, '_')}"
            else:
                uid = name
            out.append((uid, suite, path))
        out.sort()
        by_skeleton[skel] = out
    return by_skeleton


def corpus_available() -> bool:
    return CORPUS_ROOT.exists() and bool(_index())


def skeleton_counts() -> dict:
    return {k: len(v) for k, v in sorted(_index().items())}


def tests_for_skeleton(skeleton: str, limit: int | None = None) -> list[CorpusTest]:
    """Parse and return the corpus tests for one GUI skeleton."""
    entries = _index().get(skeleton, [])
    if limit is not None:
        entries = entries[:limit]
    out = []
    for uid, suite, path in entries:
        try:
            out.append(parse_litmus(Path(path).read_text(), path, suite=suite, uid=uid))
        except Exception:
            continue
    return out


@lru_cache(maxsize=4096)
def _judge_cached(text: str) -> HerdVerdict:
    return herd_judge(text)


def judge(test: CorpusTest) -> HerdVerdict:
    """herd7 verdict for a test's exists outcome (memoised by file text)."""
    return _judge_cached(test.text)

