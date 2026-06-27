import pytest

from corpus_ir import corpus_to_ir, _kind, _addr_map
from corpus_riscv import CorpusTest


def _mk(harts, init=(), name="MP+po+po", cycle="Rfe PodRR Fre PodWW"):
    return CorpusTest(
        name=name, family="MP", skeleton="MP", path="x.litmus",
        init_lines=tuple(init), harts=tuple(tuple(h) for h in harts),
        exists="(1:x5=1 /\\ 1:x7=0)", cycle=cycle, nprocs=len(harts), text="",
    )


def test_kind_classification() -> None:
    assert _kind("sw x5,0(x6)") == "store"
    assert _kind("lw x5,0(x8)") == "load"
    assert _kind("amoor.w.aq x5,x0,(x6)") == "amo"
    assert _kind("fence rw,rw") == "fence"
    assert _kind("bne x5,x0,LC00") == "branch"
    assert _kind("xor x7,x5,x5") == "dep"


def test_addr_map_from_init() -> None:
    amap = _addr_map(["0:x5=1", "0:x6=x", "1:x8=x"])
    assert amap[(0, "x6")] == "x"
    assert amap[(1, "x8")] == "x"
    assert (0, "x5") not in amap  # value, not an address symbol


def test_corpus_to_ir_builds_events_and_po() -> None:
    t = _mk(
        harts=[["sw x5,0(x6)", "sw x5,0(x7)"], ["lw x5,0(x6)", "lw x7,0(x8)"]],
        init=["0:x5=1", "0:x6=x", "0:x7=y", "1:x6=y", "1:x8=x"],
    )
    ir = corpus_to_ir(t)
    assert len(ir.harts) == 2
    assert [e.kind for e in ir.harts[0]] == ["store", "store"]
    assert ir.harts[0][0].location == "x"
    # one po edge per hart (between the two memory events)
    po = [r for r in ir.relations if r.kind == "po"]
    assert len(po) == 2
    assert all(r.local for r in po)
    assert ir.cycle == "Rfe PodRR Fre PodWW"


def test_corpus_to_ir_attaches_verdict_outcome() -> None:
    from toolchain import HerdVerdict
    t = _mk(harts=[["sw x5,0(x6)"], ["lw x5,0(x6)"]])
    v = HerdVerdict("forbidden", False, "Never", 0, 3, 3, "exists (...)", "")
    ir = corpus_to_ir(t, v)
    assert ir.expected_outcome == "forbidden"
    obs = HerdVerdict("observable", True, "Sometimes", 1, 3, 4, "exists (...)", "")
    assert corpus_to_ir(t, obs).expected_outcome == "observable"
