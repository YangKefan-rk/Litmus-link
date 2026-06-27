import pytest

from corpus_riscv import (
    corpus_available,
    parse_litmus,
    skeleton_counts,
    tests_for_skeleton as get_tests_for_skeleton,
    judge,
    family_to_skeleton,
)


def test_family_to_skeleton_mapping() -> None:
    assert family_to_skeleton("MP") == "MP"
    assert family_to_skeleton("SB") == "SB"
    assert family_to_skeleton("LB") == "LB"
    assert family_to_skeleton("WRC") == "WRC"
    assert family_to_skeleton("IRIW") == "IRIW"
    assert family_to_skeleton("CoRR") == "Co"


def test_parse_litmus_structure() -> None:
    text = (
        "RISCV MP+po+poaqp+NEW\n"
        '"PodWW RfePAq PodRRAqP Fre"\n'
        "Cycle=Fre PodWW RfePAq PodRRAqP\n"
        "{\n0:x5=1; 0:x6=x; 0:x7=y;\n1:x6=y; 1:x8=x;\n}\n"
        " P0          | P1                    ;\n"
        " sw x5,0(x6) | amoor.w.aq x5,x0,(x6) ;\n"
        " sw x5,0(x7) | lw x7,0(x8)           ;\n"
        "exists\n(1:x5=1 /\\ 1:x7=0)\n"
    )
    t = parse_litmus(text, "x.litmus")
    assert t.name == "MP+po+poaqp+NEW"
    assert t.skeleton == "MP"
    assert t.nprocs == 2
    assert t.harts[0] == ("sw x5,0(x6)", "sw x5,0(x7)")
    assert t.harts[1] == ("amoor.w.aq x5,x0,(x6)", "lw x7,0(x8)")
    assert t.exists == "(1:x5=1 /\\ 1:x7=0)"
    assert t.cycle == "Fre PodWW RfePAq PodRRAqP"


pytestmark_corpus = pytest.mark.skipif(
    not corpus_available(), reason="RISC-V litmus corpus not present"
)


@pytestmark_corpus
def test_corpus_has_full_mp_family() -> None:
    counts = skeleton_counts()
    # The corpus must expose hundreds of MP tests, not a handful -- this is the
    # core fix for "check MP -> only 6 tests".
    assert counts.get("MP", 0) > 500
    mp = get_tests_for_skeleton("MP")
    assert len(mp) == counts["MP"]
    assert all(t.harts and t.exists for t in mp)


@pytestmark_corpus
def test_corpus_herd_verdicts_are_per_outcome() -> None:
    from toolchain import tools_available
    if not tools_available():
        pytest.skip("herd7 not installed")
    by_name = {t.name: t for t in get_tests_for_skeleton("MP")}
    # A plain MP variant: weak outcome observable.
    plain = next(t for n, t in by_name.items() if n in ("MP", "MP+po+po"))
    assert judge(plain).outcome == "observable"
    # A double-fenced MP: forbidden.
    fenced = next((t for n, t in by_name.items() if "fence.rw.rw+addr" in n), None)
    if fenced is not None:
        assert judge(fenced).outcome == "forbidden"
