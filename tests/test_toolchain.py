import pytest
from types import SimpleNamespace

import litmus_link.toolchain as toolchain
from litmus_link.toolchain import ToolchainError, diy_generate, tools_available, diycross_generate, herd_judge, toolchain_info, _parse_herd, _strip_nondeterministic


requires_tools = pytest.mark.skipif(
    not tools_available(), reason="herdtools7 (diycross7/herd7/riscv.cat) not installed"
)


# --- Pure parser tests: no herd7 binary needed, so they run unconditionally. ---


def test_strip_nondeterministic_removes_only_time_line() -> None:
    raw = (
        "Test T Allowed\nStates 4\nOk\nWitnesses\nPositive: 1 Negative: 3\n"
        "Condition exists (1:x5=1)\nObservation T Sometimes 1 3\n"
        "Time T 0.01\nHash=abc123\n"
    )
    stripped = _strip_nondeterministic(raw)
    assert "Time " not in stripped
    # Every deterministic line is preserved verbatim.
    assert "Observation T Sometimes 1 3" in stripped
    assert "Hash=abc123" in stripped
    assert "States 4" in stripped
    # Idempotent + parse still works after stripping.
    assert _strip_nondeterministic(stripped) == stripped
    v = _parse_herd(stripped)
    assert v.outcome == "observable" and v.allowed is True

def test_parse_herd_never_is_forbidden() -> None:
    v = _parse_herd("Observation T Never 0 5\n")
    assert v.outcome == "forbidden" and v.allowed is False


def test_parse_herd_sometimes_is_observable() -> None:
    v = _parse_herd("Observation T Sometimes 2 5\n")
    assert v.outcome == "observable" and v.allowed is True


def test_parse_herd_no_observation_is_unknown() -> None:
    v = _parse_herd("nothing recognizable here")
    assert v.outcome == "unknown" and v.allowed is None


def test_parse_herd_contradictory_counts_are_unknown() -> None:
    # Defensive consistency check (solver Fix I): the Observation word and the
    # witness counts must agree (Never <=> positive==0). A contradiction must
    # degrade to unknown rather than emit a possibly-wrong verdict.
    never_with_witness = _parse_herd("Observation T Never 3 5\n")
    assert never_with_witness.outcome == "unknown" and never_with_witness.allowed is None
    sometimes_without_witness = _parse_herd("Observation T Sometimes 0 5\n")
    assert sometimes_without_witness.outcome == "unknown" and sometimes_without_witness.allowed is None


def test_toolchain_info_has_stable_shape() -> None:
    info = toolchain_info()
    assert set(info["tools"]) == {"diy7", "diycross7", "herd7"}
    assert info["model"]["path"].endswith("riscv.cat")


def test_herd_judge_passes_mixed_unaligned_variants(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    executable = tmp_path / "herd7"
    model = tmp_path / "riscv.cat"
    executable.touch()
    model.touch()
    monkeypatch.setattr(toolchain, "HERD", executable)
    monkeypatch.setattr(toolchain, "RISCV_CAT", model)
    seen = {}

    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        seen["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout="States 1\nCondition exists (1:x5=1)\nObservation T Sometimes 1 0\n",
            stderr="",
        )

    monkeypatch.setattr(toolchain.subprocess, "run", fake_run)
    verdict = herd_judge("RISCV T\n{}\n P0;\n nop;\nexists (1=1)\n", variants=("mixed", "unaligned"))
    assert verdict.allowed is True
    assert seen["command"][seen["command"].index("-variant") + 1] == "mixed,unaligned"


def test_herd_judge_rejects_stderr_only_user_error(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    executable = tmp_path / "herd7"
    model = tmp_path / "riscv.cat"
    executable.touch()
    model.touch()
    monkeypatch.setattr(toolchain, "HERD", executable)
    monkeypatch.setattr(toolchain, "RISCV_CAT", model)
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="Mixed mode not implemented for architecture RISCV",
        ),
    )
    with pytest.raises(ToolchainError, match="Mixed mode not implemented"):
        herd_judge("RISCV T\n{}\n P0;\n nop;\nexists (1=1)\n", variants=("mixed", "unaligned"))


@requires_tools
def test_diy_generates_basic_cycles() -> None:
    tests = diy_generate(
        safe=["Rfe", "Fre", "Wse", "Fence.rw.rwd**", "DpAddrdR", "DpAddrdW"],
        relax=["PodRR", "PodRW", "PodWR", "PodWW"],
        size=4,
        nprocs=2,
        timeout=30,
    )
    names = {test.name for test in tests}
    assert "LB" in names and "SB" in names
    assert any(name.startswith("MP+") for name in names)
    assert all(test.text.startswith("RISCV ") for test in tests)


@requires_tools
def test_diycross_generates_cartesian_product() -> None:
    # Crossing a 2-alternative reader edge with a 2-alternative writer edge must
    # yield the full 2x2 product, plus diycross's base case.
    tests = diycross_generate("MP", [
        "Rfe",
        "PodRR,Fence.rw.rwdRR",
        "Fre",
        "PodWW,Fence.rw.rwdWW",
    ])
    names = {t.name for t in tests}
    assert len(tests) >= 4, f"expected a cartesian product, got {names}"
    assert any("fence.rw.rw" in n for n in names)
    assert all(t.text.startswith("RISCV ") for t in tests)


@requires_tools
def test_herd_judges_outcome_observable_vs_forbidden() -> None:
    tests = {t.name: t for t in diycross_generate("MP", [
        "Rfe",
        "PodRR,Fence.rw.rwdRR,DpAddrdR",
        "Fre",
        "PodWW,Fence.rw.rwdWW",
    ])}
    # Plain MP (no ordering on either leg) -> the weak outcome is observable.
    plain = herd_judge(tests["MP"].text)
    assert plain.outcome == "observable"
    assert plain.allowed is True
    # Both legs fenced -> forbidden.
    fenced = herd_judge(tests["MP+fence.rw.rw+addr"].text)
    assert fenced.outcome == "forbidden"
    assert fenced.allowed is False


@requires_tools
def test_herd_verdict_is_outcome_property_not_test_label() -> None:
    # A control dependency on the reader R->R edge does NOT order load->load
    # under RVWMO, so the outcome stays observable even with a writer fence.
    tests = {t.name: t for t in diycross_generate("MP", [
        "Rfe", "PodRR,DpCtrldR", "Fre", "Fence.rw.rwdWW",
    ])}
    ctrl = herd_judge(tests["MP+fence.rw.rw+ctrl"].text)
    assert ctrl.outcome == "observable"
