from solver import parse_herd_output, solve_generated_case
from renderer import render_cases
from rules import evaluate
from models import Combination


def test_parse_herd_never_as_forbidden() -> None:
    parsed = parse_herd_output("Test MP Allowed\nObservation MP Never 0 10\n")
    assert parsed["verdict"] == "forbidden"
    assert parsed["allowed"] is False


def test_parse_herd_sometimes_as_allowed() -> None:
    parsed = parse_herd_output("Test MP Allowed\nObservation MP Sometimes 1 9\n")
    assert parsed["verdict"] == "allowed"
    assert parsed["allowed"] is True


def test_parse_herd_unparsed_is_unknown() -> None:
    parsed = parse_herd_output("no recognizable verdict")
    assert parsed["verdict"] == "unknown"
    assert parsed["allowed"] is None


def _scalar_case(variant: str):
    combination = Combination("test", "rvwmo_base", "MP", "scalar_pair", "cacheable", params={"variant": variant})
    return render_cases(combination, evaluate(combination))[0]


def test_solver_uses_native_checker_without_herd7() -> None:
    # The native checker renders a verdict even when herd7 is not installed.
    case = _scalar_case("fence_rw_rw")
    result = solve_generated_case(case)
    assert result.status == "verified"
    assert result.verdict == "forbidden"
    assert result.allowed is False
    assert result.edges, "native checker should attach per-edge reasoning"
    # herd7 is absent in this environment, so it is native-only.
    assert result.cross_check in {"herd7_absent", "agree"}


def test_solver_native_allows_base_mp() -> None:
    case = _scalar_case("base")
    result = solve_generated_case(case)
    assert result.status == "verified"
    assert result.verdict == "allowed"
    assert result.allowed is True


def test_solver_not_applicable_for_fusion() -> None:
    combination = Combination("test", "vector_mem", "MP", "vector_load", "cacheable", vector="unit_load")
    case = render_cases(combination, evaluate(combination))[0]
    result = solve_generated_case(case)
    assert result.status == "not_applicable"
    assert result.allowed is None

