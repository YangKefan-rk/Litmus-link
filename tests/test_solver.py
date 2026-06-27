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
    # LB vector stays an observation/fusion case (MP vector cacheable now gets a
    # real cycle verdict), so the solver makes no formal claim here.
    combination = Combination("test", "vector_mem", "LB", "vector_load", "cacheable", vector="unit_load")
    case = render_cases(combination, evaluate(combination))[0]
    result = solve_generated_case(case)
    assert result.status == "not_applicable"
    assert result.allowed is None


def _nc_case(variant: str):
    combination = Combination("test", "pbmt_nc", "MP", "scalar_pair", "pbmt_nc", params={"variant": variant})
    return render_cases(combination, evaluate(combination))[0]


def test_solver_nc_scalar_gets_native_verdict() -> None:
    # PBMT=NC is non-cacheable main memory and obeys RVWMO (Svpbmt), so an NC
    # scalar case must receive a real native verdict, not "not_applicable".
    case = _nc_case("fence_rw_rw")
    result = solve_generated_case(case)
    assert result.status == "verified"
    assert result.verdict == "forbidden"
    assert result.allowed is False
    assert result.edges, "NC scalar case should carry per-edge ppo reasoning"


def test_solver_nc_verdict_equals_cacheable_twin() -> None:
    # The soundness claim: RVWMO's PPO rules never reference cacheability, so an
    # NC scalar test has the same forbidden/allowed verdict as its cacheable
    # twin for every ordering variant.
    for variant in ["base", "fence_rw_rw", "fence_w_w_r_rw", "addr_dep", "ctrl_dep", "ctrl_fencei"]:
        cacheable = solve_generated_case(_scalar_case(variant))
        nc = solve_generated_case(_nc_case(variant))
        assert nc.allowed == cacheable.allowed, f"NC/cacheable verdict mismatch for {variant}"
        assert nc.verdict == cacheable.verdict, f"NC/cacheable verdict mismatch for {variant}"


def _vector_mp_case(variant: str, vector: str = "unit_store", memory_event: str = "vector_store"):
    combination = Combination("test", "vector_mem", "MP", memory_event, "cacheable", vector=vector, params={"variant": variant})
    return render_cases(combination, evaluate(combination))[0]


def test_solver_vector_mp_gets_native_verdict() -> None:
    # RVV reduces vector memory ordering to per-element RVWMO; a FENCE orders the
    # element accesses like scalar, so a vector-MP cycle gets a real verdict.
    forbidden = solve_generated_case(_vector_mp_case("fence_rw_rw"))
    assert forbidden.status == "verified"
    assert forbidden.verdict == "forbidden"
    assert forbidden.allowed is False
    assert forbidden.edges, "vector cycle case should carry per-edge ppo reasoning"
    allowed = solve_generated_case(_vector_mp_case("base"))
    assert allowed.verdict == "allowed"


def test_solver_vector_verdict_equals_scalar_twin() -> None:
    # Soundness: the vector-MP verdict equals the scalar-MP twin for every FENCE
    # variant, for both vector loads and vector stores, regardless of the vector
    # form (the intra-instruction element order does not affect cross-hart
    # ordering, which the FENCE governs).
    forms = [("unit_store", "vector_store"), ("unit_load", "vector_load"),
             ("indexed_unordered_load", "vector_load"), ("strided_store", "vector_store")]
    for variant in ["base", "fence_rw_rw", "fence_w_w_r_rw"]:
        scalar = solve_generated_case(_scalar_case(variant))
        for vector, memory_event in forms:
            vec = solve_generated_case(_vector_mp_case(variant, vector, memory_event))
            assert vec.allowed == scalar.allowed, f"vector/scalar mismatch for {variant}/{vector}"
            assert vec.verdict == scalar.verdict, f"vector/scalar mismatch for {variant}/{vector}"



