from litmus_ir import _mp_case, _lb_case, _sb_case, _wrc_case, _rwc_case, _iriw_case
from models import Combination
from rvwmo import check_rvwmo


BUILDERS = {
    "MP": _mp_case,
    "LB": _lb_case,
    "SB": _sb_case,
    "WRC": _wrc_case,
    "RWC": _rwc_case,
    "IRIW": _iriw_case,
}

# Expected RVWMO verdict per (skeleton, variant): True == forbidden (bad
# outcome cannot occur), False == allowed. Derived from the PPO rules and
# cross-checked against the standard RVWMO results for these shapes.
#
# Key reasoning:
#  - base: no ordering anywhere -> cycle broken -> allowed everywhere.
#  - fence_rw_rw: full fence on every ordered hart -> forbidden everywhere.
#  - fence_w_w_r_rw: writers get "fence w,w" (orders W->W only), readers get
#    "fence r,rw" (orders R->anything). So it forbids MP/LB/WRC/RWC/IRIW but
#    NOT SB, whose writer edges are W->R (R not in the w,w successor set).
#  - addr_dep / ctrl_dep / ctrl_fencei only ever decorate *reader* harts.
#      addr_dep  -> rule 9: orders load->{load,store}
#      ctrl_dep  -> rule 11: orders load->store ONLY
#      ctrl_fencei == ctrl_dep (fence.i has no data-ordering power in RVWMO)
EXPECTED = {
    "MP":   {"base": False, "fence_rw_rw": True, "fence_w_w_r_rw": True,  "addr_dep": False, "ctrl_dep": False, "ctrl_fencei": False},
    "LB":   {"base": False, "fence_rw_rw": True, "fence_w_w_r_rw": True,  "addr_dep": True,  "ctrl_dep": True,  "ctrl_fencei": True},
    "SB":   {"base": False, "fence_rw_rw": True, "fence_w_w_r_rw": False, "addr_dep": False, "ctrl_dep": False, "ctrl_fencei": False},
    "WRC":  {"base": False, "fence_rw_rw": True, "fence_w_w_r_rw": True,  "addr_dep": True,  "ctrl_dep": False, "ctrl_fencei": False},
    "RWC":  {"base": False, "fence_rw_rw": True, "fence_w_w_r_rw": True,  "addr_dep": True,  "ctrl_dep": False, "ctrl_fencei": False},
    "IRIW": {"base": False, "fence_rw_rw": True, "fence_w_w_r_rw": True,  "addr_dep": True,  "ctrl_dep": False, "ctrl_fencei": False},
}

VARIANTS = ["base", "fence_rw_rw", "fence_w_w_r_rw", "addr_dep", "ctrl_dep", "ctrl_fencei"]


def _combo() -> Combination:
    return Combination("test", "rvwmo_base", "MP", "scalar_pair", "cacheable")


def test_rvwmo_verdict_table() -> None:
    failures = []
    for skeleton, builder in BUILDERS.items():
        for variant in VARIANTS:
            case = builder(_combo(), variant, f"chk_{skeleton}_{variant}")
            result = check_rvwmo(case)
            assert result.status == "verified", f"{skeleton}/{variant} not verified"
            forbidden = result.allowed is False
            want = EXPECTED[skeleton][variant]
            if forbidden != want:
                failures.append(
                    f"{skeleton}/{variant}: got {'forbidden' if forbidden else 'allowed'}, "
                    f"want {'forbidden' if want else 'allowed'} :: {result.reason}"
                )
    assert not failures, "\n".join(failures)


def test_rvwmo_cites_rules() -> None:
    case = _mp_case(_combo(), "fence_rw_rw", "chk_rules")
    result = check_rvwmo(case)
    rules = {edge.rule for edge in result.edges if edge.kind == "po"}
    assert "rvwmo-4" in rules  # fence-ordered po edges
    assert result.verdict == "forbidden"


def test_rvwmo_addr_dep_breaks_only_writer_side_of_mp() -> None:
    case = _mp_case(_combo(), "addr_dep", "chk_mp_addr")
    result = check_rvwmo(case)
    assert result.verdict == "allowed"
    po = {(e.src, e.dst): e for e in result.edges if e.kind == "po"}
    # reader R->R is preserved by the address dependency (rule 9)...
    assert po[("p1_ry", "p1_rx")].preserved
    assert po[("p1_ry", "p1_rx")].rule == "rvwmo-9"
    # ...but the writer W->W has no ordering, which is what breaks the cycle.
    assert not po[("p0_wx", "p0_wy")].preserved


def test_rvwmo_control_dependency_orders_stores_only() -> None:
    # WRC/ctrl_dep: P1 edge is R->W (ordered by rule 11) but P2 edge is R->R
    # (NOT ordered, because control dependencies order later stores only).
    case = _wrc_case(_combo(), "ctrl_dep", "chk_wrc_ctrl")
    result = check_rvwmo(case)
    assert result.verdict == "allowed"
    po = {(e.src, e.dst): e for e in result.edges if e.kind == "po"}
    assert po[("p1_rx", "p1_wy")].preserved and po[("p1_rx", "p1_wy")].rule == "rvwmo-11"
    assert not po[("p2_ry", "p2_rx")].preserved


def test_rvwmo_not_applicable_for_non_scalar_model() -> None:
    case = _mp_case(_combo(), "base", "chk_model")
    object.__setattr__(case, "model", "pbmt-nc")
    result = check_rvwmo(case)
    assert result.status == "not_applicable"
    assert result.allowed is None
