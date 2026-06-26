from pathlib import Path

import math

import pytest

from PIL import ImageDraw, Image

import diagram as D
from diagram import render_diagram, _draw_harts, _route_relations
from litmus_ir import _mp_case, _lb_case, _sb_case, _wrc_case, _rwc_case, _iriw_case
from renderer import render_cases
from rules import evaluate
from models import Combination
from solver import solve_generated_case


def test_render_diagram_writes_png_and_summary(tmp_path: Path) -> None:
    combination = Combination("test", "rvwmo_base", "MP", "scalar_pair", "cacheable")
    case = render_cases(combination, evaluate(combination))[0]
    solver = solve_generated_case(case).to_json()
    result = render_diagram(case.case_ir, solver, tmp_path)  # type: ignore[arg-type]
    assert result.png_path.exists()
    assert result.json_path.exists()
    assert result.summary["schema"] == "litmus-link.diagram.v1"
    assert result.summary["relations"]
    assert result.png_path.stat().st_size > 1000


def _routed(case):
    """Build hart boxes + routed relation paths without producing a file."""
    image = Image.new("RGB", (D.WIDTH, D.HEIGHT))
    draw = ImageDraw.Draw(image)
    fonts = D._fonts()
    boxes, columns = _draw_harts(draw, fonts, case)
    return boxes, _route_relations(case, boxes, columns)


def _close_count(p1, p2, threshold: float = 7.0) -> int:
    return sum(1 for a in p1 if min(math.dist(a, b) for b in p2) < threshold)


SKELETON_BUILDERS = [_mp_case, _lb_case, _sb_case, _wrc_case, _rwc_case, _iriw_case]


@pytest.mark.parametrize("builder", SKELETON_BUILDERS)
def test_relation_routing_is_clean(builder) -> None:
    combination = Combination("test", "rvwmo_base", "MP", "scalar_pair", "cacheable")
    case = builder(combination, "base", f"route_{builder.__name__}")
    boxes, routed = _routed(case)
    assert routed, "expected at least one cross-hart relation"

    for relation in routed:
        points = relation["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # Stay inside the canvas margins and never cut above the arch band.
        assert 62 <= min(xs) and max(xs) <= 1538, f"{relation['label']} x out of bounds"
        assert min(ys) >= D.LANE_TOP - 2, f"{relation['label']} routed above the lane band"
        # A relation path may touch its own endpoint boxes but must not slice
        # through the interior of any event box.
        for event_id, box in boxes.items():
            inside = sum(1 for (x, y) in points if box[0] + 4 < x < box[2] - 4 and box[1] + 4 < y < box[3] - 4)
            assert inside <= 1, f"{relation['label']} cuts through box {event_id}"

    # No two relations may run parallel (sustained closeness). A single shared
    # crossing point is fine; a long overlapping run is not.
    for i in range(len(routed)):
        for j in range(i + 1, len(routed)):
            overlap = _close_count(routed[i]["points"], routed[j]["points"])
            assert overlap <= 5, f"{routed[i]['label']} overlaps {routed[j]['label']} ({overlap} pts)"

