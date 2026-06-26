from pathlib import Path

from diagram import render_diagram
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
