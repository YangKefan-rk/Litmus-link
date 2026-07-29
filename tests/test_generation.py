import json
import shutil
from pathlib import Path

import pytest

from litmus_link import rule_file
from litmus_link.generator import audit_profile, audit_summary, generate_combinations, generate_profile, write_audit, write_audit_for_combinations
from litmus_link.workflow import materialize_preview_diagram, preview_payload
from litmus_link.models import Combination
from litmus_link.profiles import profile_combinations
from litmus_link.rule_file import RuleFileError, load_rule_data, load_rule_file
from litmus_link.validator import ValidationError, validate_path


def test_smoke_generation_round_trip(tmp_path: Path) -> None:
    report = generate_profile("smoke", tmp_path)
    assert report["generated"] == 8
    assert report["generated_litmus"] > report["generated"]
    assert report["solver"]["verified"] > 0
    entries = validate_path(tmp_path / "@all")
    assert len(entries) == report["generated_litmus"]
    first_meta = json.loads((tmp_path / entries[0]).with_suffix(".meta.json").read_text())
    first_solver = json.loads((tmp_path / entries[0]).with_suffix(".solver.json").read_text())
    assert first_meta["schema"] == "litmus-link.meta.v1"
    assert first_meta["case_ir"]["variant"]
    assert first_meta["solver"] == first_solver
    assert "diagram" not in first_meta
    assert not (tmp_path / entries[0]).with_suffix(".diagram.png").exists()
    assert report["diagram_mode"] == "on_demand"
    assert report["generated_diagrams"] == 0
    assert first_solver["status"] in {"verified", "conflict", "not_applicable"}
    assert first_meta["test_description"]["summary"]
    assert first_meta["test_description"]["features"]


def test_validate_profile_solver_sidecar(tmp_path: Path) -> None:
    generate_profile("smoke", tmp_path)
    entry = validate_path(tmp_path)[0]
    solver_path = (tmp_path / entry).with_suffix(".solver.json")

    solver_path.write_text('{}\n', encoding="utf-8")
    with pytest.raises(ValidationError, match="solver JSON differs from metadata"):
        validate_path(tmp_path)

    solver_path.unlink()
    with pytest.raises(ValidationError, match="missing solver file"):
        validate_path(tmp_path)


@pytest.mark.parametrize(("body", "message"), [("[]", "expected a JSON object"), ("{}", "invalid combination metadata")])
def test_validate_profile_reports_malformed_metadata(tmp_path: Path, body: str, message: str) -> None:
    generate_profile("smoke", tmp_path)
    entry = validate_path(tmp_path)[0]
    meta_path = (tmp_path / entry).with_suffix(".meta.json")
    meta_path.write_text(body, encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        validate_path(tmp_path)


def test_profile_generation_can_explicitly_write_diagrams(tmp_path: Path) -> None:
    report = generate_profile("smoke", tmp_path, diagrams=True)
    entries = validate_path(tmp_path / "@all")
    first_meta = json.loads((tmp_path / entries[0]).with_suffix(".meta.json").read_text())
    first_diagram = json.loads((tmp_path / entries[0]).with_suffix(".diagram.json").read_text())
    assert first_meta["diagram"] == first_diagram
    assert (tmp_path / entries[0]).with_suffix(".diagram.png").exists()
    assert report["diagram_mode"] == "eager"
    assert report["generated_diagrams"] == report["generated_litmus"]


def test_profile_generation_reports_exact_case_progress(tmp_path: Path) -> None:
    updates = []
    report = generate_profile(
        "smoke",
        tmp_path,
        progress_callback=lambda current, total, message: updates.append((current, total, message)),
    )
    determinate = [(current, total) for current, total, _message in updates if total > 0]
    assert determinate
    assert determinate[0] == (0, report["generated_litmus"])
    assert determinate[-1] == (report["generated_litmus"], report["generated_litmus"])
    assert [current for current, _total in determinate] == sorted(
        current for current, _total in determinate
    )


def test_full_cross_has_no_missing(tmp_path: Path) -> None:
    report = write_audit("full-cross", tmp_path)
    baseline = json.loads(Path("tests/baselines/full-cross.json").read_text())
    for key in ["profile", "total_combinations", "generated", "excluded_illegal", "excluded_unsupported", "hand_required", "missing"]:
        assert report[key] == baseline[key]
    assert (tmp_path / "missing.json").read_text() == ""


def test_stress_large_summary_matches_baseline() -> None:
    report = audit_summary("stress-large", profile_combinations("stress-large"))
    baseline = json.loads(Path("tests/baselines/stress-large.json").read_text())
    assert report == baseline


def test_stress_large_names_are_unique_and_short() -> None:
    names = set()
    max_len = 0
    for combination in profile_combinations("stress-large"):
        name = combination.name
        assert len(name) <= 180
        assert name not in names
        names.add(name)
        max_len = max(max_len, len(name))
    assert len(names) == 108600
    assert max_len <= 112


def test_summary_only_audit_skips_detail_json(tmp_path: Path) -> None:
    report = write_audit("stress-large", tmp_path, summary_only=True)
    assert report["total_combinations"] == 108600
    assert (tmp_path / "audit-report.json").exists()
    assert (tmp_path / "cross-coverage.md").exists()
    assert not (tmp_path / "covered.json").exists()
    assert not (tmp_path / "excluded-unsupported.json").exists()


def test_audit_lists_unsupported_combinations(tmp_path: Path) -> None:
    rules = load_rule_data({"axes": {"attribute": ["cacheable", "pbmt_nc"], "vector": "unit_load"}})
    report = write_audit_for_combinations(rules.name, rules.combinations, tmp_path)
    excluded = json.loads((tmp_path / "excluded-unsupported.json").read_text(encoding="utf-8"))

    assert len(excluded) == report["excluded_unsupported"] == 1
    assert excluded[0]["combination"]["attribute"] == "pbmt_nc"
    assert excluded[0]["decision"]["status"] == "excluded_unsupported"
    assert "restricted to cacheable mappings" in excluded[0]["decision"]["reason"]


def test_vector_profile_contains_only_nanhu_supported_memory_types() -> None:
    rows, report = audit_profile("vector_mem")
    assert report["total_combinations"] == 609840
    assert report["generated"] == 609840
    assert report["generated_litmus"] == 1829520
    assert report["excluded_illegal"] == 0
    assert report["excluded_unsupported"] == 0
    assert report["hand_required"] == 0
    assert report["missing"] == 0
    combinations = [combination for combination, _decision in rows]
    indexed = [combination for combination in combinations if "indexed" in combination.vector]
    non_indexed = [combination for combination in combinations if "indexed" not in combination.vector]
    assert {combination.params["index_eew"] for combination in indexed} == {"ei8", "ei16", "ei32", "ei64"}
    assert all("index_eew" not in combination.params for combination in non_indexed)
    assert {combination.params["vl"] for combination in combinations} == {
        "vl1", "vl2", "vl4", "vl8", "vl16", "vl32", "vl64", "vlmax"
    }


def test_rule_file_generation(tmp_path: Path) -> None:
    rule_file = tmp_path / "custom-rules.json"
    rule_file.write_text(
        json.dumps(
            {
                "name": "custom-vector-cmo",
                "defaults": {"skeleton": "MP", "attribute": "cacheable"},
                "axes": {
                    "vector": ["unit_load", "unit_store"],
                    "cmo": ["no_cmo", "flush"],
                },
                "exclude": [{"vector": "unit_store", "cmo": "flush"}],
                "limit": 10,
            }
        ),
        encoding="utf-8",
    )
    rule_set = load_rule_file(rule_file)
    assert len(rule_set.combinations) == 3
    report = generate_combinations(rule_set.name, rule_set.combinations, tmp_path / "out", source=str(rule_file))
    assert report["total_combinations"] == 3
    assert report["source"] == str(rule_file)
    assert validate_path(tmp_path / "out" / "@all")


def test_rule_file_rejects_nonexistent_vector_forms(tmp_path: Path) -> None:
    rule_file = tmp_path / "illegal-rules.json"
    rule_file.write_text(
        json.dumps({"name": "custom-illegal", "axes": {"vector": ["fof_strided"]}, "limit": 10}),
        encoding="utf-8",
    )
    try:
        load_rule_file(rule_file)
    except RuleFileError as exc:
        assert "invalid vector value 'fof_strided'" in str(exc)
    else:
        raise AssertionError("fof_strided must be rejected before generation")


def test_rule_file_reports_deferred_vector_form_as_unsupported(tmp_path: Path) -> None:
    rule_file = tmp_path / "deferred-vector.json"
    rule_file.write_text(
        json.dumps({"name": "deferred-vector", "axes": {"vector": ["fof_load"]}, "limit": 10}),
        encoding="utf-8",
    )
    rule_set = load_rule_file(rule_file)
    assert len(rule_set.combinations) == 1
    report = generate_combinations(rule_set.name, rule_set.combinations, tmp_path / "out")
    assert report["generated"] == 0
    assert report["excluded_unsupported"] == 1
    assert not list((tmp_path / "out").glob("*.litmus"))


def test_rule_file_vector_cmo_cross_renders_both_operations(tmp_path: Path) -> None:
    rule_file = tmp_path / "cross-rules.json"
    rule_file.write_text(
        json.dumps({"name": "custom-cross", "axes": {"vector": ["unit_load"], "cmo": ["flush"]}, "limit": 10}),
        encoding="utf-8",
    )
    rule_set = load_rule_file(rule_file)
    assert rule_set.combinations[0].category == "cross"
    report = generate_combinations(rule_set.name, rule_set.combinations, tmp_path / "out", source=str(rule_file))
    assert report["generated"] == 1
    litmus = next((tmp_path / "out").glob("*.litmus")).read_text()
    assert "vle32.v" in litmus
    assert "cbo.flush" in litmus


@pytest.mark.parametrize(
    ("vector", "cmo", "params", "hart", "constants", "instructions"),
    [
        (
            "strided_load", "flush", {"vl": "vl32", "stride_bytes": 16}, 0,
            {"x11": 32, "x20": 16}, ["vsetvli x10,x11", "vlse32.v v8,(x6),x20"],
        ),
        (
            "strided_load", "no_cmo", {"vl": "vl64", "footprint": "cross_line"}, 1,
            {"x11": 64, "x20": 8}, ["vsetvli x10,x11", "vlse32.v v8,(x8),x20"],
        ),
        (
            "segment_indexed_unordered_load", "flush", {"nf": "nf3", "vl": "vl2"}, 0,
            {"x20": 12}, ["vmul.vx v16,v16,x20", "vluxseg3ei32.v v8,(x6),v16"],
        ),
    ],
)
def test_vector_observation_initializes_setup_registers(
    tmp_path: Path, vector: str, cmo: str, params: dict, hart: int,
    constants: dict[str, int], instructions: list[str],
) -> None:
    rules = load_rule_data({"cases": [{"vector": vector, "cmo": cmo, "params": params}]})
    report = generate_combinations(rules.name, rules.combinations, tmp_path)
    assert report["generated_litmus"] == 1
    litmus = next(tmp_path.glob("*.litmus")).read_text(encoding="utf-8")
    meta = json.loads(next(tmp_path.glob("*.meta.json")).read_text(encoding="utf-8"))
    case_ir = meta["case_ir"]
    assert case_ir["variant"] == "observation"
    for register, value in constants.items():
        initializer = f"{hart}:{register}={value};"
        assert initializer in case_ir["init_lines"]
        assert initializer in litmus
    setup_instructions = [event["instruction"] for event in case_ir["harts"][hart]]
    for instruction in instructions:
        assert any(rendered.startswith(instruction) for rendered in setup_instructions)


def test_preview_payload_includes_litmus_and_analysis() -> None:
    preview = preview_payload(
        {
            "mode": "scalar",
            "engine": "native_templates",
            "skeletons": ["MP"],
            "mechanisms": ["po"],
            "annotations": ["P"],
            "include_same": False,
            "sample_limit": 4,
            "judge": True,
            "solver_backend": "embedded",
        }
    )
    generated = [item for item in preview["sample"] if item.get("litmus")]
    assert len(generated) == 1
    first = generated[0]
    assert first["litmus"].startswith("RISCV ")
    assert first["case_ir"]["relations"]
    assert first["solver"]["status"] in {"verified", "conflict", "not_applicable"}
    assert first["diagram"]["schema"] == "litmus-link.diagram.v1"
    assert first["diagram"]["status"] in {"deferred", "ready"}
    deferred_png = Path(first["diagram"]["png"])
    shutil.rmtree(deferred_png.parent, ignore_errors=True)
    assert not deferred_png.exists()
    rendered = materialize_preview_diagram(first)
    assert rendered["status"] == "ready"
    assert rendered["cached"] is False
    assert Path(rendered["png"]).exists()
    cached = materialize_preview_diagram(first)
    assert cached["status"] == "ready"
    assert cached["cached"] is True
    assert first["analysis"]["solver_status"] == first["solver"]["status"]
    assert first["analysis"]["cycle"]
    assert first["analysis"]["exists"]
    assert "forbidden_outcome" in first["analysis"]


def test_preview_payload_does_not_eagerly_call_diagram_renderer(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import litmus_link.workflow as workflow

    def fail_if_called(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("preview must not render PNG diagrams")

    monkeypatch.setattr(workflow, "render_diagram", fail_if_called)
    preview = workflow.preview_payload(
        {
            "mode": "scalar",
            "engine": "native_templates",
            "skeletons": ["MP"],
            "mechanisms": ["po"],
            "annotations": ["P"],
            "include_same": False,
            "sample_limit": 1,
            "judge": False,
        }
    )
    generated = [item for item in preview["sample"] if item.get("litmus")]
    assert generated
    assert all(item["diagram"]["status"] in {"deferred", "ready"} for item in generated)


def test_rule_file_param_axes_expand_into_params(tmp_path: Path) -> None:
    rule_file = tmp_path / "param-rules.json"
    rule_file.write_text(
        json.dumps(
            {
                "name": "custom-param-axis",
                "axes": {"vector": ["unit_load"], "attribute": ["cacheable", "pbmt_nc"]},
                "param_axes": {"sew": ["e32", "e64"], "footprint": ["same_line", "cross_page"]},
                "param_defaults": {"stress": "load_queue_replay"},
                "limit": 20,
            }
        ),
        encoding="utf-8",
    )
    rule_set = load_rule_file(rule_file)
    assert len(rule_set.combinations) == 8
    assert {combination.params["sew"] for combination in rule_set.combinations} == {"e32", "e64"}
    assert {combination.params["footprint"] for combination in rule_set.combinations} == {"same_line", "cross_page"}
    assert all(combination.params["stress"] == "load_queue_replay" for combination in rule_set.combinations)


def test_rule_limit_is_checked_before_creating_parameter_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_expansion(*args, **kwargs) -> None:
        pytest.fail("over-limit parameter rows should never be expanded")

    monkeypatch.setattr(rule_file, "_expand_params", unexpected_expansion)
    with pytest.raises(RuleFileError, match="more than 3 combinations"):
        load_rule_data({"axes": {}, "param_axes": {"a": list(range(30)), "b": list(range(40))}, "limit": 3})


def test_rule_limit_counts_only_included_axes_and_cases() -> None:
    rule_set = load_rule_data(
        {
            "axes": {"skeleton": ["MP", "SB"]},
            "param_axes": {"width": ["w32", "w64"]},
            "cases": [{"skeleton": "LB", "params": {"extra": True}}],
            "exclude": [{"skeleton": "SB"}],
            "limit": 4,
        }
    )
    assert [(row.skeleton, row.params) for row in rule_set.combinations] == [
        ("MP", {"width": "w32"}),
        ("MP", {"width": "w64"}),
        ("LB", {"width": "w32", "extra": "true"}),
        ("LB", {"width": "w64", "extra": "true"}),
    ]


def test_vector_params_render_into_observation_body(tmp_path: Path) -> None:
    combination = Combination(
        "test",
        "vector_mem",
        "MP",
        "vector_load",
        "cacheable",
        vector="unit_load",
        params={"sew": "e64", "lmul": "m4", "mask": "masked", "tail": "tu_mu", "vl": "vl2", "footprint": "cross_page"},
    )
    cases = generate_combinations("vector-param-render", [combination], tmp_path)
    assert cases["generated_litmus"] == 1
    litmus = (tmp_path / f"{combination.name}.litmus").read_text()
    assert "vsetivli x10,2,e64,m4,tu,mu" in litmus
    assert "vle64.v v8,(x8),v0.t" in litmus


def test_long_parameterized_names_are_hashed() -> None:
    combination = Combination(
        "test",
        "cross",
        "IRIW",
        "vector_load",
        "cacheable_nc_alias",
        cmo="flush",
        vector="indexed_unordered_load",
        params={
            "alias": "cacheable_nc",
            "footprint": "cross_page",
            "lmul": "m1",
            "mask": "masked",
            "sew": "e16",
            "stress": "store_buffer_full",
            "sync": "full_alias_sync",
            "tail": "ta_mu",
            "vl": "vl2",
        },
    )
    assert len(combination.name) <= 180
    assert "+Cfg." in combination.name
    assert "cacheable" not in combination.name
    assert "no_tlb" not in combination.name
    assert "no_cmo" not in combination.name
