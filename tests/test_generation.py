import json
from pathlib import Path

from generator import audit_profile, audit_summary, generate_combinations, generate_profile, write_audit
from gui import preview_payload
from models import Combination
from profiles import profile_combinations
from rule_file import RuleFileError, load_rule_file
from validator import validate_path


def test_smoke_generation_round_trip(tmp_path: Path) -> None:
    report = generate_profile("smoke", tmp_path)
    assert report["generated"] == 8
    assert report["generated_litmus"] > report["generated"]
    entries = validate_path(tmp_path / "@all")
    assert len(entries) == report["generated_litmus"]
    first_meta = json.loads((tmp_path / entries[0]).with_suffix(".meta.json").read_text())
    assert first_meta["schema"] == "litmus-link.meta.v1"
    assert first_meta["case_ir"]["variant"]
    assert first_meta["test_description"]["summary"]
    assert first_meta["test_description"]["features"]


def test_full_cross_has_no_missing(tmp_path: Path) -> None:
    report = write_audit("full-cross", tmp_path)
    baseline = json.loads(Path("specs/profiles/full-cross-baseline.json").read_text())
    for key in ["profile", "total_combinations", "generated", "excluded_illegal", "excluded_unsupported", "hand_required", "missing"]:
        assert report[key] == baseline[key]
    assert (tmp_path / "missing.json").read_text() == ""


def test_stress_large_summary_matches_baseline() -> None:
    report = audit_summary("stress-large", profile_combinations("stress-large"))
    baseline = json.loads(Path("specs/profiles/stress-large-baseline.json").read_text())
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
    assert len(names) == 250360
    assert max_len == 180


def test_summary_only_audit_skips_detail_json(tmp_path: Path) -> None:
    report = write_audit("stress-large", tmp_path, summary_only=True)
    assert report["total_combinations"] == 250360
    assert (tmp_path / "audit-report.json").exists()
    assert (tmp_path / "cross-coverage.md").exists()
    assert not (tmp_path / "covered.json").exists()


def test_vector_profile_has_illegal_and_hand_buckets() -> None:
    _rows, report = audit_profile("vector_mem")
    assert report["excluded_illegal"] > 0
    assert report["hand_required"] > 0
    assert report["missing"] == 0


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


def test_preview_payload_includes_litmus_and_analysis() -> None:
    preview = preview_payload({"mode": "profile", "profile": "smoke", "sample_limit": 4})
    generated = [item for item in preview["sample"] if item.get("litmus")]
    assert len(generated) > 4
    first = generated[0]
    assert first["litmus"].startswith("RISCV ")
    assert first["case_ir"]["relations"]
    assert first["analysis"]["cycle"]
    assert first["analysis"]["exists"]
    assert "forbidden_outcome" in first["analysis"]


def test_mp_cacheable_expands_to_multiple_variants() -> None:
    preview = preview_payload(
        {
            "mode": "rule",
            "rule": {"name": "mp-cacheable", "axes": {"skeleton": ["MP"], "attribute": ["cacheable"]}, "limit": 10},
            "sample_limit": 1,
        }
    )
    generated = [item for item in preview["sample"] if item.get("litmus")]
    variants = {item["case_ir"]["variant"] for item in generated}
    assert preview["report"]["total_combinations"] == 1
    assert preview["report"]["generated_litmus"] >= 6
    assert len(generated) >= 6
    assert {"base", "fence_rw_rw", "addr_dep"}.issubset(variants)


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
            "elem_order": "ordered_elements",
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
    assert "params_" in combination.name
