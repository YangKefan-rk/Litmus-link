import json
from pathlib import Path

from generator import audit_profile, audit_summary, generate_combinations, generate_profile, write_audit
from models import Combination
from profiles import profile_combinations
from rule_file import load_rule_file
from validator import validate_path


def test_smoke_generation_round_trip(tmp_path: Path) -> None:
    report = generate_profile("smoke", tmp_path)
    assert report["generated"] == 8
    entries = validate_path(tmp_path / "@all")
    assert len(entries) == 8
    first_meta = json.loads((tmp_path / entries[0]).with_suffix(".meta.json").read_text())
    assert first_meta["schema"] == "litmus-link.meta.v1"
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
    assert len(names) == 557840
    assert max_len == 180


def test_summary_only_audit_skips_detail_json(tmp_path: Path) -> None:
    report = write_audit("stress-large", tmp_path, summary_only=True)
    assert report["total_combinations"] == 557840
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


def test_rule_file_illegal_combinations_are_audited(tmp_path: Path) -> None:
    rule_file = tmp_path / "illegal-rules.json"
    rule_file.write_text(
        json.dumps({"name": "custom-illegal", "axes": {"vector": ["fof_strided"]}, "limit": 10}),
        encoding="utf-8",
    )
    rule_set = load_rule_file(rule_file)
    report = generate_combinations(rule_set.name, rule_set.combinations, tmp_path / "out", source=str(rule_file))
    assert report["total_combinations"] == 1
    assert report["generated"] == 0
    assert report["excluded_illegal"] == 1
    assert (tmp_path / "out" / "@all").read_text() == ""
    excluded = json.loads((tmp_path / "out" / "excluded.json").read_text())
    assert excluded[0]["decision"]["status"] == "excluded_illegal"


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


def test_long_parameterized_names_are_hashed() -> None:
    combination = Combination(
        "test",
        "cross",
        "IRIW",
        "vector_load",
        "cacheable_nc_alias",
        cmo="inval_as_flush",
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
