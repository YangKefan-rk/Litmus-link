import json
from pathlib import Path

from generator import audit_profile, generate_combinations, generate_profile, write_audit
from rule_file import load_rule_file
from validator import validate_path


def test_smoke_generation_round_trip(tmp_path: Path) -> None:
    report = generate_profile("smoke", tmp_path)
    assert report["generated"] == 8
    entries = validate_path(tmp_path / "@all")
    assert len(entries) == 8
    first_meta = json.loads((tmp_path / entries[0]).with_suffix(".meta.json").read_text())
    assert first_meta["schema"] == "litmus-link.meta.v1"


def test_full_cross_has_no_missing(tmp_path: Path) -> None:
    report = write_audit("full-cross", tmp_path)
    baseline = json.loads(Path("specs/profiles/full-cross-baseline.json").read_text())
    for key in ["profile", "total_combinations", "generated", "excluded_illegal", "excluded_unsupported", "hand_required", "missing"]:
        assert report[key] == baseline[key]
    assert (tmp_path / "missing.json").read_text() == ""


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
