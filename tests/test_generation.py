import json
from pathlib import Path

from litmus_link.generator import audit_profile, generate_profile, write_audit
from litmus_link.validator import validate_path


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
