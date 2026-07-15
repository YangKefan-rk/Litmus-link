from pathlib import Path

from litmus_link.upstream import import_upstream


def test_import_upstream_indexes_without_copying(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "MP.litmus").write_text("RISCV MP\n", encoding="utf-8")
    out = tmp_path / "out"
    index = import_upstream(src, "riscv", out)
    assert index["count"] == 1
    assert (out / "riscv-index.json").exists()
