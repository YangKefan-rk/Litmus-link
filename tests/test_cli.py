from pathlib import Path

from litmus_link.cli import main


def test_cli_generate_and_validate(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "smoke"
    assert main(["generate", "--profile", "smoke", "--out", str(out)]) == 0
    assert main(["validate", str(out / "@all")]) == 0
    captured = capsys.readouterr()
    assert "validated 8 litmus files" in captured.out


def test_cli_list_rules(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["list", "rules"]) == 0
    assert "pbmt_leaf_only" in capsys.readouterr().out
