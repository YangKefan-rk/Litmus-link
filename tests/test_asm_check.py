from pathlib import Path

from litmus_link.asm_check import asm_check, extract_litmus_instructions


def test_extract_litmus_instructions_ignores_init_and_exists() -> None:
    text = """RISCV T
{
0:x5=1; 0:x6=x;
}
 P0          | P1          ;
 sw x5,0(x6) | lw x7,0(x8) ;
 fence rw,rw |             ;
exists
(1:x7=0)
"""
    assert extract_litmus_instructions(text) == ["sw x5,0(x6)", "lw x7,0(x8)", "fence rw,rw"]


def test_asm_check_invokes_selected_compiler(tmp_path: Path) -> None:
    litmus = tmp_path / "T.litmus"
    litmus.write_text(
        """RISCV T
{
0:x5=1; 0:x6=x;
}
 P0          | P1          ;
 sw x5,0(x6) | lw x7,0(x8) ;
exists
(1:x7=0)
""",
        encoding="utf-8",
    )
    atfile = tmp_path / "@all"
    atfile.write_text("T.litmus\n", encoding="utf-8")
    log = tmp_path / "gcc.log"
    fake_gcc = tmp_path / "fake-gcc"
    fake_gcc.write_text(
        f"""#!/bin/sh
echo \"$@\" >> {log}
exit 0
""",
        encoding="utf-8",
    )
    fake_gcc.chmod(0o755)
    lines = asm_check(atfile, str(fake_gcc))
    assert lines == [f"asm-check passed: 1 litmus files assembled with {fake_gcc}"]
    assert "-march=rv64gcv_zifencei_zicbom_zicboz" in log.read_text(encoding="utf-8")


def test_asm_check_skips_missing_compiler(tmp_path: Path) -> None:
    atfile = tmp_path / "@all"
    atfile.write_text("", encoding="utf-8")
    lines = asm_check(atfile, "definitely-not-a-riscv-gcc")
    assert lines == ["asm-check skipped: definitely-not-a-riscv-gcc not found"]
