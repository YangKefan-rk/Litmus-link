from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    ("instruction", "extra_extension"),
    [("sw x5,0(x6)", ""), ("amoswap.h.aqrl x0,x5,(x6)", "_zabha")],
)
def test_asm_check_invokes_selected_compiler(
    tmp_path: Path, instruction: str, extra_extension: str
) -> None:
    litmus = tmp_path / "T.litmus"
    litmus.write_text(
        f"""RISCV T
{{
0:x5=1; 0:x6=x;
}}
 P0          | P1          ;
 {instruction} | lw x7,0(x8) ;
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
    assert f"-march=rv64gcv_zifencei_zicbom_zicboz{extra_extension}" in log.read_text(encoding="utf-8").split()


def test_asm_check_rejects_indexed_case_without_instructions(tmp_path: Path) -> None:
    litmus = tmp_path / "truncated.litmus"
    litmus.write_text("RISCV truncated\n{}\nP0;\n", encoding="utf-8")
    atfile = tmp_path / "@all"
    atfile.write_text("truncated.litmus\n", encoding="utf-8")
    fake_gcc = tmp_path / "fake-gcc"
    fake_gcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_gcc.chmod(0o755)

    assert asm_check(atfile, str(fake_gcc)) == [
        "asm-check failed: 1 of 1 litmus files rejected",
        f"{litmus}: no thread-table instructions found",
    ]


def test_asm_check_skips_missing_compiler(tmp_path: Path) -> None:
    atfile = tmp_path / "@all"
    atfile.write_text("", encoding="utf-8")
    lines = asm_check(atfile, "definitely-not-a-riscv-gcc")
    assert lines == ["asm-check skipped: definitely-not-a-riscv-gcc not found"]
