from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List


DEFAULT_MARCH = "rv64gcv_zifencei_zicbom_zicboz"
DEFAULT_MABI = "lp64d"


def asm_check(atfile: Path, gcc: str) -> List[str]:
    """Assembler smoke-check generated litmus instruction bodies.

    This intentionally checks syntax only: it extracts thread-table assembly
    instructions from each .litmus file, wraps them in tiny standalone .S files,
    and asks the selected RISC-V GCC to assemble them with ``-c``. It does not
    validate litmus semantics, register initialization, or expected outcomes.
    """
    if not atfile.exists():
        raise FileNotFoundError(atfile)
    gcc_path = _resolve_gcc(gcc)
    if not gcc_path:
        return [f"asm-check skipped: {gcc} not found"]
    base = atfile.parent
    entries = _read_atfile(atfile)
    errors: list[str] = []
    checked = 0
    with tempfile.TemporaryDirectory(prefix="ll-asm-check-") as tmp:
        tmpdir = Path(tmp)
        for entry in entries:
            litmus = base / entry
            instructions = extract_litmus_instructions(litmus.read_text(encoding="utf-8"))
            if not instructions:
                continue
            checked += 1
            source = tmpdir / f"case_{checked}.S"
            source.write_text(_assembly_source(instructions), encoding="utf-8")
            obj = tmpdir / f"case_{checked}.o"
            proc = subprocess.run(
                [gcc_path, "-c", f"-march={DEFAULT_MARCH}", f"-mabi={DEFAULT_MABI}", str(source), "-o", str(obj)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout).strip().splitlines()
                errors.append(f"{litmus}: assembler rejected extracted instructions: {detail[0] if detail else 'unknown error'}")
    if errors:
        return [f"asm-check failed: {len(errors)} of {checked} litmus files rejected", *errors]
    return [f"asm-check passed: {checked} litmus files assembled with {gcc_path}"]


def extract_litmus_instructions(text: str) -> list[str]:
    instructions: list[str] = []
    in_table = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            if in_table:
                break
            continue
        low = stripped.lower()
        if low.startswith(("exists", "forall", "~exists", "observed", "locations", "filter")):
            if in_table:
                break
            continue
        if "|" not in stripped and not stripped.endswith(";"):
            continue
        cells = [cell.strip() for cell in stripped.rstrip(";").split("|")]
        if cells and all(_is_hart_header(cell) or cell == "" for cell in cells):
            in_table = True
            continue
        if not in_table:
            continue
        for cell in cells:
            instruction = _clean_instruction(cell)
            if instruction:
                instructions.append(instruction)
    return instructions


def _read_atfile(path: Path) -> list[str]:
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(stripped)
    return entries


def _resolve_gcc(gcc: str) -> str:
    resolved = shutil.which(gcc) or (gcc if Path(gcc).exists() else "")
    if resolved:
        return resolved
    if gcc in {"auto", "riscv64-linux-gnu-gcc"}:
        for candidate in ["riscv64-unknown-elf-gcc", "riscv64-linux-gnu-gcc"]:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
    return ""


def _assembly_source(instructions: Iterable[str]) -> str:
    body = ["    " + _rewrite_pseudo_operands(inst) for inst in instructions]
    return "\n".join(
        [
            "    .option norvc",
            "    .text",
            "    .globl _start",
            "_start:",
            *body,
            "    ret",
            "    .data",
            "    .balign 64",
            "x: .dword 0",
            "y: .dword 0",
            "z: .dword 0",
            "a: .dword 0",
            "b: .dword 0",
            "c: .dword 0",
            "",
        ]
    )


def _is_hart_header(cell: str) -> bool:
    return cell.startswith("P") and cell[1:].isdigit()


def _looks_like_instruction(cell: str) -> bool:
    return bool(cell) and not _is_hart_header(cell) and not cell.startswith(("{", "}", "(*", "//"))


def _clean_instruction(cell: str) -> str:
    cell = cell.strip().rstrip(";").strip()
    if not _looks_like_instruction(cell):
        return ""
    return cell


def _rewrite_pseudo_operands(instruction: str) -> str:
    # diy/herd litmus occasionally uses symbolic register initializers in the
    # init block, but the instruction table itself is ordinary assembly. Keep a
    # tiny escape hatch for labels that appear as immediates in hand/corpus rows.
    return instruction.replace("#", "//")
