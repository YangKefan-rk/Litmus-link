from __future__ import annotations

"""Nanhu-supported RISC-V AMO decoding and fixed-width arithmetic."""

import re
from dataclasses import dataclass


AMO_OPERATIONS = (
    "swap",
    "add",
    "xor",
    "and",
    "or",
    "min",
    "max",
    "minu",
    "maxu",
)
AMO_WIDTH_BYTES = (4, 8)
AMO_ORDERINGS = ("relaxed", "aq", "rl", "aqrl")


class AmoError(ValueError):
    pass


@dataclass(frozen=True)
class AmoSpec:
    operation: str
    width_bytes: int
    ordering: str

    def __post_init__(self) -> None:
        if self.operation not in AMO_OPERATIONS:
            raise AmoError(f"unsupported Nanhu AMO operation: {self.operation}")
        if self.width_bytes not in AMO_WIDTH_BYTES:
            raise AmoError(
                f"Nanhu AMOs support only W/D widths, got {self.width_bytes} bytes"
            )
        if self.ordering not in AMO_ORDERINGS:
            raise AmoError(f"unsupported AMO ordering: {self.ordering}")

    @property
    def suffix(self) -> str:
        return {
            "relaxed": "",
            "aq": ".aq",
            "rl": ".rl",
            # RISC-V encodes the two ordering bits with the single ``aqrl``
            # suffix.  ``.aq.rl`` is not an ISA mnemonic accepted by the
            # assembler.
            "aqrl": ".aqrl",
        }[self.ordering]

    @property
    def width_suffix(self) -> str:
        return {4: "w", 8: "d"}[self.width_bytes]

    @property
    def mnemonic(self) -> str:
        return f"amo{self.operation}.{self.width_suffix}{self.suffix}"


_AMO_RE = re.compile(
    r"^amo(swap|add|xor|and|or|min|max|minu|maxu)\.([wd])"
    r"(?:(\.aqrl)|(\.aq)|(\.rl))?$"
)


def parse_amo_mnemonic(instruction: str) -> AmoSpec:
    mnemonic = instruction.strip().split(maxsplit=1)[0].lower()
    match = _AMO_RE.fullmatch(mnemonic)
    if match is None:
        raise AmoError(f"unsupported Nanhu AMO instruction: {mnemonic or instruction!r}")
    ordering = (
        "aqrl"
        if match.group(3)
        else "aq"
        if match.group(4)
        else "rl"
        if match.group(5)
        else "relaxed"
    )
    return AmoSpec(match.group(1), 4 if match.group(2) == "w" else 8, ordering)


def apply_amo(operation: str, width_bytes: int, old: int, operand: int) -> int:
    """Return the W/D bit pattern written by one AMO transaction."""

    AmoSpec(operation, width_bytes, "relaxed")
    bits = width_bytes * 8
    mask = (1 << bits) - 1
    old_u = old & mask
    operand_u = operand & mask

    if operation == "swap":
        return operand_u
    if operation == "add":
        return (old_u + operand_u) & mask
    if operation == "xor":
        return old_u ^ operand_u
    if operation == "and":
        return old_u & operand_u
    if operation == "or":
        return old_u | operand_u
    if operation == "minu":
        return min(old_u, operand_u)
    if operation == "maxu":
        return max(old_u, operand_u)

    old_s = _signed(old_u, bits)
    operand_s = _signed(operand_u, bits)
    if operation == "min":
        return old_u if old_s <= operand_s else operand_u
    if operation == "max":
        return old_u if old_s >= operand_s else operand_u
    raise AmoError(f"unsupported Nanhu AMO operation: {operation}")


def amo_read_result(width_bytes: int, old: int, *, xlen: int = 64) -> int:
    """Return the XLEN-wide value written to rd by an AMO.W/AMO.D."""

    if width_bytes not in AMO_WIDTH_BYTES:
        raise AmoError(
            f"Nanhu AMOs support only W/D widths, got {width_bytes} bytes"
        )
    bits = width_bytes * 8
    if xlen < bits:
        raise AmoError(f"XLEN {xlen} is narrower than AMO width {bits}")
    value = old & ((1 << bits) - 1)
    if bits == xlen:
        return value
    signed = _signed(value, bits)
    return signed & ((1 << xlen) - 1)


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value
