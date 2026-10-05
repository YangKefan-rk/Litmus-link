from __future__ import annotations

import re

import pytest

from litmus_link.corpus_riscv import parse_litmus
from litmus_link.memory_layout import MemoryLayoutConfig
from litmus_link.native_cycles import NativeCycle
from litmus_link.native_edges import edge_by_label
from litmus_link.native_scalar import lower_native_cycle


def _memory_addresses(text: str, *, read_value: int = 0) -> list[list[int]]:
    """Evaluate rendered address arithmetic, supplying a chosen memory read value."""
    parsed = parse_litmus(text)
    addresses = []
    for hart, instructions in enumerate(parsed.harts):
        registers = {"x0": 0}
        for line in parsed.init_lines:
            match = re.fullmatch(rf"{hart}:(x\d+)=(.+)", line)
            if match:
                register, value = match.groups()
                registers[register] = {"x": 0x1000, "y": 0x2000}.get(value)
                if registers[register] is None:
                    registers[register] = int(value, 0)
        accesses = []
        for instruction in instructions:
            opcode, *operands = re.split(r"[\s,()]+", instruction.strip(" ()"))
            if opcode in {"addi", "andi", "add", "xor"}:
                dst, src, rhs = operands
                left = registers[src]
                right = int(rhs, 0) if opcode.endswith("i") else registers[rhs]
                if opcode in {"addi", "add"}:
                    registers[dst] = left + right
                elif opcode == "andi":
                    registers[dst] = left & right
                else:
                    registers[dst] = left ^ right
            elif opcode.startswith("amo"):
                dst, _data, base = operands
                accesses.append(registers[base])
                registers[dst] = read_value
            elif opcode in {"lhu", "lwu", "ld", "sh", "sw", "sd"}:
                data, offset, base = operands
                accesses.append(registers[base] + int(offset, 0))
                if opcode.startswith("l"):
                    registers[data] = read_value
            else:
                raise AssertionError(f"unexpected instruction: {instruction}")
            registers["x0"] = 0
        addresses.append(accesses)
    return addresses


@pytest.mark.parametrize("annotation", ["AMO", "Aq", "Rl", "AR"])
@pytest.mark.parametrize("overlap", ["same_start", "partial_overlap"])
@pytest.mark.parametrize("widths", [(8, 4, 2, 4), (2, 8, 4, 2)])
@pytest.mark.parametrize("same_location", [False, True])
def test_atomic_mixed_rendered_addresses_match_layout(
    annotation: str, overlap: str, widths: tuple[int, ...], same_location: bool
) -> None:
    local = "s" if same_location else "d"
    cycle = NativeCycle(
        tuple(edge_by_label(label) for label in ("Rfe", f"Po{local}RR", "Fre", f"Po{local}WW")),
        "MP",
        (annotation,) * 4,
    )
    layout = MemoryLayoutConfig("atomic_mixed", overlap=overlap, width_pattern=widths)
    case = lower_native_cycle(cycle, memory_layout=layout)
    expected = [
        [
            {"x": 0x1000, "y": 0x2000}[event.location] + event.memory_access.offset_bytes
            for event in events
            if event.role == "cycle-event"
        ]
        for events in case.case_ir.harts
    ]
    assert _memory_addresses(case.litmus) == expected


def test_offset_amo_does_not_shift_a_later_plain_access() -> None:
    cycle = NativeCycle(
        tuple(edge_by_label(label) for label in ("Rfe", "PosRR", "Fre", "PosWW")),
        "MP",
        ("AMO", "AMO", "P", "P"),
    )
    layout = MemoryLayoutConfig(
        "atomic_mixed", overlap="partial_overlap", width_pattern=(4, 4, 8, 8)
    )
    case = lower_native_cycle(cycle, memory_layout=layout)
    assert _memory_addresses(case.litmus) == [
        [0x1000, 0x1004],
        [0x1004, 0x1000],
    ]


@pytest.mark.parametrize("realdep", [False, True])
def test_offset_amo_preserves_address_dependency(realdep: bool) -> None:
    cycle = NativeCycle(
        tuple(edge_by_label(label) for label in ("Rfe", "DpAddrsR", "Fre", "PosWW")),
        "MP",
        ("P", "P", "AMO", "AMO"),
    )
    layout = MemoryLayoutConfig(
        "atomic_mixed", overlap="partial_overlap", width_pattern=(8, 8, 4, 4)
    )
    case = lower_native_cycle(cycle, memory_layout=layout, realdep=realdep)
    dependency_offset = 128 if realdep else 0
    assert _memory_addresses(case.litmus, read_value=128) == [
        [0x1004, 0x1000],
        [0x1000, 0x1004 + dependency_offset],
    ]


def test_offset_amos_do_not_exhaust_registers_in_long_dependency_chain() -> None:
    labels = ("Rfe",) + ("DpAddrsR",) * 7 + ("Fre", "PosWW")
    cycle = NativeCycle(
        tuple(edge_by_label(label) for label in labels), "Long", ("AMO",) * 10
    )
    layout = MemoryLayoutConfig(
        "atomic_mixed",
        overlap="partial_overlap",
        width_pattern=(8,) + (4,) * 8 + (8,),
    )
    case = lower_native_cycle(cycle, memory_layout=layout)
    assert _memory_addresses(case.litmus) == [
        [0x1000, 0x1000],
        [0x1004] * 8,
    ]
