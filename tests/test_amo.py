from __future__ import annotations

import pytest

from litmus_link.amo import (
    AMO_OPERATIONS,
    AMO_ORDERINGS,
    AmoError,
    AmoSpec,
    amo_read_result,
    apply_amo,
    parse_amo_mnemonic,
)


@pytest.mark.parametrize("operation", AMO_OPERATIONS)
@pytest.mark.parametrize("width_bytes", [4, 8])
def test_all_nanhu_amo_operations_are_fixed_width(
    operation: str, width_bytes: int
) -> None:
    mask = (1 << (width_bytes * 8)) - 1
    result = apply_amo(operation, width_bytes, mask - 1, 3)
    assert 0 <= result <= mask


@pytest.mark.parametrize(
    ("operation", "old", "operand", "expected"),
    [
        ("swap", 0x10, 0x03, 0x03),
        ("add", 0xFFFFFFFF, 0x02, 0x01),
        ("xor", 0xF0, 0x33, 0xC3),
        ("and", 0xF0, 0x33, 0x30),
        ("or", 0xF0, 0x33, 0xF3),
        ("min", 0xFFFFFFFF, 0x01, 0xFFFFFFFF),
        ("max", 0xFFFFFFFF, 0x01, 0x01),
        ("minu", 0xFFFFFFFF, 0x01, 0x01),
        ("maxu", 0xFFFFFFFF, 0x01, 0xFFFFFFFF),
    ],
)
def test_amo_operation_semantics(
    operation: str, old: int, operand: int, expected: int
) -> None:
    assert apply_amo(operation, 4, old, operand) == expected


@pytest.mark.parametrize("ordering", AMO_ORDERINGS)
def test_amo_mnemonic_round_trip(ordering: str) -> None:
    spec = AmoSpec("maxu", 8, ordering)
    assert parse_amo_mnemonic(spec.mnemonic) == spec


@pytest.mark.parametrize(
    ("ordering", "mnemonic"),
    [
        ("relaxed", "amoadd.w"),
        ("aq", "amoadd.w.aq"),
        ("rl", "amoadd.w.rl"),
        ("aqrl", "amoadd.w.aqrl"),
    ],
)
def test_amo_ordering_uses_isa_mnemonic_suffix(
    ordering: str, mnemonic: str
) -> None:
    assert AmoSpec("add", 4, ordering).mnemonic == mnemonic


def test_amo_parser_rejects_non_isa_split_aq_rl_suffix() -> None:
    with pytest.raises(AmoError, match="unsupported Nanhu AMO"):
        parse_amo_mnemonic("amoadd.w.aq.rl x1,x2,(x3)")


def test_amo_w_read_result_is_sign_extended_on_rv64() -> None:
    assert amo_read_result(4, 0x80000000) == 0xFFFFFFFF80000000
    assert amo_read_result(4, 0x7FFFFFFF) == 0x7FFFFFFF
    assert amo_read_result(8, 0xFFFFFFFFFFFFFFFF) == 0xFFFFFFFFFFFFFFFF


@pytest.mark.parametrize("instruction", ["amoadd.b x1,x2,(x3)", "amoadd.h x1,x2,(x3)"])
def test_nanhu_rejects_zabha_amo_widths(instruction: str) -> None:
    with pytest.raises(AmoError, match="unsupported Nanhu AMO"):
        parse_amo_mnemonic(instruction)
