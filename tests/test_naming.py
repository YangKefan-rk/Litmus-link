from __future__ import annotations

import re

from litmus_link.litmus_ir import build_litmus_ir_cases
from litmus_link.models import Combination
from litmus_link.naming import vector_native_case_identity
from litmus_link.native_scalar import lower_native_cycle, native_template_cycles
from litmus_link.rules import evaluate


def _case_names(combination: Combination) -> list[str]:
    return [case.name for case in build_litmus_ir_cases(combination, evaluate(combination))]


def test_scalar_names_follow_family_plus_relation_modifier_style() -> None:
    combination = Combination(
        "test",
        "rvwmo_base",
        "MP",
        "scalar_pair",
        "cacheable",
    )
    assert _case_names(combination) == [
        "MP",
        "MP+fence.rw.rw",
        "MP+fence.w.w+fence.r.rw",
        "MP+addr",
        "MP+ctrl",
        "MP+ctrlfencei",
    ]
    assert all("cacheable" not in name for name in _case_names(combination))
    assert all("no_tlb" not in name and "no_cmo" not in name for name in _case_names(combination))


def test_vector_names_expose_instruction_endpoint_and_vector_semantics() -> None:
    combination = Combination(
        "test",
        "vector_mem",
        "MP",
        "vector_load",
        "cacheable",
        vector="indexed_ordered_load",
        params={
            "sew": "e64",
            "lmul": "m2",
            "index_eew": "ei32",
            "mask": "masked",
            "tail": "tu_mu",
            "vl": "vl8",
            "footprint": "same_line",
            "vector_event": "p1_rx",
        },
    )
    cases = build_litmus_ir_cases(combination, evaluate(combination))
    assert [case.name for case in cases] == [
        "MP+vloxei32.v-E64-P1.Rx+M2+VL8+TU.MU+Mask",
        "MP+vloxei32.v-E64-P1.Rx+M2+VL8+TU.MU+Mask+fence.rw.rw",
        "MP+vloxei32.v-E64-P1.Rx+M2+VL8+TU.MU+Mask+fence.w.w+fence.r.rw",
    ]
    assert all(case.display_name == case.name for case in cases)


def test_native_named_family_uses_diy_relation_vocabulary() -> None:
    cycles, _audit = native_template_cycles(["MP"], ["po"], include_same=False)
    assert lower_native_cycle(cycles[0]).name == "MP+po.WW+po.RR"


def test_vector_native_name_separates_relation_map_and_file_identity() -> None:
    choices = [
        {
            "choice_id": "vector:unit_store:e16",
            "category": "vector",
            "direction": "W",
            "annotation": "P",
            "vector_form": "unit_store",
            "params": {"sew": "e16", "lmul": "m1"},
        },
        {
            "choice_id": "scalar:P:R",
            "category": "scalar",
            "direction": "R",
            "annotation": "P",
            "vector_form": None,
            "params": {},
        },
        {
            "choice_id": "scalar:P:R2",
            "category": "scalar",
            "direction": "R",
            "annotation": "P",
            "vector_form": None,
            "params": {},
        },
        {
            "choice_id": "vector:indexed_ordered_load:e16:ei32",
            "category": "vector",
            "direction": "R",
            "annotation": "P",
            "vector_form": "indexed_ordered_load",
            "params": {"sew": "e16", "index_eew": "ei32", "lmul": "m1"},
        },
    ]
    identity = vector_native_case_identity(
        "MP",
        {"family": "MP", "labels": ["PodWW", "Rfe", "PodRR", "Fre"]},
        ["PodWW", "Rfe", "PodRR", "Fre"],
        choices,
        "aligned",
    )
    repeated = vector_native_case_identity(
        "MP",
        {"family": "MP", "labels": ["PodWW", "Rfe", "PodRR", "Fre"]},
        ["PodWW", "Rfe", "PodRR", "Fre"],
        choices,
        "aligned",
    )

    assert identity["display_name"] == (
        "MP+{PodWW>Rfe>PodRR>Fre}+E{E0:VSE16,E1:SR32,E2:SR32,"
        "E3:VLOXEI32/E16}+O{same_start}"
    )
    assert re.fullmatch(r"LLV-MP-[0-9a-f]{64}", identity["machine_name"])
    assert identity["file_name"] == identity["machine_name"] + ".litmus"
    assert repeated == identity


def test_segment_nfields_is_visible_and_part_of_file_identity() -> None:
    def identity(nf: str):
        return vector_native_case_identity(
            "R",
            {"family": "R", "labels": ["Rfe", "PodRR", "Fre", "PodWW"]},
            ["Rfe", "PodRR", "Fre", "PodWW"],
            [
                {
                    "choice_id": f"vector:segment_unit_load:e32:m1:{nf}",
                    "category": "vector",
                    "direction": "R",
                    "annotation": "P",
                    "vector_form": "segment_unit_load",
                    "params": {"sew": "e32", "lmul": "m1", "nf": nf},
                }
            ],
            "aligned",
        )

    nf2 = identity("nf2")
    nf3 = identity("nf3")

    assert "E0:VLSEG2E32" in nf2["display_name"]
    assert "E0:VLSEG3E32" in nf3["display_name"]
    assert nf2["machine_name"] != nf3["machine_name"]


def test_whole_register_nreg_is_visible_and_part_of_file_identity() -> None:
    def identity(nreg: str):
        return vector_native_case_identity(
            "MP",
            {"family": "MP", "labels": ["PodWW", "Rfe", "PodRR", "Fre"]},
            ["PodWW", "Rfe", "PodRR", "Fre"],
            [
                {
                    "choice_id": f"vector:whole_register_load:e64:{nreg}",
                    "category": "vector",
                    "direction": "R",
                    "annotation": "P",
                    "vector_form": "whole_register_load",
                    "params": {"sew": "e64", "whole_nreg": nreg},
                }
            ],
            "aligned",
        )

    nreg1 = identity("nreg1")
    nreg8 = identity("nreg8")
    assert "E0:VL1RE64" in nreg1["display_name"]
    assert "E0:VL8RE64" in nreg8["display_name"]
    assert nreg1["machine_name"] != nreg8["machine_name"]
