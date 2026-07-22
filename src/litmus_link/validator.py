from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List

from .corpus_riscv import parse_litmus
from .models import Combination, GENERATED
from .rules import evaluate


class ValidationError(Exception):
    pass


def validate_path(path: Path) -> List[str]:
    atfile = _resolve_atfile(path)
    base = atfile.parent
    errors: List[str] = []
    for entry in _read_atfile(atfile):
        litmus_path = base / entry
        if not litmus_path.exists():
            errors.append(f"missing litmus file: {litmus_path}")
            continue
        if litmus_path.suffix != ".litmus":
            errors.append(f"@all entry is not a .litmus file: {entry}")
            continue
        meta_path = litmus_path.with_suffix(".meta.json")
        if not meta_path.exists():
            errors.append(f"missing metadata file: {meta_path}")
            continue
        errors.extend(_validate_pair(litmus_path, meta_path))
    if errors:
        raise ValidationError("\n".join(errors))
    return _read_atfile(atfile)


def _resolve_atfile(path: Path) -> Path:
    if path.is_dir():
        return path / "@all"
    return path


def _read_atfile(path: Path) -> List[str]:
    if not path.exists():
        raise ValidationError(f"missing @all file: {path}")
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(stripped)
    return entries


def _validate_pair(litmus_path: Path, meta_path: Path) -> List[str]:
    errors: List[str] = []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") == "litmus-link.scalar-meta.v1":
        return _validate_scalar_pair(litmus_path, meta_path, meta)
    if meta.get("schema") == "litmus-link.native-scalar-meta.v1":
        return _validate_native_scalar_pair(litmus_path, meta_path, meta)
    combination = Combination.from_json(meta["combination"])
    decision = evaluate(combination)
    litmus_lines = litmus_path.read_text(encoding="utf-8").splitlines()
    header = litmus_lines[0].strip() if litmus_lines else ""
    expected_header = f"RISCV {meta.get('name', combination.name)}"
    if header != expected_header:
        errors.append(f"{litmus_path}: expected header {expected_header!r}, got {header!r}")
    case_ir = meta.get("case_ir") if isinstance(meta.get("case_ir"), dict) else {}
    if case_ir and case_ir.get("combination_name") != combination.name:
        errors.append(
            f"{meta_path}: case IR combination_name does not match combination identity"
        )
    if litmus_path.stem != str(meta.get("name", "")):
        errors.append(f"{litmus_path}: filename does not match metadata name")
    if meta.get("file_name") and meta.get("file_name") != litmus_path.name:
        errors.append(f"{litmus_path}: metadata file_name does not match actual filename")
    if case_ir and case_ir.get("name") != meta.get("name"):
        errors.append(f"{meta_path}: case IR name does not match metadata name")
    if case_ir and meta.get("display_name") != case_ir.get("display_name"):
        errors.append(f"{meta_path}: display_name does not match case IR")
    if case_ir.get("variant") == "vector-native-cycle":
        errors.extend(_validate_vector_file_identity(litmus_path, meta_path, case_ir))
    if meta.get("legality_status") != GENERATED:
        errors.append(f"{meta_path}: generated corpus contains non-generated status")
    if decision.status != GENERATED:
        errors.append(f"{meta_path}: rule engine now classifies as {decision.status}: {decision.reason}")
    for key in ["axes", "requires", "rvwmo_class", "expected_kind", "generated_from"]:
        if key not in meta:
            errors.append(f"{meta_path}: missing key {key}")
    return errors


def _validate_vector_file_identity(
    litmus_path: Path,
    meta_path: Path,
    case_ir: dict,
) -> List[str]:
    errors: List[str] = []
    name = str(case_ir.get("name", ""))
    if not re.fullmatch(r"LLV-[A-Za-z0-9_.-]+-[0-9a-f]{64}", name):
        errors.append(f"{meta_path}: invalid Vector LLV file identity {name!r}")
        return errors
    metadata = case_ir.get("metadata")
    identity = metadata.get("file_identity") if isinstance(metadata, dict) else None
    canonical = identity.get("canonical") if isinstance(identity, dict) else None
    if not isinstance(canonical, dict):
        errors.append(f"{meta_path}: missing canonical Vector file identity")
        return errors
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    family = re.sub(
        r"[^A-Za-z0-9_.-]+",
        ".",
        str(canonical.get("family", "Cycle")),
    ).strip("._-") or "x"
    expected_name = f"LLV-{family}-{digest}"
    if name != expected_name:
        errors.append(
            f"{meta_path}: Vector file identity hash does not match canonical case"
        )
    if identity.get("sha256") != digest:
        errors.append(f"{meta_path}: Vector file identity sha256 is inconsistent")
    if identity.get("file_name") != litmus_path.name:
        errors.append(f"{meta_path}: Vector file identity names the wrong file")
    return errors


def _validate_scalar_pair(litmus_path: Path, meta_path: Path, meta: dict) -> List[str]:
    errors: List[str] = []
    text = litmus_path.read_text(encoding="utf-8")
    parsed = parse_litmus(text, str(litmus_path))
    name = str(meta.get("name", ""))
    if parsed.name != name:
        errors.append(f"{litmus_path}: header name {parsed.name!r} does not match metadata {name!r}")
    if litmus_path.stem != name:
        errors.append(f"{litmus_path}: filename does not match metadata name {name!r}")
    if meta.get("architecture") != "RISCV":
        errors.append(f"{meta_path}: scalar architecture must be RISCV")
    if meta.get("generated_from") != "herdtools7":
        errors.append(f"{meta_path}: scalar test must record generated_from=herdtools7")
    if meta.get("cycle", "") != parsed.cycle:
        errors.append(f"{meta_path}: Cycle metadata does not match the litmus source")
    if meta.get("exists", "") != parsed.exists:
        errors.append(f"{meta_path}: exists metadata does not match the litmus source")
    generator = meta.get("generator")
    if not isinstance(generator, dict) or generator.get("engine") not in {"diy7", "diycross7"}:
        errors.append(f"{meta_path}: missing or invalid scalar generator engine")
    if not isinstance(meta.get("requires"), list) or "RV64I" not in meta.get("requires", []):
        errors.append(f"{meta_path}: scalar test must declare RV64I")
    solver = meta.get("solver")
    if not isinstance(solver, dict) or solver.get("status") not in {
        "verified", "unknown", "unchecked", "inconclusive", "conflict", "unavailable", "not_applicable"
    }:
        errors.append(f"{meta_path}: missing or invalid scalar solver result")
    solver_path = litmus_path.with_suffix(".solver.json")
    if not solver_path.exists():
        errors.append(f"missing scalar solver file: {solver_path}")
    elif solver != json.loads(solver_path.read_text(encoding="utf-8")):
        errors.append(f"{solver_path}: solver JSON differs from metadata")
    return errors


def _validate_native_scalar_pair(litmus_path: Path, meta_path: Path, meta: dict) -> List[str]:
    errors: List[str] = []
    text = litmus_path.read_text(encoding="utf-8")
    parsed = parse_litmus(text, str(litmus_path))
    name = str(meta.get("name", ""))
    if parsed.name != name or litmus_path.stem != name:
        errors.append(f"{litmus_path}: native filename/header/metadata names do not match")
    if meta.get("architecture") != "RISCV":
        errors.append(f"{meta_path}: native scalar architecture must be RISCV")
    if meta.get("generated_from") != "litmus-link-native":
        errors.append(f"{meta_path}: native test must record generated_from=litmus-link-native")
    if meta.get("cycle", "") != parsed.cycle:
        errors.append(f"{meta_path}: native Cycle metadata does not match source")
    if meta.get("exists", "") != parsed.exists:
        errors.append(f"{meta_path}: native exists metadata does not match source")
    generator = meta.get("generator")
    if not isinstance(generator, dict) or generator.get("engine") != "litmus-link-native":
        errors.append(f"{meta_path}: missing native generator identity")
    case_ir = meta.get("case_ir")
    if not isinstance(case_ir, dict) or case_ir.get("cycle") != parsed.cycle:
        errors.append(f"{meta_path}: missing or inconsistent native case IR")
    solver = meta.get("solver")
    if not isinstance(solver, dict) or solver.get("status") not in {
        "verified", "unknown", "unchecked", "inconclusive", "conflict", "unavailable", "not_applicable"
    }:
        errors.append(f"{meta_path}: missing or invalid native solver result")
    solver_path = litmus_path.with_suffix(".solver.json")
    if not solver_path.exists():
        errors.append(f"missing native solver file: {solver_path}")
    elif solver != json.loads(solver_path.read_text(encoding="utf-8")):
        errors.append(f"{solver_path}: native solver JSON differs from metadata")
    return errors
