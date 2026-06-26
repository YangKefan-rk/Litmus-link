from __future__ import annotations

import json
from pathlib import Path
from typing import List

from models import Combination, GENERATED
from rules import evaluate


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
    combination = Combination.from_json(meta["combination"])
    decision = evaluate(combination)
    header = litmus_path.read_text(encoding="utf-8").splitlines()[0].strip()
    expected_header = f"RISCV {meta.get('name', combination.name)}"
    if header != expected_header:
        errors.append(f"{litmus_path}: expected header {expected_header!r}, got {header!r}")
    if not str(meta.get("name", "")).startswith(combination.name):
        errors.append(f"{meta_path}: metadata name does not derive from combination")
    if meta.get("legality_status") != GENERATED:
        errors.append(f"{meta_path}: generated corpus contains non-generated status")
    if decision.status != GENERATED:
        errors.append(f"{meta_path}: rule engine now classifies as {decision.status}: {decision.reason}")
    for key in ["axes", "requires", "rvwmo_class", "expected_kind", "generated_from"]:
        if key not in meta:
            errors.append(f"{meta_path}: missing key {key}")
    return errors
