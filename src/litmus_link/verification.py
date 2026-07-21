from __future__ import annotations

"""Independent verification entry points for generated or upstream litmus."""

import json
from collections import Counter
from pathlib import Path

from .litmus_ir import LitmusCaseIR, LitmusEvent, LitmusRelation
from .rvwmo_solver import solve_rvwmo
from .toolchain import ToolchainError, herd_judge
from .vector_solver import is_vector_case, solve_vector_case


class VerificationError(ValueError):
    pass


VERIFY_BACKENDS = ("embedded", "herd7", "crosscheck")


def verify_path(
    path: Path,
    *,
    backend: str = "embedded",
    timeout: int = 120,
    max_candidates: int = 100_000,
    write: bool = False,
) -> dict:
    selected = _backend(backend)
    files = _litmus_files(path)
    results = []
    counts: Counter[str] = Counter()
    for litmus_path in files:
        result = verify_litmus_file(
            litmus_path,
            backend=selected,
            timeout=timeout,
            max_candidates=max_candidates,
        )
        counts[result["status"]] += 1
        results.append({"file": str(litmus_path), "result": result})
        if write:
            _write_result(litmus_path, result)
    return {
        "schema": "litmus-link.verification-report.v1",
        "backend": selected,
        "path": str(path),
        "tests": len(files),
        "counts": dict(sorted(counts.items())),
        "results": results,
        "written": write,
    }


def verify_litmus_file(
    litmus_path: Path,
    *,
    backend: str = "embedded",
    timeout: int = 120,
    max_candidates: int = 100_000,
) -> dict:
    selected = _backend(backend)
    if not litmus_path.exists() or litmus_path.suffix != ".litmus":
        raise VerificationError(f"not a .litmus file: {litmus_path}")
    embedded = None
    external = None
    case_ir = _load_case_ir(litmus_path)
    vector_case = case_ir is not None and is_vector_case(case_ir)
    if selected in {"embedded", "crosscheck"}:
        if case_ir is None:
            embedded = {
                "schema": "litmus-link.embedded-rvwmo.v1",
                "status": "not_applicable",
                "verdict": "unmodeled",
                "allowed": None,
                "backend": "embedded",
                "reason": "Embedded verification requires Litmus-link case_ir metadata.",
            }
        else:
            embedded = (
                solve_vector_case(
                    case_ir,
                    max_candidates=max_candidates,
                    timeout_seconds=float(timeout),
                ).to_json()
                if vector_case
                else solve_rvwmo(
                    case_ir,
                    max_candidates=max_candidates,
                    timeout_seconds=float(timeout),
                ).to_json()
            )
    if selected in {"herd7", "crosscheck"}:
        external = (
            {
                "schema": "litmus-link.herd7-verification.v1",
                "status": "not_applicable",
                "verdict": "unmodeled",
                "allowed": None,
                "backend": "herd7",
                "model": "riscv.cat",
                "reason": "Stock herd7/riscv.cat does not parse or model RVV memory instructions.",
            }
            if vector_case
            else _herd_result(litmus_path, timeout=timeout)
        )
    if selected == "embedded":
        return embedded or _missing_result("embedded")
    if selected == "herd7":
        return external or _missing_result("herd7")
    assert embedded is not None and external is not None
    if vector_case:
        return {
            "schema": "litmus-link.verification.v1",
            "status": embedded.get("status", "not_applicable"),
            "verdict": embedded.get("verdict", "unmodeled"),
            "allowed": embedded.get("allowed"),
            "backend": "vector-aware-embedded",
            "reason": "Vector-aware embedded result returned; no stock herd7 RVV model exists for cross-check.",
            "cross_check": "no_external_vector_model",
            "embedded": embedded,
            "herd7": external,
        }
    agree = (
        embedded.get("status") == "verified"
        and external.get("status") == "verified"
        and embedded.get("allowed") == external.get("allowed")
    )
    if agree:
        return {
            "schema": "litmus-link.verification.v1",
            "status": "verified",
            "verdict": embedded["verdict"],
            "allowed": embedded["allowed"],
            "backend": "crosscheck",
            "reason": "Embedded RVWMO and herd7 agree.",
            "embedded": embedded,
            "herd7": external,
        }
    return {
        "schema": "litmus-link.verification.v1",
        "status": "conflict",
        "verdict": "conflict",
        "allowed": None,
        "backend": "crosscheck",
        "reason": "Both backends must return the same verified allowed value.",
        "embedded": embedded,
        "herd7": external,
    }


def _herd_result(path: Path, *, timeout: int) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        verdict = herd_judge(text, timeout=timeout)
    except ToolchainError as exc:
        return {
            "schema": "litmus-link.herd7-verification.v1",
            "status": "unavailable",
            "verdict": "unknown",
            "allowed": None,
            "backend": "herd7",
            "model": "riscv.cat",
            "reason": str(exc),
        }
    return {
        "schema": "litmus-link.herd7-verification.v1",
        "status": "verified" if verdict.outcome in {"observable", "forbidden"} else "unknown",
        "verdict": verdict.outcome,
        "allowed": verdict.allowed,
        "backend": "herd7",
        "model": "riscv.cat",
        "reason": f"herd7 observation: {verdict.observation or verdict.outcome}",
        "observation": verdict.observation,
        "positive": verdict.positive,
        "negative": verdict.negative,
        "states": verdict.states,
        "condition": verdict.condition,
        "raw_output": verdict.raw,
    }


def _litmus_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".litmus":
        return [path]
    atfile = path / "@all" if path.is_dir() else path
    if not atfile.exists():
        raise VerificationError(f"missing path: {path}")
    base = atfile.parent
    files = [
        base / line.strip()
        for line in atfile.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [file for file in files if not file.exists()]
    if missing:
        raise VerificationError(f"@all references missing file: {missing[0]}")
    return files


def _load_case_ir(litmus_path: Path) -> LitmusCaseIR | None:
    meta_path = litmus_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8")).get("case_ir")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return LitmusCaseIR.from_json(data)
    except (KeyError, TypeError, ValueError):
        return None


def _write_result(litmus_path: Path, result: dict) -> None:
    solver_path = litmus_path.with_suffix(".solver.json")
    solver_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta_path = litmus_path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["solver"] = result
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _backend(value: str) -> str:
    selected = str(value).lower()
    if selected not in VERIFY_BACKENDS:
        raise VerificationError(f"unknown verification backend: {value}")
    return selected


def _missing_result(backend: str) -> dict:
    return {
        "schema": "litmus-link.verification.v1",
        "status": "unavailable",
        "verdict": "unknown",
        "allowed": None,
        "backend": backend,
        "reason": f"{backend} result unavailable",
    }
