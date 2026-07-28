from __future__ import annotations

"""Independent verification entry points for generated or upstream litmus."""

import json
import re
from collections import Counter
from pathlib import Path

from .litmus_ir import LitmusCaseIR
from .renderer import render_ir
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
    if not litmus_path.is_file() or litmus_path.suffix != ".litmus":
        raise VerificationError(f"not a .litmus file: {litmus_path}")
    source = litmus_path.read_text(encoding="utf-8")
    case_ir = _load_case_ir(litmus_path)
    if case_ir is not None:
        if _litmus_body(source) != _litmus_body(render_ir(case_ir)):
            raise VerificationError(f"{litmus_path}: litmus body differs from case_ir metadata")
        if case_ir.variant == "observation" or case_ir.expected_outcome == "manual_oracle_required":
            return {
                "schema": "litmus-link.verification.v1",
                "status": "not_applicable",
                "verdict": "unmodeled",
                "allowed": None,
                "backend": selected,
                "reason": case_ir.metadata.get("formal_scope")
                or "This generated case requires a hardware observation or manual oracle.",
            }
    embedded = None
    vector_case = case_ir is not None and is_vector_case(case_ir)
    vector_result = None
    if vector_case:
        vector_result = solve_vector_case(
            case_ir,
            max_candidates=max_candidates,
            timeout_seconds=float(timeout),
            external_check=selected in {"herd7", "crosscheck"},
            external_timeout=timeout,
        )
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
                vector_result.to_json()
                if vector_result is not None
                else solve_rvwmo(
                    case_ir,
                    max_candidates=max_candidates,
                    timeout_seconds=float(timeout),
                ).to_json()
            )
    if selected == "embedded":
        return embedded or _missing_result("embedded")
    if vector_result is not None:
        if vector_result.external is None:
            external = {
                "schema": "litmus-link.vector-herd-reference.v1",
                "status": vector_result.status,
                "verdict": "unknown",
                "allowed": None,
                "reason": f"Scalar projection was skipped. {vector_result.reason}",
            }
        else:
            external = dict(vector_result.external)
    else:
        mixed = case_ir is not None and any(
            event.memory_access is not None
            and event.memory_access.atomicity_model == "byte_level_no_mag"
            for event in case_ir.events()
        )
        external = _herd_result(source, timeout=timeout, variants=("mixed", "unaligned") if mixed else ())
    if selected == "herd7":
        return _vector_external_result(external) if vector_case else external
    assert embedded is not None
    if vector_case:
        external_status = str(external.get("status", "external_unsupported"))
        return {
            "schema": "litmus-link.verification.v1",
            "status": embedded.get("status", "not_applicable"),
            "verdict": embedded.get("verdict", "unmodeled"),
            "allowed": embedded.get("allowed"),
            "backend": "vector-aware-crosscheck",
            "reason": embedded.get("reason", "Vector-aware RVWMO result."),
            "cross_check": external_status,
            "embedded": embedded,
            "herd7": external,
        }
    if embedded.get("status") != "verified":
        return {
            "schema": "litmus-link.verification.v1",
            "status": embedded.get("status", "inconclusive"),
            "verdict": embedded.get("verdict", "unknown"),
            "allowed": embedded.get("allowed"),
            "backend": "crosscheck",
            "cross_check": "embedded_inconclusive",
            "reason": "Embedded RVWMO did not produce a verified verdict; no conflict is claimed.",
            "embedded": embedded,
            "herd7": external,
        }
    if external.get("status") != "verified":
        return {
            "schema": "litmus-link.verification.v1",
            "status": "verified",
            "verdict": embedded["verdict"],
            "allowed": embedded["allowed"],
            "backend": "crosscheck",
            "cross_check": f"external_{external.get('status', 'unknown')}",
            "reason": "Embedded RVWMO result retained because the external reference is unavailable.",
            "embedded": embedded,
            "herd7": external,
        }
    agree = embedded.get("allowed") == external.get("allowed")
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


def _herd_result(text: str, *, timeout: int, variants: tuple[str, ...]) -> dict:
    try:
        verdict = herd_judge(text, timeout=timeout, variants=variants)
    except (OSError, ToolchainError) as exc:
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


def _vector_external_result(external: dict) -> dict:
    if external.get("allowed") is None:
        return external
    if not external["projection"]["exact"]:
        return {**external, "verdict": "unknown", "allowed": None}
    return {
        **external,
        "status": "verified",
        "backend": "herd7-scalar-projection",
        "cross_check": external["status"],
    }


def _litmus_body(source: str) -> str:
    # Both renderers emit a standalone opening brace. Compare the body while
    # ignoring comments and table padding.
    text = re.sub(r"\(\*.*?\*\)", "", source, flags=re.DOTALL)
    start = re.search(r"^[ \t]*\{[ \t]*$", text, flags=re.MULTILINE)
    if start is None or not text.lstrip().startswith("RISCV "):
        return ""
    return " ".join(text[start.start():].split())


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
    for file in files:
        if not file.is_file():
            raise VerificationError(f"@all references missing file: {file}")
        if file.suffix != ".litmus":
            raise VerificationError(f"@all entry is not a .litmus file: {file}")
    return files


def _load_case_ir(litmus_path: Path) -> LitmusCaseIR | None:
    meta_path = litmus_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{meta_path}: invalid metadata JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise VerificationError(f"{meta_path}: expected a JSON object")
    data = meta.get("case_ir")
    if data is None:
        return None
    if not isinstance(data, dict):
        raise VerificationError(f"{meta_path}: case_ir must be a JSON object")
    try:
        return LitmusCaseIR.from_json(data)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"{meta_path}: invalid case_ir: {exc}") from exc


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
