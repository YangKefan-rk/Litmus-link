from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models import GeneratedCase
from rvwmo import check_rvwmo
from fusion import analyze_fusion


@dataclass(frozen=True)
class SolverResult:
    status: str
    verdict: str
    allowed: bool | None
    model: str
    tool: str
    reason: str
    cross_check: str = "native_only"
    edges: list[dict[str, Any]] = field(default_factory=list)
    fusion: dict[str, Any] | None = None
    observation: str = ""
    raw_output: str = ""
    command: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "litmus-link.solver.v1",
            "status": self.status,
            "verdict": self.verdict,
            "allowed": self.allowed,
            "model": self.model,
            "tool": self.tool,
            "reason": self.reason,
            "cross_check": self.cross_check,
            "edges": list(self.edges),
            "fusion": self.fusion,
            "observation": self.observation,
            "raw_output": self.raw_output,
            "command": list(self.command or []),
        }


def solve_generated_case(case: GeneratedCase, herd: str = "herd7") -> SolverResult:
    case_ir = case.case_ir
    if case_ir is None or case_ir.model != "rvwmo" or case.decision.expected_kind != "rvwmo-herd":
        # Not a pure scalar RVWMO case: no formal forbidden/allowed verdict is
        # made. For fusion (vector/CMO/PBMT/TLB) cases we still attach the
        # extension-prose ordering analysis so consumers get something better
        # than a bare "not modeled" -- but it never carries a herd verdict.
        fusion = analyze_fusion(case_ir).to_json() if case_ir is not None else None
        reason = "Only pure scalar RVWMO main-memory cases receive a formal verdict."
        if fusion is not None and fusion.get("status") == "analyzed":
            reason = "Extension-prose fusion analysis (informative, no herd verdict): " + fusion["reason"]
        return SolverResult(
            status="not_applicable",
            verdict="unmodeled",
            allowed=None,
            model=case_ir.model if case_ir is not None else case.decision.rvwmo_class,
            tool="none",
            reason=reason,
            fusion=fusion,
        )

    # Primary path: the native axiomatic RVWMO checker. It always renders a
    # verdict for scalar cycle cases and needs no external tool.
    native = check_rvwmo(case_ir)
    edges = [edge.to_json() for edge in native.edges]

    # Optional cross-validation against herd7 if it happens to be installed.
    herd_check = _run_herd(case, herd)
    if herd_check is None:
        return SolverResult(
            status="verified",
            verdict=native.verdict,
            allowed=native.allowed,
            model="rvwmo-native",
            tool="native",
            reason=native.reason + " (herd7 not on PATH; no cross-check performed.)",
            cross_check="herd7_absent",
            edges=edges,
        )

    herd_status, herd_parsed, raw, command = herd_check
    if herd_status != "ok":
        return SolverResult(
            status="verified",
            verdict=native.verdict,
            allowed=native.allowed,
            model="rvwmo-native",
            tool="native",
            reason=native.reason + f" (herd7 cross-check unavailable: {herd_status}.)",
            cross_check=f"herd7_{herd_status}",
            edges=edges,
            raw_output=raw,
            command=command,
        )

    if herd_parsed["allowed"] == native.allowed:
        return SolverResult(
            status="verified",
            verdict=native.verdict,
            allowed=native.allowed,
            model="rvwmo-native+riscv.cat",
            tool="native+herd7",
            reason=native.reason + " Confirmed by herd7/riscv.cat.",
            cross_check="agree",
            edges=edges,
            observation=herd_parsed.get("observation", ""),
            raw_output=raw,
            command=command,
        )

    return SolverResult(
        status="conflict",
        verdict=native.verdict,
        allowed=native.allowed,
        model="rvwmo-native+riscv.cat",
        tool="native+herd7",
        reason=(
            f"Native checker says {native.verdict} but herd7 says {herd_parsed['verdict']}. "
            "Native verdict is reported as primary; investigate the disagreement."
        ),
        cross_check="conflict",
        edges=edges,
        observation=herd_parsed.get("observation", ""),
        raw_output=raw,
        command=command,
    )


def _run_herd(case: GeneratedCase, herd: str) -> tuple[str, dict[str, Any], str, list[str]] | None:
    """Run herd7 if available. Returns None when herd7 is not on PATH.

    Otherwise returns (status, parsed, raw_output, command) where status is
    "ok", "error", or "unparsed".
    """
    herd_path = shutil.which(herd)
    if herd_path is None:
        return None
    with tempfile.TemporaryDirectory(prefix="litmus-link-herd-") as tmp:
        litmus_path = Path(tmp) / f"{case.name}.litmus"
        litmus_path.write_text(case.litmus, encoding="utf-8")
        command = [herd_path, "-model", "riscv.cat", str(litmus_path)]
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return "error", {}, str(exc), command
    raw = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return "error", {}, raw, command
    parsed = parse_herd_output(raw)
    if parsed["allowed"] is None:
        return "unparsed", parsed, raw, command
    return "ok", parsed, raw, command


def parse_herd_output(output: str) -> dict[str, Any]:
    observation = _last_match(output, r"^Observation\s+[^\s]+\s+(.+)$")
    if observation:
        normalized = observation.strip().lower()
        if normalized.startswith("never"):
            return {"verdict": "forbidden", "allowed": False, "observation": observation.strip()}
        if normalized.startswith("sometimes") or normalized.startswith("always"):
            return {"verdict": "allowed", "allowed": True, "observation": observation.strip()}

    condition = _last_match(output, r"^Condition\s+(.+)$")
    if condition:
        lowered = condition.lower()
        if "is forbidden" in lowered or "forbidden" == lowered.strip():
            return {"verdict": "forbidden", "allowed": False, "observation": condition.strip()}
        if "is allowed" in lowered or "allowed" == lowered.strip():
            return {"verdict": "allowed", "allowed": True, "observation": condition.strip()}

    if re.search(r"\bNever\b", output):
        return {"verdict": "forbidden", "allowed": False, "observation": "Never"}
    if re.search(r"\bSometimes\b|\bAlways\b", output):
        return {"verdict": "allowed", "allowed": True, "observation": "Sometimes/Always"}
    return {"verdict": "unknown", "allowed": None, "observation": "unparsed"}


def _last_match(text: str, pattern: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else ""
