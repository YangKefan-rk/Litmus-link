from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import GeneratedCase


@dataclass(frozen=True)
class SolverResult:
    status: str
    verdict: str
    allowed: bool | None
    model: str
    tool: str
    reason: str
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
            "observation": self.observation,
            "raw_output": self.raw_output,
            "command": list(self.command or []),
        }


def solve_generated_case(case: GeneratedCase, herd: str = "herd7") -> SolverResult:
    case_ir = case.case_ir
    if case_ir is None or case_ir.model != "rvwmo" or case.decision.expected_kind != "rvwmo-herd":
        return SolverResult(
            status="not_applicable",
            verdict="unmodeled",
            allowed=None,
            model=case_ir.model if case_ir is not None else case.decision.rvwmo_class,
            tool="none",
            reason="Only pure scalar RVWMO main-memory cases are sent to herd7.",
        )

    herd_path = shutil.which(herd)
    if herd_path is None:
        return SolverResult(
            status="solver_unavailable",
            verdict="unchecked",
            allowed=None,
            model="riscv.cat",
            tool=herd,
            reason="herd7 was not found on PATH; no allowed/forbidden claim was made.",
        )

    with tempfile.TemporaryDirectory(prefix="litmus-link-herd-") as tmp:
        litmus_path = Path(tmp) / f"{case.name}.litmus"
        litmus_path.write_text(case.litmus, encoding="utf-8")
        command = [herd_path, "-model", "riscv.cat", str(litmus_path)]
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    raw = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return SolverResult(
            status="solver_error",
            verdict="unchecked",
            allowed=None,
            model="riscv.cat",
            tool=herd_path,
            reason=f"herd7 exited with status {result.returncode}.",
            raw_output=raw,
            command=command,
        )
    parsed = parse_herd_output(raw)
    return SolverResult(
        status="verified",
        verdict=parsed["verdict"],
        allowed=parsed["allowed"],
        model="riscv.cat",
        tool=herd_path,
        reason="herd7 completed successfully; verdict parsed from its output.",
        observation=parsed.get("observation", ""),
        raw_output=raw,
        command=command,
    )


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
