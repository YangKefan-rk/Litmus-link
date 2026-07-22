from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import GeneratedCase
from .rvwmo import check_rvwmo
from .fusion import analyze_fusion


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
    vector: dict[str, Any] | None = None

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
            "vector": self.vector,
        }


def solve_generated_case(
    case: GeneratedCase,
    herd: str = "herd7",
    *,
    vector_external_check: bool = False,
) -> SolverResult:
    case_ir = case.case_ir
    if case.decision.expected_kind == "rvwmo-vector":
        return _solve_vector_generated_case(
            case, external_check=vector_external_check
        )
    if case_ir is None or case_ir.model != "rvwmo" or case.decision.expected_kind not in {"rvwmo-herd", "rvwmo-nc"}:
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

    # NC scalar tests are RVWMO-decidable too (Svpbmt: NC is non-cacheable main
    # memory, RVWMO-ordered) -- same verdict as the cacheable twin, plus one
    # Svpbmt prose dependency we record here.
    reason_base = native.reason
    if case.decision.expected_kind == "rvwmo-nc":
        reason_base = (
            native.reason
            + " (PBMT=NC is non-cacheable main memory and obeys RVWMO per Svpbmt, so this"
            + " verdict equals the cacheable twin; the plain body lets herd7 cross-check it.)"
        )

    herd_check = _run_herd(case, herd)
    if herd_check is None:
        return SolverResult(
            status="verified",
            verdict=native.verdict,
            allowed=native.allowed,
            model="rvwmo-native",
            tool="native",
            reason=reason_base + " (herd7 not on PATH; no cross-check performed.)",
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
            reason=reason_base + f" (herd7 cross-check unavailable: {herd_status}.)",
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
            reason=reason_base + " Confirmed by herd7/riscv.cat.",
            cross_check="agree",
            edges=edges,
            observation=herd_parsed.get("observation", ""),
            raw_output=raw,
            command=command,
        )

    # Native and herd7 disagree. herd7 + riscv.cat is the OFFICIAL RISC-V memory
    # model; the native checker is a hand-rolled approximation that mis-handles
    # topologies whose forbiddenness is not captured by "all po edges preserved".
    # herd7 judges
    # the actual rendered body, so it is authoritative for the verdict; native's
    # per-edge reasoning is kept for explanation and the disagreement is surfaced
    # via cross_check rather than silently picking the (possibly wrong) native one.
    return SolverResult(
        status="verified",
        verdict=herd_parsed["verdict"],
        allowed=herd_parsed["allowed"],
        model="riscv.cat+native",
        tool="herd7(authoritative)+native",
        reason=(
            f"herd7/riscv.cat (authoritative RISC-V model) says {herd_parsed['verdict']}; "
            f"the native approximation said {native.verdict} and is overridden. "
            "Native per-edge reasoning is retained for explanation only."
        ),
        cross_check="native_disagrees",
        edges=edges,
        observation=herd_parsed.get("observation", ""),
        raw_output=raw,
        command=command,
    )


def _solve_vector_generated_case(
    case: GeneratedCase, *, external_check: bool = False
) -> SolverResult:
    from .vector_solver import solve_vector_case

    if case.case_ir is None:
        return SolverResult(
            status="not_applicable",
            verdict="unmodeled",
            allowed=None,
            model="riscv.cat+rvv-elements",
            tool="litmus-link-vector-rvwmo",
            reason="Vector-aware verification requires case_ir metadata.",
            cross_check="not_applicable",
        )

    result = solve_vector_case(
        case.case_ir,
        external_check=external_check,
    )
    payload = result.to_json()
    edges: list[dict[str, Any]] = []
    execution = result.embedded.execution if result.embedded is not None else None
    if execution is not None:
        for rule, pairs in sorted(execution.ppo_rules.items()):
            for source, target in sorted(pairs):
                edges.append(
                    {
                        "src": source,
                        "dst": target,
                        "kind": "ppo",
                        "label": rule,
                        "preserved": True,
                        "reason": "RVWMO/vector-aware preserved program order",
                    }
                )
    verdict = "allowed" if result.verdict == "observable" else result.verdict
    external_status = (
        str(result.external.get("status", "not_run"))
        if result.external is not None
        else "not_run"
    )
    cross_check = (
        "external_unsupported"
        if external_status == "external_unsupported"
        else external_status
    )
    return SolverResult(
        status=result.status,
        verdict=verdict,
        allowed=result.allowed,
        model=(
            "riscv.cat+rvv-elements+herd-scalar-projection"
            if external_status in {"agree", "conflict"}
            else "riscv.cat+rvv-elements"
        ),
        tool=(
            "litmus-link-vector-rvwmo+herd7"
            if external_status in {"agree", "conflict"}
            else "litmus-link-vector-rvwmo"
        ),
        reason=result.reason,
        cross_check=cross_check,
        edges=edges,
        vector=payload,
    )


_HERD_BODY_CACHE: dict[str, Any] = {}


def _semantic_body(litmus_text: str) -> str:
    """The init+threads+exists block (from the first '{') fully determines
    herd7's verdict; the `RISCV <name>` header, cycle-label string and comment
    lines before it do not. Keying the herd cache on this lets the many
    stress-profile cases that render byte-identical bodies under different names
    share a single herd7 invocation."""
    idx = litmus_text.find("{")
    return litmus_text[idx:] if idx != -1 else litmus_text


def _herd_judge_cached(litmus_text: str):
    """herd_judge memoised on the semantic body. Sound because the verdict is a
    pure function of the body; herd is still run on the FULL text on a miss (it
    needs the `RISCV <name>` header to parse). Note: a cache hit reuses the
    first body's raw output, so raw_output's echoed test-name may differ from
    the current case -- cosmetic only, the verdict/observation are identical."""
    from . import toolchain

    body = _semantic_body(litmus_text)
    cached = _HERD_BODY_CACHE.get(body)
    if cached is not None:
        return cached
    verdict = toolchain.herd_judge(litmus_text)
    if len(_HERD_BODY_CACHE) < 50000:  # soft cap for long-lived server processes
        _HERD_BODY_CACHE[body] = verdict
    return verdict


def _judge_litmus_via_toolchain(litmus_text: str, command_label: str) -> tuple[str, dict[str, Any], str, list[str]] | None:
    """Judge one litmus text with the real herd7 via the toolchain wrapper.

    Uses toolchain's explicit binary path + ``-I <libdir>`` so the cross-check
    works even when herd7 is not on PATH (it is not, in this environment -- the
    previous ``shutil.which`` lookup silently disabled every scalar cross-check,
    and the command was also missing ``-I`` so riscv.cat could not be found).
    Returns None when herd7 is unavailable, else (status, parsed, raw, command)
    with status "ok" | "error" | "unparsed".
    """
    from . import toolchain

    if not toolchain.HERD.exists() or not toolchain.RISCV_CAT.exists():
        return None
    command = [str(toolchain.HERD), "-I", str(toolchain.HERDTOOLS_LIB),
               "-model", str(toolchain.RISCV_CAT), command_label]
    try:
        verdict = _herd_judge_cached(litmus_text)
    except toolchain.ToolchainError as exc:
        return "error", {}, str(exc), command
    if verdict.allowed is None:
        return "unparsed", {"verdict": "unknown", "allowed": None, "observation": verdict.observation}, verdict.raw, command
    parsed = {
        "verdict": "forbidden" if verdict.allowed is False else "allowed",
        "allowed": verdict.allowed,
        "observation": verdict.observation,
    }
    return "ok", parsed, verdict.raw, command


def _run_herd(case: GeneratedCase, herd: str) -> tuple[str, dict[str, Any], str, list[str]] | None:
    """Cross-check a scalar case against the real herd7. None if unavailable."""
    return _judge_litmus_via_toolchain(case.litmus, f"<{case.name}.litmus>")


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
