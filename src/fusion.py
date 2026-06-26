from __future__ import annotations

"""Extension-prose ordering analysis for fusion litmus scenarios.

Stock RVWMO (and herd7's riscv.cat) models only scalar main-memory ordering.
It says NOTHING about the vector (V), cache-management (Zicbom/Zicboz),
page-based memory types (Svpbmt), instruction-fetch (Zifencei) or address-
translation (sfence.vma) extensions that Litmus-link also fuses into tests.

This module therefore does NOT and MUST NOT emit a formal "herd-forbidden"
verdict for fusion cases. Instead it reports, with citations to the relevant
extension prose, whether a fusion test contains the synchronisation those
extensions *document* for its features:

  * ordering-documented -- every orderable fusion feature is paired with the
    barrier the spec documents (FENCE for NC/vector ordering, the
    fence;cbo.flush;fence pattern for an NC alias). Under the cited prose the
    producer effect is ordered before its observation.
  * ordering-absent -- an orderable feature lacks its documented barrier, so
    the weak / reordered outcome is permitted by the extension prose. That is
    a hardware-observation outcome: it needs a stress window plus coverage
    instrumentation to be exhibited, never a herd assertion.
  * prose-spec -- the test rests on a prose-only / hand-required property
    (cross-hart TLB shootdown) that no local barrier can settle.

Every result keeps allowed=None and formal_forbidden_claim=False.
"""

from dataclasses import dataclass, field
from typing import Any

from litmus_ir import LitmusCaseIR, LitmusEvent

VECTOR_FORMS = {
    "unit_load", "unit_store", "strided_load", "strided_store",
    "indexed_ordered_load", "indexed_unordered_load",
    "indexed_ordered_store", "indexed_unordered_store",
    "segment_load", "segment_store", "fof_load", "fof_segment_load",
}
CMO_FORMS = {"clean", "flush", "inval", "zero"}
ATTRIBUTES = {"cacheable", "pbmt_nc", "nc_alias", "cacheable_nc_alias"}
TLB_FORMS = {
    "local_sfence", "remote_sfence", "pte_remap", "permission_fault",
    "ad_update", "asid_global", "satp_switch", "pte_update", "pbmte_sfence",
}

EXTENSION_MODEL = "extension-prose"

@dataclass(frozen=True)
class FusionFinding:
    """One fusion feature and whether its documented barrier is present."""

    feature: str          # e.g. "vector:unit_store", "cmo:flush", "attr:pbmt_nc"
    citation: str         # spec section the rule comes from
    covered: bool         # documented synchronisation present in the test?
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "citation": self.citation,
            "covered": self.covered,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FusionResult:
    status: str                       # "analyzed" | "not_applicable"
    verdict: str                      # "ordering-documented" | "ordering-absent" | "prose-spec"
    allowed: None                     # ALWAYS None: no formal RVWMO boolean for fusion
    model: str
    formal_forbidden_claim: bool      # ALWAYS False
    reason: str
    findings: list[FusionFinding] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "litmus-link.fusion.v1",
            "status": self.status,
            "verdict": self.verdict,
            "allowed": self.allowed,
            "model": self.model,
            "formal_forbidden_claim": self.formal_forbidden_claim,
            "reason": self.reason,
            "findings": [finding.to_json() for finding in self.findings],
        }

# __APPEND2__


def analyze_fusion(case_ir: LitmusCaseIR) -> FusionResult:
    """Report documented extension ordering for a non-scalar fusion case."""
    if case_ir.model == "rvwmo":
        return FusionResult(
            status="not_applicable",
            verdict="prose-spec",
            allowed=None,
            model=EXTENSION_MODEL,
            formal_forbidden_claim=False,
            reason="Pure scalar RVWMO case is handled by the native checker, not fusion analysis.",
            findings=[],
        )

    events = case_ir.events()
    has_data_fence = any(_is_data_fence(e) for e in events)
    has_io_fence = any(_is_io_fence(e) for e in events)
    cmo_wrapped = _cmo_fence_wrapped(case_ir)
    cmo_trailing_fence = _cmo_has_trailing_fence(case_ir)
    tags = set(case_ir.tags)

    findings: list[FusionFinding] = []
    orderable = 0
    uncovered = 0

    # --- Vector memory ordering (V extension) ---
    vector_form = next((t for t in tags if t in VECTOR_FORMS), "")
    if vector_form:
        orderable += 1
        covered = has_data_fence or has_io_fence
        if not covered:
            uncovered += 1
        findings.append(
            FusionFinding(
                feature=f"vector:{vector_form}",
                citation="V-spec, 'Vector Memory Consistency Model' (vector accesses obey RVWMO per element; FENCE orders them against other accesses)",
                covered=covered,
                detail=_vector_detail(vector_form, covered),
            )
        )

    # --- Cache-management ordering (Zicbom / Zicboz) ---
    cmo_form = next((t for t in tags if t in CMO_FORMS), "")
    if cmo_form:
        orderable += 1
        covered = cmo_wrapped or cmo_trailing_fence
        if not covered:
            uncovered += 1
        findings.append(
            FusionFinding(
                feature=f"cmo:{cmo_form}",
                citation="Zicbom/Zicboz: a CBO is ordered by FENCE with matching PRED/SUCC bits; the robust pattern is fence;cbo;fence",
                covered=covered,
                detail=_cmo_detail(cmo_wrapped, cmo_trailing_fence),
            )
        )

    # --- Page-based memory type / NC alias (Svpbmt) ---
    attribute = next((t for t in tags if t in ATTRIBUTES and t != "cacheable"), "")
    if attribute:
        orderable += 1
        if attribute == "cacheable_nc_alias":
            covered = cmo_wrapped and has_io_fence
            citation = "Svpbmt + alias rule: a cacheable/NC alias needs fence iorw,iorw; cbo.flush; fence iorw,iorw to avoid losing coherence/order"
        else:  # pbmt_nc / nc_alias
            covered = has_data_fence or has_io_fence
            citation = "Svpbmt: NC main-memory accesses are weakly ordered; FENCE (rw,rw / iorw,iorw) supplies the ordering"
        if not covered:
            uncovered += 1
        findings.append(
            FusionFinding(
                feature=f"attr:{attribute}",
                citation=citation,
                covered=covered,
                detail=("Documented barrier present." if covered else "No FENCE orders the NC access; weak outcome is permitted by the prose."),
            )
        )

    # --- Address translation / TLB (sfence.vma) -- prose-only, hand territory ---
    tlb_form = next((t for t in tags if t in TLB_FORMS), "")
    if tlb_form:
        findings.append(
            FusionFinding(
                feature=f"tlb:{tlb_form}",
                citation="Privileged spec, sfence.vma: cross-hart shootdown ordering is a prose/hand-required property, not a local-barrier outcome",
                covered=False,
                detail="Cross-hart TLB coherence cannot be settled by a local FENCE; treat as prose-spec/hand-required.",
            )
        )

    return _verdict(orderable, uncovered, tlb_form, findings)


def _verdict(orderable: int, uncovered: int, tlb_form: str, findings: list[FusionFinding]) -> FusionResult:
    tlb_caveat = " A cross-hart TLB shootdown property remains prose-spec/hand-required." if tlb_form else ""
    if orderable == 0:
        return FusionResult(
            status="analyzed",
            verdict="prose-spec",
            allowed=None,
            model=EXTENSION_MODEL,
            formal_forbidden_claim=False,
            reason="No locally-orderable fusion feature; outcome rests on prose-spec/hardware-observation properties." + tlb_caveat,
            findings=findings,
        )
    if uncovered == 0:
        return FusionResult(
            status="analyzed",
            verdict="ordering-documented",
            allowed=None,
            model=EXTENSION_MODEL,
            formal_forbidden_claim=False,
            reason="Every orderable fusion feature carries the synchronisation its extension documents; the producer effect is ordered before observation (informative, model-extended -- not a herd verdict)." + tlb_caveat,
            findings=findings,
        )
    return FusionResult(
        status="analyzed",
        verdict="ordering-absent",
        allowed=None,
        model=EXTENSION_MODEL,
        formal_forbidden_claim=False,
        reason=f"{uncovered} orderable fusion feature(s) lack the documented barrier; the reordered outcome is permitted by the extension prose and needs a stress window + coverage instrumentation to exhibit (never a herd forbidden claim)." + tlb_caveat,
        findings=findings,
    )

# __APPEND3__


def _is_data_fence(event: LitmusEvent) -> bool:
    # "fence rw,rw" / "fence w,w" / ... but NOT "fence.i".
    return event.kind == "fence" and event.instruction.startswith("fence ") and "," in event.instruction


def _is_io_fence(event: LitmusEvent) -> bool:
    return event.kind == "fence" and "iorw,iorw" in event.instruction.replace(" ", "")


def _cmo_fence_wrapped(case_ir: LitmusCaseIR) -> bool:
    """A CBO op with a *data* FENCE both before and after it in program order.

    fence.i is deliberately excluded: it synchronises instruction fetch, not
    the CBO's data effect, so it never counts toward data ordering.
    """
    for hart in case_ir.harts:
        for index, event in enumerate(hart):
            if event.kind == "cmo":
                before = any(_is_data_fence(h) for h in hart[:index])
                after = any(_is_data_fence(h) for h in hart[index + 1 :])
                if before and after:
                    return True
    return False


def _cmo_has_trailing_fence(case_ir: LitmusCaseIR) -> bool:
    for hart in case_ir.harts:
        for index, event in enumerate(hart):
            if event.kind == "cmo" and any(_is_data_fence(h) for h in hart[index + 1 :]):
                return True
    return False


def _vector_detail(vector_form: str, covered: bool) -> str:
    notes = {
        "indexed_ordered_load": "Ordered-indexed: element accesses keep program order among themselves.",
        "indexed_ordered_store": "Ordered-indexed: element accesses keep program order among themselves.",
        "indexed_unordered_load": "Unordered-indexed: no inter-element order is guaranteed.",
        "indexed_unordered_store": "Unordered-indexed: no inter-element order is guaranteed.",
        "fof_load": "Fault-only-first: only the first element may fault; later elements are speculative.",
        "fof_segment_load": "Fault-only-first segment load: only the first segment may fault.",
        "segment_load": "Segment: each field is a separate element access.",
        "segment_store": "Segment: each field is a separate element access.",
    }
    base = notes.get(vector_form, "Vector element accesses each obey RVWMO as scalar accesses.")
    order = "A FENCE orders these element accesses against other harts' accesses." if covered else "No FENCE present; element accesses may reorder relative to the observer."
    return f"{base} {order}"


def _cmo_detail(wrapped: bool, trailing: bool) -> str:
    if wrapped:
        return "fence;cbo;fence wrap present: the CBO effect is ordered both before and after (robust pattern)."
    if trailing:
        return "A trailing FENCE orders the CBO effect before the observation, though it is not the full fence;cbo;fence wrap."
    return "No FENCE orders the CBO; its effect may reorder relative to the observer (weak outcome permitted)."

