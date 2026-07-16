from __future__ import annotations

"""RISC-V diy-style cycle policy implemented without invoking diy7.

This module mirrors the generation stages in herdtools7 ``gen/relax.ml`` and
``gen/alt.ml`` for the scalar RISC-V edge subset represented by NativeEdge:
relaxation parsing/macro expansion, safe-vs-relax selection, adjacency policy,
bounded cycle construction, legality checks, and rotation canonicalisation.
"""

import fnmatch
import itertools
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Sequence

from .native_cycles import NativeCycle, location_ids, validate_cycle
from .native_edges import (
    DIFFERENT,
    EXTERNAL,
    LOCAL,
    NativeEdge,
    edge_catalog,
    edges_for_shape,
)


DIY_MODES = (
    "default",
    "sc",
    "thin",
    "uni",
    "critical",
    "free",
    "ppo",
    "transitive",
    "total",
    "mixedcheck",
)
DIY_OBSERVERS = ("avoid", "accept", "force", "local", "three", "four", "infinity")
DIY_OBSERVER_TYPES = ("straight", "fenced", "loop")


class NativeDiyError(ValueError):
    pass


@dataclass(frozen=True)
class Relaxation:
    source: str
    edges: tuple[NativeEdge, ...]

    def __post_init__(self) -> None:
        if not self.edges:
            raise NativeDiyError("a relaxation cannot be empty")
        for left, right in zip(self.edges, self.edges[1:]):
            if left.dst != right.src:
                raise NativeDiyError(
                    f"relaxation {self.source!r} has incompatible directions: "
                    f"{left.label} -> {right.label}"
                )

    @property
    def label(self) -> str:
        if len(self.edges) == 1:
            return self.edges[0].label
        return "[" + ",".join(edge.label for edge in self.edges) + "]"

    @property
    def first(self) -> NativeEdge:
        return self.edges[0]

    @property
    def last(self) -> NativeEdge:
        return self.edges[-1]

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "label": self.label,
            "edges": [edge.label for edge in self.edges],
        }


@dataclass(frozen=True)
class DiyConfig:
    safe: tuple[str, ...]
    relax: tuple[str, ...]
    reject: tuple[str, ...] = ()
    prefixes: tuple[tuple[str, ...], ...] = ()
    size: int = 4
    min_size: int = 2
    nprocs: int = 2
    exact_procs: bool = False
    upto: bool = True
    mode: str = "default"
    mix: bool = False
    min_relax: int = 1
    max_relax: int = 1
    max_accesses_per_proc: int | None = None
    include_same: bool = False
    include_internal: bool = True
    observer: str = "avoid"
    observer_type: str = "straight"
    realdep: bool = False
    unrollatomic: int | None = None
    moreedges: bool = False

    def validate(self) -> None:
        if not self.safe:
            raise NativeDiyError("safe edge list cannot be empty")
        if not self.relax:
            raise NativeDiyError("relax edge list cannot be empty")
        if self.min_size < 2 or self.size < self.min_size:
            raise NativeDiyError("cycle size range must satisfy 2 <= min_size <= size")
        if self.nprocs < 1:
            raise NativeDiyError("nprocs must be at least 1")
        if self.max_accesses_per_proc is not None and self.max_accesses_per_proc < 1:
            raise NativeDiyError("max accesses per hart must be at least 1")
        if self.mode not in DIY_MODES:
            raise NativeDiyError(f"unknown diy mode: {self.mode}")
        if self.observer not in DIY_OBSERVERS:
            raise NativeDiyError(f"unknown observer policy: {self.observer}")
        if self.observer_type not in DIY_OBSERVER_TYPES:
            raise NativeDiyError(f"unknown observer type: {self.observer_type}")
        if self.min_relax < 0 or self.max_relax < self.min_relax:
            raise NativeDiyError("relaxation count must satisfy 0 <= min_relax <= max_relax")
        if not self.mix and not (self.min_relax <= 1 <= self.max_relax):
            raise NativeDiyError("non-mixed diy generation requires relaxation cardinality to include one")
        if self.unrollatomic is not None and self.unrollatomic < 0:
            raise NativeDiyError("unrollatomic must be non-negative")
        if self.moreedges:
            raise NativeDiyError(
                "native diy --moreedges requires mixed-size atom lowering, which is not implemented"
            )
        if self.unrollatomic is not None:
            raise NativeDiyError(
                "native diy --unrollatomic requires explicit LR/SC atomic idioms, which are not implemented"
            )

    def to_json(self) -> dict:
        return {
            "safe": list(self.safe),
            "relax": list(self.relax),
            "reject": list(self.reject),
            "prefixes": [list(prefix) for prefix in self.prefixes],
            "size": self.size,
            "min_size": self.min_size,
            "nprocs": self.nprocs,
            "exact_procs": self.exact_procs,
            "upto": self.upto,
            "mode": self.mode,
            "mix": self.mix,
            "min_relax": self.min_relax,
            "max_relax": self.max_relax,
            "max_accesses_per_proc": self.max_accesses_per_proc,
            "include_same": self.include_same,
            "include_internal": self.include_internal,
            "observer": self.observer,
            "observer_type": self.observer_type,
            "realdep": self.realdep,
            "unrollatomic": self.unrollatomic,
            "moreedges": self.moreedges,
        }


DEFAULT_DIY_SAFE = (
    "Rfe",
    "Fre",
    "Wse",
    "Fence.rw.rwd**",
    "Fence.rw.wd*W",
    "Fence.w.wdWW",
    "DpAddrdR",
    "DpAddrdW",
    "DpDatadW",
    "DpCtrldR",
    "DpCtrldW",
)
DEFAULT_DIY_RELAX = ("PodRR", "PodRW", "PodWR", "PodWW")


def expand_relaxations(
    tokens: Sequence[str],
    *,
    include_same: bool,
    include_internal: bool,
    moreedges: bool = False,
) -> tuple[Relaxation, ...]:
    mechanisms = ("communication", "po", "fence", "dependency")
    domain = edge_catalog(mechanisms, include_same, include_internal)
    expanded: list[Relaxation] = []
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        token_expansion = _expand_one(
            token,
            domain,
            include_same=include_same,
            moreedges=moreedges,
        )
        if not token_expansion:
            raise NativeDiyError(f"relaxation has no direction-compatible expansion: {token}")
        expanded.extend(token_expansion)
    unique: dict[tuple[str, ...], Relaxation] = {}
    for relaxation in expanded:
        unique.setdefault(tuple(edge.label for edge in relaxation.edges), relaxation)
    if not unique:
        raise NativeDiyError(f"no usable relaxation expanded from: {', '.join(tokens)}")
    return tuple(sorted(unique.values(), key=lambda relaxation: relaxation.label))


def enumerate_diy_cycles(config: DiyConfig) -> tuple[list[NativeCycle], dict]:
    config.validate()
    if config.mode == "thin":
        return [], {
            "schema": "litmus-link.native-diy-audit.v1",
            "config": config.to_json(),
            "expanded_safe": [],
            "expanded_relax": [],
            "expanded_reject": [],
            "expanded_prefixes": [],
            "candidates": 0,
            "accepted": 0,
            "duplicate": 0,
            "excluded": {"riscv_thin_has_no_arch_ppo_relaxations": 1},
        }
    safe = expand_relaxations(
        config.safe,
        include_same=config.include_same,
        include_internal=config.include_internal,
        moreedges=config.moreedges,
    )
    relax = expand_relaxations(
        config.relax,
        include_same=config.include_same,
        include_internal=config.include_internal,
        moreedges=config.moreedges,
    )
    reject = expand_relaxations(
        config.reject,
        include_same=config.include_same,
        include_internal=config.include_internal,
        moreedges=config.moreedges,
    ) if config.reject else ()
    prefixes = _expand_prefixes(
        config.prefixes,
        include_same=config.include_same,
        include_internal=config.include_internal,
        moreedges=config.moreedges,
    )
    relax_keys = {tuple(edge.label for edge in item.edges) for item in relax}
    # herdtools removes safe relaxations that are also explicitly relaxed.
    safe = tuple(item for item in safe if tuple(edge.label for edge in item.edges) not in relax_keys)

    exclusions: Counter[str] = Counter()
    candidates = 0
    duplicate = 0
    seen: set[tuple[str, ...]] = set()
    cycles: list[NativeCycle] = []
    sizes = range(config.min_size, config.size + 1) if config.upto else (config.size,)
    prefix_sizes = range(1, config.size + 1) if config.upto else (config.size,)

    if config.mix:
        domains = [("mixed", safe + relax)]
    else:
        domains = [(item.label, safe + (item,)) for item in relax]

    prefix_variants = prefixes or ((),)
    for _focus, domain in domains:
        for prefix in prefix_variants:
            selected_sizes = prefix_sizes if prefix else sizes
            for size in selected_sizes:
                for selected, suffix in _relaxation_cycles(domain, size, config.mode, safe, prefix=prefix):
                    candidates += 1
                    selected_relax = {
                        tuple(edge.label for edge in item.edges)
                        for item in suffix
                        if tuple(edge.label for edge in item.edges) in relax_keys
                    }
                    if not selected_relax:
                        exclusions["no_relaxation"] += 1
                        continue
                    if config.mix and not (config.min_relax <= len(selected_relax) <= config.max_relax):
                        exclusions["relaxation_cardinality"] += 1
                        continue
                    if not config.mix and len(selected_relax) != 1:
                        exclusions["mixed_relaxations_disabled"] += 1
                        continue
                    flattened = tuple(edge for item in selected for edge in item.edges)
                    if _contains_rejected(flattened, reject):
                        exclusions["rejected_subsequence"] += 1
                        continue
                    if not prefix and _cannot_normalise(selected, config.mode):
                        exclusions["cannot_normalise"] += 1
                        continue
                    if not config.include_same and sum(edge.location == DIFFERENT for edge in flattened) < 2:
                        exclusions["insufficient_location_changes"] += 1
                        continue
                    decision = validate_cycle(
                        flattened,
                        max_procs=config.nprocs,
                        exact_procs=config.exact_procs,
                        max_accesses_per_proc=config.max_accesses_per_proc,
                    )
                    if not decision.accepted:
                        exclusions[decision.reason] += 1
                        continue
                    observer_reason = _observer_rejection(flattened, config.observer)
                    if observer_reason:
                        exclusions[observer_reason] += 1
                        continue
                    cycle = NativeCycle(flattened, "DIY").canonical()
                    if cycle.canonical_key in seen:
                        duplicate += 1
                        continue
                    seen.add(cycle.canonical_key)
                    cycles.append(cycle)

    cycles.sort(key=lambda cycle: cycle.canonical_key)
    audit = {
        "schema": "litmus-link.native-diy-audit.v1",
        "config": config.to_json(),
        "expanded_safe": [item.to_json() for item in safe],
        "expanded_relax": [item.to_json() for item in relax],
        "expanded_reject": [item.to_json() for item in reject],
        "expanded_prefixes": [
            [item.to_json() for item in prefix]
            for prefix in prefixes
        ],
        "candidates": candidates,
        "accepted": len(cycles),
        "duplicate": duplicate,
        "excluded": dict(sorted(exclusions.items())),
    }
    return cycles, audit


def _expand_one(
    token: str,
    domain: Sequence[NativeEdge],
    *,
    include_same: bool,
    moreedges: bool,
) -> list[Relaxation]:
    if token == "PPO":
        return [Relaxation(token, (edge,)) for edge in domain if edge.is_local and edge.preserved]
    macro = re.fullmatch(r"(all|some)(RR|RW|WR|WW)", token)
    if macro:
        shape = macro.group(2)
        edges = tuple(
            edge
            for edge in edges_for_shape(
                shape,
                ("po", "fence", "dependency"),
                include_same=include_same,
            )
            if edge.location == DIFFERENT
        )
        if macro.group(1) == "some":
            default_dependency = {
                "RR": "DpAddrdR",
                "RW": "DpDatadW",
                "WR": None,
                "WW": None,
            }[shape]
            edges = tuple(
                edge
                for edge in edges
                if edge.relation in {"po", "fence"}
                or edge.label == default_dependency
            )
        return [Relaxation(token, (edge,)) for edge in edges]
    cumulative = re.fullmatch(r"(ABC|AC|BC)(.+)", token)
    if cumulative:
        prefix, body = cumulative.groups()
        bodies = _edge_matches(body, domain)
        rfe = _single_edge("Rfe", domain)
        out = []
        for edge in bodies:
            sequence = {
                "AC": (rfe, edge),
                "BC": (edge, rfe),
                "ABC": (rfe, edge, rfe),
            }[prefix]
            try:
                out.append(Relaxation(token, sequence))
            except NativeDiyError:
                continue
        return out
    if token.startswith("[") and token.endswith("]"):
        parts = [part.strip() for part in token[1:-1].split(",") if part.strip()]
        if not parts:
            raise NativeDiyError("empty relaxation sequence")
        choices = [_edge_matches(part, domain) for part in parts]
        out = []
        for selected in itertools.product(*choices):
            try:
                out.append(Relaxation(token, tuple(selected)))
            except NativeDiyError:
                continue
        return out
    return [Relaxation(token, (edge,)) for edge in _edge_matches(token, domain)]


def _edge_matches(pattern: str, domain: Sequence[NativeEdge]) -> tuple[NativeEdge, ...]:
    original = pattern
    aliases = {
        "Rf": "Rf*",
        "Fr": "Fr*",
        "Ws": "Ws*",
        "Co": "Ws*",
        "Coe": "Wse",
        "Coi": "Wsi",
        "Po": "Po*",
    }
    pattern = aliases.get(pattern, pattern)
    # Our canonical spelling is Wse/Wsi while newer herdtools prints Coe/Coi.
    normalized = pattern.replace("Coe", "Wse").replace("Coi", "Wsi")
    fence_lowering = _fence_pattern_lowering(normalized)
    if normalized.startswith("Ctrl"):
        normalized = "Dp" + normalized
    if normalized in {"Dp", "DpAddr", "DpData", "DpCtrl", "DpCtrlFenceI"}:
        normalized += "*"
    fence_location = re.fullmatch(r"Fence([ds])(.*)", normalized)
    if fence_location:
        normalized = f"Fence.*{fence_location.group(1)}{fence_location.group(2)}"
    if normalized == "Fence":
        normalized = "Fence.*"
    elif normalized in {
        "Fence.i",
        "Fence.r.r",
        "Fence.r.w",
        "Fence.r.rw",
        "Fence.w.r",
        "Fence.w.w",
        "Fence.w.rw",
        "Fence.rw.r",
        "Fence.rw.w",
        "Fence.rw.rw",
        "Fence.tso",
        "Fence.iorw.iorw",
    }:
        normalized += "[ds]*"
    matches = tuple(edge for edge in domain if fnmatch.fnmatchcase(edge.label, normalized))
    if fence_lowering is not None:
        matches = tuple(edge for edge in matches if edge.lowering == fence_lowering)
    if original.startswith(("Ctrl", "DpCtrl")) and "CtrlFenceI" not in original:
        matches = tuple(edge for edge in matches if edge.mechanism == "ctrl")
    elif "CtrlFenceI" in original:
        matches = tuple(edge for edge in matches if edge.mechanism == "ctrl_fencei")
    if not matches:
        raise NativeDiyError(f"unknown or unavailable native relaxation: {pattern}")
    return matches


def _fence_pattern_lowering(pattern: str) -> str | None:
    barriers = {
        "Fence.iorw.iorw": "fence iorw,iorw",
        "Fence.rw.rw": "fence rw,rw",
        "Fence.rw.r": "fence rw,r",
        "Fence.rw.w": "fence rw,w",
        "Fence.r.rw": "fence r,rw",
        "Fence.w.rw": "fence w,rw",
        "Fence.r.r": "fence r,r",
        "Fence.r.w": "fence r,w",
        "Fence.w.r": "fence w,r",
        "Fence.w.w": "fence w,w",
        "Fence.tso": "fence.tso",
        "Fence.i": "fence.i",
    }
    for name in sorted(barriers, key=len, reverse=True):
        if pattern == name:
            return barriers[name]
        if pattern.startswith(name) and pattern[len(name) : len(name) + 1] in {"*", "d", "s", "["}:
            return barriers[name]
    return None


def _single_edge(label: str, domain: Sequence[NativeEdge]) -> NativeEdge:
    matches = [edge for edge in domain if edge.label == label]
    if len(matches) != 1:
        raise NativeDiyError(f"required edge {label} is unavailable")
    return matches[0]


def _expand_prefixes(
    prefixes: Sequence[Sequence[str]],
    *,
    include_same: bool,
    include_internal: bool,
    moreedges: bool,
) -> tuple[tuple[Relaxation, ...], ...]:
    expanded: list[tuple[Relaxation, ...]] = []
    for raw_prefix in prefixes:
        tokens = tuple(token.strip() for token in raw_prefix if token.strip())
        if not tokens:
            raise NativeDiyError("a diy prefix cannot be empty")
        choices = [
            expand_relaxations(
                (token,),
                include_same=include_same,
                include_internal=include_internal,
                moreedges=moreedges,
            )
            for token in tokens
        ]
        expanded.extend(tuple(selected) for selected in itertools.product(*choices))

    unique: dict[tuple[str, ...], tuple[Relaxation, ...]] = {}
    for prefix in expanded:
        key = tuple(item.label for item in prefix)
        unique.setdefault(key, prefix)
    return tuple(unique[key] for key in sorted(unique))


def _relaxation_cycles(
    domain: Sequence[Relaxation],
    size: int,
    mode: str,
    safe: Sequence[Relaxation],
    *,
    prefix: Sequence[Relaxation] = (),
) -> Iterator[tuple[tuple[Relaxation, ...], tuple[Relaxation, ...]]]:
    safe_edges = {item.label for item in safe}
    fixed = tuple(prefix)

    if any(
        not _can_precede(left, right, mode, safe_edges)
        for left, right in zip(fixed, fixed[1:])
    ):
        return

    def extend(suffix: tuple[Relaxation, ...]) -> Iterator[tuple[tuple[Relaxation, ...], tuple[Relaxation, ...]]]:
        selected = fixed + suffix
        if len(suffix) == size:
            if selected and _can_precede(selected[-1], selected[0], mode, safe_edges):
                yield selected, suffix
            return
        for item in domain:
            if not selected or _can_precede(selected[-1], item, mode, safe_edges):
                yield from extend(suffix + (item,))

    yield from extend(())


def _can_precede(
    left_relax: Relaxation,
    right_relax: Relaxation,
    mode: str,
    safe_edges: set[str],
) -> bool:
    left = left_relax.last
    right = right_relax.first
    if left.dst != right.src:
        return False
    if (left.relation, right.relation) in {("co", "co"), ("fr", "co"), ("rf", "fr")}:
        if left.scope == right.scope:
            return False
    if mode == "thin":
        return True
    if mode == "critical":
        if left.relation in {"co", "fr"} and right.relation == "rf":
            return True
        return left.scope != right.scope
    if mode == "mixedcheck":
        if left.scope == EXTERNAL and right.scope == EXTERNAL:
            return False
        if left.scope == LOCAL and right.scope == LOCAL:
            return not (left.location == DIFFERENT and right.location == DIFFERENT)
        return True
    if mode == "uni":
        if (left.relation, right.relation) in {("co", "co"), ("fr", "co"), ("rf", "fr")}:
            return left.scope != right.scope
        if left.relation == "po" and right.relation == "po":
            return False
        return True
    if mode in {"free", "ppo", "transitive", "total"}:
        if (left.relation, right.relation) in {("co", "co"), ("fr", "co"), ("rf", "fr")}:
            if mode in {"free", "total"}:
                return left.scope != right.scope
            return False
        if mode == "ppo":
            if left.label == right.label:
                return False
        if mode == "transitive" and left.scope == LOCAL and right.scope == LOCAL:
            compact = _compact_local(left, right)
            if compact and compact in safe_edges:
                return False
        return True
    if mode == "sc":
        # SC mode rejects adjacent internal relations unless their composition
        # is not already represented by a safe local edge.
        if left.scope == LOCAL and right.scope == LOCAL:
            compact = _compact_local(left, right)
            return compact is None or compact not in safe_edges
        return True
    # herdtools default accepts external boundaries and a small set of useful
    # internal compositions; all other internal/internal pairs are redundant.
    if left.scope == EXTERNAL or right.scope == EXTERNAL:
        return True
    if left.relation in {"rf", "fr", "co"} and right.relation == "po" and right.location == DIFFERENT:
        return True
    if left.relation == "po" and left.location == DIFFERENT and right.relation in {"rf", "fr", "co"}:
        return True
    if left.relation == "dependency" and left.location == DIFFERENT and right.relation == "po" and right.location == DIFFERENT:
        return True
    if left.relation == "po" and left.location == DIFFERENT and right.relation == "dependency" and right.location == DIFFERENT:
        return True
    if left.relation == "rf" and left.scope == LOCAL and right.relation == "po" and right.location != DIFFERENT:
        return True
    if left.relation == "po" and left.location != DIFFERENT and right.relation == "rf" and right.scope == LOCAL:
        return True
    return False


def _compact_local(left: NativeEdge, right: NativeEdge) -> str | None:
    if left.scope != LOCAL or right.scope != LOCAL or left.src not in "RW" or right.dst not in "RW":
        return None
    location = "s" if left.location != DIFFERENT and right.location != DIFFERENT else "d"
    if left.relation == "fence":
        return left.label
    if right.relation == "fence":
        return right.label
    if left.relation == "dependency":
        return left.label
    return f"Po{location}{left.src}{right.dst}"


def _contains_rejected(edges: Sequence[NativeEdge], reject: Sequence[Relaxation]) -> bool:
    if not reject:
        return False
    labels = tuple(edge.label for edge in edges)
    doubled = labels + labels
    for relaxation in reject:
        needle = tuple(edge.label for edge in relaxation.edges)
        for start in range(len(labels)):
            if doubled[start : start + len(needle)] == needle:
                return True
    return False


def _observer_rejection(edges: Sequence[NativeEdge], policy: str) -> str | None:
    # The critical cycle's write count is a conservative proxy for whether a
    # complete coherence chain needs an observer process.  Lowering always
    # records the final value; policies that explicitly permit longer chains
    # retain them for straight/fenced/loop observer insertion.
    locations = location_ids(edges)
    writes_per_location = Counter(
        location for edge, location in zip(edges, locations) if edge.src == "W"
    )
    maximum = max(writes_per_location.values(), default=0)
    if policy == "three" and maximum > 3:
        return "observer_limit_three"
    if policy == "four" and maximum > 4:
        return "observer_limit_four"
    return None


def _cannot_normalise(relaxations: Sequence[Relaxation], mode: str) -> bool:
    if mode not in {"default", "sc", "transitive"}:
        return False
    edges = tuple(edge for item in relaxations for edge in item.edges)
    # Normaliser.family rejects the non-critical same-scope compositions that
    # default diy generation may temporarily admit while growing a prefix.
    # Internal communication is the exception for which those compositions
    # carry a recognised family meaning.
    if any(edge.scope == LOCAL and edge.relation in {"rf", "fr", "co"} for edge in edges):
        return False
    for index, left in enumerate(relaxations):
        right = relaxations[(index + 1) % len(relaxations)]
        if left.last.scope != right.first.scope:
            continue
        # A cumulative AC/ABC relaxation intentionally begins with Rfe.  The
        # incoming external communication is part of that grouped idiom and
        # is accepted by herdtools' normalizer rather than compacted away.
        if len(right.edges) > 1 and right.first.label == "Rfe":
            continue
        return True
    return False
