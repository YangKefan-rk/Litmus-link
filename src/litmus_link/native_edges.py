from __future__ import annotations

"""Native RISC-V litmus relation grammar.

This module is deliberately independent of herdtools and of any pre-generated
corpus.  It describes the finite edge domain consumed by Litmus-link's native
cycle enumerator.  An edge connects two memory events in the critical cycle;
the source/target directions make incompatible sequences rejectable before
assembly is generated.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


READ = "R"
WRITE = "W"
LOCAL = "local"
EXTERNAL = "external"
SAME = "same"
DIFFERENT = "different"

ACCESS_DIRECTIONS = (READ, WRITE)
ACCESS_SHAPES = tuple(f"{src}{dst}" for src in ACCESS_DIRECTIONS for dst in ACCESS_DIRECTIONS)
FENCE_SETS = ("r", "w", "rw")


@dataclass(frozen=True)
class NativeEdge:
    label: str
    relation: str
    src: str
    dst: str
    scope: str
    location: str
    mechanism: str
    lowering: str = ""
    preserved: bool = True

    def __post_init__(self) -> None:
        if self.src not in ACCESS_DIRECTIONS or self.dst not in ACCESS_DIRECTIONS:
            raise ValueError(f"invalid edge direction: {self.src}->{self.dst}")
        if self.scope not in {LOCAL, EXTERNAL}:
            raise ValueError(f"invalid edge scope: {self.scope}")
        if self.location not in {SAME, DIFFERENT}:
            raise ValueError(f"invalid edge location relation: {self.location}")

    @property
    def shape(self) -> str:
        return self.src + self.dst

    @property
    def is_external(self) -> bool:
        return self.scope == EXTERNAL

    @property
    def is_local(self) -> bool:
        return self.scope == LOCAL

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "relation": self.relation,
            "src": self.src,
            "dst": self.dst,
            "scope": self.scope,
            "location": self.location,
            "mechanism": self.mechanism,
            "lowering": self.lowering,
            "preserved": self.preserved,
        }


def communication_edges(*, include_internal: bool = True) -> tuple[NativeEdge, ...]:
    edges = [
        NativeEdge("Rfe", "rf", WRITE, READ, EXTERNAL, SAME, "communication"),
        NativeEdge("Fre", "fr", READ, WRITE, EXTERNAL, SAME, "communication"),
        NativeEdge("Wse", "co", WRITE, WRITE, EXTERNAL, SAME, "communication"),
    ]
    if include_internal:
        edges.extend(
            [
                NativeEdge("Rfi", "rf", WRITE, READ, LOCAL, SAME, "communication"),
                NativeEdge("Fri", "fr", READ, WRITE, LOCAL, SAME, "communication"),
                NativeEdge("Wsi", "co", WRITE, WRITE, LOCAL, SAME, "communication"),
            ]
        )
    return tuple(edges)


def po_edges(*, include_same: bool = True) -> tuple[NativeEdge, ...]:
    out: list[NativeEdge] = []
    locations = (DIFFERENT, SAME) if include_same else (DIFFERENT,)
    for location in locations:
        prefix = "Pod" if location == DIFFERENT else "Pos"
        for shape in ACCESS_SHAPES:
            out.append(
                NativeEdge(
                    f"{prefix}{shape}",
                    "po",
                    shape[0],
                    shape[1],
                    LOCAL,
                    location,
                    "po",
                    preserved=False,
                )
            )
    return tuple(out)


def fence_edges(*, include_same: bool = True) -> tuple[NativeEdge, ...]:
    out: list[NativeEdge] = []
    locations = (DIFFERENT, SAME) if include_same else (DIFFERENT,)
    for location in locations:
        location_code = "d" if location == DIFFERENT else "s"
        for shape in ACCESS_SHAPES:
            out.append(
                NativeEdge(
                    f"Fence.i{location_code}{shape}",
                    "fence",
                    shape[0],
                    shape[1],
                    LOCAL,
                    location,
                    "fence",
                    lowering="fence.i",
                    preserved=False,
                )
            )
        for pred in FENCE_SETS:
            for succ in FENCE_SETS:
                for shape in ACCESS_SHAPES:
                    orders = _fence_covers(pred, shape[0]) and _fence_covers(succ, shape[1])
                    out.append(
                        NativeEdge(
                            f"Fence.{pred}.{succ}{location_code}{shape}",
                            "fence",
                            shape[0],
                            shape[1],
                            LOCAL,
                            location,
                            "fence",
                            lowering=f"fence {pred},{succ}",
                            preserved=orders,
                        )
                    )
        for shape in ACCESS_SHAPES:
            out.append(
                NativeEdge(
                    f"Fence.iorw.iorw{location_code}{shape}",
                    "fence",
                    shape[0],
                    shape[1],
                    LOCAL,
                    location,
                    "fence",
                    lowering="fence iorw,iorw",
                    preserved=True,
                )
            )
            out.append(
                NativeEdge(
                    f"Fence.tso{location_code}{shape}",
                    "fence",
                    shape[0],
                    shape[1],
                    LOCAL,
                    location,
                    "fence",
                    lowering="fence.tso",
                    preserved=shape != "WR",
                )
            )
    return tuple(out)


def dependency_edges(*, include_same: bool = True) -> tuple[NativeEdge, ...]:
    out: list[NativeEdge] = []
    locations = (DIFFERENT, SAME) if include_same else (DIFFERENT,)
    mechanisms = {
        READ: ("Addr", "Ctrl", "CtrlFenceI"),
        WRITE: ("Addr", "Data", "Ctrl", "CtrlFenceI"),
    }
    for location in locations:
        location_code = "d" if location == DIFFERENT else "s"
        for dst, dependencies in mechanisms.items():
            for dependency in dependencies:
                preserved = (
                    dependency in {"Addr", "CtrlFenceI"}
                    or (dst == WRITE and dependency in {"Data", "Ctrl"})
                )
                out.append(
                    NativeEdge(
                        f"Dp{dependency}{location_code}{dst}",
                        "dependency",
                        READ,
                        dst,
                        LOCAL,
                        location,
                        _dependency_mechanism(dependency),
                        lowering=_dependency_lowering(dependency),
                        preserved=preserved,
                    )
                )
    return tuple(out)


@lru_cache(maxsize=None)
def edge_catalog(
    mechanisms: tuple[str, ...] = ("communication", "po", "fence", "dependency"),
    include_same: bool = True,
    include_internal: bool = True,
) -> tuple[NativeEdge, ...]:
    selected = set(mechanisms)
    unknown = selected - {"communication", "po", "fence", "dependency"}
    if unknown:
        raise ValueError(f"unknown native edge mechanisms: {', '.join(sorted(unknown))}")
    out: list[NativeEdge] = []
    if "communication" in selected:
        out.extend(communication_edges(include_internal=include_internal))
    if "po" in selected:
        out.extend(po_edges(include_same=include_same))
    if "fence" in selected:
        out.extend(fence_edges(include_same=include_same))
    if "dependency" in selected:
        out.extend(dependency_edges(include_same=include_same))
    labels = [edge.label for edge in out]
    if len(labels) != len(set(labels)):
        raise AssertionError("native edge catalog contains duplicate labels")
    return tuple(out)


def edges_for_shape(
    shape: str,
    mechanisms: Iterable[str] = ("po", "fence", "dependency"),
    *,
    include_same: bool = True,
) -> tuple[NativeEdge, ...]:
    if shape not in ACCESS_SHAPES:
        raise ValueError(f"invalid local access shape: {shape}")
    selected = tuple(dict.fromkeys(str(value) for value in mechanisms))
    return tuple(
        edge
        for edge in edge_catalog(selected, include_same, False)
        if edge.is_local and edge.shape == shape
    )


def edge_by_label(label: str) -> NativeEdge:
    for edge in edge_catalog():
        if edge.label == label:
            return edge
    raise ValueError(f"unknown native edge: {label}")


def _fence_covers(mode: str, direction: str) -> bool:
    return (direction == READ and "r" in mode) or (direction == WRITE and "w" in mode)


def _dependency_mechanism(name: str) -> str:
    return {
        "Addr": "addr",
        "Data": "data",
        "Ctrl": "ctrl",
        "CtrlFenceI": "ctrl_fencei",
    }[name]


def _dependency_lowering(name: str) -> str:
    return {
        "Addr": "addr",
        "Data": "data",
        "Ctrl": "ctrl",
        "CtrlFenceI": "ctrl_fencei",
    }[name]
