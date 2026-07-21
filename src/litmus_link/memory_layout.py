from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

from .litmus_ir import MemoryAccess


MEMORY_LAYOUT_MODES = ("aligned", "misaligned", "mixed", "atomic", "atomic_mixed")
MISALIGNED_BOUNDARIES = ("same16", "cross16", "cross64")
MISALIGNED_WIDTHS = (2, 4, 8)
ATOMIC_OVERLAPS = ("same_start", "partial_overlap")


@dataclass(frozen=True)
class MemoryLayoutConfig:
    mode: str = "aligned"
    width_bytes: int = 4
    boundary: str = "same16"
    overlap: str = "same_start"
    width_pattern: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in MEMORY_LAYOUT_MODES:
            raise ValueError(f"unknown scalar memory layout mode: {self.mode}")
        if self.width_bytes not in MISALIGNED_WIDTHS:
            raise ValueError(f"misaligned scalar width must be 2, 4, or 8 bytes: {self.width_bytes}")
        if self.boundary not in MISALIGNED_BOUNDARIES:
            raise ValueError(f"unknown misaligned boundary: {self.boundary}")
        if self.overlap not in ATOMIC_OVERLAPS:
            raise ValueError(f"unknown atomic overlap shape: {self.overlap}")
        if self.width_pattern and any(width not in MISALIGNED_WIDTHS for width in self.width_pattern):
            raise ValueError("width_pattern must contain only 2, 4, or 8 bytes")
        if self.mode == "atomic_mixed":
            if len(self.width_pattern) < 2:
                raise ValueError("atomic_mixed requires at least two widths in width_pattern")
            if len(set(self.width_pattern)) < 2:
                raise ValueError("atomic_mixed requires at least two distinct widths")

    @property
    def is_aligned(self) -> bool:
        return self.mode in {"aligned", "atomic", "atomic_mixed"}

    @property
    def is_atomic_mixed(self) -> bool:
        return self.mode == "atomic_mixed"

    @property
    def id(self) -> str:
        if self.mode == "aligned":
            return "aligned-w32"
        if self.mode == "atomic":
            return f"atomic-w{self.width_bytes * 8}"
        if self.mode == "atomic_mixed":
            pattern = "-".join(str(width * 8) for width in self.width_pattern)
            return f"atomic-mixed-{pattern}-{self.overlap}"
        if self.mode == "mixed":
            pattern = "-".join(
                str(width * 8) for width in (self.width_pattern or MISALIGNED_WIDTHS)
            )
            width = f"mixed-{pattern}"
        else:
            width = f"w{self.width_bytes * 8}"
        return f"{self.mode}-{width}-{self.boundary}-no-mag"

    def size_for_event(self, event_ordinal: int) -> int:
        if self.mode == "atomic_mixed":
            if event_ordinal >= len(self.width_pattern):
                raise ValueError(
                    f"layout {self.id} has no width for event ordinal {event_ordinal}"
                )
            return self.width_pattern[event_ordinal]
        if self.mode == "mixed":
            pattern = self.width_pattern or MISALIGNED_WIDTHS
            return (
                pattern[event_ordinal]
                if event_ordinal < len(pattern)
                else pattern[event_ordinal % len(pattern)]
            )
        if self.mode == "atomic":
            return self.width_bytes
        return self.width_bytes

    def access_for(self, base_symbol: str, event_ordinal: int) -> MemoryAccess:
        if self.mode == "aligned":
            return MemoryAccess.create(base_symbol, 0, 4)
        if self.mode == "atomic":
            return MemoryAccess.create(base_symbol, 0, self.width_bytes)
        if self.mode == "atomic_mixed":
            size = self.size_for_event(event_ordinal)
            if self.overlap == "same_start":
                offset = 0
            else:
                # Keep every access naturally aligned while placing the
                # 2/4/8-byte footprints inside one shared 8-byte region.
                offset = {2: 6, 4: 4, 8: 0}[size]
            access = MemoryAccess.create(base_symbol, offset, size, "mixed_size_atomic")
            if not access.natural_aligned or access.boundary != "same16":
                raise AssertionError(f"invalid atomic mixed layout {self.id}: {access}")
            return access
        size = self.size_for_event(event_ordinal)
        anchor = {"same16": 7, "cross16": 15, "cross64": 63}[self.boundary]
        # Every generated interval contains the anchor.  For cross16/cross64 it
        # also contains anchor+1, so the selected instruction really crosses
        # the requested boundary.  All offsets are non-natural alignments.
        offset = anchor - (size // 2 - 1)
        access = MemoryAccess.create(base_symbol, offset, size)
        expected_boundary = {
            "same16": "same16",
            "cross16": "cross16_same_line",
            "cross64": "cross64",
        }[self.boundary]
        if access.natural_aligned or access.boundary != expected_boundary:
            raise AssertionError(f"invalid generated layout {self.id}: {access}")
        return access

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "width_bytes": None if self.mode in {"mixed", "atomic_mixed"} else self.width_bytes,
            "mixed_width_bytes": list(self.width_pattern or MISALIGNED_WIDTHS)
            if self.mode in {"mixed", "atomic_mixed"} else [],
            "width_pattern_bytes": [width * 8 for width in self.width_pattern],
            "width_domain_bytes": sorted({width * 8 for width in self.width_pattern})
            if self.width_pattern else [],
            "boundary": self.boundary,
            "overlap": self.overlap,
            "atomicity_model": (
                "mixed_size_atomic"
                if self.mode == "atomic_mixed"
                else "aligned_atomic" if self.is_aligned else "byte_level_no_mag"
            ),
            "mag_bytes": None,
        }


ALIGNED_LAYOUT = MemoryLayoutConfig()


def expand_memory_layouts(
    modes: Sequence[str] = ("aligned",),
    widths: Sequence[int] = MISALIGNED_WIDTHS,
    boundaries: Sequence[str] = MISALIGNED_BOUNDARIES,
    atomic_overlaps: Sequence[str] = ATOMIC_OVERLAPS,
    event_count: int | None = None,
) -> tuple[MemoryLayoutConfig, ...]:
    selected_modes = tuple(dict.fromkeys(str(value) for value in modes))
    selected_widths = tuple(dict.fromkeys(int(value) for value in widths))
    selected_boundaries = tuple(dict.fromkeys(str(value) for value in boundaries))
    layouts: list[MemoryLayoutConfig] = []
    if "aligned" in selected_modes:
        layouts.append(ALIGNED_LAYOUT)
    if "misaligned" in selected_modes:
        layouts.extend(
            MemoryLayoutConfig("misaligned", width, boundary)
            for width, boundary in product(selected_widths, selected_boundaries)
        )
    if "mixed" in selected_modes:
        if len(selected_widths) < 2:
            raise ValueError("mixed requires at least two selected widths")
        pattern_length = event_count if event_count is not None else len(selected_widths)
        if pattern_length < 2:
            raise ValueError("mixed event_count must be at least two")
        layouts.extend(
            MemoryLayoutConfig("mixed", 4, boundary, "same_start", pattern)
            for boundary in selected_boundaries
            for pattern in product(selected_widths, repeat=pattern_length)
            if len(set(pattern)) >= 2
        )
    if "atomic" in selected_modes:
        layouts.extend(MemoryLayoutConfig("atomic", width, "same16") for width in selected_widths)
    if "atomic_mixed" in selected_modes:
        selected_overlaps = tuple(dict.fromkeys(str(value) for value in atomic_overlaps))
        unknown_overlaps = set(selected_overlaps) - set(ATOMIC_OVERLAPS)
        if unknown_overlaps:
            raise ValueError(f"unknown atomic overlap shape: {', '.join(sorted(unknown_overlaps))}")
        if len(selected_widths) < 2:
            raise ValueError("atomic_mixed requires at least two selected widths")
        pattern_length = event_count if event_count is not None else len(selected_widths)
        if pattern_length < 2:
            raise ValueError("atomic_mixed event_count must be at least two")
        layouts.extend(
            MemoryLayoutConfig("atomic_mixed", 4, "same16", overlap, pattern)
            for pattern in product(selected_widths, repeat=pattern_length)
            if len(set(pattern)) >= 2
            for overlap in selected_overlaps
        )
    unknown = set(selected_modes) - set(MEMORY_LAYOUT_MODES)
    if unknown:
        raise ValueError(f"unknown scalar memory layout modes: {', '.join(sorted(unknown))}")
    if not layouts:
        raise ValueError("at least one scalar memory layout must be selected")
    return tuple(layouts)


def all_accesses_overlap(accesses: Iterable[MemoryAccess]) -> bool:
    byte_sets = [set(access.covered_bytes) for access in accesses]
    return bool(byte_sets) and bool(set.intersection(*byte_sets))
