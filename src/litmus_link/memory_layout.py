from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

from .litmus_ir import MemoryAccess


MEMORY_LAYOUT_MODES = ("aligned", "misaligned", "mixed")
MISALIGNED_BOUNDARIES = ("same16", "cross16", "cross64")
MISALIGNED_WIDTHS = (2, 4, 8)


@dataclass(frozen=True)
class MemoryLayoutConfig:
    mode: str = "aligned"
    width_bytes: int = 4
    boundary: str = "same16"

    def __post_init__(self) -> None:
        if self.mode not in MEMORY_LAYOUT_MODES:
            raise ValueError(f"unknown scalar memory layout mode: {self.mode}")
        if self.width_bytes not in MISALIGNED_WIDTHS:
            raise ValueError(f"misaligned scalar width must be 2, 4, or 8 bytes: {self.width_bytes}")
        if self.boundary not in MISALIGNED_BOUNDARIES:
            raise ValueError(f"unknown misaligned boundary: {self.boundary}")

    @property
    def is_aligned(self) -> bool:
        return self.mode == "aligned"

    @property
    def id(self) -> str:
        if self.is_aligned:
            return "aligned-w32"
        width = "mixed16-32-64" if self.mode == "mixed" else f"w{self.width_bytes * 8}"
        return f"{self.mode}-{width}-{self.boundary}-no-mag"

    def size_for_event(self, event_ordinal: int) -> int:
        if self.mode == "mixed":
            return MISALIGNED_WIDTHS[event_ordinal % len(MISALIGNED_WIDTHS)]
        return self.width_bytes

    def access_for(self, base_symbol: str, event_ordinal: int) -> MemoryAccess:
        if self.is_aligned:
            return MemoryAccess.create(base_symbol, 0, 4)
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
            "width_bytes": None if self.mode == "mixed" else self.width_bytes,
            "mixed_width_bytes": list(MISALIGNED_WIDTHS) if self.mode == "mixed" else [],
            "boundary": self.boundary,
            "atomicity_model": "aligned_atomic" if self.is_aligned else "byte_level_no_mag",
            "mag_bytes": None,
        }


ALIGNED_LAYOUT = MemoryLayoutConfig()


def expand_memory_layouts(
    modes: Sequence[str] = ("aligned",),
    widths: Sequence[int] = MISALIGNED_WIDTHS,
    boundaries: Sequence[str] = MISALIGNED_BOUNDARIES,
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
        layouts.extend(MemoryLayoutConfig("mixed", 4, boundary) for boundary in selected_boundaries)
    unknown = set(selected_modes) - set(MEMORY_LAYOUT_MODES)
    if unknown:
        raise ValueError(f"unknown scalar memory layout modes: {', '.join(sorted(unknown))}")
    if not layouts:
        raise ValueError("at least one scalar memory layout must be selected")
    return tuple(layouts)


def all_accesses_overlap(accesses: Iterable[MemoryAccess]) -> bool:
    byte_sets = [set(access.covered_bytes) for access in accesses]
    return bool(byte_sets) and bool(set.intersection(*byte_sets))
