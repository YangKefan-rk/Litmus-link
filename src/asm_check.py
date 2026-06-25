from __future__ import annotations

from pathlib import Path
from typing import List


def asm_check(atfile: Path, gcc: str) -> List[str]:
    if not atfile.exists():
        raise FileNotFoundError(atfile)
    return [f"asm-check placeholder: {gcc} available; litmus assembly extraction is not implemented for {atfile}"]
