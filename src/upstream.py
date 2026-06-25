from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


PATTERNS = {
    "riscv": "*.litmus",
    "ifetch": "*.litmus",
    "aarch64-vmsa": "*.litmus.toml",
}


def import_upstream(src: Path, kind: str, out_dir: Path) -> Dict[str, object]:
    if kind not in PATTERNS:
        raise ValueError(f"unknown upstream kind: {kind}")
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob(PATTERNS[kind]))
    entries: List[Dict[str, str]] = []
    for file_path in files:
        entries.append({"path": str(file_path.relative_to(src)), "name": file_path.name})
    index = {"schema": "litmus-link.upstream-index.v1", "kind": kind, "src": str(src), "count": len(entries), "entries": entries}
    (out_dir / f"{kind}-index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index
