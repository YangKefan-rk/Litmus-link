# Litmus-link

Litmus-link is a RISC-V litmus-test generator focused on scenarios that are not covered by the public scalar RVWMO corpus: Vector memory operations, Zicbom/Zicboz CMO instructions, Svpbmt/PBMT=NC and NC aliases, TLB/page-table interactions, and cross combinations of those features.

The project intentionally avoids blind Cartesian generation. Every generated test first passes through ISA legality checks, RVWMO classification, and coverage audit accounting. Combinations that are illegal, unsupported, or require hand-written setup are reported instead of silently dropped.

## Reference Inputs

- `litmus-tests-riscv`: `.litmus`, `@all`, and RVWMO relation naming conventions.
- `litmus-tests-riscv-ifetch`: RISC-V instruction-fetch, code patching, `fence.i`, and `sfence.vma` sketches.
- `litmus-tests-armv8a-system-vmsa`: page-table-as-test-data style, including alias, BBM, TLB invalidation, and exception-handler structure. Litmus-link borrows the structure, not AArch64 instruction semantics.

## Quick Start

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
litmus-link list profiles
litmus-link generate --profile smoke --out out/smoke
litmus-link validate out/smoke/@all
litmus-link audit --profile full-cross --out out/audit
```

The code is compatible with Python 3.10 for local bring-up. Python 3.11+ is recommended for future development and CI.

## Commands

- `litmus-link generate --profile <name> --out <dir>` generates `.litmus`, `.meta.json`, `@all`, and `audit-report.json`.
- `litmus-link validate <dir-or-@all>` validates index references, metadata, naming, and legality status.
- `litmus-link audit --profile <name>` expands the profile without writing tests and reports generated, excluded, HAND-required, and missing combinations.
- `litmus-link list profiles|axes|rules|hand` prints available profiles, generation axes, legality rules, or HAND categories.
- `litmus-link import-upstream --src <repo> --kind riscv|ifetch|aarch64-vmsa --out <dir>` writes a compact index of upstream tests without copying the corpus.

## Design Boundary

Pure scalar main-memory tests can be checked against `herd7 -model riscv.cat`. Vector, CMO, PBMT, `FENCE.I`, and `SFENCE.VMA` interactions are emitted as specification-constrained or hardware-observation tests because those interactions are not fully formalized in the standard RVWMO herd model.
