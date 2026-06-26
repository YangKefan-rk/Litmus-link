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
litmus-link audit --profile stress-large --summary-only --out out/audit-stress-large
litmus-link generate --rule-file specs/rule-files/example-vector-cmo.json --out out/custom
```

The code is compatible with Python 3.10 for local bring-up. Python 3.11+ is recommended for future development and CI.

## Commands

- `litmus-link generate --profile <name> --out <dir>` generates `.litmus`, `.meta.json`, `@all`, and `audit-report.json`.
- `litmus-link generate --rule-file <json> --out <dir>` generates from user-defined axes or explicit cases instead of a built-in profile.
- `litmus-link validate <dir-or-@all>` validates index references, metadata, naming, and legality status.
- `litmus-link audit --profile <name>` or `litmus-link audit --rule-file <json>` expands the domain without writing tests and reports generated, excluded, HAND-required, and missing combinations.
- `litmus-link audit --summary-only` skips large detail JSON files and writes only `audit-report.json` plus coverage markdown.
- `litmus-link list profiles|axes|rules|features|hand` prints available profiles, generation axes, legality rules, feature descriptions, or HAND categories.
- `litmus-link import-upstream --src <repo> --kind riscv|ifetch|aarch64-vmsa --out <dir>` writes a compact index of upstream tests without copying the corpus.

## Large Profiles

The small profiles are for smoke tests and targeted debugging. The large profiles are intended to cover the multicore stress space across RVWMO skeletons, Vector memory, CMO, PBMT/NC aliases, TLB/VM transitions, and microarchitecture pressure axes.

| Profile | Total combinations | Generated `.litmus` | HAND-required | Excluded illegal |
| --- | ---: | ---: | ---: | ---: |
| `stress-large` | 557,840 | 73,440 | 355,760 | 128,640 |
| `stress-all` | 6,113,560 | 1,599,840 | 2,659,600 | 1,854,120 |

Use `stress-large` as the practical large profile. Use `stress-all` only when you intentionally want the multi-million combination domain. Start with summary audit before generating files:

```sh
litmus-link audit --profile stress-large --summary-only --out out/audit-stress-large
litmus-link audit --profile stress-all --summary-only --out out/audit-stress-all
```

Every generated `.meta.json` includes a `test_description` section that explains the selected skeleton, feature axes, and stress knobs. Use `litmus-link list features` to inspect the description catalog.

## User Rule Files

Rule files let users define a bounded generation domain without editing Python. The input is JSON with `name`, optional `defaults`, cross-product `axes`, explicit `cases`, optional `exclude` patterns, and a `limit` guardrail. All expanded combinations still pass through the same ISA legality and RVWMO classification rules as built-in profiles.

Use `litmus-link list axes` to see accepted values. A minimal rule file can be as small as:

```json
{
  "name": "my-cmo-smoke",
  "axes": {
    "cmo": ["flush", "zero"],
    "attribute": ["cacheable", "pbmt_nc"]
  },
  "limit": 20
}
```

See `specs/rule-files/README.md` and `specs/rule-files/example-vector-cmo.json` for a larger example that combines Vector and CMO axes.

## Design Boundary

Pure scalar main-memory tests can be checked against `herd7 -model riscv.cat`. Vector, CMO, PBMT, `FENCE.I`, and `SFENCE.VMA` interactions are emitted as specification-constrained or hardware-observation tests because those interactions are not fully formalized in the standard RVWMO herd model.
