# Litmus-link

Litmus-link is a native RISC-V litmus-test generator plus an extension-scenario generator for Vector memory operations, Zicbom/Zicboz CMO instructions, Svpbmt/PBMT=NC and NC aliases, TLB/page-table interactions, and cross combinations of those features.

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
git submodule update --init nexus-am
litmus-link list profiles
litmus-link native catalog
litmus-link native diy --out out/native-diy
litmus-link verify out/native-diy/@all --backend embedded
litmus-link native templates --skeleton MP --no-judge --out out/native-mp
litmus-link validate out/native-mp/@all
litmus-link generate --profile smoke --out out/smoke
litmus-link validate out/smoke/@all
litmus-link audit --profile stress-large --summary-only --out out/audit-stress-large
litmus-link generate --rule-file examples/rules/vector-cmo.json --out out/custom
litmus-link asm-check out/custom/@all --gcc auto
litmus-link qt-gui --check
```

The code is compatible with Python 3.10 for local bring-up. Python 3.11+ is recommended for future development and CI.

## Commands

- `litmus-link generate --profile <name> --out <dir>` generates `.litmus`, `.meta.json`, `.solver.json`, diagram files when an IR is available, `@all`, and `audit-report.json`.
- `litmus-link generate --rule-file <json> --out <dir>` generates from user-defined axes or explicit cases instead of a built-in profile.
- `litmus-link validate <dir-or-@all>` validates index references, metadata, naming, and legality status.
- `litmus-link asm-check <@all> --gcc <tool>` optionally extracts generated instruction bodies and asks a RISC-V assembler to accept them. This is a syntax smoke check, not a litmus semantic proof.
- `litmus-link audit --profile <name>` or `litmus-link audit --rule-file <json>` expands the domain without writing tests and reports generated, excluded, HAND-required, and missing combinations.
- `litmus-link audit --summary-only` skips large detail JSON files and writes only `audit-report.json` plus coverage markdown.
- `litmus-link list profiles|axes|rules|features|hand` prints available profiles, generation axes, legality rules, feature descriptions, or HAND categories.
- `litmus-link qt-gui` starts an optional PyQt/PySide desktop UI when a Qt binding is installed. Custom-rule generation computes solver results and PNG diagrams by default; use the advanced defer switch only for very large corpus dumps.
- `litmus-link import-upstream --src <repo> --kind riscv|ifetch|aarch64-vmsa --out <dir>` writes a compact index of upstream tests without copying the corpus.
- `litmus-link native templates` exhausts the configured variants of named scalar skeletons without invoking diy7/diycross7 or reading an existing corpus.
- `litmus-link native enumerate` enumerates every canonical cycle in a user-bounded native relation grammar.
- `litmus-link native diy` applies a native safe/relax/reject/prefix cycle strategy compatible with the supported RISC-V `diy7` domain.
- `litmus-link verify <file|dir|@all> --backend embedded|herd7|crosscheck` checks each generated `exists` outcome and can update solver metadata with `--write`.
- `litmus-link native catalog` prints the native grammar and exact finite-domain counts.
- `litmus-link scalar cross` generates named scalar skeleton families with `diycross7`.
- `litmus-link scalar enumerate` enumerates bounded scalar cycles with `diy7`.
- `litmus-link scalar tools|catalog` is the legacy/reference herdtools path and is not used by the default GUI generator.

## Native Scalar Generation

The default scalar path is implemented inside Litmus-link. It does not invoke
`diy7`/`diycross7` and does not copy or index an existing corpus. The native
pipeline performs relation expansion, direction matching, hart and location
constraint solving, rotation canonicalization, RISC-V register/address/value
allocation, dependency lowering, `exists` construction, and file rendering.

The current finite grammar covers external/internal `rf/fr/co`, different- and
same-location `po`, all nonempty R/W fence predecessor/successor subsets,
address/data/control/control+`fence.i` dependencies, and event annotations
`P/Aq/Rl/AR`. Non-plain annotations lower to ISA-valid `amoor.w`/`amoswap.w`
forms and declare the A extension; the generator does not emit pseudo
`lw.aq`/`sw.rl` instructions.

The exact named-template domain currently contains 74,873,888 cases. MP alone
contains 106,496 cases. These are finite-grammar counts, not a claim that the
set of all possible programs is finite. Audit metadata records the selected
grammar, base cycles, accepted cases, canonical duplicates, and every rejected
constraint class.

Generate the complete native MP domain without any external process:

```sh
litmus-link native templates --skeleton MP --no-judge --out out/native-mp
litmus-link validate out/native-mp/@all
```

Limit files while retaining the full-domain audit count:

```sh
litmus-link native templates --skeleton MP --limit 100 --no-judge --out out/native-mp-100
```

Enumerate cycles independently of named MP/LB/SB templates:

```sh
litmus-link native enumerate --min-size 2 --size 4 --nprocs 2 \
  --mechanism po --mechanism fence --mechanism dependency \
  --limit 1000 --no-judge --out out/native-cycles
```

Use the native diy-compatible safe/relax strategy, including fixed prefixes:

```sh
litmus-link native diy \
  --safe 'Rfe,Fre,Wse,Fence.rw.rwd**,DpAddrdR,DpAddrdW,DpDatadW' \
  --relax 'PodRR,PodRW,PodWR,PodWW' \
  --prefix 'PodWW Rfe' \
  --solver-backend embedded \
  --out out/native-diy
litmus-link verify out/native-diy/@all --backend crosscheck --write
```

By default, native generation uses Litmus-link's offline RVWMO execution-graph
solver. Select `--solver-backend herd7` to use the official model, or
`--solver-backend crosscheck` to require both implementations to agree. The
legacy `scalar cross/enumerate`
commands remain available for comparison, but the Qt GUI does not use them.

The native diy strategy has cycle-set differential tests against official
`diy7` for all ten generation modes, exact size/hart bounds, mixed tested
relaxations, observer policies, and fixed prefixes. This compatibility claim
is limited to the current scalar, naturally aligned RISC-V edge domain.
Mixed-size/partial-overlap atoms and explicit LR/SC success/failure scaffolding
are not silently approximated; they remain unsupported until their lowering,
outcomes, and RVWMO execution events can be checked end to end.

## GUI

The project has one GUI implementation: the local Qt desktop application. Install one Qt binding and run:

```sh
python3 -m pip install PyQt6
litmus-link qt-gui
```

Check Qt availability with:

```sh
litmus-link qt-gui --check
```

The Qt window opens on the machine where the command runs. On a server, use X forwarding or a remote desktop session. The GUI does not open a network socket.

The Qt GUI opens on `Scalar Litmus`, backed by the native generator. Named-family mode exhausts the selected skeleton/mechanism/annotation domain; relation-cycle mode ignores family names and enumerates every canonical cycle within the selected size/hart bounds; diy-compatible mode exposes safe/relax/reject lists and the cycle policy. `Generate the complete accepted domain` has no hidden file cap, while `Maximum preview rows` limits only the scrollable preview. Embedded RVWMO verification is enabled by default and works on a closed server; the GUI can instead use herd7 or require a cross-check. Actions run in the background and update the status bar, progress indicator, log, summary, and case-inspector tabs.

## Large Profiles

The small profiles are for smoke tests and targeted debugging. The large profiles are intended to cover the multicore stress space across RVWMO skeletons, Vector memory, CMO, PBMT/NC aliases, TLB/VM transitions, and microarchitecture pressure axes.

| Profile | Total combinations | Generated `.litmus` | HAND-required | Excluded illegal |
| --- | ---: | ---: | ---: | ---: |
| `stress-large` | 250,360 | 38,400 | 205,640 | 4,880 |
| `stress-all` | 3,890,180 | 1,489,728 | 2,309,140 | 88,480 |

Use `stress-large` as the practical large profile. Use `stress-all` only when you intentionally want the multi-million combination domain. Start with summary audit before generating files:

```sh
litmus-link audit --profile stress-large --summary-only --out out/audit-stress-large
litmus-link audit --profile stress-all --summary-only --out out/audit-stress-all
```

After generation, `make asm-check OUT=out/smoke` can be used when a RISC-V GCC is available. The target uses `--gcc auto`, so it will try common RISC-V compiler names and otherwise report a clean skip.

Every generated `.meta.json` includes a `test_description` section that explains the selected skeleton, feature axes, and stress knobs. Use `litmus-link list features` to inspect the description catalog.

## User Rule Files

Rule files let users define a bounded generation domain without editing Python. The input is JSON with `name`, optional `defaults`, cross-product `axes`, optional `param_axes`, explicit `cases`, optional `exclude` patterns, and a `limit` guardrail. All expanded combinations still pass through the same ISA legality and RVWMO classification rules as built-in profiles.

Use `litmus-link list axes` to see accepted values. A minimal rule file can be as small as:

```json
{
  "name": "my-cmo-smoke",
  "axes": {
    "cmo": ["flush", "zero"],
    "attribute": ["cacheable", "pbmt_nc"]
  },
  "param_axes": {
    "footprint": ["same_line", "cross_page"],
    "sync": ["none", "post_fence"],
    "stress": ["none", "store_buffer_full"]
  },
  "limit": 20
}
```

See `examples/rules/vector-cmo.json` for a larger example that combines Vector and CMO axes.

## Design Boundary

Verification is split by semantic scope:

- **Pure scalar main-memory tests** are checked by the independent execution-
  graph solver in `src/litmus_link/rvwmo_solver.py`. It enumerates `rf` and
  per-location `co`, derives `fr`, implements RISC-V PPO rules `r1-r13`, and
  checks the Coherence, Model, and Atomic axioms from `riscv.cat`. A forbidden
  verdict is returned only after exhaustive bounded search; a timeout or
  candidate limit returns `inconclusive`.
- **External comparison** uses `herd7 + riscv.cat`. Select `crosscheck` to
  require both implementations to return the same verified `allowed` value.
  The embedded backend currently consumes Litmus-link `case_ir` metadata;
  arbitrary upstream `.litmus` files without that metadata must use the
  `herd7` backend.
- **The simple MP vector-memory subset** can be judged by scalar element
  lowering when the selected axes are exactly renderable by the current IR.
  More complex Vector parameters such as cross-page footprints, masks, non-base
  SEW/LMUL/VL, CMO, PBMT aliases, `FENCE.I`, and `SFENCE.VMA` interactions are
  emitted as hardware-observation or prose-spec cases, not formal forbidden
  claims.
- **Unrendered stress axes** are not counted as formal coverage. For example,
  `dep=data/aq/rl/aqrl` is reported as unsupported until a real instruction body
  exists for that relation shape.

## Repository Layout

```text
src/litmus_link/   Python package: CLI, rules, generation, solvers, diagrams, Qt GUI
src/cli.py         Compatibility entry point for PYTHONPATH=src python3 -m cli
examples/rules/    User-editable JSON generation rule examples
tests/             Unit tests and machine-checked profile baselines
nexus-am/          Git submodule reserved for the future ELF build backend
out/               Generated corpora and audit output; ignored by Git
```

`nexus-am` is intentionally kept as a submodule. It is not yet called by the
current `.litmus` generation path; the planned ELF backend will use it as the
runtime/build harness. Clone with `--recurse-submodules`, or initialize it later
with `git submodule update --init nexus-am`.
