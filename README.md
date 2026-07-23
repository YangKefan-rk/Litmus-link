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

- `litmus-link generate --profile <name> --out <dir>` generates `.litmus`, `.meta.json`, `.solver.json`, `@all`, and `audit-report.json`. PNG diagrams are rendered on demand in the GUI; pass `--diagrams` only when an eager batch is required.
- `litmus-link generate --rule-file <json> --out <dir>` generates from user-defined axes or explicit cases instead of a built-in profile.
- `litmus-link validate <dir-or-@all>` validates index references, metadata, naming, and legality status.
- `litmus-link asm-check <@all> --gcc <tool>` optionally extracts generated instruction bodies and asks a RISC-V assembler to accept them. This is a syntax smoke check, not a litmus semantic proof.
- `litmus-link audit --profile <name>` or `litmus-link audit --rule-file <json>` expands the domain without writing tests and reports generated, excluded, HAND-required, and missing combinations.
- `litmus-link audit --summary-only` skips large detail JSON files and writes only `audit-report.json` plus coverage markdown.
- `litmus-link list profiles|axes|rules|features|hand` prints available profiles, generation axes, legality rules, feature descriptions, or HAND categories.
- `litmus-link qt-gui` starts an optional PyQt/PySide desktop UI when a Qt binding is installed. Its two workflows are `Scalar Litmus` and `Vector Litmus`; preview and generation defer PNG rendering until a case is opened.
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
`P/AMO/Aq/Rl/AR`. `P` lowers to a plain load/store; `AMO` is a relaxed AMO
without an aq/rl suffix; the remaining forms lower to ISA-valid
`amoor.{h,w,d}`/`amoswap.{h,w,d}` variants with the selected ordering bits.
The generator does not emit pseudo `lw.aq`/`sw.rl` instructions.

The exact named-template domain currently contains 284,286,625 cases. MP alone
contains 260,000 cases with the full annotation set. These are finite-grammar counts, not a claim that the
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
is limited to cycle construction: Litmus-link performs its own mixed-size and
misaligned address-layout lowering after the logical cycle has been built.
Explicit LR/SC success/failure scaffolding remains unsupported and is not
silently approximated.

## Scalar Mixed-Size And Misaligned Accesses

The native generator can expand every selected logical cycle over real scalar
mixed-size and misaligned layouts. The target model has no MAG16/MAG32 option.
Ordinary misaligned load/store uses `byte_level_no_mag` and is expanded into
byte sub-events for `rf/co/fr/po-loc`,
while program order, dependencies, and fences are lifted from the parent
instruction to all of its bytes. Torn observations are therefore permitted,
but the split bytes cannot escape RVWMO ordering constraints.

Both `mixed` and `atomic_mixed` are event-level products: every memory event
gets an independent 16/32/64-bit choice, including repeated widths and every
ordering of those choices. Uniform ordinary layouts are already covered by
`misaligned`; uniform aligned atomic layouts are covered by `atomic-w16`,
`atomic-w32`, and `atomic-w64`, so mixed modes keep only patterns containing at
least two widths. `atomic_mixed` additionally expands `same_start` and
`partial_overlap` while preserving natural alignment for every AMO.

Generate MP with all homogeneous 16/32/64-bit misaligned widths, mixed-width
partial overlaps, and within-16-B/cross-16-B/cross-64-B layouts:

```sh
litmus-link native templates --skeleton MP --annotation P \
  --memory-layout aligned \
  --memory-layout misaligned \
  --memory-layout mixed \
  --solver-backend embedded \
  --out out/native-mp-misaligned
```

Select one boundary and width for a targeted run:

```sh
litmus-link native templates --skeleton MP --annotation P \
  --memory-layout misaligned \
  --misalign-width 64 \
  --misalign-boundary cross64 \
  --solver-backend embedded \
  --out out/native-mp-cross64
```

This domain contains only legal cacheable ordinary scalar loads/stores. It does
not generate misaligned LR/SC/AMO, NC/IO fault cases, Vector misalignment, or
exception/CSR checks. Non-plain `AMO/Aq/Rl/AR` AMO annotations are machine-counted
as excluded when combined with a misaligned layout. Fixed-width aligned atomic
layouts and `atomic_mixed` layouts additionally cover naturally aligned
16/32/64-bit AMOs, same-start overlap, and partial overlap. A 16-bit AMO
records `Zabha` in `requires`; mixed-size atomic cases are generated but marked
manual-oracle-required because the embedded solver does not yet model their
overlapping atomic footprint. The generated metadata
records the exact byte range, boundary, `mag_bytes: null`, and
Ordinary no-MAG records `whole_access_atomic: false`; aligned atomic records
`true`. Mixed-size atomic records `true` for each individual footprint, but
their cross-width execution is marked `manual_oracle_required` until a solver
with a mixed-size atomic model is available.

External comparison requests `herd7 -variant mixed,unaligned`. Some herdtools7
revisions, including the locally pinned RISC-V build, parse those variants but
report that mixed mode is not implemented for RISC-V. In that environment use
the offline embedded backend; Litmus-link reports an explicit external-tool
error instead of claiming an `unknown` verdict.

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

The Qt application intentionally contains only two configuration pages: `Scalar Litmus` and `Vector Litmus`. `Scalar Litmus` is backed by the native generator. Named-family mode exhausts the selected skeleton/mechanism/annotation domain; relation-cycle mode ignores family names and enumerates every canonical cycle within the selected size/hart bounds; diy-compatible mode exposes safe/relax/reject lists and the cycle policy. `Vector Litmus` uses the same relation-cycle engine. Its core axes select skeletons, relation mechanisms, endpoint categories (`Vector`, `Scalar`, `AMO`), endpoint compositions (`V only`, `V+S`, `V+A`, `V+S+A`), and all supported RVV forms. Parameter axes independently select scalar B/H/W/D, AMO opcode/W-D/relaxed-aq-rl-aqrl, overlap layout, SEW/LMUL/index EEW/Segment NFIELDS/mask/tail/VL. NFIELDS is active only for Segment forms and illegal `EMUL * NFIELDS > 8` register groups are filtered before generation. The currently implemented verification scope is recorded in each case's metadata and audit report rather than presented as a permanent GUI constraint; future PBMT, alignment, CMO, and TLB axes can be added as their models become available. `Random preview cases` is a reproducible sample from the full finite domain, keyed by `Random seed`; it is not the first N Cartesian-product rows. `Balanced skeleton coverage` assigns near-equal quotas to selected skeleton families before covering endpoint, Vector-form, alignment, and relation-mechanism strata. `Domain-weighted random` instead preserves the natural cardinality of the complete case space. Generation independently offers balanced sampling, domain-weighted sampling, or `All legal combinations`. Exhaustive mode streams every legal assignment under the current filters, ignores the sampled-generation limit, and requires an explicit warning confirmation because a broad configuration can contain an astronomical number of files. The browser uses a virtual table model, supports text/family/status/verdict filters, and does not allocate widgets or PNG images per case. `Preview Cases` is the fast unchecked path; `Verify Preview` runs the selected embedded or herd projection backend where it is applicable. The Vector page's `Basic configuration` section controls the requested solver process count; Embedded batches use 16 local processes by default and accept up to 64. Set `LITMUS_LINK_SOLVER_WORKERS=<n>` to impose a lower hard cap on a shared machine. The verification-effort selector bounds interactive searches; hitting a budget produces `inconclusive`, never a guessed `forbidden` verdict. Double-clicking a case renders its PNG on demand and scales the complete diagram to the detail window. Actions run in the background and update determinate progress whenever the generator has a finite work count.

## Naming

Generated names follow the upstream litmus convention of putting the relation
family first and appending semantic modifiers with `+`. Scalar examples are
`MP`, `MP+addr`, and `MP+fence.rw.rw`. Native relation names retain edge
direction where it is needed to distinguish thread roles, for example
`MP+po.RR+po.WW`. Relation-cycle Vector cases separate their human-readable
name from their file identity. The display form is
`Family+{relation>ring}+E{event:instruction,...}`; for example,
`MP+{PodWW>Rfe>PodRR>Fre}+E{E0:VSE16,E3:VLOXEI32/E16}`. Cycle labels therefore
remain short relation tokens, while the event map shows exactly which cycle
vertices use Vector memory instructions. Vector misalignment appends `U16`,
`X16`, or `X64` to the affected event. Files use
`LLV-<family>-<full SHA-256>.litmus`; the digest covers the canonical cycle,
every endpoint choice, all Vector parameters, and alignment. Metadata retains
both identities and generation aborts rather than allowing two canonical cases
to overwrite the same file. Existing one-endpoint `vector_mem` profile names
are retained only for compatibility with older rule files.

Default generation axes such as `cacheable`, `no_tlb`, and `no_cmo` are not
part of a case name. Non-default architectural features such as `PBMT.NC`,
`cbo.flush`, and `sfence.vma.remote` remain visible. Large stress-only axis
sets use a short stable `Cfg.<id>` suffix; their complete values remain in the
adjacent `.meta.json` and are never discarded.

## Large Profiles

The small profiles are for smoke tests and targeted debugging. The large profiles are intended to cover the multicore stress space across RVWMO skeletons, Vector memory, CMO, PBMT/NC aliases, TLB/VM transitions, and microarchitecture pressure axes. In the Nanhu target profile, Vector memory is restricted to cacheable mappings; PBMT/IO/NC-alias coverage remains in scalar, CMO, and VM paths.

| Profile | Total combinations | Generated `.litmus` | HAND-required | Excluded illegal | Excluded unsupported |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stress-large` | 108,600 | 13,184 | 78,680 | 0 | 16,800 |
| `stress-all` | 1,892,100 | 231,504 | 1,619,060 | 0 | 42,560 |

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
  candidate limit returns `inconclusive`. Misaligned cases use the explicitly
  labeled `riscv.cat+byte_level_no_mag` extension and byte-addressed events.
- **External comparison** uses `herd7 + riscv.cat`. Select `crosscheck` to
  compare both implementations. Startup probes check actual scalar, W/D AMO,
  and `-variant mixed` semantics; an unsupported parser/model capability is
  reported as `external_unsupported`, not as a solver conflict. Only two
  completed, opposite verdicts produce `conflict`.
  The embedded backend currently consumes Litmus-link `case_ir` metadata;
  arbitrary upstream `.litmus` files without that metadata must use the
  `herd7` backend.
- **The Vector-aware embedded frontend** expands active RVV elements into an
  explicit execution graph and reuses the RVWMO axiom engine. The supported
  scope is unit-stride, strided, indexed-unordered, indexed-ordered, and their
  non-FOF Segment load/store forms on cacheable memory, with `SEW`, `LMUL`, deterministic
  `vl1/vl2/vl4/vl8/vl16/vl32/vl64/vlmax`, indexed `EEW=ei8/ei16/ei32/ei64`,
  Segment `NFIELDS=2..8`, and an explicit even-element mask. Nanhu `VLEN=128`
  is part of the model. Segment instructions expand to `eN.fM` transactions;
  a mask disables all fields of a segment element. Fields within one element
  remain unordered, while ordered-indexed Segment forms order every field of an
  earlier element before every field of a later element.
  The legacy built-in `vector_mem` profile covers all ten named scalar skeletons,
  every load/store endpoint, all 8 supported load/store forms, legal
  `SEW`/`LMUL` pairs, all four indexed EEWs, `unmasked/masked`, and the finite
  `vl` boundary domain above. Combinations
  whose active footprint crosses 64B are excluded from this same-line formal
  profile. The current domain also covers all four legal tail policies and
  contains 609,840 combinations and 1,829,520 files after the three ordering
  variants are expanded. The Qt `Vector Litmus` workflow supersedes that
  one-endpoint profile with multi-endpoint relation-cycle generation.
- **Vector external checking uses scalar element projections.** Stock
  `herd7/riscv.cat` never receives RVV syntax. `VL=1` projections and bounded
  unordered-element permutations are exact differential oracles; multi-element
  ordered-indexed projections are advisory because ordinary scalar `po` is
  stronger than isolated Vector sibling order. Segment projection is reported
  as `external_unsupported`, because scalar program order would incorrectly
  order fields from one instruction. Unsupported mixed-size or AMO
  capabilities retain the embedded result and report `external_unsupported`.
  FOF/fault trimming, segment partial
  completion, nonzero `vstart`/restart, whole-register transfer, complex
  indexed aliases, PBMT/PMA, and Vector+CMO/TLB interactions remain outside the
  formal solver and cannot produce a verified forbidden claim.
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
