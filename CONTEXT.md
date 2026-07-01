# Litmus-link Current Context

## Goal

Build `Litmus-link` as a RISC-V litmus generation tool that can cover multicore
Vector/CMO/PBMT/NC/TLB/VM/stress scenarios at the scale of hundreds of
thousands to millions of combinations, while filtering ISA-illegal cases and
separating HAND-required cases — and decide allowed/forbidden honestly,
without fabricating ordering the spec does not give.

## Module chain

```
rule/profile -> litmus_ir / corpus_riscv (case expansion)
             -> solver + herd7/riscv.cat or fusion extension-prose analysis
             -> diagram (PNG topology) -> gui / qt_gui
             -> optional asm_check syntax smoke test
```

## Verification architecture (see RVWMO-verification.md)

Two layers, split because stock RVWMO/herd7 only model scalar main memory:

1. **herd7 / riscv.cat** — authoritative for pure scalar RVWMO corpus tests
   when the bundled herdtools path is present. `src/corpus_riscv.py` indexes
   the real RISC-V corpus and `src/toolchain.py` strips nondeterministic herd
   timing from stored outputs.
2. **Native axiomatic checker** — `src/rvwmo.py`. Provides edge explanations
   and fallback verdicts for renderable scalar IR. A scalar test is forbidden
   iff every `po` edge of its critical cycle is a preserved (`ppo`) edge.
3. **Fusion (vector/CMO/PBMT/TLB)** — `src/fusion.py`. NEVER a formal
   forbidden claim: always `allowed = None`, `formal_forbidden_claim = False`.
   Reports `ordering-documented` / `ordering-absent` / `prose-spec` with spec
   citations, attached to the solver result under the `fusion` field.

Solver status set: `verified` | `conflict` | `not_applicable`.

## Diagram rendering

`src/diagram.py` routes cross-hart relations as clean curves: adjacent harts
get cubic side curves through the column gap (the Rfe/Fre X crossing), distant
harts get nested over-the-top arches. Relations sharing a box edge are spread
to distinct attachment points so they never run parallel. Colour by kind
(rfe=green, fre=red, co=blue, obs=gray); white label chips; gray program-order
spine. Pure routing lives in `_route_relations()`; invariants pinned in
`tests/test_diagram.py` (in bounds, no box cut, no parallel overlap).

## Important files

- `src/toolchain.py` — herd7/diycross wrappers and deterministic output parsing.
- `src/corpus_riscv.py` — real scalar RVWMO corpus indexing and herd judging.
- `src/rvwmo.py` — native scalar RVWMO checker (PPO oracle / fallback).
- `src/fusion.py` — extension-prose fusion ordering analysis.
- `src/solver.py` — verdict assembly, herd7 cross-check, `SolverResult`.
- `src/diagram.py` — topology PNG rendering and relation routing.
- `src/litmus_ir.py` — `LitmusCaseIR`, event/relation builders per skeleton.
- `src/generator.py` — generation/audit flow, `solver_counts`.
- `src/gui.py` / `src/qt_gui.py` — browser and Qt workflows, including preview
  case lists, solver/diagram summaries, and generation limit reporting.
- `src/asm_check.py` — optional assembler syntax smoke check for generated
  instruction bodies.
- `RVWMO-verification.md` — the two-layer verification design.

## Verified commands

```sh
make test       # pytest suite, or tests/run_tests.py fallback
make smoke      # regenerate + validate the 8-combination smoke corpus
make verify     # regenerate smoke + report native/herd7/conflict/fusion counts
make herd7      # opam install herdtools7 (optional cross-validation)
make asm-check  # syntax-check generated instruction bodies if a RISC-V GCC exists
```

Latest smoke profile counts: `generated: 8`, `generated_litmus: 22`,
`solver.verified: 18`, `solver.conflict: 0`, `solver.not_applicable: 4`.

## Possible next steps

- Install herd7 (`make herd7`) and run `make verify` to cross-validate the
  native scalar verdicts against riscv.cat; resolve any conflicts.
- Extend the native PPO oracle if new scalar variants (e.g. data deps, AMO/LR-SC
  pairs) are added to the IR.
- Broaden the fusion citation catalog as more extension scenarios are generated.
