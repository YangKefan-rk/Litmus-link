# Litmus-link Current Context

## Goal

Build `Litmus-link` as a RISC-V litmus generation tool that can cover multicore
Vector/CMO/PBMT/NC/TLB/VM/stress scenarios at the scale of hundreds of
thousands to millions of combinations, while filtering ISA-illegal cases and
separating HAND-required cases — and decide allowed/forbidden honestly,
without fabricating ordering the spec does not give.

## Module chain

```
rule/profile -> litmus_ir (LitmusCaseIR variant expansion)
             -> rvwmo (native scalar checker) + fusion (extension-prose analysis)
             -> solver (verdict assembly + optional herd7 cross-check)
             -> diagram (PNG topology) -> gui / qt_gui
```

## Verification architecture (see RVWMO-verification.md)

Two layers, split because stock RVWMO/herd7 only model scalar main memory:

1. **Native axiomatic checker** — `src/rvwmo.py`. Primary verdict source for
   pure scalar RVWMO cycle cases (`model == "rvwmo"`). A scalar test is
   forbidden iff every `po` edge of its critical cycle is a preserved (`ppo`)
   edge. Implements PPO rules 1–3 (same address), 4 (FENCE), 9 (addr dep),
   10 (data dep), 11 (ctrl dep, store-only). `fence.i` is NOT a data fence.
   Each edge carries its rule + reason. Verdict table pinned in
   `tests/test_rvwmo.py`.
2. **herd7 cross-validation** — optional. `src/solver.py` runs
   `herd7 -model riscv.cat` when on PATH and compares: `agree` /
   `conflict` / `herd7_absent` / `herd7_error` (recorded in `cross_check`).
   `make herd7` installs it via opam; `make verify` exits non-zero on conflict.
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

- `src/rvwmo.py` — native scalar RVWMO checker (PPO oracle).
- `src/fusion.py` — extension-prose fusion ordering analysis.
- `src/solver.py` — verdict assembly, herd7 cross-check, `SolverResult`.
- `src/diagram.py` — topology PNG rendering and relation routing.
- `src/litmus_ir.py` — `LitmusCaseIR`, event/relation builders per skeleton.
- `src/generator.py` — generation/audit flow, `solver_counts`.
- `RVWMO-verification.md` — the two-layer verification design.

## Verified commands

```sh
make test       # 62 tests pass (pytest), or tests/run_tests.py fallback
make smoke      # regenerate + validate the 8-combination smoke corpus
make verify     # regenerate smoke + report native/herd7/conflict/fusion counts
make herd7      # opam install herdtools7 (optional cross-validation)
```

Latest smoke solver counts: `verified: 6, conflict: 0, not_applicable: 7`.

## Possible next steps

- Install herd7 (`make herd7`) and run `make verify` to cross-validate the
  native scalar verdicts against riscv.cat; resolve any conflicts.
- Extend the native PPO oracle if new scalar variants (e.g. data deps, AMO/LR-SC
  pairs) are added to the IR.
- Broaden the fusion citation catalog as more extension scenarios are generated.
