# RVWMO verification

Litmus-link decides the allowed/forbidden outcome of generated tests with two
layers. The split exists because the standard formal model only covers part of
what this project generates.

## Layer 1 — native axiomatic checker (primary)

`src/rvwmo.py` is a small, faithful checker for **pure scalar RVWMO** cases
(the `model == "rvwmo"` cycle tests: MP/LB/SB/WRC/RWC/IRIW and their ordering
variants). It needs no external tool and always renders a verdict.

Principle: each scalar case IR encodes exactly one critical cycle that
alternates local program-order (`po`) edges with external communication edges
(`rfe`/`fre`/`co`). Under RVWMO's Global Memory Order acyclicity rule the
test's `exists` (weak) outcome is **forbidden iff every `po` edge of that cycle
is a preserved (`ppo`) edge** — because then `ppo ∪ comm` is cyclic, which
RVWMO disallows. If any `po` edge is not preserved, the cycle is broken and the
outcome is **allowed**.

The whole problem therefore reduces to a PPO oracle over consecutive same-hart
memory operations. The implemented rules, each tagged with its number from the
RISC-V unprivileged spec ("RVWMO Memory Consistency Model", PPO list):

| Rule | Mechanism |
|------|-----------|
| 1–3  | overlapping-address ordering |
| 4    | `FENCE` with matching predecessor/successor sets |
| 9    | address dependency from a load (orders load→{load,store}) |
| 10   | data dependency from a load to a store |
| 11   | control dependency from a load to a store (**store only**) |

`fence.i` is deliberately **not** a data fence: RVWMO gives it no data-memory
ordering power, so "control dep + `fence.i`" orders exactly what a bare control
dependency orders. Every edge in the result carries its `preserved` flag, the
rule that ordered it, and a human-readable reason for audit.

## Layer 2 — herd7 cross-validation (optional)

If `herd7` is on `PATH`, `src/solver.py` runs it (`herd7 -model riscv.cat`) on
the same scalar case and compares:

- **agree** — native and herd7 give the same verdict (`cross_check: "agree"`,
  badge shows `(native+herd7)`).
- **conflict** — they disagree (`status: "conflict"`). The native verdict is
  still reported as primary, but both raw outputs are retained so the
  disagreement can be investigated. `make verify` exits non-zero on any
  conflict.
- **absent / error** — herd7 missing or failed; the native verdict stands
  (`cross_check: "herd7_absent"` / `"herd7_error"`).

### Installing herd7

```
make herd7        # opam install herdtools7
```

This needs OCaml + opam. On Debian/Ubuntu:

```
sudo apt-get install opam
opam init -y && eval $(opam env)
make herd7
make verify       # regenerate smoke corpus and cross-check
```

herd7 ships its own `riscv.cat`; nothing extra is vendored here.

## Fusion scenarios — extension-prose analysis (never a herd verdict)

Stock RVWMO / `riscv.cat` models **only scalar main memory**. It says nothing
about the vector (V), cache-management (Zicbom/Zicboz), page-based memory types
(Svpbmt), instruction-fetch (Zifencei) or address-translation (`sfence.vma`)
extensions that Litmus-link fuses into tests. Claiming a herd-forbidden verdict
for those would be fabrication.

`src/fusion.py` therefore produces an **informative, model-extended** result
for non-scalar cases, with citations to the relevant extension prose. It
reports whether the test contains the synchronisation those extensions
*document* for its features, and **always** keeps `allowed = None` and
`formal_forbidden_claim = False`.

| Verdict | Meaning |
|---------|---------|
| `ordering-documented` | every orderable feature carries its documented barrier (`FENCE` for NC/vector ordering; `fence;cbo.flush;fence` for an NC alias). The producer effect is ordered before observation. |
| `ordering-absent` | an orderable feature lacks its barrier, so the reordered outcome is permitted by the prose. This is a hardware-observation: it needs a stress window + coverage instrumentation to exhibit, never a herd assertion. |
| `prose-spec` | rests on a prose-only / hand-required property (cross-hart TLB shootdown) that no local barrier can settle. |

This is exposed on the solver result under the `fusion` field and surfaced in
the diagram badge (`<verdict> (ext-prose)`) and the GUI outcome text.
