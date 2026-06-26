# Profiles

Profiles define finite generation domains. A profile is not an unbounded Cartesian product; every axis value is declared and every expanded combination must be classified as generated, excluded, HAND-required, or missing.

Implemented profiles:

- `smoke`: small generated corpus for tests and examples.
- `rvwmo_base`: scalar RVWMO skeletons.
- `vector_mem`: vector memory operations crossed with memory attributes.
- `cmo_pbmt`: CMO operations crossed with PBMT/cacheability attributes.
- `vm_tlb`: PTE/TLB/sfence.vma scenarios.
- `full-cross`: combined audit profile covering CMO/PBMT/Vector/TLB interactions.
- `stress-large`: practical large multicore profile. Current baseline is 250,360 combinations, 39,840 generated `.litmus`, 205,640 HAND-required, 4,880 illegal, and 0 missing.
- `stress-all`: exhaustive stress profile. Current baseline is 3,890,180 combinations, 1,492,560 generated `.litmus`, 2,309,140 HAND-required, 88,480 illegal, and 0 missing.

Large profiles should usually be audited with `--summary-only` first:

```sh
PYTHONPATH=src python3 -m cli audit --profile stress-large --summary-only --out out/audit-stress-large
```

The `stress-large` profile is intended for day-to-day large generation. `stress-all` keeps the full multi-million combination domain available when a release or long-running generation job needs it.
