# Profiles

Profiles define finite generation domains. A profile is not an unbounded Cartesian product; every axis value is declared and every expanded combination must be classified as generated, excluded, HAND-required, or missing.

Implemented profiles:

- `smoke`: small generated corpus for tests and examples.
- `rvwmo_base`: scalar RVWMO skeletons.
- `vector_mem`: vector memory operations crossed with memory attributes.
- `cmo_pbmt`: CMO operations crossed with PBMT/cacheability attributes.
- `vm_tlb`: PTE/TLB/sfence.vma scenarios.
- `full-cross`: combined audit profile covering CMO/PBMT/Vector/TLB interactions.
