# Upstream Sources

Litmus-link references existing public litmus repositories without copying their full corpora.

## litmus-tests-riscv

- Provides standard RISC-V `.litmus` syntax, `@all` recursive indexes, and RVWMO relation naming.
- Generated tests in Litmus-link preserve the standard `RISCV <name>` header and simple thread-table layout where possible.

## litmus-tests-riscv-ifetch

- Provides RISC-V instruction-fetch shapes involving patched code, `fence.i`, and one `sfence.vma` virtual-memory sketch.
- Litmus-link uses those as references for `ifetch` and VM/ifetch HAND scenarios.

## litmus-tests-armv8a-system-vmsa

- Provides page-table object modeling in `.litmus.toml`, including BBM, aliasing, TLB invalidation, and exception handlers.
- Litmus-link reuses the organization idea and converts it into RISC-V concepts: PTE writes, `sfence.vma`, `satp`, PBMT, local/remote shootdown, and trap handlers.
