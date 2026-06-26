# Instruction Families

Instruction families are represented in generated metadata through `requires` and axes.

- Scalar: RV64I load/store and optional A-extension AMO cases.
- Vector: V memory instructions with legal `vsetvli` setup. Fault-only-first coverage is limited to standard unit-stride `vle*ff.v` and unit-stride segment `vlseg*ff.v` loads; strided or indexed FOF forms are not valid RISC-V Vector instructions and are not accepted as generation axes.
- CMO: Zicbom and Zicboz operations.
- VM/TLB: S-mode, Sv39, PTE update, `satp`, and `sfence.vma` scenarios.
- Ifetch: Zifencei and self/cross-modifying-code observations.
