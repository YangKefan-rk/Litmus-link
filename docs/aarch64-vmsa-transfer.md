# AArch64 VMSA Lessons For RISC-V VM Litmus

The AArch64 VMSA corpus is useful because it treats translation state as part of the test, not as hidden setup. Litmus-link adapts that idea to RISC-V.

Useful patterns:

- Page-table setup as data: RISC-V metadata records PTE level, leaf/non-leaf status, PBMT, ASID, and initial validity.
- BBM: RISC-V HAND cases use invalid PTE, `sfence.vma`, new PTE, and observer loads/fetches.
- TLB update races: RISC-V HAND cases distinguish local `sfence.vma` from remote shootdown.
- Alias tests: RISC-V cases model same PA through cacheable and PBMT=NC aliases.
- Exception handlers: vector fault and page-fault HAND cases record `mcause/scause`, `mtval/stval`, `vstart`, and `vl` expectations.

Instruction semantics are not copied from AArch64. RISC-V cases use `sfence.vma`, `fence`, `fence.i`, `satp`, and RISC-V PTE fields.
