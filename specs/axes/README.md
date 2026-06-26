# Axes

Generation axes are intentionally finite and machine-audited.

- `skeleton`: MP, LB, SB, WRC, RWC, IRIW, ISA2, R, S, Co.
- `attribute`: cacheable, PBMT=NC, PBMT=IO, NC alias, cacheable/NC alias.
- `vector`: unit-stride, strided, indexed ordered/unordered, segment, unit-stride FOF, unit-stride segment FOF. Cross-page is a footprint parameter, not a vector instruction.
- `cmo`: clean, flush, inval, zero. Alias synchronization is represented by `params.sync=full_alias_sync`, not by a fake CMO operation.
- `tlb`: local/remote sfence, PTE remap, permission fault, A/D update, ASID/global, satp switch.

Reserved PBMT, non-leaf PBMT, non-zero CBO offsets, CSR-denied CBO, and nonexistent strided/indexed FOF forms remain in the rule engine only as negative/HAND-required checks; they are not public generation axes.

Use `litmus-link list axes` for the executable list.
