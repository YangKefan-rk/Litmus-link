# Axes

Generation axes are intentionally finite and machine-audited.

- `skeleton`: MP, LB, SB, WRC, RWC, IRIW, ISA2, R, S, Co.
- `attribute`: cacheable, PBMT=NC, PBMT=IO, NC alias, cacheable/NC alias, reserved PBMT.
- `vector`: unit, strided, indexed ordered/unordered, segment, FOF, cross-page.
- `cmo`: clean, flush, inval, inval-as-flush, zero, alias sync, negative CSR/offset variants.
- `tlb`: local/remote sfence, PTE remap, permission fault, A/D update, ASID/global, satp switch.

Use `litmus-link list axes` for the executable list.
