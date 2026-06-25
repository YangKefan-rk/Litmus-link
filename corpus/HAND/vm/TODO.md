# VM HAND TODO

- BBM: invalid PTE, `sfence.vma`, new PTE, observer load/fetch.
- UpdateFromTLB: stale translation versus updated PTE observation.
- Remote shootdown: hart0 updates PTE, hart1 executes explicit `sfence.vma` after notification.
- ASID/global: stale and synchronized translations across ASID or global mappings.
- `satp` switch: local translation-context switch with explicit fences.

These need privilege setup, page-table allocation, and trap recovery, so they remain HAND until the backend owns a complete VM harness.
