# Legality Rules

The authoritative executable rules live in `src/litmus_link/rules.py` and are exposed by:

```sh
litmus-link list rules
```

Rules cover:

- Svpbmt leaf-only use and reserved PBMT encodings.
- PBMTE and `sfence.vma` synchronization requirements.
- Cacheable/NC alias synchronization requirements.
- CBO offset, xenvcfg, and `cbo.zero` non-atomic behavior.
- Vector FOF legality, vector memory ordering, and non-idempotent memory restrictions.
- Local-only nature of `fence.i` and `sfence.vma`.
- RVWMO scope limits for CMO/PMA/FENCE.I/SFENCE.VMA interactions.
