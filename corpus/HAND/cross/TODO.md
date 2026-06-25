# Cross HAND TODO

- Vector footprint crosses one cacheable page and one PBMT=NC page.
- PTE attribute switches between cacheable and PBMT=NC before/after CBO.
- Vector store followed by CBO and remote scalar observer.
- CMO plus remote shootdown plus `fence.i` for executable page remap.
- Alias page with vector indexed access where elements hit different attributes.
