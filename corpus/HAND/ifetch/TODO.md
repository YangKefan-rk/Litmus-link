# Ifetch HAND TODO

- Cross-hart CMC where writer performs data store and executing hart performs `fence.i`.
- Code page remap with `sfence.vma` and `fence.i`, based on the RISC-V ifetch VM sketch.
- Patched instruction crossing fetch block or page boundary.
- CMO plus `fence.i` sequences for code visibility.
