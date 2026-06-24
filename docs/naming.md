# Naming

Generated names use a stable, machine-readable structure:

```text
LL_<profile>_<skeleton>_<memory-event>_<attribute>_<tlb>_<cmo>_<vector>
```

Rules:

- All names are ASCII and contain only letters, numbers, `_`, `+`, `.`, and `-` after rendering.
- A generated `.litmus` file has a matching `.meta.json` file with the same stem.
- `generated_from` in metadata records the profile, skeleton, axes, and rule decisions.
- HAND cases may use descriptive names but must include metadata with `hand_required=true`.
