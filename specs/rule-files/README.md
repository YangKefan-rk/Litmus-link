# User Rule Files

Users can generate custom finite domains with JSON rule files instead of built-in profiles. Rule files are intentionally data-only: they choose combinations, while the normal legality engine still decides whether each combination is generated, excluded, or moved to HAND.

Example:

```json
{
  "name": "custom-vector-cmo",
  "defaults": {
    "skeleton": "MP",
    "attribute": "cacheable"
  },
  "axes": {
    "vector": ["unit_load", "unit_store"],
    "cmo": ["no_cmo", "flush"]
  },
  "exclude": [
    {"vector": "unit_store", "cmo": "flush"}
  ],
  "limit": 100
}
```

Run it with:

```sh
PYTHONPATH=src python3 -m cli generate --rule-file specs/rule-files/example-vector-cmo.json --out out/custom
PYTHONPATH=src python3 -m cli audit --rule-file specs/rule-files/example-vector-cmo.json --out out/custom-audit
```

Fields:

- `name`: profile name recorded in metadata.
- `defaults`: default values for combination fields.
- `axes`: finite axis values to cross-product.
- `param_defaults`: default values for fine-grained parameter fields stored in `Combination.params`.
- `param_axes`: finite parameter-axis values to cross-product, such as `sew`, `lmul`, `footprint`, `sync`, `vm`, `pte`, or `stress`.
- `cases`: optional explicit cases; can be used together with `axes`.
- `exclude`: optional patterns removed after expansion.
- `limit`: maximum expanded combinations, used as a guardrail.

Supported combination fields are `category`, `skeleton`, `memory_event`, `attribute`, `tlb`, `cmo`, and `vector`. Use `litmus-link list axes` to print accepted values. Use `litmus-link list features` to see common parameter meanings.

Inference rules keep simple files short:

- Setting `vector` infers `memory_event=vector_load` or `vector_store` when `memory_event` is omitted.
- Setting `cmo` infers `memory_event=cmo` when no vector/TLB event already selected it.
- Setting `tlb` infers `memory_event=pte_update` when no vector/CMO event already selected it.
- Combining Vector with CMO or TLB infers `category=cross` unless `category` is explicitly set.
- Setting a non-cacheable attribute without another feature infers `category=pbmt_nc`.

Explicit `cases` may carry a free-form `params` object. Renderers preserve it in `.meta.json`; future backends can use it for addresses, element widths, page layout, or harness controls.

Example with parameter axes:

```json
{
  "name": "my-vector-params",
  "axes": {
    "skeleton": ["MP", "LB"],
    "vector": ["unit_load", "indexed_ordered_load"],
    "attribute": ["cacheable", "pbmt_nc"]
  },
  "param_axes": {
    "sew": ["e32", "e64"],
    "lmul": ["m1", "m2"],
    "footprint": ["same_line", "cross_page"],
    "stress": ["none", "load_queue_replay"]
  },
  "limit": 1000
}
```

Example with explicit cases:

```json
{
  "name": "my-explicit-cases",
  "defaults": {"skeleton": "MP", "attribute": "cacheable"},
  "cases": [
    {"vector": "fof_load", "params": {"fault_element": 3}},
    {"cmo": "flush_sync", "attribute": "cacheable_nc_alias"}
  ],
  "limit": 10
}
```
