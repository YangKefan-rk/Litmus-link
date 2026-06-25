# Full-Cross Audit Baseline

This baseline is produced by:

```sh
litmus-link audit --profile full-cross --out out/audit
```

Current expected counts:

| Field | Count |
|---|---:|
| Total combinations | 186 |
| Generated | 85 |
| Excluded illegal | 51 |
| Excluded unsupported | 0 |
| HAND-required | 50 |
| Missing | 0 |

`Missing` must remain zero. If a profile change alters these counts, update this file in the same change that updates the rule tests.
