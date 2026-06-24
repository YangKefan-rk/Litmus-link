# Coverage Audit

`litmus-link audit` expands a profile into a finite combination domain and classifies each combination.

Classifications:

- `generated`: legal and supported by the automatic renderer.
- `excluded_illegal`: explicitly illegal by RISC-V ISA or privileged architecture rules.
- `excluded_unsupported`: legal but outside the current generator backend.
- `hand_required`: legal but requires hand-written trap, page-table, or platform setup.
- `missing`: legal and expected to be generated, but no generator path exists. Any `missing` entry fails `make audit`.

The audit report is the project self-check: every defined combination must either be emitted or have a machine-readable reason for exclusion.
