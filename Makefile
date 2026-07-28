PYTHON ?= python3
PROFILE ?= smoke
OUT ?= out/$(PROFILE)

.PHONY: test smoke scalar-smoke audit audit-large validate clean asm-check herd7 verify

test:
	@if $(PYTHON) -c 'import pytest' >/dev/null 2>&1; then \
		PYTHONPATH=src $(PYTHON) -m pytest; \
	else \
		PYTHONPATH=src $(PYTHON) tests/run_tests.py; \
	fi

herd7:
	@if command -v herd7 >/dev/null; then \
		echo "herd7 already installed: $$(command -v herd7)"; \
	else \
		command -v opam >/dev/null || { echo "opam not found. Install OCaml/opam first."; exit 1; }; \
		opam install -y herdtools7 && \
			echo "herd7 installed. Re-run 'make smoke' to cross-validate scalar verdicts against riscv.cat."; \
	fi

verify:
	PYTHONPATH=src $(PYTHON) -m litmus_link generate --profile smoke --out out/smoke
	@$(PYTHON) -c "import json; from pathlib import Path; \
root=Path('out/smoke'); entries=(root/'@all').read_text(encoding='utf-8').splitlines(); \
rows=[json.loads((root/entry).with_suffix('.solver.json').read_text(encoding='utf-8')) for entry in entries]; \
ag=sum(r.get('cross_check')=='agree' for r in rows); \
cf=sum(r.get('status')=='conflict' for r in rows); \
na=sum(r.get('status')=='not_applicable' for r in rows); \
print(f'native verdicts: {sum(r[\"status\"]==\"verified\" for r in rows)}  herd7-agree: {ag}  conflict: {cf}  fusion(not_applicable): {na}'); \
exit(1 if cf else 0)"


smoke:
	rm -rf out/smoke
	PYTHONPATH=src $(PYTHON) -m litmus_link generate --profile smoke --out out/smoke
	PYTHONPATH=src $(PYTHON) -m litmus_link validate out/smoke/@all

scalar-smoke:
	rm -rf out/scalar-smoke
	PYTHONPATH=src $(PYTHON) -m litmus_link scalar cross --skeleton MP --mechanism po --out out/scalar-smoke
	PYTHONPATH=src $(PYTHON) -m litmus_link validate out/scalar-smoke/@all

audit:
	mkdir -p out/audit
	PYTHONPATH=src $(PYTHON) -m litmus_link audit --profile full-cross --out out/audit
	test ! -s out/audit/missing.json

audit-large:
	mkdir -p out/audit-stress-large
	PYTHONPATH=src $(PYTHON) -m litmus_link audit --profile stress-large --summary-only --out out/audit-stress-large

validate:
	PYTHONPATH=src $(PYTHON) -m litmus_link validate $(OUT)/@all

asm-check:
	PYTHONPATH=src $(PYTHON) -m litmus_link asm-check $(OUT)/@all --gcc auto

clean:
	rm -rf out .pytest_cache
