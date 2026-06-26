PYTHON ?= python3
PROFILE ?= smoke
OUT ?= out/$(PROFILE)

.PHONY: test smoke audit audit-large validate clean asm-check herd7 verify

test:
	@if $(PYTHON) -c 'import pytest' >/dev/null 2>&1; then \
		PYTHONPATH=src $(PYTHON) -m pytest; \
	else \
		PYTHONPATH=src $(PYTHON) tests/run_tests.py; \
	fi

herd7:
	@command -v herd7 >/dev/null && { echo "herd7 already installed: $$(command -v herd7)"; exit 0; } || true
	@command -v opam  >/dev/null || { echo "opam not found. Install OCaml/opam first (see RVWMO-verification.md)."; exit 1; }
	opam install -y herdtools7
	@echo "herd7 installed. Re-run 'make smoke' to cross-validate scalar verdicts against riscv.cat."

verify:
	PYTHONPATH=src $(PYTHON) -m cli generate --profile smoke --out corpus/smoke/generated
	@PYTHONPATH=src $(PYTHON) -c "import json,glob; \
rows=[json.load(open(f)) for f in glob.glob('corpus/smoke/generated/*.solver.json')]; \
ag=sum(r.get('cross_check')=='agree' for r in rows); \
cf=sum(r.get('status')=='conflict' for r in rows); \
na=sum(r.get('status')=='not_applicable' for r in rows); \
print(f'native verdicts: {sum(r[\"status\"]==\"verified\" for r in rows)}  herd7-agree: {ag}  conflict: {cf}  fusion(not_applicable): {na}'); \
exit(1 if cf else 0)"


smoke:
	rm -rf corpus/smoke/generated
	PYTHONPATH=src $(PYTHON) -m cli generate --profile smoke --out corpus/smoke/generated
	PYTHONPATH=src $(PYTHON) -m cli validate corpus/smoke/generated/@all

audit:
	mkdir -p out/audit
	PYTHONPATH=src $(PYTHON) -m cli audit --profile full-cross --out out/audit
	test ! -s out/audit/missing.json

audit-large:
	mkdir -p out/audit-stress-large
	PYTHONPATH=src $(PYTHON) -m cli audit --profile stress-large --summary-only --out out/audit-stress-large

validate:
	PYTHONPATH=src $(PYTHON) -m cli validate $(OUT)/@all

asm-check:
	@command -v riscv64-linux-gnu-gcc >/dev/null || { echo "riscv64-linux-gnu-gcc not found; skipping"; exit 0; }
	PYTHONPATH=src $(PYTHON) -m cli asm-check $(OUT)/@all --gcc riscv64-linux-gnu-gcc

clean:
	rm -rf out corpus/smoke/generated .pytest_cache
