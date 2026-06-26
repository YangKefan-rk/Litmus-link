PYTHON ?= python3
PROFILE ?= smoke
OUT ?= out/$(PROFILE)

.PHONY: test smoke audit audit-large validate clean asm-check

test:
	@if $(PYTHON) -c 'import pytest' >/dev/null 2>&1; then \
		PYTHONPATH=src $(PYTHON) -m pytest; \
	else \
		PYTHONPATH=src $(PYTHON) tests/run_tests.py; \
	fi

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
