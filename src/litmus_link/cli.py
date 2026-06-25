from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .asm_check import asm_check
from .generator import generate_profile, write_audit
from .profiles import HAND_CATEGORIES, axis_values, list_profiles
from .rules import list_rules
from .upstream import import_upstream
from .validator import ValidationError, validate_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="litmus-link")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate litmus tests")
    gen.add_argument("--profile", required=True)
    gen.add_argument("--out", required=True, type=Path)

    audit = sub.add_parser("audit", help="audit a profile without generating litmus files")
    audit.add_argument("--profile", required=True)
    audit.add_argument("--out", type=Path)

    validate = sub.add_parser("validate", help="validate generated corpus")
    validate.add_argument("path", type=Path)

    list_cmd = sub.add_parser("list", help="list known profiles, axes, rules, or hand categories")
    list_cmd.add_argument("what", choices=["profiles", "axes", "rules", "hand"])

    upstream = sub.add_parser("import-upstream", help="index an upstream litmus repository")
    upstream.add_argument("--src", required=True, type=Path)
    upstream.add_argument("--kind", required=True, choices=["riscv", "ifetch", "aarch64-vmsa"])
    upstream.add_argument("--out", required=True, type=Path)

    asm = sub.add_parser("asm-check", help="optional assembler smoke check")
    asm.add_argument("atfile", type=Path)
    asm.add_argument("--gcc", default="riscv64-linux-gnu-gcc")

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            report = generate_profile(args.profile, args.out)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report.get("missing", 0) == 0 else 1
        if args.command == "audit":
            out = args.out or Path("out") / "audit"
            report = write_audit(args.profile, out)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report.get("missing", 0) == 0 else 1
        if args.command == "validate":
            entries = validate_path(args.path)
            print(f"validated {len(entries)} litmus files")
            return 0
        if args.command == "list":
            _print_list(args.what)
            return 0
        if args.command == "import-upstream":
            index = import_upstream(args.src, args.kind, args.out)
            print(json.dumps({"kind": index["kind"], "count": index["count"]}, indent=2, sort_keys=True))
            return 0
        if args.command == "asm-check":
            for line in asm_check(args.atfile, args.gcc):
                print(line)
            return 0
    except (ValueError, FileNotFoundError, ValidationError) as exc:
        print(f"litmus-link: error: {exc}", file=sys.stderr)
        return 2
    return 2


def _print_list(what: str) -> None:
    if what == "profiles":
        for name, description in list_profiles().items():
            print(f"{name}\t{description}")
    elif what == "axes":
        print(json.dumps(axis_values(), indent=2, sort_keys=True))
    elif what == "rules":
        for name, description in list_rules().items():
            print(f"{name}\t{description}")
    elif what == "hand":
        for category in HAND_CATEGORIES:
            print(category)
