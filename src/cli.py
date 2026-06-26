from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from asm_check import asm_check
from descriptions import feature_description_catalog
from generator import generate_combinations, generate_profile, write_audit, write_audit_for_combinations
from gui import run_gui
from profiles import HAND_CATEGORIES, axis_values, list_profiles
from qt_gui import QtGuiError, qt_binding_status, run_qt_gui
from rule_file import RuleFileError, load_rule_file, rule_field_values
from rules import list_rules
from upstream import import_upstream
from validator import ValidationError, validate_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="litmus-link")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate litmus tests")
    gen.add_argument("--profile")
    gen.add_argument("--rule-file", type=Path, help="JSON file describing user-defined generation axes or cases")
    gen.add_argument("--out", required=True, type=Path)

    audit = sub.add_parser("audit", help="audit a profile without generating litmus files")
    audit.add_argument("--profile")
    audit.add_argument("--rule-file", type=Path, help="JSON file describing user-defined generation axes or cases")
    audit.add_argument("--out", type=Path)
    audit.add_argument("--summary-only", action="store_true", help="only write audit-report.json and coverage markdown; skip large detail JSON files")

    validate = sub.add_parser("validate", help="validate generated corpus")
    validate.add_argument("path", type=Path)

    list_cmd = sub.add_parser("list", help="list known profiles, axes, rules, features, or hand categories")
    list_cmd.add_argument("what", choices=["profiles", "axes", "rules", "features", "hand"])

    upstream = sub.add_parser("import-upstream", help="index an upstream litmus repository")
    upstream.add_argument("--src", required=True, type=Path)
    upstream.add_argument("--kind", required=True, choices=["riscv", "ifetch", "aarch64-vmsa"])
    upstream.add_argument("--out", required=True, type=Path)

    asm = sub.add_parser("asm-check", help="optional assembler smoke check")
    asm.add_argument("atfile", type=Path)
    asm.add_argument("--gcc", default="riscv64-linux-gnu-gcc")

    gui = sub.add_parser("gui", help="start the local graphical configuration UI")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", default=8765, type=int)
    gui.add_argument("--no-open", action="store_true", help="do not open a browser automatically")

    qt_gui = sub.add_parser("qt-gui", help="start the optional PyQt/PySide desktop GUI")
    qt_gui.add_argument("--check", action="store_true", help="only print Qt binding availability")

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            _require_profile_or_rule_file(args.profile, args.rule_file)
            if args.rule_file:
                rule_set = load_rule_file(args.rule_file)
                report = generate_combinations(rule_set.name, rule_set.combinations, args.out, source=str(args.rule_file))
            else:
                report = generate_profile(args.profile, args.out)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report.get("missing", 0) == 0 else 1
        if args.command == "audit":
            out = args.out or Path("out") / "audit"
            _require_profile_or_rule_file(args.profile, args.rule_file)
            if args.rule_file:
                rule_set = load_rule_file(args.rule_file)
                report = write_audit_for_combinations(rule_set.name, rule_set.combinations, out, source=str(args.rule_file), summary_only=args.summary_only)
            else:
                report = write_audit(args.profile, out, summary_only=args.summary_only)
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
        if args.command == "gui":
            run_gui(args.host, args.port, open_browser=not args.no_open)
            return 0
        if args.command == "qt-gui":
            if args.check:
                print(json.dumps(qt_binding_status(), indent=2, sort_keys=True))
                return 0
            return run_qt_gui()
    except (ValueError, FileNotFoundError, ValidationError, RuleFileError, QtGuiError) as exc:
        print(f"litmus-link: error: {exc}", file=sys.stderr)
        return 2
    return 2


def _require_profile_or_rule_file(profile: str | None, rule_file: Path | None) -> None:
    if bool(profile) == bool(rule_file):
        raise ValueError("provide exactly one of --profile or --rule-file")


def _print_list(what: str) -> None:
    if what == "profiles":
        for name, description in list_profiles().items():
            print(f"{name}\t{description}")
    elif what == "axes":
        values = axis_values()
        values["rule_file_fields"] = rule_field_values()
        print(json.dumps(values, indent=2, sort_keys=True))
    elif what == "rules":
        for name, description in list_rules().items():
            print(f"{name}\t{description}")
    elif what == "features":
        print(json.dumps(feature_description_catalog(), indent=2, sort_keys=True))
    elif what == "hand":
        for category in HAND_CATEGORIES:
            print(category)


if __name__ == "__main__":
    raise SystemExit(main())
