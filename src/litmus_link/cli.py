from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .asm_check import asm_check
from .descriptions import feature_description_catalog
from .generator import generate_combinations, generate_profile, write_audit, write_audit_for_combinations
from .profiles import HAND_CATEGORIES, axis_values, list_profiles
from .native_scalar import (
    DEFAULT_NATIVE_MECHANISMS,
    NATIVE_ANNOTATIONS,
    NATIVE_PRESETS,
    NativeGenerationError,
    generate_native_relations,
    generate_native_templates,
    native_catalog,
)
from .qt_gui import QtGuiError, qt_binding_status, run_qt_gui
from .rule_file import RuleFileError, load_rule_file, rule_field_values
from .rules import list_rules
from .scalar import (
    DEFAULT_MECHANISMS,
    DEFAULT_RELAX_EDGES,
    DEFAULT_SAFE_EDGES,
    MECHANISM_EDGES,
    SCALAR_PRESETS,
    ScalarGenerationError,
    generate_scalar_cross,
    generate_scalar_enumerated,
    parse_custom_cycle,
    scalar_catalog,
)
from .toolchain import ToolchainError, toolchain_info
from .upstream import import_upstream
from .validator import ValidationError, validate_path


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

    qt_gui = sub.add_parser("qt-gui", help="start the optional PyQt/PySide desktop GUI")
    qt_gui.add_argument("--check", action="store_true", help="only print Qt binding availability")

    scalar = sub.add_parser("scalar", help="generate scalar RISC-V litmus tests with herdtools7")
    scalar_sub = scalar.add_subparsers(dest="scalar_command", required=True)
    scalar_sub.add_parser("catalog", help="list scalar skeletons, edge mechanisms, and defaults")
    scalar_sub.add_parser("tools", help="show resolved diy7/diycross7/herd7 paths and versions")

    scalar_cross = scalar_sub.add_parser("cross", help="cross local-edge mechanisms over named skeletons")
    scalar_cross.add_argument("--skeleton", action="append", choices=sorted(SCALAR_PRESETS), help="repeat to generate multiple skeletons; default: MP")
    scalar_cross.add_argument("--mechanism", action="append", choices=sorted(MECHANISM_EDGES), help="repeat to select po/fence/dependency; default: all")
    scalar_cross.add_argument("--name", help="name for a custom --cycle")
    scalar_cross.add_argument("--cycle", help="custom diycross cycle: semicolon-separated positions, comma-separated alternatives")
    scalar_cross.add_argument("--out", required=True, type=Path)
    scalar_cross.add_argument("--limit", type=int, help="maximum number of generated litmus files")
    scalar_cross.add_argument("--no-judge", action="store_true", help="skip herd7/riscv.cat outcome judging")
    scalar_cross.add_argument("--timeout", type=int, default=180, help="timeout in seconds for each tool invocation")

    scalar_enum = scalar_sub.add_parser("enumerate", help="enumerate scalar cycles with diy7")
    scalar_enum.add_argument("--safe", action="append", help="comma-separated safe edges; repeatable")
    scalar_enum.add_argument("--relax", action="append", help="comma-separated relaxed edges; repeatable")
    scalar_enum.add_argument("--size", type=int, default=4, help="maximum cycle edge count")
    scalar_enum.add_argument("--nprocs", type=int, default=2, help="maximum hart count")
    scalar_enum.add_argument("--exact", action="store_true", help="generate cycles with exactly --size edges")
    scalar_enum.add_argument("--one", action="store_true", help="ask diy7 for one relaxation occurrence per cycle")
    scalar_enum.add_argument("--mode", default="default", choices=["default", "sc", "uni", "thin", "critical", "free", "ppo", "transitive", "total", "mixedcheck"])
    scalar_enum.add_argument("--obstype", default="fenced", choices=["fenced", "loop", "straight"])
    scalar_enum.add_argument("--realdep", action="store_true", help="emit real dependency instruction sequences")
    scalar_enum.add_argument("--moreedges", action="store_true", help="use diy7's more complete edge set")
    scalar_enum.add_argument("--unrollatomic", type=int, help="unroll atomic idioms by this amount")
    scalar_enum.add_argument("--out", required=True, type=Path)
    scalar_enum.add_argument("--limit", type=int, help="maximum number of generated litmus files")
    scalar_enum.add_argument("--no-judge", action="store_true", help="skip herd7/riscv.cat outcome judging")
    scalar_enum.add_argument("--timeout", type=int, default=180, help="timeout in seconds for each tool invocation")

    native = sub.add_parser("native", help="generate scalar RISC-V litmus tests without diy7/diycross7")
    native_sub = native.add_subparsers(dest="native_command", required=True)
    native_sub.add_parser("catalog", help="show the native relation grammar and template counts")

    native_templates = native_sub.add_parser("templates", help="exhaust every configured variant of named relation families")
    native_templates.add_argument("--skeleton", action="append", choices=sorted(NATIVE_PRESETS), help="repeat to select families; default: MP")
    native_templates.add_argument("--mechanism", action="append", choices=sorted(DEFAULT_NATIVE_MECHANISMS), help="repeat to select po/fence/dependency; default: all")
    native_templates.add_argument("--different-location-only", action="store_true", help="exclude same-location local edges")
    native_templates.add_argument("--annotation", action="append", choices=NATIVE_ANNOTATIONS, help="repeat to select P/Aq/Rl/AR; default: all")
    native_templates.add_argument("--out", required=True, type=Path)
    native_templates.add_argument("--limit", type=int, help="maximum files to write; audit still reports the complete finite domain")
    native_templates.add_argument("--no-judge", action="store_true", help="skip the independent herd7/riscv.cat cross-check")
    native_templates.add_argument("--diagrams", action="store_true", help="write PNG and diagram JSON for every generated case")
    native_templates.add_argument("--timeout", type=int, default=180)

    native_cycles = native_sub.add_parser("enumerate", help="enumerate all canonical cycles in a bounded native edge domain")
    native_cycles.add_argument("--mechanism", action="append", choices=sorted(DEFAULT_NATIVE_MECHANISMS), help="repeat to select local mechanisms; communication is always included")
    native_cycles.add_argument("--different-location-only", action="store_true")
    native_cycles.add_argument("--no-internal-communication", action="store_true")
    native_cycles.add_argument("--annotation", action="append", choices=NATIVE_ANNOTATIONS, help="repeat to vary event annotations; default: P")
    native_cycles.add_argument("--min-size", type=int, default=2)
    native_cycles.add_argument("--size", type=int, default=4, help="maximum relation-cycle edge count")
    native_cycles.add_argument("--nprocs", type=int, default=2, help="maximum hart count")
    native_cycles.add_argument("--exact-procs", action="store_true")
    native_cycles.add_argument("--max-accesses-per-proc", type=int, default=4)
    native_cycles.add_argument("--out", required=True, type=Path)
    native_cycles.add_argument("--limit", type=int)
    native_cycles.add_argument("--no-judge", action="store_true")
    native_cycles.add_argument("--diagrams", action="store_true")
    native_cycles.add_argument("--timeout", type=int, default=180)

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
            lines = asm_check(args.atfile, args.gcc)
            for line in lines:
                print(line)
            return 1 if lines and lines[0].startswith("asm-check failed:") else 0
        if args.command == "qt-gui":
            if args.check:
                print(json.dumps(qt_binding_status(), indent=2, sort_keys=True))
                return 0
            return run_qt_gui()
        if args.command == "scalar":
            if args.scalar_command == "catalog":
                print(json.dumps(scalar_catalog(), indent=2, sort_keys=True))
                return 0
            if args.scalar_command == "tools":
                info = toolchain_info()
                print(json.dumps(info, indent=2, sort_keys=True))
                return 0 if info["available"] else 1
            if args.scalar_command == "cross":
                if args.cycle and args.skeleton:
                    raise ScalarGenerationError("--cycle cannot be combined with --skeleton")
                if args.name and not args.cycle:
                    raise ScalarGenerationError("--name is only valid with --cycle")
                cycle = parse_custom_cycle(args.cycle) if args.cycle else None
                report = generate_scalar_cross(
                    out_dir=args.out,
                    presets=args.skeleton or ([] if cycle else ["MP"]),
                    mechanisms=args.mechanism or DEFAULT_MECHANISMS,
                    limit=args.limit,
                    judge=not args.no_judge,
                    timeout=args.timeout,
                    custom_name=args.name,
                    custom_cycle=cycle,
                )
            else:
                report = generate_scalar_enumerated(
                    out_dir=args.out,
                    safe=_split_edges(args.safe) if args.safe else DEFAULT_SAFE_EDGES,
                    relax=_split_edges(args.relax) if args.relax else DEFAULT_RELAX_EDGES,
                    size=args.size,
                    nprocs=args.nprocs,
                    exact=args.exact,
                    one=args.one,
                    mode=args.mode,
                    obstype=args.obstype,
                    realdep=args.realdep,
                    moreedges=args.moreedges,
                    unrollatomic=args.unrollatomic,
                    limit=args.limit,
                    judge=not args.no_judge,
                    timeout=args.timeout,
                )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1 if report.get("verdicts", {}).get("unknown", 0) else 0
        if args.command == "native":
            if args.native_command == "catalog":
                print(json.dumps(native_catalog(), indent=2, sort_keys=True))
                return 0
            if args.native_command == "templates":
                report = generate_native_templates(
                    out_dir=args.out,
                    presets=args.skeleton or ["MP"],
                    mechanisms=args.mechanism or DEFAULT_NATIVE_MECHANISMS,
                    include_same=not args.different_location_only,
                    annotations=args.annotation or NATIVE_ANNOTATIONS,
                    limit=args.limit,
                    judge=not args.no_judge,
                    diagrams=args.diagrams,
                    timeout=args.timeout,
                )
            else:
                report = generate_native_relations(
                    out_dir=args.out,
                    mechanisms=["communication", *(args.mechanism or DEFAULT_NATIVE_MECHANISMS)],
                    include_same=not args.different_location_only,
                    include_internal=not args.no_internal_communication,
                    min_size=args.min_size,
                    max_size=args.size,
                    max_procs=args.nprocs,
                    exact_procs=args.exact_procs,
                    max_accesses_per_proc=args.max_accesses_per_proc,
                    annotations=args.annotation or ("P",),
                    limit=args.limit,
                    judge=not args.no_judge,
                    diagrams=args.diagrams,
                    timeout=args.timeout,
                )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1 if report.get("verdicts", {}).get("unknown", 0) else 0
    except (ValueError, FileNotFoundError, ValidationError, RuleFileError, QtGuiError, ScalarGenerationError, NativeGenerationError, ToolchainError) as exc:
        print(f"litmus-link: error: {exc}", file=sys.stderr)
        return 2
    return 2


def _require_profile_or_rule_file(profile: str | None, rule_file: Path | None) -> None:
    if bool(profile) == bool(rule_file):
        raise ValueError("provide exactly one of --profile or --rule-file")


def _split_edges(values: list[str]) -> tuple[str, ...]:
    edges = tuple(edge.strip() for value in values for edge in value.split(",") if edge.strip())
    if not edges:
        raise ScalarGenerationError("edge list cannot be empty")
    return edges


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
