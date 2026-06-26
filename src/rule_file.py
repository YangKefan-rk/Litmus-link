from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from models import Combination
from profiles import ATTRIBUTES, CMO_OPS, HAND_CATEGORIES, SKELETONS, TLB_OPS, VECTOR_OPS


COMBINATION_FIELDS = {
    "category",
    "skeleton",
    "memory_event",
    "attribute",
    "tlb",
    "cmo",
    "vector",
}

MEMORY_EVENTS = {
    "scalar_pair",
    "vector_load",
    "vector_store",
    "cmo",
    "pte_update",
    "ifetch",
    "amo",
}

CATEGORIES = {
    "rvwmo_base",
    "vector_mem",
    "pbmt_nc",
    "cmo",
    "vm_tlb",
    "ifetch",
    "cross",
    "exception",
    "custom",
    *HAND_CATEGORIES,
}

KNOWN_VALUES = {
    "category": CATEGORIES,
    "skeleton": set(SKELETONS),
    "memory_event": MEMORY_EVENTS,
    "attribute": set(ATTRIBUTES),
    "tlb": {"no_tlb", *TLB_OPS},
    "cmo": {"no_cmo", *CMO_OPS},
    "vector": {"none", *VECTOR_OPS},
}

DEFAULTS = {
    "category": "custom",
    "skeleton": "MP",
    "memory_event": "scalar_pair",
    "attribute": "cacheable",
    "tlb": "no_tlb",
    "cmo": "no_cmo",
    "vector": "none",
}


class RuleFileError(ValueError):
    pass


@dataclass(frozen=True)
class RuleSet:
    name: str
    combinations: List[Combination]
    source: Path


def rule_field_values() -> Dict[str, List[str]]:
    return {field: sorted(values) for field, values in sorted(KNOWN_VALUES.items())}


def load_rule_file(path: Path) -> RuleSet:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuleFileError(f"invalid JSON in {path}: {exc}") from exc
    return load_rule_data(data, path)


def load_rule_data(data: Mapping[str, Any], source: Path | None = None) -> RuleSet:
    if not isinstance(data, dict):
        raise RuleFileError("rule file root must be a JSON object")

    source_path = source or Path("<inline-rule>")
    name = _string(data.get("name", source_path.stem), "name")
    defaults = _defaults(data.get("defaults", {}))
    param_defaults = _param_defaults(data.get("param_defaults", {}))
    param_axes = _param_axes(data.get("param_axes", {}))
    limit = _limit(data.get("limit", 10000))
    combinations: List[Combination] = []

    if "axes" in data:
        combinations.extend(_expand_axes(name, defaults, _mapping(data["axes"], "axes"), param_defaults, param_axes))
    if "cases" in data:
        combinations.extend(_expand_cases(name, defaults, data["cases"], param_defaults, param_axes))

    if not combinations:
        raise RuleFileError("rule file must define at least one combination via axes or cases")

    if "exclude" in data:
        exclusions = _exclude_patterns(data["exclude"])
        combinations = [combination for combination in combinations if not _is_excluded(combination, exclusions)]

    if len(combinations) > limit:
        raise RuleFileError(f"rule file expands to {len(combinations)} combinations, above limit {limit}")

    _reject_duplicates(combinations)
    return RuleSet(name=name, combinations=combinations, source=source_path)


def _expand_axes(
    name: str,
    defaults: Mapping[str, str],
    axes: Mapping[str, Any],
    param_defaults: Mapping[str, str],
    param_axes: Mapping[str, Sequence[str]],
) -> List[Combination]:
    unknown = sorted(set(axes) - COMBINATION_FIELDS)
    if unknown:
        raise RuleFileError(f"unknown axes fields: {', '.join(unknown)}")
    normalized = {field: _values(field, value) for field, value in axes.items()}
    fields = list(normalized)
    combinations = []
    for selected in product(*(normalized[field] for field in fields)):
        values = dict(defaults)
        values.update(dict(zip(fields, selected)))
        values = _infer_values(values)
        _validate_values(values)
        for params in _expand_params(param_defaults, param_axes):
            combinations.append(_combination(name, values, params=params))
    return combinations


def _expand_cases(
    name: str,
    defaults: Mapping[str, str],
    raw_cases: Any,
    param_defaults: Mapping[str, str],
    param_axes: Mapping[str, Sequence[str]],
) -> List[Combination]:
    if not isinstance(raw_cases, list):
        raise RuleFileError("cases must be a list of objects")
    combinations = []
    for index, raw_case in enumerate(raw_cases):
        case = _mapping(raw_case, f"cases[{index}]")
        unknown = sorted(set(case) - COMBINATION_FIELDS - {"params"})
        if unknown:
            raise RuleFileError(f"unknown fields in cases[{index}]: {', '.join(unknown)}")
        values = dict(defaults)
        for field in COMBINATION_FIELDS:
            if field in case:
                values[field] = _string(case[field], f"cases[{index}].{field}")
        values = _infer_values(values)
        _validate_values(values)
        case_params = case.get("params", {})
        if not isinstance(case_params, dict):
            raise RuleFileError(f"cases[{index}].params must be an object")
        for params in _expand_params(param_defaults, param_axes):
            merged_params = dict(params)
            for key, value in case_params.items():
                merged_params[_string(key, f"cases[{index}].params key")] = _param_value(value, f"cases[{index}].params.{key}")
            combinations.append(_combination(name, values, params=merged_params))
    return combinations


def _infer_values(values: Dict[str, str]) -> Dict[str, str]:
    inferred = dict(values)
    if inferred["vector"] != "none" and inferred["memory_event"] == "scalar_pair":
        inferred["memory_event"] = "vector_store" if inferred["vector"].endswith("store") else "vector_load"
    if inferred["cmo"] != "no_cmo" and inferred["memory_event"] == "scalar_pair":
        inferred["memory_event"] = "cmo"
    if inferred["tlb"] != "no_tlb" and inferred["memory_event"] == "scalar_pair":
        inferred["memory_event"] = "pte_update"
    if inferred["category"] == "custom":
        if inferred["vector"] != "none" and (inferred["cmo"] != "no_cmo" or inferred["tlb"] != "no_tlb"):
            inferred["category"] = "cross"
        elif inferred["cmo"] != "no_cmo" and inferred["tlb"] != "no_tlb":
            inferred["category"] = "cross"
        elif inferred["vector"] != "none":
            inferred["category"] = "vector_mem"
        elif inferred["cmo"] != "no_cmo":
            inferred["category"] = "cmo"
        elif inferred["tlb"] != "no_tlb":
            inferred["category"] = "vm_tlb"
        elif inferred["attribute"] != "cacheable":
            inferred["category"] = "pbmt_nc"
    return inferred


def _combination(name: str, values: Mapping[str, str], params: Mapping[str, Any] | None = None) -> Combination:
    return Combination(
        profile=name,
        category=values["category"],
        skeleton=values["skeleton"],
        memory_event=values["memory_event"],
        attribute=values["attribute"],
        tlb=values["tlb"],
        cmo=values["cmo"],
        vector=values["vector"],
        params=dict(params or {}),
    )


def _validate_values(values: Mapping[str, str]) -> None:
    for field, value in values.items():
        known = KNOWN_VALUES[field]
        if value not in known:
            allowed = ", ".join(sorted(known))
            raise RuleFileError(f"invalid {field} value {value!r}; allowed values: {allowed}")


def _defaults(raw_defaults: Any) -> Dict[str, str]:
    defaults = dict(DEFAULTS)
    raw = _mapping(raw_defaults, "defaults")
    unknown = sorted(set(raw) - COMBINATION_FIELDS)
    if unknown:
        raise RuleFileError(f"unknown defaults fields: {', '.join(unknown)}")
    for field, value in raw.items():
        defaults[field] = _string(value, f"defaults.{field}")
    _validate_values(defaults)
    return defaults


def _param_defaults(raw_defaults: Any) -> Dict[str, str]:
    raw = _mapping(raw_defaults, "param_defaults")
    return {_string(key, "param_defaults key"): _param_value(value, f"param_defaults.{key}") for key, value in raw.items()}


def _param_axes(raw_axes: Any) -> Dict[str, Sequence[str]]:
    raw = _mapping(raw_axes, "param_axes")
    return {_string(key, "param_axes key"): _param_values(value, f"param_axes.{key}") for key, value in raw.items()}


def _expand_params(defaults: Mapping[str, str], axes: Mapping[str, Sequence[str]]) -> List[Dict[str, str]]:
    if not axes:
        return [dict(defaults)]
    keys = list(axes)
    expanded = []
    for selected in product(*(axes[key] for key in keys)):
        params = dict(defaults)
        params.update(dict(zip(keys, selected)))
        expanded.append(params)
    return expanded


def _exclude_patterns(raw_exclude: Any) -> List[Mapping[str, Sequence[str]]]:
    if not isinstance(raw_exclude, list):
        raise RuleFileError("exclude must be a list of objects")
    patterns = []
    for index, raw_pattern in enumerate(raw_exclude):
        pattern = _mapping(raw_pattern, f"exclude[{index}]")
        unknown = sorted(set(pattern) - COMBINATION_FIELDS)
        if unknown:
            raise RuleFileError(f"unknown fields in exclude[{index}]: {', '.join(unknown)}")
        normalized = {field: _values(field, value) for field, value in pattern.items()}
        patterns.append(normalized)
    return patterns


def _is_excluded(combination: Combination, patterns: Iterable[Mapping[str, Sequence[str]]]) -> bool:
    axes = combination.axes()
    for pattern in patterns:
        if all(axes[field] in values for field, values in pattern.items()):
            return True
    return False


def _reject_duplicates(combinations: Sequence[Combination]) -> None:
    seen = set()
    duplicates = []
    for combination in combinations:
        if combination.name in seen:
            duplicates.append(combination.name)
        seen.add(combination.name)
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise RuleFileError(f"rule file expands to duplicate generated names: {sample}")


def _values(field: str, raw_value: Any) -> List[str]:
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, list) and raw_value and all(isinstance(item, str) for item in raw_value):
        values = list(raw_value)
    else:
        raise RuleFileError(f"{field} must be a string or non-empty list of strings")
    for value in values:
        _validate_values({field: value})
    return values


def _param_values(raw_value: Any, name: str) -> List[str]:
    if isinstance(raw_value, (str, int, bool)):
        values = [_param_value(raw_value, name)]
    elif isinstance(raw_value, list) and raw_value:
        values = [_param_value(item, f"{name}[]") for item in raw_value]
    else:
        raise RuleFileError(f"{name} must be a string, integer, boolean, or non-empty list of those values")
    return values


def _param_value(raw_value: Any, name: str) -> str:
    if isinstance(raw_value, bool):
        return "true" if raw_value else "false"
    if isinstance(raw_value, int):
        return str(raw_value)
    if isinstance(raw_value, str) and raw_value:
        return raw_value
    raise RuleFileError(f"{name} must be a non-empty string, integer, or boolean")


def _limit(raw_limit: Any) -> int:
    if not isinstance(raw_limit, int) or raw_limit <= 0:
        raise RuleFileError("limit must be a positive integer")
    return raw_limit


def _mapping(raw_value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(raw_value, dict):
        raise RuleFileError(f"{name} must be an object")
    return raw_value


def _string(raw_value: Any, name: str) -> str:
    if not isinstance(raw_value, str) or not raw_value:
        raise RuleFileError(f"{name} must be a non-empty string")
    return raw_value
