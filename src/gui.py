from __future__ import annotations

import json
import re
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Iterable, Tuple

from diagram import render_diagram
from descriptions import feature_description_catalog
from generator import audit_summary, generate_combinations, generate_profile, write_audit, write_audit_for_combinations
from models import Combination, GENERATED
from profiles import (
    ALIAS_MODES,
    ATTRIBUTES,
    CMO_OPS,
    CMO_SYNC_SEQUENCES,
    ELEMENT_ORDERS,
    SKELETONS,
    STRESSORS,
    TLB_OPS,
    VECTOR_FOOTPRINTS,
    VECTOR_LENGTHS,
    VECTOR_LMULS,
    VECTOR_MASKS,
    VECTOR_OPS,
    VECTOR_TAILS,
    VECTOR_WIDTHS,
    VM_CONTEXTS,
    PTE_STATES,
    SHOOTDOWN_SCOPES,
    list_profiles,
    profile_combinations,
)
from rule_file import RuleFileError, load_rule_data, rule_field_values
from rules import evaluate
from renderer import render_cases
from solver import solve_generated_case


PARAM_AXIS_VALUES: Dict[str, list[str]] = {
    "dep": ["addr", "data", "ctrl", "ctrl_fence", "aq", "rl", "aqrl"],
    "width": ["w8", "w16", "w32", "w64"],
    "outcome": ["allowed", "forbidden", "mixed_size"],
    "sew": list(VECTOR_WIDTHS),
    "lmul": list(VECTOR_LMULS),
    "mask": list(VECTOR_MASKS),
    "tail": list(VECTOR_TAILS),
    "footprint": list(VECTOR_FOOTPRINTS),
    "vl": list(VECTOR_LENGTHS),
    "elem_order": list(ELEMENT_ORDERS),
    "sync": list(CMO_SYNC_SEQUENCES),
    "vm": list(VM_CONTEXTS),
    "shootdown": list(SHOOTDOWN_SCOPES),
    "pte": list(PTE_STATES),
    "alias": list(ALIAS_MODES),
    "stress": list(STRESSORS),
}


def run_gui(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), _GuiHandler)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"litmus-link GUI listening on {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nlitmus-link GUI stopped")
    finally:
        server.server_close()


def options_payload() -> Dict[str, Any]:
    rule_fields = rule_field_values()
    return {
        "profiles": list_profiles(),
        "axes": {
            "skeleton": list(SKELETONS),
            "attribute": list(ATTRIBUTES),
            "vector": ["none", *VECTOR_OPS],
            "cmo": ["no_cmo", *CMO_OPS],
            "tlb": ["no_tlb", *TLB_OPS],
        },
        "rule_file_fields": rule_fields,
        "param_axes": PARAM_AXIS_VALUES,
        "features": feature_description_catalog(),
    }


def preview_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    name, combinations, source = _combinations_from_payload(payload)
    sample_limit = int(payload.get("sample_limit", 10))
    sample = []
    cached = list(combinations) if isinstance(combinations, list) else None
    iterator = iter(cached if cached is not None else combinations)
    for combination in islice(iterator, max(sample_limit, 0)):
        decision = evaluate(combination)
        rendered_cases = []
        if decision.status == GENERATED:
            rendered_cases = render_cases(combination, decision)
        if rendered_cases:
            for case in rendered_cases:
                solver = solve_generated_case(case).to_json()
                diagram = None
                if case.case_ir is not None:
                    diagram = render_diagram(case.case_ir, solver, Path(gettempdir()) / "litmus-link-preview-diagrams").summary
                sample.append(_preview_item(combination, decision.to_json(), case.name, case.litmus, case.case_ir.to_json() if case.case_ir else None, solver, diagram))
        else:
            sample.append(_preview_item(combination, decision.to_json(), combination.name, "", None, None, None))
    summary_combinations = cached if cached is not None else _combinations_from_payload(payload)[1]
    return {
        "profile": name,
        "source": source,
        "report": audit_summary(name, summary_combinations, source=source),
        "sample": sample,
    }


def _preview_item(
    combination: Combination,
    decision: Dict[str, Any],
    name: str,
    litmus: str,
    case_ir: Dict[str, Any] | None,
    solver: Dict[str, Any] | None,
    diagram: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "combination": combination.to_json(),
        "decision": decision,
        "litmus": litmus,
        "case_ir": case_ir,
        "solver": solver,
        "diagram": diagram,
        "analysis": _preview_analysis(combination, decision, litmus, case_ir, solver),
    }


def _preview_analysis(
    combination: Combination,
    decision: Dict[str, Any],
    litmus: str,
    case_ir: Dict[str, Any] | None = None,
    solver: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    exists = str(case_ir.get("exists")) if case_ir else _extract_exists(litmus)
    cycle = str(case_ir.get("cycle")) if case_ir else _cycle_text(combination, litmus)
    tokens = [relation.get("label", relation.get("kind", "")) for relation in case_ir.get("relations", [])] if case_ir else _cycle_tokens(cycle)
    forbidden = _forbidden_text(combination, decision, exists, solver)
    return {
        "cycle": cycle,
        "cycle_tokens": tokens,
        "exists": exists,
        "forbidden_outcome": forbidden,
        "solver_status": solver.get("status") if solver else "not_applicable",
        "solver_verdict": solver.get("verdict") if solver else "unmodeled",
        "harts": [f"P{index}" for index, _hart in enumerate(case_ir.get("harts", []))] if case_ir else _harts_from_litmus(litmus),
        "memory_locations": _memory_locations_from_ir(case_ir) if case_ir else _memory_locations_from_litmus(litmus),
    }


def _extract_exists(litmus: str) -> str:
    match = re.search(r"\bexists\b\s*(.*)$", litmus, flags=re.DOTALL)
    return " ".join(match.group(1).split()) if match else "No exists clause in rendered preview."


def _cycle_text(combination: Combination, litmus: str) -> str:
    quoted = re.findall(r'"([^"]+)"', litmus)
    if quoted and _looks_like_cycle(quoted[0]):
        return quoted[0]
    skeleton_cycles = {
        "MP": "po/W->W -> rfe -> fre -> po/R->R",
        "LB": "po/R->W -> rfe -> po/R->W -> rfe",
        "SB": "po/W->R -> fre -> po/W->R -> fre",
        "WRC": "wse -> rfe -> po/R->W -> rfe -> fre",
        "RWC": "rfe -> po/R->W -> wse -> rfe -> fre",
        "IRIW": "wse -> rfe -> fre -> rfe -> fre",
        "ISA2": "ppo/dependency-or-fence -> rf/fr/co observation",
        "R": "read-shape rf/fr/dependency cycle",
        "S": "store-shape co/propagation cycle",
        "Co": "coherence per-location co/rf/fr cycle",
    }
    base = skeleton_cycles.get(combination.skeleton, f"{combination.skeleton} relation cycle")
    features = [value for value in [combination.vector, combination.cmo, combination.tlb, combination.attribute] if value not in {"none", "no_cmo", "no_tlb", "cacheable"}]
    return base + (" with " + ", ".join(features) if features else "")


def _looks_like_cycle(text: str) -> bool:
    tokens = {"po", "pod", "rfe", "rfi", "fre", "fri", "co", "wse", "rf", "fr", "ctrl", "addr", "data"}
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _cycle_tokens(cycle: str) -> list[str]:
    raw = re.split(r"\s*(?:->|,|\s+)\s*", cycle.strip())
    return [token for token in raw if token]


def _forbidden_text(combination: Combination, decision: Dict[str, Any], exists: str, solver: Dict[str, Any] | None = None) -> str:
    if solver:
        status = solver.get("status")
        verdict = solver.get("verdict")
        cross = solver.get("cross_check", "")
        tool = "native RVWMO checker" + (" (confirmed by herd7/riscv.cat)" if cross == "agree" else "")
        if status == "verified" and verdict == "forbidden":
            return f"Verified forbidden by {tool}: {exists}"
        if status == "verified" and verdict == "allowed":
            return f"Verified allowed by {tool}: {exists}"
        if status == "conflict":
            return f"Conflict between native checker and herd7 ({solver.get('reason', '')}); native verdict {verdict} reported as primary."
        if status == "not_applicable":
            fusion = solver.get("fusion") or {}
            if fusion.get("status") == "analyzed":
                return f"Extension-prose ordering analysis ({fusion.get('verdict')}, informative -- not a herd verdict): {fusion.get('reason', '')}"
            return "No formal RVWMO forbidden assertion is emitted for this extension/prose-spec case."
    outcome = str(combination.params.get("outcome", ""))
    if outcome == "forbidden":
        return f"Requested forbidden outcome, but solver verification is still required: {exists}"
    if decision.get("expected_kind") == "rvwmo-herd":
        return "RVWMO decides whether the exists outcome is allowed or forbidden for this scalar main-memory case."
    return "No formal forbidden assertion is emitted; this is a hardware-observation/prose-spec outcome."


def _harts_from_litmus(litmus: str) -> list[str]:
    harts = sorted(set(re.findall(r"(?:^|[\s{;])(\d+):", litmus)))
    return [f"P{hart}" for hart in harts] or ["P0", "P1"]


def _memory_locations_from_litmus(litmus: str) -> list[str]:
    init_match = re.search(r"\{(.*?)\}", litmus, flags=re.DOTALL)
    if not init_match:
        return ["x", "y"]
    locations = sorted(set(re.findall(r"=([A-Za-z_][A-Za-z0-9_]*)(?:[;\s]|$)", init_match.group(1))))
    return [location for location in locations if not location.startswith("P")][:4] or ["x", "y"]


def _memory_locations_from_ir(case_ir: Dict[str, Any] | None) -> list[str]:
    if not case_ir:
        return ["x", "y"]
    locations: list[str] = []
    for hart in case_ir.get("harts", []):
        for event in hart:
            location = event.get("location")
            if location and location not in locations:
                locations.append(location)
    return locations or ["x", "y"]


def audit_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(str(payload.get("out") or "out/gui-audit"))
    summary_only = bool(payload.get("summary_only", True))
    mode = str(payload.get("mode", "profile"))
    if mode == "profile":
        return write_audit(str(payload.get("profile") or "smoke"), out_dir, summary_only=summary_only)
    name, combinations, source = _combinations_from_payload(payload)
    return write_audit_for_combinations(name, combinations, out_dir, source=source, summary_only=summary_only)


def generate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(str(payload.get("out") or "out/gui-generated"))
    mode = str(payload.get("mode", "profile"))
    if mode == "profile":
        return generate_profile(str(payload.get("profile") or "smoke"), out_dir)
    name, combinations, source = _combinations_from_payload(payload)
    return generate_combinations(name, combinations, out_dir, source=source)


def _combinations_from_payload(payload: Dict[str, Any]) -> Tuple[str, Iterable[Combination], str | None]:
    mode = str(payload.get("mode", "profile"))
    if mode == "profile":
        profile = str(payload.get("profile") or "smoke")
        return profile, profile_combinations(profile), None
    rule = payload.get("rule")
    if not isinstance(rule, dict):
        raise RuleFileError("custom GUI requests must include a rule object")
    rule_set = load_rule_data(rule, Path("<gui-rule>"))
    return rule_set.name, rule_set.combinations, "gui"


class _GuiHandler(BaseHTTPRequestHandler):
    server_version = "LitmusLinkGUI/0.1"

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(HTML)
        elif self.path == "/api/options":
            self._send_json(options_payload())
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/preview":
                self._send_json(preview_payload(payload))
            elif self.path == "/api/audit":
                self._send_json(audit_payload(payload))
            elif self.path == "/api/generate":
                self._send_json(generate_payload(payload))
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, RuleFileError, FileNotFoundError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, data: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Litmus-link GUI</title>
  <style>
    :root { color-scheme: light; --bg: #f5f7fa; --panel: #ffffff; --ink: #17202a; --muted: #5f6b7a; --line: #d7dee8; --accent: #0f766e; --accent2: #334155; --danger: #b42318; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    header { padding: 18px 24px; background: #111827; color: white; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    header h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }
    header .status { font-size: 13px; color: #cbd5e1; }
    main { display: grid; grid-template-columns: minmax(360px, 520px) minmax(420px, 1fr); gap: 16px; padding: 16px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    h2 { margin: 0 0 12px; font-size: 15px; }
    h3 { margin: 14px 0 8px; font-size: 13px; color: var(--accent2); }
    label { display: block; font-size: 12px; font-weight: 650; color: var(--accent2); margin: 10px 0 4px; }
    input[type="text"], input[type="number"], select, textarea { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px 9px; font: inherit; background: white; color: var(--ink); }
    textarea { min-height: 220px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.4; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .tabs { display: flex; gap: 6px; margin-bottom: 12px; }
    .tab, button { border: 1px solid var(--line); background: white; color: var(--ink); border-radius: 6px; padding: 8px 10px; font-weight: 650; cursor: pointer; }
    .tab.active, button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    button.secondary { background: #e8eef5; border-color: #c8d2df; }
    button.danger { color: var(--danger); }
    .buttons { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .group { border-top: 1px solid var(--line); padding-top: 10px; margin-top: 10px; }
    .checks { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 6px 10px; max-height: 220px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .checks label { margin: 0; display: flex; align-items: center; gap: 7px; font-weight: 500; color: var(--ink); }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.5; }
    .hidden { display: none; }
    pre { margin: 0; padding: 12px; background: #0f172a; color: #dbeafe; border-radius: 8px; overflow: auto; min-height: 280px; font-size: 12px; line-height: 1.45; }
    .split { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .metric { display: grid; grid-template-columns: repeat(5, minmax(90px, 1fr)); gap: 8px; margin-bottom: 12px; }
    .metric div { border: 1px solid var(--line); border-radius: 6px; padding: 9px; background: #f8fafc; }
    .metric b { display: block; font-size: 18px; }
    .metric span { color: var(--muted); font-size: 11px; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } .metric { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <header><h1>Litmus-link GUI</h1><div class="status" id="status">loading</div></header>
  <main>
    <section>
      <div class="tabs"><button class="tab active" data-mode="profile">Profile</button><button class="tab" data-mode="rule">Custom Rule</button></div>
      <div id="profilePanel">
        <label>Profile</label><select id="profileSelect"></select>
        <label>Output directory</label><input id="profileOut" type="text" value="out/gui-profile">
      </div>
      <div id="rulePanel" class="hidden">
        <div class="row"><div><label>Name</label><input id="ruleName" type="text" value="gui-custom"></div><div><label>Limit</label><input id="ruleLimit" type="number" value="10000" min="1"></div></div>
        <label>Output directory</label><input id="ruleOut" type="text" value="out/gui-custom">
        <div id="axisGroups"></div>
        <div class="group"><h3>Rule JSON</h3><textarea id="ruleJson"></textarea></div>
      </div>
      <label><input id="summaryOnly" type="checkbox" checked> Summary-only audit</label>
      <div class="buttons"><button class="secondary" id="buildBtn">Build Rule</button><button id="previewBtn">Preview</button><button id="auditBtn" class="primary">Audit</button><button id="generateBtn" class="danger">Generate</button></div>
      <p class="hint">Audit classifies combinations through the same legality rules used by CLI. Generate writes .litmus, .meta.json, @all, and audit-report.json.</p>
    </section>
    <section class="split">
      <div class="metric" id="metrics"></div>
      <pre id="output">Waiting for configuration.</pre>
    </section>
  </main>
  <script>
    const state = { mode: 'profile', options: null };
    const primaryAxes = ['skeleton', 'attribute', 'vector', 'cmo', 'tlb'];
    const paramAxes = ['sew', 'lmul', 'mask', 'tail', 'footprint', 'vl', 'elem_order', 'sync', 'vm', 'shootdown', 'pte', 'alias', 'stress'];
    const $ = id => document.getElementById(id);
    function selectedValues(name) { return [...document.querySelectorAll(`[data-axis="${name}"]:checked, [data-param="${name}"]:checked`)].map(x => x.value); }
    function setStatus(text) { $('status').textContent = text; }
    function show(data) { $('output').textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2); renderMetrics(data.report || data); }
    function renderMetrics(report) {
      const keys = ['total_combinations', 'generated', 'hand_required', 'excluded_illegal', 'missing'];
      $('metrics').innerHTML = keys.map(k => `<div><b>${report && report[k] !== undefined ? report[k].toLocaleString() : '-'}</b><span>${k}</span></div>`).join('');
    }
    function checkboxGroup(kind, name, values) {
      const attr = kind === 'param' ? 'data-param' : 'data-axis';
      const defaults = new Set(kind === 'param' ? [] : [values[0]]);
      return `<div class="group"><h3>${name}</h3><div class="checks">${values.map(v => `<label><input type="checkbox" ${attr}="${name}" value="${v}" ${defaults.has(v) ? 'checked' : ''}>${v}</label>`).join('')}</div></div>`;
    }
    function buildPanels(options) {
      $('profileSelect').innerHTML = Object.entries(options.profiles).map(([name, desc]) => `<option value="${name}">${name} - ${desc}</option>`).join('');
      $('axisGroups').innerHTML = primaryAxes.map(name => checkboxGroup('axis', name, options.axes[name] || [])).join('') + paramAxes.map(name => checkboxGroup('param', name, options.param_axes[name] || [])).join('');
      buildRule();
    }
    function buildRule() {
      const axes = {};
      for (const name of primaryAxes) { const values = selectedValues(name); if (values.length) axes[name] = values; }
      const param_axes = {};
      for (const name of paramAxes) { const values = selectedValues(name); if (values.length) param_axes[name] = values; }
      const rule = { name: $('ruleName').value || 'gui-custom', axes, param_axes, limit: Number($('ruleLimit').value || 10000) };
      $('ruleJson').value = JSON.stringify(rule, null, 2);
      return rule;
    }
    function requestPayload() {
      if (state.mode === 'profile') return { mode: 'profile', profile: $('profileSelect').value, out: $('profileOut').value, summary_only: $('summaryOnly').checked };
      return { mode: 'rule', rule: JSON.parse($('ruleJson').value), out: $('ruleOut').value, summary_only: $('summaryOnly').checked };
    }
    async function post(path) {
      setStatus(path.replace('/api/', '') + ' running');
      const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestPayload()) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      setStatus('ready');
      show(data);
    }
    document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
      state.mode = tab.dataset.mode;
      document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === tab));
      $('profilePanel').classList.toggle('hidden', state.mode !== 'profile');
      $('rulePanel').classList.toggle('hidden', state.mode !== 'rule');
    }));
    $('buildBtn').addEventListener('click', () => show(buildRule()));
    $('previewBtn').addEventListener('click', () => post('/api/preview').catch(e => show({ error: e.message })));
    $('auditBtn').addEventListener('click', () => post('/api/audit').catch(e => show({ error: e.message })));
    $('generateBtn').addEventListener('click', () => post('/api/generate').catch(e => show({ error: e.message })));
    document.addEventListener('change', e => { if (e.target.matches('[data-axis], [data-param], #ruleName, #ruleLimit')) buildRule(); });
    fetch('/api/options').then(r => r.json()).then(options => { state.options = options; buildPanels(options); setStatus('ready'); }).catch(e => { setStatus('error'); show({ error: e.message }); });
  </script>
</body>
</html>
"""
