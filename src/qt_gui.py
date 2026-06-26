from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterable, Tuple

from gui import PARAM_AXIS_VALUES, audit_payload, generate_payload, options_payload, preview_payload


class QtGuiError(ValueError):
    pass


def qt_binding_status() -> Dict[str, str]:
    status = {}
    for name in ["PyQt6", "PySide6", "PyQt5", "PySide2"]:
        try:
            __import__(name)
            status[name] = "available"
        except Exception as exc:
            status[name] = f"unavailable: {type(exc).__name__}"
    return status


def run_qt_gui() -> int:
    QtWidgets, binding = _load_qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = _LitmusLinkQtWindow(QtWidgets, binding)
    window.resize(1180, 760)
    window.show()
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    return int(exec_fn())


def _load_qt_widgets() -> Tuple[Any, str]:
    errors = []
    for binding in ["PyQt6", "PySide6", "PyQt5", "PySide2"]:
        try:
            module = __import__(f"{binding}.QtWidgets", fromlist=["QtWidgets"])
            return module, binding
        except Exception as exc:
            errors.append(f"{binding}: {type(exc).__name__}: {exc}")
    raise QtGuiError(
        "No Qt binding is installed. Install one of: PyQt6, PySide6, PyQt5, or PySide2. "
        "Recommended: python3 -m pip install PyQt6. Details: " + "; ".join(errors)
    )


class _LitmusLinkQtWindow:
    def __init__(self, QtWidgets: Any, binding: str) -> None:
        self.QtWidgets = QtWidgets
        self.binding = binding
        self.options = options_payload()
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(f"Litmus-link Qt GUI ({binding})")
        self.primary_checks: Dict[str, list[Any]] = {}
        self.param_checks: Dict[str, list[Any]] = {}
        self._build_ui()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.window, name)

    def _build_ui(self) -> None:
        QtWidgets = self.QtWidgets
        root = QtWidgets.QVBoxLayout(self.window)

        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs)

        profile_tab = QtWidgets.QWidget()
        profile_layout = QtWidgets.QFormLayout(profile_tab)
        self.profile_combo = QtWidgets.QComboBox()
        for name, description in self.options["profiles"].items():
            self.profile_combo.addItem(f"{name} - {description}", name)
        self.profile_out = QtWidgets.QLineEdit("out/qt-profile")
        profile_layout.addRow("Profile", self.profile_combo)
        profile_layout.addRow("Output", self.profile_out)
        tabs.addTab(profile_tab, "Profile")

        custom_tab = QtWidgets.QWidget()
        custom_layout = QtWidgets.QVBoxLayout(custom_tab)
        form = QtWidgets.QFormLayout()
        self.rule_name = QtWidgets.QLineEdit("qt-custom")
        self.rule_limit = QtWidgets.QLineEdit("10000")
        self.rule_out = QtWidgets.QLineEdit("out/qt-custom")
        form.addRow("Name", self.rule_name)
        form.addRow("Limit", self.rule_limit)
        form.addRow("Output", self.rule_out)
        custom_layout.addLayout(form)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        for axis in ["skeleton", "attribute", "vector", "cmo", "tlb"]:
            self.primary_checks[axis] = self._add_check_group(scroll_layout, axis, self.options["axes"].get(axis, []), checked_first=True)
        for axis in ["sew", "lmul", "mask", "tail", "footprint", "vl", "elem_order", "sync", "vm", "shootdown", "pte", "alias", "stress"]:
            self.param_checks[axis] = self._add_check_group(scroll_layout, axis, PARAM_AXIS_VALUES.get(axis, []), checked_first=False)
        scroll.setWidget(scroll_content)
        custom_layout.addWidget(scroll, 1)

        self.rule_json = QtWidgets.QPlainTextEdit()
        self.rule_json.setMinimumHeight(160)
        custom_layout.addWidget(self.rule_json)
        tabs.addTab(custom_tab, "Custom Rule")
        self.tabs = tabs

        controls = QtWidgets.QHBoxLayout()
        self.summary_only = QtWidgets.QCheckBox("Summary-only audit")
        self.summary_only.setChecked(True)
        controls.addWidget(self.summary_only)
        controls.addStretch(1)
        self.build_button = QtWidgets.QPushButton("Build Rule")
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.audit_button = QtWidgets.QPushButton("Audit")
        self.generate_button = QtWidgets.QPushButton("Generate")
        for button in [self.build_button, self.preview_button, self.audit_button, self.generate_button]:
            controls.addWidget(button)
        root.addLayout(controls)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(240)
        root.addWidget(self.output)

        self.build_button.clicked.connect(self._build_rule)
        self.preview_button.clicked.connect(lambda: self._run_action("preview"))
        self.audit_button.clicked.connect(lambda: self._run_action("audit"))
        self.generate_button.clicked.connect(lambda: self._run_action("generate"))
        self._build_rule()

    def _add_check_group(self, layout: Any, title: str, values: Iterable[str], checked_first: bool) -> list[Any]:
        QtWidgets = self.QtWidgets
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(group)
        checks = []
        for index, value in enumerate(values):
            check = QtWidgets.QCheckBox(str(value))
            check.setProperty("axis_value", str(value))
            if checked_first and index == 0:
                check.setChecked(True)
            check.stateChanged.connect(self._build_rule)
            grid.addWidget(check, index // 4, index % 4)
            checks.append(check)
        layout.addWidget(group)
        return checks

    def _selected(self, checks: Iterable[Any]) -> list[str]:
        return [str(check.property("axis_value")) for check in checks if check.isChecked()]

    def _build_rule(self) -> Dict[str, Any]:
        axes = {name: values for name, checks in self.primary_checks.items() if (values := self._selected(checks))}
        param_axes = {name: values for name, checks in self.param_checks.items() if (values := self._selected(checks))}
        try:
            limit = int(self.rule_limit.text() or "10000")
        except ValueError:
            limit = 10000
        rule = {"name": self.rule_name.text() or "qt-custom", "axes": axes, "param_axes": param_axes, "limit": limit}
        self.rule_json.setPlainText(json.dumps(rule, indent=2, sort_keys=True))
        return rule

    def _payload(self) -> Dict[str, Any]:
        if self.tabs.currentIndex() == 0:
            return {
                "mode": "profile",
                "profile": self.profile_combo.currentData(),
                "out": self.profile_out.text() or "out/qt-profile",
                "summary_only": self.summary_only.isChecked(),
            }
        return {
            "mode": "rule",
            "rule": json.loads(self.rule_json.toPlainText()),
            "out": self.rule_out.text() or "out/qt-custom",
            "summary_only": self.summary_only.isChecked(),
        }

    def _run_action(self, action: str) -> None:
        try:
            payload = self._payload()
            if action == "preview":
                result = preview_payload(payload)
            elif action == "audit":
                result = audit_payload(payload)
            elif action == "generate":
                result = generate_payload(payload)
            else:
                raise ValueError(f"unknown action: {action}")
            self.output.setPlainText(json.dumps(result, indent=2, sort_keys=True))
        except Exception as exc:
            self.output.setPlainText(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
