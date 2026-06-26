from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
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
    QtWidgets, QtCore, binding = _load_qt_modules()
    os.environ.pop("SESSION_MANAGER", None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Litmus-link")
    app.setStyleSheet(_stylesheet())
    window = _LitmusLinkQtWindow(QtWidgets, QtCore, binding)
    window.resize(1440, 900)
    window.show()
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    return int(exec_fn())


def _load_qt_modules() -> Tuple[Any, Any, str]:
    errors = []
    for binding in ["PyQt6", "PySide6", "PyQt5", "PySide2"]:
        try:
            widgets = __import__(f"{binding}.QtWidgets", fromlist=["QtWidgets"])
            core = __import__(f"{binding}.QtCore", fromlist=["QtCore"])
            return widgets, core, binding
        except Exception as exc:
            errors.append(f"{binding}: {type(exc).__name__}: {exc}")
    raise QtGuiError(
        "No Qt binding is installed. Install one of: PyQt6, PySide6, PyQt5, or PySide2. "
        "Recommended: python3 -m pip install PyQt6. Details: " + "; ".join(errors)
    )


def _signal(QtCore: Any, *types: object) -> Any:
    signal_type = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal")
    return signal_type(*types)


def _slot(QtCore: Any, *types: object) -> Any:
    slot_type = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
    if slot_type is None:
        return lambda function: function
    return slot_type(*types)


def _make_worker_class(QtCore: Any) -> Any:
    class ActionWorker(QtCore.QObject):
        started = _signal(QtCore, str)
        progress = _signal(QtCore, str)
        finished = _signal(QtCore, str, object)
        failed = _signal(QtCore, str, str)

        def __init__(self, action: str, label: str, payload: Dict[str, Any]) -> None:
            super().__init__()
            self.action = action
            self.label = label
            self.payload = payload

        def run(self) -> None:
            try:
                self.started.emit(self.label)
                self.progress.emit("Preparing request payload")
                if self.action == "preview":
                    self.progress.emit("Expanding sample combinations")
                    result = preview_payload(self.payload)
                elif self.action == "audit":
                    self.progress.emit("Classifying combinations with legality rules")
                    result = audit_payload(self.payload)
                elif self.action == "generate":
                    self.progress.emit("Writing .litmus, .meta.json, @all, and audit report")
                    result = generate_payload(self.payload)
                else:
                    raise ValueError(f"unknown action: {self.action}")
                self.progress.emit("Finalizing result summary")
                self.finished.emit(self.label, result)
            except Exception as exc:
                self.failed.emit(self.label, str(exc))

    return ActionWorker


def _make_ui_receiver_class(QtCore: Any) -> Any:
    class UiReceiver(QtCore.QObject):
        def __init__(self, owner: "_LitmusLinkQtWindow") -> None:
            super().__init__()
            self.owner = owner

        @_slot(QtCore, str)
        def handle_started(self, label: str) -> None:
            self.owner._handle_started(label)

        @_slot(QtCore, str)
        def handle_progress(self, message: str) -> None:
            self.owner._append_log(message)

        @_slot(QtCore, str, object)
        def handle_finished(self, label: str, result: object) -> None:
            self.owner._handle_finished(label, result)

        @_slot(QtCore, str, str)
        def handle_failed(self, label: str, message: str) -> None:
            self.owner._handle_failed(label, message)

    return UiReceiver


class _LitmusLinkQtWindow:
    PRIMARY_AXES = ["skeleton", "attribute", "vector", "cmo", "tlb"]
    PARAM_AXES = ["sew", "lmul", "mask", "tail", "footprint", "vl", "elem_order", "sync", "vm", "shootdown", "pte", "alias", "stress"]

    def __init__(self, QtWidgets: Any, QtCore: Any, binding: str) -> None:
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.binding = binding
        self.options = options_payload()
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(f"Litmus-link Qt GUI ({binding})")
        self.primary_checks: Dict[str, list[Any]] = {}
        self.param_checks: Dict[str, list[Any]] = {}
        self.action_buttons: list[Any] = []
        self.active_thread = None
        self.active_worker = None
        self.started_at = 0.0
        self.suspend_rule_sync = False
        self.worker_class = _make_worker_class(QtCore)
        self.ui_receiver = _make_ui_receiver_class(QtCore)(self)
        self._build_ui()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.window, name)

    def _build_ui(self) -> None:
        QtWidgets = self.QtWidgets
        QtCore = self.QtCore
        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(14)

        root.addWidget(self._build_header())
        root.addWidget(self._build_flow_panel())

        splitter = QtWidgets.QSplitter(_horizontal(QtCore))
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([560, 820])
        root.addWidget(splitter, 1)

        root.addLayout(self._build_action_bar())
        root.addWidget(self._build_status_bar())
        self._sync_rule_preview()
        self._update_output_hint()

    def _build_header(self) -> Any:
        QtWidgets = self.QtWidgets
        header = QtWidgets.QFrame()
        header.setObjectName("Header")
        layout = QtWidgets.QVBoxLayout(header)
        layout.setContentsMargins(18, 14, 18, 14)
        title = QtWidgets.QLabel("Litmus-link Generator")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Configure RISC-V litmus domains, audit ISA/RVWMO legality, then generate traceable .litmus files."
        )
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _build_flow_panel(self) -> Any:
        QtWidgets = self.QtWidgets
        frame = QtWidgets.QFrame()
        frame.setObjectName("FlowPanel")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        steps = [
            ("1", "Select Scope", "Choose a profile or custom axes."),
            ("2", "Audit Rules", "Filter illegal and HAND-only cases."),
            ("3", "Generate Litmus", "Write tests and metadata."),
            ("4", "Inspect Output", "Open @all and audit-report.json."),
        ]
        for index, (number, title, text) in enumerate(steps):
            layout.addWidget(self._flow_step(number, title, text), 1)
            if index < len(steps) - 1:
                arrow = QtWidgets.QLabel("->")
                arrow.setObjectName("FlowArrow")
                arrow.setAlignment(_align_center(self.QtCore))
                layout.addWidget(arrow)
        return frame

    def _flow_step(self, number: str, title: str, text: str) -> Any:
        QtWidgets = self.QtWidgets
        step = QtWidgets.QFrame()
        step.setObjectName("FlowStep")
        layout = QtWidgets.QHBoxLayout(step)
        layout.setContentsMargins(10, 8, 10, 8)
        badge = QtWidgets.QLabel(number)
        badge.setObjectName("FlowBadge")
        badge.setAlignment(_align_center(self.QtCore))
        badge.setFixedSize(30, 30)
        copy = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(title)
        label.setObjectName("FlowTitle")
        detail = QtWidgets.QLabel(text)
        detail.setObjectName("FlowText")
        detail.setWordWrap(True)
        copy.addWidget(label)
        copy.addWidget(detail)
        layout.addWidget(badge)
        layout.addLayout(copy, 1)
        return step

    def _build_config_panel(self) -> Any:
        QtWidgets = self.QtWidgets
        panel = QtWidgets.QFrame()
        panel.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("Configuration")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)

        self.mode_tabs = QtWidgets.QTabWidget()
        self.mode_tabs.addTab(self._build_profile_tab(), "Profile Mode")
        self.mode_tabs.addTab(self._build_custom_tab(), "Custom Rule Mode")
        self.mode_tabs.currentChanged.connect(lambda _index: self._update_output_hint())
        layout.addWidget(self.mode_tabs, 1)
        return panel

    def _build_profile_tab(self) -> Any:
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(10)

        note = QtWidgets.QLabel("Use a built-in generation domain for quick smoke, focused, or large stress runs.")
        note.setObjectName("Hint")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        self.profile_combo = QtWidgets.QComboBox()
        for name, description in self.options["profiles"].items():
            self.profile_combo.addItem(f"{name} - {description}", name)
        self.profile_combo.currentIndexChanged.connect(lambda _index: self._update_output_hint())
        self.profile_out = QtWidgets.QLineEdit("out/qt-profile")
        self.profile_out.textChanged.connect(lambda _text: self._update_output_hint())
        form.addRow("Profile", self.profile_combo)
        form.addRow("Output directory", self.profile_out)
        layout.addLayout(form)
        layout.addStretch(1)
        return tab

    def _build_custom_tab(self) -> Any:
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(10)

        note = QtWidgets.QLabel("Select axes directly. The rule JSON updates automatically and can still be edited by hand.")
        note.setObjectName("Hint")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        self.rule_name = QtWidgets.QLineEdit("qt-custom")
        self.rule_limit = QtWidgets.QLineEdit("10000")
        self.rule_out = QtWidgets.QLineEdit("out/qt-custom")
        for line in [self.rule_name, self.rule_limit, self.rule_out]:
            line.textChanged.connect(lambda _text: self._sync_rule_preview())
        self.rule_out.textChanged.connect(lambda _text: self._update_output_hint())
        form.addRow("Rule name", self.rule_name)
        form.addRow("Combination limit", self.rule_limit)
        form.addRow("Output directory", self.rule_out)
        layout.addLayout(form)

        self.axis_tabs = QtWidgets.QTabWidget()
        self.axis_tabs.addTab(self._build_axis_page(self.PRIMARY_AXES, self.options["axes"], checked_first=True), "Core Axes")
        self.axis_tabs.addTab(self._build_axis_page(self.PARAM_AXES, PARAM_AXIS_VALUES, checked_first=False), "Parameter Axes")
        layout.addWidget(self.axis_tabs, 1)

        advanced = QtWidgets.QGroupBox("Advanced rule preview")
        advanced_layout = QtWidgets.QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(10, 10, 10, 10)
        self.refresh_rule_button = QtWidgets.QPushButton("Refresh Rule Preview")
        self.refresh_rule_button.clicked.connect(self._sync_rule_preview)
        advanced_layout.addWidget(self.refresh_rule_button)
        advanced_hint = QtWidgets.QLabel("The preview is optional. Generate uses the JSON shown here, including manual edits.")
        advanced_hint.setObjectName("Hint")
        advanced_hint.setWordWrap(True)
        advanced_layout.addWidget(advanced_hint)
        layout.addWidget(advanced)

        return tab

    def _build_axis_page(self, axes: Iterable[str], values_by_axis: Dict[str, Iterable[str]], checked_first: bool) -> Any:
        QtWidgets = self.QtWidgets
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 8, 0)
        scroll_layout.setSpacing(8)

        for axis in axes:
            checks = self._add_check_group(scroll_layout, axis, values_by_axis.get(axis, []), checked_first=checked_first)
            if axis in self.PRIMARY_AXES:
                self.primary_checks[axis] = checks
            else:
                self.param_checks[axis] = checks

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        page_layout.addWidget(scroll)
        return page

    def _add_check_group(self, layout: Any, title: str, values: Iterable[str], checked_first: bool) -> list[Any]:
        QtWidgets = self.QtWidgets
        group = QtWidgets.QGroupBox(title)
        group.setCheckable(False)
        grid = QtWidgets.QGridLayout(group)
        grid.setContentsMargins(10, 8, 10, 10)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        checks = []
        for index, value in enumerate(values):
            check = QtWidgets.QCheckBox(str(value))
            check.setProperty("axis_value", str(value))
            if checked_first and index == 0:
                check.setChecked(True)
            check.stateChanged.connect(lambda _state, self=self: self._sync_rule_preview())
            grid.addWidget(check, index // 3, index % 3)
            checks.append(check)
        layout.addWidget(group)
        return checks

    def _build_result_panel(self) -> Any:
        QtWidgets = self.QtWidgets
        panel = QtWidgets.QFrame()
        panel.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("Results")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)

        self.result_tabs = QtWidgets.QTabWidget()
        self.summary_view = QtWidgets.QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setPlainText("Choose a profile or custom rule, then run Preview Sample, Run Audit, or Generate Files.")
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.rule_json = QtWidgets.QPlainTextEdit()
        self.rule_json.setMinimumHeight(160)
        self.rule_json.textChanged.connect(self._mark_rule_manual_edit)
        self.raw_json = QtWidgets.QPlainTextEdit()
        self.raw_json.setReadOnly(True)
        self.result_tabs.addTab(self.summary_view, "Summary")
        self.result_tabs.addTab(self.log_view, "Log")
        self.result_tabs.addTab(self.rule_json, "Rule JSON")
        self.result_tabs.addTab(self.raw_json, "Raw JSON")
        layout.addWidget(self.result_tabs, 1)
        return panel

    def _build_action_bar(self) -> Any:
        QtWidgets = self.QtWidgets
        controls = QtWidgets.QHBoxLayout()
        self.summary_only = QtWidgets.QCheckBox("Summary-only audit")
        self.summary_only.setChecked(True)
        controls.addWidget(self.summary_only)
        self.output_hint = QtWidgets.QLabel("Output: out/qt-profile")
        self.output_hint.setObjectName("OutputHint")
        controls.addWidget(self.output_hint, 1)
        self.preview_button = QtWidgets.QPushButton("Preview Sample")
        self.audit_button = QtWidgets.QPushButton("Run Audit")
        self.generate_button = QtWidgets.QPushButton("Generate Files")
        self.generate_button.setObjectName("GenerateButton")
        self.action_buttons = [self.preview_button, self.audit_button, self.generate_button]
        self.preview_button.clicked.connect(lambda: self._run_action("preview", "Preview Sample"))
        self.audit_button.clicked.connect(lambda: self._run_action("audit", "Run Audit"))
        self.generate_button.clicked.connect(lambda: self._run_action("generate", "Generate Files"))
        for button in self.action_buttons:
            controls.addWidget(button)
        return controls

    def _build_status_bar(self) -> Any:
        QtWidgets = self.QtWidgets
        frame = QtWidgets.QFrame()
        frame.setObjectName("StatusBar")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        self.status_label = QtWidgets.QLabel("Ready")
        self.elapsed_label = QtWidgets.QLabel("Elapsed: 0.0s")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.status_label, 2)
        layout.addWidget(self.progress_bar, 3)
        layout.addWidget(self.elapsed_label)
        return frame

    def _selected(self, checks: Iterable[Any]) -> list[str]:
        return [str(check.property("axis_value")) for check in checks if check.isChecked()]

    def _build_rule(self) -> Dict[str, Any]:
        axes = {name: values for name, checks in self.primary_checks.items() if (values := self._selected(checks))}
        param_axes = {name: values for name, checks in self.param_checks.items() if (values := self._selected(checks))}
        try:
            limit = int(self.rule_limit.text() or "10000")
        except ValueError:
            limit = 10000
        return {"name": self.rule_name.text() or "qt-custom", "axes": axes, "param_axes": param_axes, "limit": limit}

    def _sync_rule_preview(self) -> None:
        if self.suspend_rule_sync or not hasattr(self, "rule_json"):
            return
        self.suspend_rule_sync = True
        try:
            self.rule_json.setPlainText(json.dumps(self._build_rule(), indent=2, sort_keys=True))
        finally:
            self.suspend_rule_sync = False
        self._update_output_hint()

    def _mark_rule_manual_edit(self) -> None:
        if not self.suspend_rule_sync:
            self.status_label.setText("Rule JSON edited manually")

    def _payload(self) -> Dict[str, Any]:
        if self.mode_tabs.currentIndex() == 0:
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

    def _run_action(self, action: str, label: str) -> None:
        if self.active_thread is not None:
            self._append_log("Another action is still running; wait for it to finish.")
            return
        try:
            payload = self._payload()
        except Exception as exc:
            self._show_error("Invalid configuration", str(exc))
            return

        thread = self.QtCore.QThread(self.window)
        worker = self.worker_class(action, label, payload)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.started.connect(self.ui_receiver.handle_started)
        worker.progress.connect(self.ui_receiver.handle_progress)
        worker.finished.connect(self.ui_receiver.handle_finished)
        worker.failed.connect(self.ui_receiver.handle_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self.active_thread = thread
        self.active_worker = worker
        thread.start()

    def _handle_started(self, label: str) -> None:
        self.started_at = time.monotonic()
        self.status_label.setText(f"{label} running")
        self.elapsed_label.setText("Elapsed: 0.0s")
        self.progress_bar.setRange(0, 0)
        self.log_view.clear()
        self.raw_json.clear()
        self._append_log(f"Started: {label}")
        for button in self.action_buttons:
            button.setEnabled(False)
        self.result_tabs.setCurrentWidget(self.log_view)

    def _handle_finished(self, label: str, result: object) -> None:
        elapsed = time.monotonic() - self.started_at
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText(f"{label} finished")
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self._append_log(f"Finished: {label} in {elapsed:.1f}s")
        if isinstance(result, dict):
            self.summary_view.setPlainText(_summary_text(label, result, self._current_out_dir()))
            self.raw_json.setPlainText(json.dumps(result, indent=2, sort_keys=True))
        else:
            self.summary_view.setPlainText(str(result))
            self.raw_json.setPlainText(json.dumps({"result": str(result)}, indent=2, sort_keys=True))
        self.result_tabs.setCurrentWidget(self.summary_view)
        for button in self.action_buttons:
            button.setEnabled(True)

    def _handle_failed(self, label: str, message: str) -> None:
        elapsed = time.monotonic() - self.started_at if self.started_at else 0.0
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"{label} failed")
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self._show_error(f"{label} failed", message)
        for button in self.action_buttons:
            button.setEnabled(True)

    def _clear_worker(self) -> None:
        self.active_thread = None
        self.active_worker = None

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def _show_error(self, title: str, message: str) -> None:
        self.summary_view.setPlainText(f"{title}\n\n{message}")
        self.raw_json.setPlainText(json.dumps({"error": message}, indent=2, sort_keys=True))
        self._append_log(f"ERROR: {message}")
        self.result_tabs.setCurrentWidget(self.summary_view)

    def _current_out_dir(self) -> str:
        if self.mode_tabs.currentIndex() == 0:
            return self.profile_out.text() or "out/qt-profile"
        return self.rule_out.text() or "out/qt-custom"

    def _update_output_hint(self) -> None:
        if hasattr(self, "output_hint"):
            self.output_hint.setText(f"Output: {self._current_out_dir()}")


def _summary_text(label: str, result: Dict[str, Any], out_dir: str) -> str:
    lines = [f"{label} complete", ""]
    if "profile" in result:
        lines.append(f"Profile: {result['profile']}")
    if "source" in result and result["source"]:
        lines.append(f"Source: {result['source']}")
    lines.extend(
        [
            f"Output directory: {out_dir}",
            "",
            "Counts:",
            f"  total combinations: {result.get('total_combinations', '-')}",
            f"  generated: {result.get('generated', '-')}",
            f"  excluded illegal: {result.get('excluded_illegal', '-')}",
            f"  excluded unsupported: {result.get('excluded_unsupported', '-')}",
            f"  HAND-required: {result.get('hand_required', '-')}",
            f"  missing: {result.get('missing', '-')}",
        ]
    )
    if label == "Generate Files":
        out_path = Path(out_dir)
        lines.extend(
            [
                "",
                "Generated artifacts:",
                f"  litmus files: {result.get('generated', 0)}",
                f"  @all: {out_path / '@all'}",
                f"  audit report: {out_path / 'audit-report.json'}",
                f"  excluded cases: {out_path / 'excluded.json'}",
            ]
        )
    elif label == "Run Audit":
        out_path = Path(out_dir)
        lines.extend(
            [
                "",
                "Audit artifacts:",
                f"  audit report: {out_path / 'audit-report.json'}",
                f"  coverage markdown: {out_path / 'cross-coverage.md'}",
            ]
        )
    if "sample" in result:
        lines.extend(["", f"Preview sample count: {len(result.get('sample', []))}"])
        for item in result.get("sample", [])[:5]:
            lines.append(f"  {item.get('name', '<unnamed>')}")
    return "\n".join(lines)


def _horizontal(QtCore: Any) -> Any:
    orientation = getattr(QtCore, "Qt").Orientation if hasattr(getattr(QtCore, "Qt"), "Orientation") else getattr(QtCore, "Qt")
    return orientation.Horizontal


def _align_center(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    if hasattr(qt, "AlignmentFlag"):
        return qt.AlignmentFlag.AlignCenter
    return qt.AlignCenter


def _stylesheet() -> str:
    return """
    QWidget { background: #f6f8fb; color: #182230; font-size: 13px; }
    QFrame#Header { background: #172033; border-radius: 8px; }
    QLabel#Title { color: #ffffff; font-size: 24px; font-weight: 700; }
    QLabel#Subtitle { color: #cbd5e1; font-size: 13px; }
    QFrame#FlowPanel, QFrame#Panel, QFrame#StatusBar { background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; }
    QFrame#FlowStep { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; }
    QLabel#FlowBadge { background: #0f766e; color: #ffffff; border-radius: 15px; font-weight: 700; }
    QLabel#FlowTitle { color: #111827; font-weight: 700; }
    QLabel#FlowText, QLabel#Hint, QLabel#OutputHint { color: #5b6778; }
    QLabel#FlowArrow { color: #64748b; font-size: 18px; font-weight: 700; }
    QLabel#SectionTitle { color: #111827; font-size: 17px; font-weight: 700; }
    QTabWidget::pane { border: 1px solid #d9e2ec; border-radius: 6px; background: #ffffff; }
    QTabBar::tab { background: #e8eef6; padding: 8px 14px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
    QTabBar::tab:selected { background: #ffffff; color: #0f766e; font-weight: 700; }
    QGroupBox { border: 1px solid #d9e2ec; border-radius: 6px; margin-top: 10px; padding-top: 10px; font-weight: 700; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QLineEdit, QComboBox, QPlainTextEdit { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px; }
    QPlainTextEdit { font-family: monospace; font-size: 12px; }
    QPushButton { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 12px; font-weight: 700; }
    QPushButton:hover { background: #eef6ff; }
    QPushButton:disabled { color: #94a3b8; background: #f1f5f9; }
    QPushButton#GenerateButton { background: #0f766e; color: #ffffff; border-color: #0f766e; }
    QProgressBar { background: #e8eef6; border: 1px solid #cbd5e1; border-radius: 6px; text-align: center; min-height: 18px; }
    QProgressBar::chunk { background: #0f766e; border-radius: 5px; }
    """
