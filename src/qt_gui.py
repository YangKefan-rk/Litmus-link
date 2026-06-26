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
    QtWidgets, QtCore, QtGui, binding = _load_qt_modules()
    os.environ.pop("SESSION_MANAGER", None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Litmus-link")
    app.setStyleSheet(_stylesheet())
    window = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    window.resize(1440, 900)
    window.show()
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    return int(exec_fn())


def _load_qt_modules() -> Tuple[Any, Any, Any, str]:
    errors = []
    for binding in ["PyQt6", "PySide6", "PyQt5", "PySide2"]:
        try:
            widgets = __import__(f"{binding}.QtWidgets", fromlist=["QtWidgets"])
            core = __import__(f"{binding}.QtCore", fromlist=["QtCore"])
            gui = __import__(f"{binding}.QtGui", fromlist=["QtGui"])
            return widgets, core, gui, binding
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
    NONE_VALUES = {"vector": "none", "cmo": "no_cmo", "tlb": "no_tlb"}
    PARAM_AXES = ["sew", "lmul", "mask", "tail", "footprint", "vl", "elem_order", "sync", "vm", "shootdown", "pte", "alias", "dep", "width", "outcome", "stress"]
    PARAM_GROUPS = {
        "Vector": ["sew", "lmul", "mask", "tail", "vl", "elem_order"],
        "Memory Footprint": ["footprint", "alias"],
        "CMO Sync": ["sync"],
        "Virtual Memory": ["vm", "shootdown", "pte"],
        "RVWMO Shape": ["dep", "width", "outcome"],
        "Stress": ["stress"],
    }

    def __init__(self, QtWidgets: Any, QtCore: Any, QtGui: Any, binding: str) -> None:
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.binding = binding
        self.options = options_payload()
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(f"Litmus-link Qt Generator ({binding})")
        self.primary_checks: Dict[str, list[Any]] = {}
        self.param_checks: Dict[str, list[Any]] = {}
        self.action_buttons: list[Any] = []
        self.axis_group_widgets: Dict[str, Any] = {}
        self.param_group_widgets: Dict[str, Any] = {}
        self.preview_items: list[Dict[str, Any]] = []
        self.active_thread = None
        self.active_worker = None
        self.elapsed_timer = QtCore.QTimer(self.window)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
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
        title = QtWidgets.QLabel("Litmus-link Qt Generator")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Configure legal RISC-V Vector, CMO, PBMT/NC, and TLB cross domains before writing traceable .litmus files."
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
            ("1", "Select Scope"),
            ("2", "Audit Rules"),
            ("3", "Generate Litmus"),
            ("4", "Inspect Output"),
        ]
        for index, (number, title) in enumerate(steps):
            layout.addWidget(self._flow_step(number, title), 1)
            if index < len(steps) - 1:
                arrow = QtWidgets.QLabel("->")
                arrow.setObjectName("FlowArrow")
                arrow.setAlignment(_align_center(self.QtCore))
                layout.addWidget(arrow)
        return frame

    def _flow_step(self, number: str, title: str) -> Any:
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
        copy.addWidget(label)
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
        self.axis_tabs.setObjectName("AxisTabs")
        self.axis_tabs.addTab(self._build_axis_page(self.PRIMARY_AXES, self.options["axes"], checked_first=True), "Core Axes")
        for group_name, axes in self.PARAM_GROUPS.items():
            self.axis_tabs.addTab(self._build_parameter_page(group_name, axes), _parameter_tab_title(group_name))
        layout.addWidget(self.axis_tabs, 1)

        advanced = QtWidgets.QGroupBox("Advanced rule preview")
        advanced_layout = QtWidgets.QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(10, 10, 10, 10)
        self.refresh_rule_button = QtWidgets.QPushButton("Refresh Rule Preview")
        self.refresh_rule_button.clicked.connect(self._sync_rule_preview)
        advanced_layout.addWidget(self.refresh_rule_button)
        layout.addWidget(advanced)

        return tab

    def _build_parameter_page(self, group_name: str, axes: Iterable[str]) -> Any:
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

        group = QtWidgets.QGroupBox(group_name)
        group.setObjectName("AxisGroup")
        group.setProperty("axis_role", "parameter")
        group.setProperty("group_state", "inactive")
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(10, 8, 10, 10)
        for axis in axes:
            checks = self._add_check_group(group_layout, axis, PARAM_AXIS_VALUES.get(axis, []), checked_first=False, role="parameter")
            self.param_checks[axis] = checks
        self.param_group_widgets[group_name] = group
        scroll_layout.addWidget(group)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        page_layout.addWidget(scroll)
        return page

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
            checks = self._add_check_group(scroll_layout, axis, values_by_axis.get(axis, []), checked_first=checked_first, role="core")
            if axis in self.PRIMARY_AXES:
                self.primary_checks[axis] = checks
            else:
                self.param_checks[axis] = checks

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        page_layout.addWidget(scroll)
        return page

    def _add_check_group(self, layout: Any, title: str, values: Iterable[str], checked_first: bool, role: str) -> list[Any]:
        QtWidgets = self.QtWidgets
        group = QtWidgets.QGroupBox(title)
        group.setObjectName("AxisGroup")
        group.setProperty("axis_role", role)
        group.setProperty("group_state", "base")
        group.setCheckable(False)
        self.axis_group_widgets[title] = group
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
            check.stateChanged.connect(lambda _state, check=check, axis=title, self=self: self._handle_check_changed(axis, check))
            grid.addWidget(check, index // 3, index % 3)
            checks.append(check)
        layout.addWidget(group)
        return checks

    def _handle_check_changed(self, axis: str, check: Any) -> None:
        if self.suspend_rule_sync:
            return
        none_value = self.NONE_VALUES.get(axis)
        if none_value is not None:
            self.suspend_rule_sync = True
            try:
                checks = self.primary_checks.get(axis, [])
                value = str(check.property("axis_value"))
                if check.isChecked() and value == none_value:
                    for other in checks:
                        if other is not check:
                            other.setChecked(False)
                elif check.isChecked():
                    for other in checks:
                        if str(other.property("axis_value")) == none_value:
                            other.setChecked(False)
                elif not self._selected(checks):
                    for other in checks:
                        if str(other.property("axis_value")) == none_value:
                            other.setChecked(True)
                            break
            finally:
                self.suspend_rule_sync = False
        self._sync_rule_preview()

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
        self.preview_list = QtWidgets.QListWidget()
        self.preview_list.setObjectName("PreviewList")
        self.preview_list.setAlternatingRowColors(True)
        self.preview_list.itemDoubleClicked.connect(self._open_preview_detail)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.rule_json = QtWidgets.QPlainTextEdit()
        self.rule_json.setMinimumHeight(160)
        self.rule_json.textChanged.connect(self._mark_rule_manual_edit)
        self.raw_json = QtWidgets.QPlainTextEdit()
        self.raw_json.setReadOnly(True)
        self.result_tabs.addTab(self.summary_view, "Summary")
        self.result_tabs.addTab(self.preview_list, "Preview Litmus")
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
            self._apply_none_switches()
            self.rule_json.setPlainText(json.dumps(self._build_rule(), indent=2, sort_keys=True))
        finally:
            self.suspend_rule_sync = False
        self._update_output_hint()

    def _apply_none_switches(self) -> None:
        vector_enabled = bool(self._selected_non_none("vector", "none"))
        cmo_enabled = bool(self._selected_non_none("cmo", "no_cmo"))
        tlb_enabled = bool(self._selected_non_none("tlb", "no_tlb"))
        memory_enabled = bool(self._selected_non_none("attribute", "cacheable")) or vector_enabled or cmo_enabled or tlb_enabled
        self._set_core_axis_visual("vector", vector_enabled)
        self._set_core_axis_visual("cmo", cmo_enabled)
        self._set_core_axis_visual("tlb", tlb_enabled)
        self._set_core_axis_visual("attribute", memory_enabled)
        self._set_core_axis_visual("skeleton", True)
        groups = {
            "Vector": vector_enabled,
            "Memory Footprint": memory_enabled,
            "CMO Sync": cmo_enabled,
            "Virtual Memory": tlb_enabled,
            "RVWMO Shape": True,
            "Stress": True,
        }
        for group_name, enabled in groups.items():
            group = self.param_group_widgets.get(group_name)
            if group is None:
                continue
            self._set_group_state(group, "active" if enabled else "inactive")
            if not enabled:
                for axis in self.PARAM_GROUPS[group_name]:
                    for check in self.param_checks.get(axis, []):
                        check.setChecked(False)
                        check.setEnabled(False)
            else:
                for axis in self.PARAM_GROUPS[group_name]:
                    for check in self.param_checks.get(axis, []):
                        check.setEnabled(True)
        self._refresh_choice_states()

    def _selected_non_none(self, axis: str, none_value: str) -> list[str]:
        return [value for value in self._selected(self.primary_checks.get(axis, [])) if value != none_value]

    def _set_core_axis_visual(self, axis: str, enabled: bool) -> None:
        group = self.axis_group_widgets.get(axis)
        if group is not None:
            self._set_group_state(group, "active" if enabled else "inactive")

    def _refresh_choice_states(self) -> None:
        for axis, checks in {**self.primary_checks, **self.param_checks}.items():
            none_value = self.NONE_VALUES.get(axis)
            for check in checks:
                value = str(check.property("axis_value"))
                selected_none = none_value is not None and value == none_value and check.isChecked()
                state = "off" if selected_none else "on" if check.isChecked() else "base"
                check.setProperty("choice_state", state)
                self._refresh_widget_style(check)

    def _set_group_state(self, group: Any, state: str) -> None:
        group.setProperty("group_state", state)
        self._refresh_widget_style(group)

    def _refresh_widget_style(self, widget: Any) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _mark_rule_manual_edit(self) -> None:
        if not self.suspend_rule_sync:
            self.status_label.setText("Rule JSON edited manually")

    def _payload(self) -> Dict[str, Any]:
        sample_limit = self._preview_sample_limit()
        if self.mode_tabs.currentIndex() == 0:
            return {
                "mode": "profile",
                "profile": self.profile_combo.currentData(),
                "out": self.profile_out.text() or "out/qt-profile",
                "summary_only": self.summary_only.isChecked(),
                "sample_limit": sample_limit,
            }
        return {
            "mode": "rule",
            "rule": json.loads(self.rule_json.toPlainText()),
            "out": self.rule_out.text() or "out/qt-custom",
            "summary_only": self.summary_only.isChecked(),
            "sample_limit": sample_limit,
        }

    def _preview_sample_limit(self) -> int:
        if self.mode_tabs.currentIndex() == 0:
            return 200
        try:
            return min(max(int(self.rule_limit.text() or "10000"), 1), 1000)
        except ValueError:
            return 200

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
        self.summary_view.setPlainText(f"{label} is running. Progress messages are shown in the Log tab.")
        self._append_log(f"Started: {label}")
        for button in self.action_buttons:
            button.setEnabled(False)
        self.refresh_rule_button.setEnabled(False)
        self.elapsed_timer.start(250)
        self.result_tabs.setCurrentWidget(self.log_view)

    def _handle_finished(self, label: str, result: object) -> None:
        elapsed = time.monotonic() - self.started_at
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText(f"{label} finished")
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self.elapsed_timer.stop()
        self._append_log(f"Finished: {label} in {elapsed:.1f}s")
        if isinstance(result, dict):
            self.summary_view.setPlainText(_summary_text(label, result, self._current_out_dir()))
            self.raw_json.setPlainText(json.dumps(result, indent=2, sort_keys=True))
            if label == "Preview Sample":
                self._populate_preview_list(result)
        else:
            self.summary_view.setPlainText(str(result))
            self.raw_json.setPlainText(json.dumps({"result": str(result)}, indent=2, sort_keys=True))
        self.result_tabs.setCurrentWidget(self.summary_view)
        for button in self.action_buttons:
            button.setEnabled(True)
        self.refresh_rule_button.setEnabled(True)

    def _handle_failed(self, label: str, message: str) -> None:
        elapsed = time.monotonic() - self.started_at if self.started_at else 0.0
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"{label} failed")
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self.elapsed_timer.stop()
        self._show_error(f"{label} failed", message)
        for button in self.action_buttons:
            button.setEnabled(True)
        self.refresh_rule_button.setEnabled(True)

    def _clear_worker(self) -> None:
        self.active_thread = None
        self.active_worker = None

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def _update_elapsed(self) -> None:
        if self.started_at:
            self.elapsed_label.setText(f"Elapsed: {time.monotonic() - self.started_at:.1f}s")

    def _show_error(self, title: str, message: str) -> None:
        self.summary_view.setPlainText(f"{title}\n\n{message}")
        self.raw_json.setPlainText(json.dumps({"error": message}, indent=2, sort_keys=True))
        self._append_log(f"ERROR: {message}")
        self.result_tabs.setCurrentWidget(self.summary_view)

    def _populate_preview_list(self, result: Dict[str, Any]) -> None:
        self.preview_list.clear()
        self.preview_items = [item for item in result.get("sample", []) if item.get("litmus")]
        for index, item in enumerate(self.preview_items, start=1):
            decision = item.get("decision", {})
            analysis = item.get("analysis", {})
            title = item.get("name", f"case-{index}")
            status = decision.get("status", "unknown")
            cycle = analysis.get("cycle", "")
            list_item = self.QtWidgets.QListWidgetItem(f"{index:03d}  {status}  {title}\n{cycle}")
            list_item.setData(_user_role(self.QtCore), index - 1)
            self.preview_list.addItem(list_item)
        if not self.preview_items:
            self.preview_list.addItem("No generated litmus cases in this preview sample.")

    def _open_preview_detail(self, item: Any) -> None:
        index = item.data(_user_role(self.QtCore))
        if index is None or index < 0 or index >= len(self.preview_items):
            return
        dialog = _LitmusPreviewDialog(self.QtWidgets, self.QtCore, self.QtGui, self.preview_items[index], self.window)
        dialog.resize(1180, 860)
        _exec_dialog(dialog)

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


class _LitmusPreviewDialog:
    def __init__(self, QtWidgets: Any, QtCore: Any, QtGui: Any, item: Dict[str, Any], parent: Any) -> None:
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.item = item
        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle(str(item.get("name", "Litmus preview")))
        self._build_ui()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dialog, name)

    def _build_ui(self) -> None:
        QtWidgets = self.QtWidgets
        layout = QtWidgets.QVBoxLayout(self.dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel(str(self.item.get("name", "Litmus preview")))
        title.setObjectName("DialogTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        splitter = QtWidgets.QSplitter(_vertical(self.QtCore))
        splitter.addWidget(self._build_diagram_view())
        splitter.addWidget(self._build_detail_tabs())
        splitter.setSizes([560, 260])
        layout.addWidget(splitter, 1)

        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.dialog.accept)
        layout.addWidget(close)

    def _build_diagram_view(self) -> Any:
        QtWidgets = self.QtWidgets
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        diagram = self.item.get("diagram") or {}
        png = Path(str(diagram.get("png", ""))) if diagram.get("png") else None
        label = QtWidgets.QLabel()
        label.setAlignment(_align_center(self.QtCore))
        if png and png.exists():
            pixmap = self.QtGui.QPixmap(str(png))
            label.setPixmap(pixmap)
            label.setMinimumSize(pixmap.size())
        else:
            label.setText(f"Diagram PNG is not available.\nExpected: {png or '<none>'}")
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll, 1)
        return container

    def _build_detail_tabs(self) -> Any:
        QtWidgets = self.QtWidgets
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._text_view(self._analysis_text()), "Summary")
        tabs.addTab(self._text_view(str(self.item.get("litmus", ""))), "Litmus")
        tabs.addTab(self._json_view(self.item.get("solver", {})), "Solver")
        tabs.addTab(self._json_view(self.item.get("case_ir", {})), "IR")
        tabs.addTab(self._json_view(self.item.get("diagram", {})), "Diagram")
        return tabs

    def _text_view(self, text: str) -> Any:
        view = self.QtWidgets.QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        return view

    def _json_view(self, data: Any) -> Any:
        return self._text_view(json.dumps(data, indent=2, sort_keys=True))

    def _analysis_text(self) -> str:
        combination = self.item.get("combination", {})
        decision = self.item.get("decision", {})
        analysis = self.item.get("analysis", {})
        solver = self.item.get("solver", {})
        diagram = self.item.get("diagram", {})
        cycle = analysis.get("cycle", "")
        tokens = " -> ".join(analysis.get("cycle_tokens", []))
        exists = analysis.get("exists", "")
        forbidden = analysis.get("forbidden_outcome", "")
        png = diagram.get("png", "")
        axes = combination.get("name", self.item.get("name", ""))
        return "\n".join(
            [
                f"Case: {axes}",
                f"Status: {decision.get('status', '-')}",
                f"RVWMO class: {decision.get('rvwmo_class', '-')}",
                f"Expected kind: {decision.get('expected_kind', '-')}",
                f"Solver: {solver.get('status', '-')} / {solver.get('verdict', '-')}",
                f"Diagram: {png or '-'}",
                "",
                f"Cycle: {cycle}",
                f"Dependency ring: {tokens}",
                "",
                f"Exists: {exists}",
                f"Forbidden outcome: {forbidden}",
            ]
        )


def _parameter_tab_title(group_name: str) -> str:
    titles = {
        "Memory Footprint": "Memory",
        "CMO Sync": "CMO Params",
        "Virtual Memory": "VM Params",
        "RVWMO Shape": "RVWMO",
    }
    return titles.get(group_name, group_name)


def _horizontal(QtCore: Any) -> Any:
    orientation = getattr(QtCore, "Qt").Orientation if hasattr(getattr(QtCore, "Qt"), "Orientation") else getattr(QtCore, "Qt")
    return orientation.Horizontal


def _vertical(QtCore: Any) -> Any:
    orientation = getattr(QtCore, "Qt").Orientation if hasattr(getattr(QtCore, "Qt"), "Orientation") else getattr(QtCore, "Qt")
    return orientation.Vertical


def _align_center(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    if hasattr(qt, "AlignmentFlag"):
        return qt.AlignmentFlag.AlignCenter
    return qt.AlignCenter


def _user_role(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    if hasattr(qt, "ItemDataRole"):
        return qt.ItemDataRole.UserRole
    return qt.UserRole


def _exec_dialog(dialog: Any) -> int:
    exec_fn = getattr(dialog, "exec", None) or getattr(dialog, "exec_", None)
    return int(exec_fn())


def _stylesheet() -> str:
    return """
    QWidget { background: #f3f6fb; color: #182230; font-size: 13px; }
    QLabel, QCheckBox { background: transparent; }
    QFrame#Header { background: #172033; border-radius: 8px; }
    QLabel#Title { color: #ffffff; font-size: 24px; font-weight: 700; }
    QLabel#Subtitle { color: #cbd5e1; font-size: 13px; }
    QFrame#FlowPanel, QFrame#Panel, QFrame#StatusBar { background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; }
    QFrame#FlowStep { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; }
    QLabel#FlowBadge { background: #0f766e; color: #ffffff; border-radius: 15px; font-weight: 700; }
    QLabel#FlowTitle { color: #111827; font-weight: 700; }
    QLabel#OutputHint { color: #5b6778; }
    QLabel#FlowArrow { color: #64748b; font-size: 18px; font-weight: 700; }
    QLabel#SectionTitle { color: #111827; font-size: 17px; font-weight: 700; }
    QTabWidget::pane { border: 1px solid #cfd9e6; border-radius: 7px; background: #ffffff; }
    QTabBar::tab { background: #e7edf5; color: #475569; padding: 8px 14px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
    QTabBar::tab:selected { background: #ffffff; color: #0f766e; font-weight: 700; border-top: 3px solid #0f766e; }
    QTabWidget#AxisTabs QTabBar::tab:first { background: #e0f2fe; color: #075985; font-weight: 700; }
    QTabWidget#AxisTabs QTabBar::tab:first:selected { background: #ffffff; color: #075985; border-top: 3px solid #0284c7; }
    QGroupBox { border: 1px solid #d9e2ec; border-radius: 6px; margin-top: 10px; padding-top: 10px; font-weight: 700; background: #ffffff; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QGroupBox#AxisGroup[axis_role="core"] { border: 2px solid #91c5f8; background: #f0f8ff; }
    QGroupBox#AxisGroup[axis_role="core"][group_state="active"] { border: 2px solid #0284c7; background: #e0f2fe; }
    QGroupBox#AxisGroup[axis_role="core"][group_state="inactive"] { border: 1px solid #cbd5e1; background: #f8fafc; }
    QGroupBox#AxisGroup[axis_role="parameter"] { border: 1px solid #d7dee8; background: #ffffff; }
    QGroupBox#AxisGroup[axis_role="parameter"][group_state="active"] { border: 2px solid #0f766e; background: #ecfdf5; }
    QGroupBox#AxisGroup[axis_role="parameter"][group_state="inactive"] { border: 1px solid #d7dee8; background: #f8fafc; }
    QLineEdit, QComboBox, QPlainTextEdit { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px; }
    QPlainTextEdit { font-family: monospace; font-size: 12px; }
    QCheckBox { spacing: 7px; padding: 3px 6px; border-radius: 5px; }
    QCheckBox[choice_state="on"] { background: #d1fae5; color: #064e3b; font-weight: 700; }
    QCheckBox[choice_state="off"] { background: #fee2e2; color: #7f1d1d; font-weight: 700; }
    QCheckBox:disabled { color: #94a3b8; background: transparent; }
    QPushButton { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 12px; font-weight: 700; }
    QPushButton:hover { background: #eef6ff; }
    QPushButton:disabled { color: #94a3b8; background: #f1f5f9; }
    QPushButton#GenerateButton { background: #0f766e; color: #ffffff; border-color: #0f766e; }
    QProgressBar { background: #e8eef6; border: 1px solid #cbd5e1; border-radius: 6px; text-align: center; min-height: 18px; }
    QProgressBar::chunk { background: #0f766e; border-radius: 5px; }
    """
