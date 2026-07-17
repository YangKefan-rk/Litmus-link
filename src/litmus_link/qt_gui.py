from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .workflow import PARAM_AXIS_VALUES, audit_payload, generate_payload, options_payload, preview_payload


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
                elif self.action == "verify":
                    self.progress.emit("Generating preview cases and checking RVWMO outcomes")
                    verify_payload = dict(self.payload)
                    verify_payload["judge"] = True
                    result = preview_payload(verify_payload)
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
        self.scalar_skeleton_checks: list[Any] = []
        self.scalar_mechanism_checks: list[Any] = []
        self.scalar_annotation_checks: list[Any] = []
        self.scalar_memory_mode_checks: list[Any] = []
        self.scalar_memory_width_checks: list[Any] = []
        self.scalar_memory_boundary_checks: list[Any] = []
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
            "Exhaustively generate scalar RVWMO cycles with Litmus-link's native engine, then configure Vector, CMO, PBMT/NC, and TLB extensions."
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
                arrow = QtWidgets.QLabel()
                arrow.setObjectName("FlowArrow")
                arrow.setAlignment(_align_center(self.QtCore))
                arrow.setPixmap(_standard_arrow_icon(QtWidgets, self.window).pixmap(22, 22))
                arrow.setFixedWidth(30)
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
        panel.setMaximumWidth(820)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("Configuration")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)

        self.mode_tabs = QtWidgets.QTabWidget()
        self.scalar_tab = self._build_scalar_tab()
        self.profile_tab = self._build_profile_tab()
        self.custom_tab = self._build_custom_tab()
        self.mode_tabs.addTab(self.scalar_tab, "Scalar Litmus")
        self.mode_tabs.addTab(self.profile_tab, "Profile Mode")
        self.mode_tabs.addTab(self.custom_tab, "Custom Rule Mode")
        self.mode_tabs.currentChanged.connect(lambda _index: self._update_output_hint())
        layout.addWidget(self.mode_tabs, 1)
        return panel

    def _build_scalar_tab(self) -> Any:
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(10)

        form = QtWidgets.QFormLayout()
        self.scalar_engine = QtWidgets.QComboBox()
        self.scalar_engine.addItem("Litmus-link native - exhaustive named families", "native_templates")
        self.scalar_engine.addItem("Litmus-link native - enumerate all relation cycles", "native_cycles")
        self.scalar_engine.addItem("Litmus-link native - diy-compatible safe/relax strategy", "native_diy")
        self.scalar_engine.currentIndexChanged.connect(lambda _index: self._update_scalar_engine())
        self.scalar_out = QtWidgets.QLineEdit("out/qt-scalar")
        self.scalar_out.textChanged.connect(lambda _text: self._update_output_hint())
        self.scalar_limit = QtWidgets.QSpinBox()
        self.scalar_limit.setRange(1, 1000000)
        self.scalar_limit.setValue(1000)
        self.scalar_all_cases = QtWidgets.QCheckBox("Generate the complete accepted domain")
        self.scalar_all_cases.setChecked(True)
        self.scalar_all_cases.toggled.connect(lambda checked: self.scalar_limit.setEnabled(not checked))
        self.scalar_limit.setEnabled(False)
        generation_scope = QtWidgets.QHBoxLayout()
        generation_scope.addWidget(self.scalar_all_cases)
        generation_scope.addWidget(QtWidgets.QLabel("Otherwise cap files at"))
        generation_scope.addWidget(self.scalar_limit)
        self.scalar_preview_limit = QtWidgets.QSpinBox()
        self.scalar_preview_limit.setRange(1, 100000)
        self.scalar_preview_limit.setValue(1000)
        verification_row = QtWidgets.QHBoxLayout()
        self.scalar_judge = QtWidgets.QCheckBox("Verify generated outcomes")
        self.scalar_judge.setChecked(True)
        self.scalar_solver_backend = QtWidgets.QComboBox()
        self.scalar_solver_backend.addItem("Embedded RVWMO (offline)", "embedded")
        self.scalar_solver_backend.addItem("External herd7 + riscv.cat", "herd7")
        self.scalar_solver_backend.addItem("Cross-check embedded and herd7", "crosscheck")
        verification_row.addWidget(self.scalar_judge)
        verification_row.addWidget(self.scalar_solver_backend, 1)
        form.addRow("Generation engine", self.scalar_engine)
        form.addRow("Output directory", self.scalar_out)
        form.addRow("Generation scope", generation_scope)
        form.addRow("Maximum preview rows", self.scalar_preview_limit)
        form.addRow("Outcome verification", verification_row)
        layout.addLayout(form)

        self.scalar_cross_group = QtWidgets.QGroupBox("Native relation domain")
        self.scalar_cross_group.setObjectName("ScalarPrimaryGroup")
        cross_layout = QtWidgets.QVBoxLayout(self.scalar_cross_group)
        self.scalar_skeleton_label = QtWidgets.QLabel("Skeletons")
        cross_layout.addWidget(self.scalar_skeleton_label)
        self.scalar_skeleton_widget = QtWidgets.QWidget()
        skeleton_grid = QtWidgets.QGridLayout()
        skeleton_grid.setContentsMargins(0, 0, 0, 0)
        self.scalar_skeleton_widget.setLayout(skeleton_grid)
        for index, name in enumerate(self.options["native_scalar"]["presets"]):
            check = QtWidgets.QCheckBox(name)
            check.setProperty("axis_value", name)
            check.setChecked(name == "MP")
            skeleton_grid.addWidget(check, index // 5, index % 5)
            self.scalar_skeleton_checks.append(check)
        cross_layout.addWidget(self.scalar_skeleton_widget)
        self.scalar_mechanism_label = QtWidgets.QLabel("Local ordering mechanisms")
        cross_layout.addWidget(self.scalar_mechanism_label)
        self.scalar_mechanism_widget = QtWidgets.QWidget()
        mechanism_row = QtWidgets.QHBoxLayout(self.scalar_mechanism_widget)
        mechanism_row.setContentsMargins(0, 0, 0, 0)
        for name in self.options["native_scalar"]["mechanisms"]:
            check = QtWidgets.QCheckBox(name)
            check.setProperty("axis_value", name)
            check.setChecked(True)
            mechanism_row.addWidget(check)
            self.scalar_mechanism_checks.append(check)
        mechanism_row.addStretch(1)
        cross_layout.addWidget(self.scalar_mechanism_widget)
        cross_layout.addWidget(QtWidgets.QLabel("Event annotations (non-P modes lower to legal AMOs)"))
        annotation_row = QtWidgets.QHBoxLayout()
        for name in self.options["native_scalar"]["annotations"]:
            check = QtWidgets.QCheckBox(name)
            check.setProperty("axis_value", name)
            check.setChecked(True)
            annotation_row.addWidget(check)
            self.scalar_annotation_checks.append(check)
        annotation_row.addStretch(1)
        cross_layout.addLayout(annotation_row)
        self.scalar_include_same = QtWidgets.QCheckBox("Include same-location (s) as well as different-location (d) edges")
        self.scalar_include_same.setChecked(True)
        cross_layout.addWidget(self.scalar_include_same)
        layout.addWidget(self.scalar_cross_group)

        self.scalar_memory_group = QtWidgets.QGroupBox("Scalar memory layout")
        self.scalar_memory_group.setObjectName("AxisGroup")
        self.scalar_memory_group.setProperty("axis_role", "parameter")
        memory_layout = QtWidgets.QFormLayout(self.scalar_memory_group)
        self.scalar_memory_enable = QtWidgets.QCheckBox("Enable mixed / misaligned accesses")
        self.scalar_memory_enable.toggled.connect(self._update_scalar_memory_layout)
        self.scalar_memory_include_aligned = QtWidgets.QCheckBox("Include aligned baseline")
        self.scalar_memory_include_aligned.setChecked(True)

        mode_row = QtWidgets.QHBoxLayout()
        for label, value in (("Misaligned", "misaligned"), ("Mixed-size misaligned", "mixed")):
            check = QtWidgets.QCheckBox(label)
            check.setProperty("axis_value", value)
            check.setChecked(True)
            self.scalar_memory_mode_checks.append(check)
            mode_row.addWidget(check)
        mode_row.addStretch(1)

        width_row = QtWidgets.QHBoxLayout()
        for width in self.options["native_scalar"]["memory_layout"]["width_bits"]:
            check = QtWidgets.QCheckBox(f"{width}-bit")
            check.setProperty("axis_value", str(width))
            check.setChecked(True)
            self.scalar_memory_width_checks.append(check)
            width_row.addWidget(check)
        width_row.addStretch(1)

        boundary_labels = {
            "same16": "Within 16 B",
            "cross16": "Cross 16 B",
            "cross64": "Cross 64 B line",
        }
        boundary_row = QtWidgets.QHBoxLayout()
        for boundary in self.options["native_scalar"]["memory_layout"]["boundaries"]:
            check = QtWidgets.QCheckBox(boundary_labels.get(boundary, boundary))
            check.setProperty("axis_value", boundary)
            check.setChecked(True)
            self.scalar_memory_boundary_checks.append(check)
            boundary_row.addWidget(check)
        boundary_row.addStretch(1)

        atomicity = QtWidgets.QLabel("byte_level_no_mag")
        atomicity.setObjectName("OutputHint")
        for control in (
            self.scalar_memory_include_aligned,
            *self.scalar_memory_mode_checks,
            *self.scalar_memory_width_checks,
            *self.scalar_memory_boundary_checks,
        ):
            control.toggled.connect(
                lambda _checked: self._update_scalar_memory_layout(
                    self.scalar_memory_enable.isChecked()
                )
            )
        memory_layout.addRow("Mode", self.scalar_memory_enable)
        memory_layout.addRow("Corpus", self.scalar_memory_include_aligned)
        memory_layout.addRow("Layouts", mode_row)
        memory_layout.addRow("Widths", width_row)
        memory_layout.addRow("Boundaries", boundary_row)
        memory_layout.addRow("Atomicity", atomicity)
        layout.addWidget(self.scalar_memory_group)

        self.scalar_enumerate_group = QtWidgets.QGroupBox("Exhaustive cycle bounds")
        self.scalar_enumerate_group.setObjectName("ScalarParameterGroup")
        enumerate_layout = QtWidgets.QFormLayout(self.scalar_enumerate_group)
        self.scalar_min_size = QtWidgets.QSpinBox()
        self.scalar_min_size.setRange(2, 16)
        self.scalar_min_size.setValue(2)
        self.scalar_size = QtWidgets.QSpinBox()
        self.scalar_size.setRange(2, 16)
        self.scalar_size.setValue(4)
        self.scalar_nprocs = QtWidgets.QSpinBox()
        self.scalar_nprocs.setRange(2, 16)
        self.scalar_nprocs.setValue(2)
        self.scalar_max_accesses = QtWidgets.QSpinBox()
        self.scalar_max_accesses.setRange(1, 16)
        self.scalar_max_accesses.setValue(4)
        self.scalar_exact_procs = QtWidgets.QCheckBox("require exactly the selected hart count")
        self.scalar_include_internal = QtWidgets.QCheckBox("include internal rf/fr/co edges")
        self.scalar_include_internal.setChecked(True)
        flags = QtWidgets.QHBoxLayout()
        flags.addWidget(self.scalar_exact_procs)
        flags.addWidget(self.scalar_include_internal)
        flags.addStretch(1)
        enumerate_layout.addRow("Minimum cycle size", self.scalar_min_size)
        enumerate_layout.addRow("Maximum cycle size", self.scalar_size)
        enumerate_layout.addRow("Maximum harts", self.scalar_nprocs)
        enumerate_layout.addRow("Maximum accesses per hart", self.scalar_max_accesses)
        enumerate_layout.addRow("Cycle constraints", flags)
        layout.addWidget(self.scalar_enumerate_group)

        self.scalar_diy_group = QtWidgets.QGroupBox("Diy-compatible generation policy")
        self.scalar_diy_group.setObjectName("ScalarParameterGroup")
        diy_layout = QtWidgets.QFormLayout(self.scalar_diy_group)
        diy_defaults = self.options["native_scalar"]["diy"]
        self.scalar_diy_safe = QtWidgets.QPlainTextEdit("\n".join(diy_defaults["safe"]))
        self.scalar_diy_relax = QtWidgets.QPlainTextEdit("\n".join(diy_defaults["relax"]))
        self.scalar_diy_reject = QtWidgets.QPlainTextEdit()
        self.scalar_diy_prefix = QtWidgets.QPlainTextEdit()
        self.scalar_diy_prefix.setPlaceholderText("Optional; one fixed prefix per line, e.g. PodWW Rfe")
        for editor in (
            self.scalar_diy_safe,
            self.scalar_diy_relax,
            self.scalar_diy_reject,
            self.scalar_diy_prefix,
        ):
            editor.setMaximumHeight(92)
        self.scalar_diy_mode = QtWidgets.QComboBox()
        for mode in diy_defaults["modes"]:
            self.scalar_diy_mode.addItem(mode, mode)
        self.scalar_diy_observer = QtWidgets.QComboBox()
        for observer in diy_defaults["observers"]:
            self.scalar_diy_observer.addItem(observer, observer)
        self.scalar_diy_obstype = QtWidgets.QComboBox()
        for observer_type in diy_defaults["observer_types"]:
            self.scalar_diy_obstype.addItem(observer_type, observer_type)
        self.scalar_diy_mix = QtWidgets.QCheckBox("mix distinct relaxations")
        self.scalar_diy_exact_size = QtWidgets.QCheckBox("require exact cycle size")
        self.scalar_diy_realdep = QtWidgets.QCheckBox("emit real dependencies")
        self.scalar_diy_same = QtWidgets.QCheckBox("allow same-location local edges")
        self.scalar_diy_min_relax = QtWidgets.QSpinBox()
        self.scalar_diy_min_relax.setRange(0, 16)
        self.scalar_diy_min_relax.setValue(1)
        self.scalar_diy_max_relax = QtWidgets.QSpinBox()
        self.scalar_diy_max_relax.setRange(0, 16)
        self.scalar_diy_max_relax.setValue(1)
        self.scalar_diy_min_relax.setEnabled(False)
        self.scalar_diy_max_relax.setEnabled(False)
        self.scalar_diy_mix.toggled.connect(self.scalar_diy_min_relax.setEnabled)
        self.scalar_diy_mix.toggled.connect(self.scalar_diy_max_relax.setEnabled)
        relax_counts = QtWidgets.QHBoxLayout()
        relax_counts.addWidget(QtWidgets.QLabel("min"))
        relax_counts.addWidget(self.scalar_diy_min_relax)
        relax_counts.addWidget(QtWidgets.QLabel("max"))
        relax_counts.addWidget(self.scalar_diy_max_relax)
        relax_counts.addStretch(1)
        diy_flags = QtWidgets.QHBoxLayout()
        for flag in (self.scalar_diy_mix, self.scalar_diy_exact_size, self.scalar_diy_realdep, self.scalar_diy_same):
            diy_flags.addWidget(flag)
        diy_flags.addStretch(1)
        diy_layout.addRow("Safe relaxations", self.scalar_diy_safe)
        diy_layout.addRow("Tested relaxations", self.scalar_diy_relax)
        diy_layout.addRow("Rejected sequences", self.scalar_diy_reject)
        diy_layout.addRow("Fixed prefixes", self.scalar_diy_prefix)
        diy_layout.addRow("Cycle policy", self.scalar_diy_mode)
        diy_layout.addRow("Relaxation cardinality", relax_counts)
        diy_layout.addRow("Observer policy", self.scalar_diy_observer)
        diy_layout.addRow("Observer implementation", self.scalar_diy_obstype)
        diy_layout.addRow("Policy flags", diy_flags)
        layout.addWidget(self.scalar_diy_group)
        layout.addStretch(1)
        self._update_scalar_memory_layout(False)
        self._update_scalar_engine()
        return tab

    def _update_scalar_memory_layout(self, enabled: bool) -> None:
        if not hasattr(self, "scalar_memory_group"):
            return
        controls = [
            self.scalar_memory_include_aligned,
            *self.scalar_memory_mode_checks,
            *self.scalar_memory_width_checks,
            *self.scalar_memory_boundary_checks,
        ]
        for control in controls:
            control.setEnabled(enabled)
            control.setProperty("choice_state", "on" if enabled and control.isChecked() else "base")
            self._refresh_widget_style(control)
        self._set_group_state(self.scalar_memory_group, "active" if enabled else "inactive")
        for check in self.scalar_annotation_checks:
            annotation = str(check.property("axis_value"))
            if enabled and annotation != "P":
                check.setChecked(False)
                check.setEnabled(False)
            else:
                check.setEnabled(True)
        if enabled:
            for check in self.scalar_annotation_checks:
                if str(check.property("axis_value")) == "P":
                    check.setChecked(True)
            self.scalar_solver_backend.setCurrentIndex(0)

    def _update_scalar_engine(self) -> None:
        if not hasattr(self, "scalar_engine"):
            return
        engine = self.scalar_engine.currentData()
        templates = engine == "native_templates"
        diy = engine == "native_diy"
        self.scalar_skeleton_label.setVisible(templates)
        self.scalar_skeleton_widget.setVisible(templates)
        self.scalar_mechanism_label.setVisible(not diy)
        self.scalar_mechanism_widget.setVisible(not diy)
        self.scalar_include_same.setVisible(not diy)
        self.scalar_enumerate_group.setVisible(not templates)
        self.scalar_diy_group.setVisible(diy)

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
        self.summary_view.setPlainText("Choose scalar generation, a profile, or a custom rule, then run Preview Cases, Run Audit, or Generate Files.")
        self.preview_page = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(self.preview_page)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        self.preview_stats_label = QtWidgets.QLabel("Preview classification: no sample loaded")
        self.preview_stats_label.setObjectName("PreviewStatsLabel")
        self.preview_stats = QtWidgets.QTreeWidget()
        self.preview_stats.setObjectName("PreviewStats")
        self.preview_stats.setHeaderLabels(["Classification", "Value", "Count"])
        self.preview_stats.setRootIsDecorated(True)
        self.preview_stats.setAlternatingRowColors(True)
        self.preview_stats.setMaximumHeight(220)
        self.preview_stats.setUniformRowHeights(True)
        self.preview_table = QtWidgets.QTableWidget()
        self.preview_table.setObjectName("PreviewTable")
        self.preview_table.setColumnCount(5)
        self.preview_table.setHorizontalHeaderLabels(["#", "Status", "Shape", "Verdict", "Case"])
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setWordWrap(False)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(_no_edit_triggers(QtWidgets))
        self.preview_table.setSelectionBehavior(_select_rows(QtWidgets))
        self.preview_table.setSelectionMode(_single_selection(QtWidgets))
        self.preview_table.itemDoubleClicked.connect(self._open_preview_detail)
        _configure_preview_header(self.preview_table, QtWidgets)
        preview_layout.addWidget(self.preview_stats_label)
        preview_layout.addWidget(self.preview_stats)
        preview_layout.addWidget(self.preview_table, 1)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.rule_json = QtWidgets.QPlainTextEdit()
        self.rule_json.setMinimumHeight(160)
        self.rule_json.textChanged.connect(self._mark_rule_manual_edit)
        self.raw_json = QtWidgets.QPlainTextEdit()
        self.raw_json.setReadOnly(True)
        self.result_tabs.addTab(self.summary_view, "Summary")
        self.result_tabs.addTab(self.preview_page, "Preview Litmus")
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
        self.defer_solver_diagram = QtWidgets.QCheckBox("Defer solver/diagram")
        self.defer_solver_diagram.setToolTip("Advanced: skip herd7 verdicts and PNG diagrams for large real-corpus generation.")
        self.defer_solver_diagram.setChecked(False)
        controls.addWidget(self.defer_solver_diagram)
        self.output_hint = QtWidgets.QLabel("Output: out/qt-profile")
        self.output_hint.setObjectName("OutputHint")
        controls.addWidget(self.output_hint, 1)
        self.preview_button = QtWidgets.QPushButton("Preview Cases")
        self.verify_button = QtWidgets.QPushButton("Verify Preview")
        self.audit_button = QtWidgets.QPushButton("Run Audit")
        self.generate_button = QtWidgets.QPushButton("Generate Files")
        self.generate_button.setObjectName("GenerateButton")
        self.action_buttons = [self.preview_button, self.verify_button, self.audit_button, self.generate_button]
        self.preview_button.clicked.connect(lambda: self._run_action("preview", "Preview Cases"))
        self.verify_button.clicked.connect(lambda: self._run_action("verify", "Verify Preview"))
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
        current = self.mode_tabs.currentWidget()
        if current is self.scalar_tab:
            engine = str(self.scalar_engine.currentData())
            payload: Dict[str, Any] = {
                "mode": "scalar",
                "engine": engine,
                "out": self.scalar_out.text() or "out/qt-scalar",
                "limit": None if self.scalar_all_cases.isChecked() else self.scalar_limit.value(),
                "sample_limit": sample_limit,
                "judge": self.scalar_judge.isChecked(),
                "solver_backend": str(self.scalar_solver_backend.currentData()),
                "summary_only": self.summary_only.isChecked(),
            }
            payload["mechanisms"] = self._selected(self.scalar_mechanism_checks)
            payload["annotations"] = self._selected(self.scalar_annotation_checks)
            payload["memory_layout"] = {
                "enabled": self.scalar_memory_enable.isChecked(),
                "include_aligned": self.scalar_memory_include_aligned.isChecked(),
                "modes": self._selected(self.scalar_memory_mode_checks),
                "width_bits": [
                    int(value) for value in self._selected(self.scalar_memory_width_checks)
                ],
                "boundaries": self._selected(self.scalar_memory_boundary_checks),
                "atomicity_model": "byte_level_no_mag",
                "mag_bytes": None,
            }
            payload["include_same"] = self.scalar_diy_same.isChecked() if engine == "native_diy" else self.scalar_include_same.isChecked()
            if engine == "native_templates":
                payload["skeletons"] = self._selected(self.scalar_skeleton_checks)
            else:
                payload.update(
                    {
                        "min_size": self.scalar_min_size.value(),
                        "size": self.scalar_size.value(),
                        "nprocs": self.scalar_nprocs.value(),
                        "max_accesses_per_proc": self.scalar_max_accesses.value(),
                        "exact_procs": self.scalar_exact_procs.isChecked(),
                        "include_internal": self.scalar_include_internal.isChecked(),
                    }
                )
                if engine == "native_diy":
                    payload["diy"] = {
                        "safe": _text_relaxations(self.scalar_diy_safe.toPlainText()),
                        "relax": _text_relaxations(self.scalar_diy_relax.toPlainText()),
                        "reject": _text_relaxations(self.scalar_diy_reject.toPlainText()),
                        "prefixes": _text_prefixes(self.scalar_diy_prefix.toPlainText()),
                        "mode": str(self.scalar_diy_mode.currentData()),
                        "mix": self.scalar_diy_mix.isChecked(),
                        "min_relax": self.scalar_diy_min_relax.value(),
                        "max_relax": self.scalar_diy_max_relax.value(),
                        "observer": str(self.scalar_diy_observer.currentData()),
                        "observer_type": str(self.scalar_diy_obstype.currentData()),
                        "exact_size": self.scalar_diy_exact_size.isChecked(),
                        "realdep": self.scalar_diy_realdep.isChecked(),
                        "moreedges": False,
                    }
            return payload
        if current is self.profile_tab:
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
            "compute_verdicts": not self.defer_solver_diagram.isChecked(),
        }

    def _preview_sample_limit(self) -> int:
        current = self.mode_tabs.currentWidget()
        if current is self.scalar_tab:
            return self.scalar_preview_limit.value()
        if current is self.profile_tab:
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
            if label in {"Preview Cases", "Verify Preview"}:
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
        QtWidgets = self.QtWidgets
        table = self.preview_table
        table.clearContents()
        # Show every sampled case -- generated, hand-required, and illegal --
        # so the count matches the audit summary instead of silently dropping
        # everything without a rendered litmus body.
        self.preview_items = list(result.get("sample", []))
        self._populate_preview_statistics(result.get("classification_counts", {}) or {})
        table.setRowCount(len(self.preview_items))
        for row, item in enumerate(self.preview_items):
            decision = item.get("decision", {}) or {}
            analysis = item.get("analysis", {}) or {}
            status = decision.get("status", "unknown")
            cells = [
                str(row + 1),
                _status_label(status),
                _shape_label(item.get("combination", {}) or {}),
                _verdict_label(item.get("solver"), decision),
                str(item.get("name", f"case-{row + 1}")),
            ]
            for column, text in enumerate(cells):
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setData(_user_role(self.QtCore), row)
                if column == 1:
                    _tint_cell(cell, self.QtGui, _status_color(status))
                cell.setToolTip(analysis.get("cycle", "") or status)
                table.setItem(row, column, cell)
        if not self.preview_items:
            table.setRowCount(1)
            empty = QtWidgets.QTableWidgetItem("No cases in this preview sample.")
            table.setItem(0, 0, empty)
        table.resizeColumnsToContents()

    def _populate_preview_statistics(self, statistics: Dict[str, Any]) -> None:
        QtWidgets = self.QtWidgets
        tree = self.preview_stats
        tree.clear()
        displayed = int(statistics.get("displayed_cases", len(self.preview_items)) or 0)
        self.preview_stats_label.setText(
            f"Preview classification: {displayed} displayed case{'s' if displayed != 1 else ''}"
        )
        labels = {
            "status": "Generation status",
            "verdict": "Solver verdict",
            "skeleton": "Skeleton",
            "category": "Category",
            "memory_layout": "Memory layout",
            "attribute": "Memory attribute",
            "memory_event": "Memory event",
            "vector": "Vector axis",
            "cmo": "CMO axis",
            "tlb": "TLB axis",
        }
        groups = statistics.get("groups", {}) or {}
        for key, values in groups.items():
            if not isinstance(values, dict):
                continue
            total = sum(int(count) for count in values.values())
            parent = QtWidgets.QTreeWidgetItem([labels.get(key, key), "", str(total)])
            for value, count in sorted(values.items(), key=lambda item: (-int(item[1]), str(item[0]))):
                parent.addChild(QtWidgets.QTreeWidgetItem(["", str(value), str(count)]))
            parent.setExpanded(True)
            tree.addTopLevelItem(parent)
        for column in range(3):
            tree.resizeColumnToContents(column)

    def _open_preview_detail(self, item: Any) -> None:
        index = item.data(_user_role(self.QtCore))
        if index is None or index < 0 or index >= len(self.preview_items):
            return
        dialog = _LitmusPreviewDialog(self.QtWidgets, self.QtCore, self.QtGui, self.preview_items[index], self.window)
        dialog.resize(1180, 860)
        _exec_dialog(dialog)

    def _current_out_dir(self) -> str:
        current = self.mode_tabs.currentWidget()
        if current is self.scalar_tab:
            return self.scalar_out.text() or "out/qt-scalar"
        if current is self.profile_tab:
            return self.profile_out.text() or "out/qt-profile"
        return self.rule_out.text() or "out/qt-custom"

    def _update_output_hint(self) -> None:
        if hasattr(self, "output_hint"):
            self.output_hint.setText(f"Output: {self._current_out_dir()}")
        if hasattr(self, "defer_solver_diagram") and hasattr(self, "mode_tabs"):
            self.defer_solver_diagram.setVisible(self.mode_tabs.currentWidget() is self.custom_tab)


def _summary_text(label: str, result: Dict[str, Any], out_dir: str) -> str:
    counts = result.get("report") if isinstance(result.get("report"), dict) else result
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
            f"  total combinations: {counts.get('total_combinations', counts.get('available_litmus', '-'))}",
            f"  generated: {counts.get('generated', counts.get('generated_litmus', '-'))}",
            f"  excluded illegal: {counts.get('excluded_illegal', '-')}",
            f"  excluded unsupported: {counts.get('excluded_unsupported', '-')}",
            f"  HAND-required: {counts.get('hand_required', '-')}",
            f"  missing: {counts.get('missing', '-')}",
        ]
    )
    if label == "Generate Files":
        out_path = Path(out_dir)
        solver = result.get("solver") if isinstance(result.get("solver"), dict) else result.get("verdicts", {})
        solver_files = sum(int(solver.get(key, 0)) for key in ["verified", "conflict", "not_applicable", "unchecked", "unknown"])
        generation_errors = int(result.get("generation_errors", 0) or 0)
        generation_limit = result.get("generation_limit")
        lines.extend(
            [
                "",
                "Generated artifacts:",
                f"  generated combinations: {result.get('generated', result.get('generated_litmus', 0))}",
                f"  litmus files: {result.get('generated_litmus', result.get('generated', 0))}",
                f"  available litmus: {result.get('available_litmus', result.get('generated_litmus', result.get('generated', 0)))}",
                f"  solver results: {solver_files}",
                f"  verdict mode: {result.get('verdict_mode', 'computed')}",
                f"  @all: {out_path / '@all'}",
                f"  generation report: {out_path / 'generation-report.json'}" if str(result.get("schema", "")).startswith("litmus-link.scalar-") else f"  audit report: {out_path / 'audit-report.json'}",
            ]
        )
        if not str(result.get("schema", "")).startswith("litmus-link.scalar-"):
            lines.append(f"  excluded cases: {out_path / 'excluded.json'}")
        if result.get("generation_limited"):
            lines.append(f"  generation limit: truncated to {generation_limit} litmus files")
        if generation_errors:
            lines.append(f"  generation errors: {out_path / 'generation-errors.json'} ({generation_errors})")
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
        available = counts.get("available_litmus", result.get("available_litmus", len(result.get("sample", []))))
        lines.extend(
            [
                "",
                f"Available cases: {available}",
                f"Displayed cases: {len(result.get('sample', []))}",
            ]
        )
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
        combination = self.item.get("combination", {}) or {}
        decision = self.item.get("decision", {}) or {}
        analysis = self.item.get("analysis", {}) or {}
        solver = self.item.get("solver", {}) or {}
        diagram = self.item.get("diagram", {}) or {}
        cycle = analysis.get("cycle", "")
        tokens = " -> ".join(analysis.get("cycle_tokens", []))
        exists = analysis.get("exists", "")
        outcome = analysis.get("outcome_interpretation", analysis.get("forbidden_outcome", ""))
        png = diagram.get("png", "")
        axes = combination.get("name", self.item.get("name", ""))
        lines = [
            f"Case: {axes}",
            f"Status: {decision.get('status', '-')}",
            f"RVWMO class: {decision.get('rvwmo_class', '-')}",
            f"Expected kind: {decision.get('expected_kind', '-')}",
            f"Solver: {solver.get('status', '-')} / {solver.get('verdict', '-')}",
            f"Diagram: {png or '-'}",
        ]
        reason = decision.get("reason")
        if reason:
            lines.append(f"Reason: {reason}")
        lines += [
            "",
            f"Cycle: {cycle}",
            f"Dependency ring: {tokens}",
            "",
            f"Exists: {exists}",
            f"Outcome interpretation: {outcome}",
        ]
        return "\n".join(lines)


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


def _standard_arrow_icon(QtWidgets: Any, widget: Any) -> Any:
    style = getattr(QtWidgets, "QStyle")
    standard = style.StandardPixmap if hasattr(style, "StandardPixmap") else style
    return widget.style().standardIcon(standard.SP_ArrowRight)


def _exec_dialog(dialog: Any) -> int:
    exec_fn = getattr(dialog, "exec", None) or getattr(dialog, "exec_", None)
    return int(exec_fn())


def _no_edit_triggers(QtWidgets: Any) -> Any:
    abstract = QtWidgets.QAbstractItemView
    triggers = getattr(abstract, "EditTrigger", abstract)
    return triggers.NoEditTriggers


def _select_rows(QtWidgets: Any) -> Any:
    abstract = QtWidgets.QAbstractItemView
    behavior = getattr(abstract, "SelectionBehavior", abstract)
    return behavior.SelectRows


def _single_selection(QtWidgets: Any) -> Any:
    abstract = QtWidgets.QAbstractItemView
    mode = getattr(abstract, "SelectionMode", abstract)
    return mode.SingleSelection


def _configure_preview_header(table: Any, QtWidgets: Any) -> None:
    header = table.horizontalHeader()
    resize = getattr(QtWidgets.QHeaderView, "ResizeMode", QtWidgets.QHeaderView)
    # Stretch the final "Case" column; size the rest to their contents.
    header.setStretchLastSection(True)
    for column in range(table.columnCount() - 1):
        header.setSectionResizeMode(column, resize.ResizeToContents)


_STATUS_LABELS = {
    "generated": "GEN",
    "hand_required": "HAND",
    "excluded_illegal": "ILLEGAL",
    "excluded_unsupported": "UNSUPP",
}
_STATUS_COLORS = {
    "generated": "#16a34a",
    "hand_required": "#d97706",
    "excluded_illegal": "#dc2626",
    "excluded_unsupported": "#64748b",
}


def _status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status.upper())


def _text_relaxations(text: str) -> list[str]:
    """Split GUI relaxation text on newlines/top-level commas.

    Bracketed compositional relaxations such as ``[Rfe,Fence.rw.rwdRR]``
    remain one token.
    """
    out: list[str] = []
    current: list[str] = []
    depth = 0
    for character in text:
        if character == "[":
            depth += 1
        elif character == "]":
            depth = max(depth - 1, 0)
        if character in {",", "\n", ";"} and depth == 0:
            token = "".join(current).strip()
            if token:
                out.append(token)
            current = []
        else:
            current.append(character)
    token = "".join(current).strip()
    if token:
        out.append(token)
    return out


def _text_prefixes(text: str) -> list[list[str]]:
    return [
        [token for token in line.replace(";", " ").split() if token]
        for line in text.splitlines()
        if line.strip()
    ]


def _status_color(status: str) -> str:
    return _STATUS_COLORS.get(status, "#475569")


def _tint_cell(cell: Any, QtGui: Any, color_hex: str) -> None:
    cell.setForeground(QtGui.QColor(color_hex))
    font = cell.font()
    font.setBold(True)
    cell.setFont(font)


def _shape_label(combination: Dict[str, Any]) -> str:
    skeleton = combination.get("skeleton", "?")
    extras = []
    for key, none_value in (("vector", "none"), ("cmo", "no_cmo"), ("tlb", "no_tlb")):
        value = combination.get(key)
        if value and value != none_value:
            extras.append(value)
    attribute = combination.get("attribute")
    if attribute and attribute != "cacheable":
        extras.append(attribute)
    return f"{skeleton} + {', '.join(extras)}" if extras else skeleton


def _verdict_label(solver: Dict[str, Any] | None, decision: Dict[str, Any]) -> str:
    if solver:
        status = solver.get("status")
        if status == "verified":
            return solver.get("verdict", "verified")
        if status == "conflict":
            return f"conflict:{solver.get('verdict', '?')}"
        fusion = solver.get("fusion") or {}
        if fusion.get("status") == "analyzed":
            return f"{fusion.get('verdict', 'prose-spec')} (ext)"
    status = decision.get("status", "")
    if status == "hand_required":
        return "hand-required"
    if status == "excluded_illegal":
        return "illegal"
    if status == "excluded_unsupported":
        return "unsupported"
    return "-"


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
    QTableWidget#PreviewTable { background: #ffffff; border: 1px solid #cfd9e6; border-radius: 7px; gridline-color: #e7edf5; font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 12px; }
    QTableWidget#PreviewTable::item { padding: 5px 8px; }
    QTableWidget#PreviewTable::item:selected { background: #e0f2fe; color: #0c4a6e; }
    QLabel#PreviewStatsLabel { color: #0f766e; font-weight: 700; padding: 2px 4px; }
    QTreeWidget#PreviewStats { background: #f8fafc; border: 1px solid #cfd9e6; border-radius: 6px; alternate-background-color: #ffffff; }
    QTreeWidget#PreviewStats::item { padding: 3px 6px; }
    QHeaderView::section { background: #172033; color: #ffffff; padding: 6px 8px; border: none; border-right: 1px solid #2a3650; font-weight: 700; }
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
