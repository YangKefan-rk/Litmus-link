from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .profiles import vector_effective_vl
from .workflow import (
    PARAM_AXIS_VALUES,
    audit_payload,
    generate_payload,
    materialize_preview_diagram,
    options_payload,
    preview_payload,
)


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
    app.aboutToQuit.connect(window.shutdown)
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
        progress = _signal(QtCore, object)
        finished = _signal(QtCore, str, object)
        failed = _signal(QtCore, str, str)

        def __init__(self) -> None:
            super().__init__()
            self.request_count = 0
            self._last_percent = -1
            self._last_message = ""
            self._last_progress_emit = 0.0

        def _stage(self, message: str) -> None:
            self.progress.emit({"current": 0, "total": 0, "message": message})

        def _report_progress(self, current: int, total: int, message: str) -> None:
            current = max(int(current), 0)
            total = max(int(total), 0)
            percent = min(100, int(current * 100 / total)) if total else -1
            now = time.monotonic()
            if (
                total
                and percent == self._last_percent
                and current < total
                and now - self._last_progress_emit < 0.5
            ):
                return
            self._last_percent = percent
            self._last_message = message
            self._last_progress_emit = now
            self.progress.emit({"current": current, "total": total, "message": message})

        @_slot(QtCore, str, str, object)
        def run_request(self, action: str, label: str, payload: Dict[str, Any]) -> None:
            try:
                self.request_count += 1
                self._last_percent = -1
                self._last_message = ""
                self._last_progress_emit = 0.0
                self.started.emit(label)
                self._stage("Preparing request payload")
                if action == "preview":
                    self._stage("Expanding sample combinations")
                    preview_request = dict(payload)
                    preview_request["judge"] = False
                    preview_request["compute_verdicts"] = False
                    result = preview_payload(
                        preview_request,
                        progress_callback=self._report_progress,
                    )
                elif action == "verify":
                    self._stage("Generating preview cases and checking RVWMO outcomes")
                    verify_payload = dict(payload)
                    verify_payload["judge"] = True
                    verify_payload["compute_verdicts"] = True
                    result = preview_payload(
                        verify_payload,
                        progress_callback=self._report_progress,
                    )
                elif action == "audit":
                    self._stage("Classifying combinations with legality rules")
                    result = audit_payload(payload)
                elif action == "generate":
                    self._stage("Preparing output and counting cases")
                    result = generate_payload(
                        payload,
                        progress_callback=self._report_progress,
                    )
                else:
                    raise ValueError(f"unknown action: {action}")
                self._report_progress(1, 1, "Finalizing result summary")
                self.finished.emit(label, result)
            except Exception as exc:
                self.failed.emit(label, str(exc))

    return ActionWorker


def _make_action_bus_class(QtCore: Any) -> Any:
    class ActionBus(QtCore.QObject):
        request = _signal(QtCore, str, str, object)

    return ActionBus


def _make_preview_model_class(QtCore: Any, QtGui: Any) -> Any:
    """Create a virtualized table model for the preview browser.

    QTableWidget allocates several QObject-backed cell objects per row.  That
    is convenient for a handful of tests but becomes the dominant cost when a
    user previews tens of thousands of cases.  This model keeps the result
    dictionaries as the source of truth and formats only rows Qt asks to draw.
    """
    class PreviewCaseModel(QtCore.QAbstractTableModel):
        HEADERS = ["#", "Status", "Family", "Verdict", "File name", "Case"]

        def __init__(self) -> None:
            super().__init__()
            self.items: list[Dict[str, Any]] = []
            self.visible: Any = range(0)
            self.search = ""
            self.status_filter = ""
            self.skeleton_filter = ""
            self.verdict_filter = ""

        def rowCount(self, parent: Any = None) -> int:
            return 0 if parent is not None and parent.isValid() else len(self.visible)

        def columnCount(self, parent: Any = None) -> int:
            return 0 if parent is not None and parent.isValid() else len(self.HEADERS)

        def headerData(self, section: int, orientation: Any, role: Any = None) -> Any:
            if role != _display_role(QtCore):
                return None
            if orientation == _horizontal_orientation(QtCore):
                return self.HEADERS[section]
            return None

        def data(self, index: Any, role: Any = None) -> Any:
            if not index.isValid() or index.row() >= len(self.visible):
                return None
            source_index = self.visible[index.row()]
            item = self.items[source_index]
            values = _preview_table_values(item, source_index)
            if role == _display_role(QtCore):
                return values[index.column()]
            if role == _tooltip_role(QtCore):
                if index.column() == 4:
                    return str(item.get("file_name") or "No generated file")
                if index.column() == 5:
                    return f"{values[5]}\nCycle: {values[6]}"
                return values[6] or values[5]
            if role == _foreground_role(QtCore) and index.column() == 1:
                status = str((item.get("decision") or {}).get("status", ""))
                return QtGui.QColor(_status_color(status))
            if role == _user_role(QtCore):
                return source_index
            return None

        def set_items(self, items: Iterable[Dict[str, Any]]) -> None:
            self.beginResetModel()
            self.items = list(items)
            self.visible = range(len(self.items))
            self.endResetModel()
            if any((self.search, self.status_filter, self.skeleton_filter, self.verdict_filter)):
                self.apply_filters()

        def set_filters(
            self,
            search: str = "",
            status: str = "",
            skeleton: str = "",
            verdict: str = "",
        ) -> None:
            self.search = search.strip().lower()
            self.status_filter = status
            self.skeleton_filter = skeleton
            self.verdict_filter = verdict
            self.apply_filters()

        def apply_filters(self) -> None:
            self.beginResetModel()
            if not any((self.search, self.status_filter, self.skeleton_filter, self.verdict_filter)):
                self.visible = range(len(self.items))
                self.endResetModel()
                return
            visible: list[int] = []
            for source_index, item in enumerate(self.items):
                values = _preview_table_values(item, source_index)
                status = str((item.get("decision") or {}).get("status", ""))
                skeleton = str((item.get("combination") or {}).get("skeleton", ""))
                verdict = _preview_filter_verdict(item)
                haystack = " ".join(str(value) for value in values).lower()
                if self.search and self.search not in haystack:
                    continue
                if self.status_filter and status != self.status_filter:
                    continue
                if self.skeleton_filter and skeleton != self.skeleton_filter:
                    continue
                if self.verdict_filter and verdict != self.verdict_filter:
                    continue
                visible.append(source_index)
            self.visible = visible
            self.endResetModel()

        def source_index(self, row: int) -> int | None:
            if row < 0 or row >= len(self.visible):
                return None
            return self.visible[row]

    return PreviewCaseModel


def _make_ui_receiver_class(QtCore: Any) -> Any:
    class UiReceiver(QtCore.QObject):
        def __init__(self, owner: "_LitmusLinkQtWindow") -> None:
            super().__init__()
            self.owner = owner

        @_slot(QtCore, str)
        def handle_started(self, label: str) -> None:
            self.owner._handle_started(label)

        @_slot(QtCore, object)
        def handle_progress(self, payload: object) -> None:
            self.owner._handle_progress(payload)

        @_slot(QtCore, str, object)
        def handle_finished(self, label: str, result: object) -> None:
            self.owner._handle_finished(label, result)
            self.owner._clear_worker()

        @_slot(QtCore, str, str)
        def handle_failed(self, label: str, message: str) -> None:
            self.owner._handle_failed(label, message)
            self.owner._clear_worker()

    return UiReceiver


def _make_responsive_flow_panel_class(QtWidgets: Any, QtCore: Any) -> Any:
    class ResponsiveFlowPanel(QtWidgets.QFrame):
        """Switch between a horizontal workflow and a compact 2x2 layout."""

        # The configuration/results splitter gives the window a relatively
        # wide minimum size.  Switch before the four cards start compressing,
        # based on the flow panel's own width rather than an idealized phone
        # viewport.
        BREAKPOINT = 1280

        def __init__(self, step_factory: Any, arrow_factory: Any) -> None:
            super().__init__()
            self.setObjectName("FlowPanel")
            self.layout_mode = "wide"
            self.stack = QtWidgets.QStackedLayout(self)
            self.stack.setContentsMargins(0, 0, 0, 0)
            self.wide_page, self.wide_cards = self._build_wide_page(
                step_factory, arrow_factory
            )
            self.compact_page, self.compact_cards = self._build_compact_page(
                step_factory, arrow_factory
            )
            self.stack.addWidget(self.wide_page)
            self.stack.addWidget(self.compact_page)
            self._set_compact(False)

        def _build_wide_page(self, step_factory: Any, arrow_factory: Any) -> tuple[Any, list[Any]]:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(page)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(6)
            cards: list[Any] = []
            for index, (number, title) in enumerate(_FLOW_STEPS):
                card = step_factory(number, title)
                cards.append(card)
                layout.addWidget(card, 1)
                if index < len(_FLOW_STEPS) - 1:
                    layout.addWidget(arrow_factory("right"))
            return page, cards

        def _build_compact_page(self, step_factory: Any, arrow_factory: Any) -> tuple[Any, list[Any]]:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QGridLayout(page)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setHorizontalSpacing(8)
            layout.setVerticalSpacing(6)
            cards = [step_factory(number, title) for number, title in _FLOW_STEPS]
            layout.addWidget(cards[0], 0, 0)
            layout.addWidget(arrow_factory("right"), 0, 1)
            layout.addWidget(cards[1], 0, 2)
            layout.addWidget(arrow_factory("down"), 1, 2)
            layout.addWidget(cards[3], 2, 0)
            layout.addWidget(arrow_factory("left"), 2, 1)
            layout.addWidget(cards[2], 2, 2)
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(2, 1)
            layout.setRowStretch(0, 1)
            layout.setRowStretch(2, 1)
            return page, cards

        def resizeEvent(self, event: Any) -> None:
            compact = event.size().width() < self.BREAKPOINT
            self._set_compact(compact)
            super().resizeEvent(event)

        def _set_compact(self, compact: bool) -> None:
            self.layout_mode = "compact" if compact else "wide"
            self.stack.setCurrentIndex(1 if compact else 0)
            self.setFixedHeight(158 if compact else 76)
            self.updateGeometry()

    return ResponsiveFlowPanel


def _make_fit_scroll_area_class(QtWidgets: Any) -> Any:
    class FitScrollArea(QtWidgets.QScrollArea):
        def __init__(self, resize_callback: Any) -> None:
            super().__init__()
            self._resize_callback = resize_callback

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            self._resize_callback()

    return FitScrollArea


_FLOW_STEPS = (
    ("1", "Select Scope"),
    ("2", "Audit Rules"),
    ("3", "Generate Litmus"),
    ("4", "Inspect Output"),
)


class _LitmusLinkQtWindow:
    def __init__(self, QtWidgets: Any, QtCore: Any, QtGui: Any, binding: str) -> None:
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.binding = binding
        self.options = options_payload()
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(f"Litmus-link Qt Generator ({binding})")
        self.scalar_skeleton_checks: list[Any] = []
        self.scalar_mechanism_checks: list[Any] = []
        self.scalar_annotation_checks: list[Any] = []
        self.scalar_memory_mode_checks: list[Any] = []
        self.scalar_memory_width_checks: list[Any] = []
        self.scalar_memory_boundary_checks: list[Any] = []
        self.scalar_memory_atomic_overlap_checks: list[Any] = []
        self.vector_checks: Dict[str, list[Any]] = {}
        self.vector_filter_groups: list[Any] = []
        self.vector_group_by_key: Dict[str, Any] = {}
        self.vector_category_checks: list[Any] = []
        self.vector_composition_checks: list[Any] = []
        self.vector_core_groups: list[Any] = []
        self.vector_parameter_groups: list[Any] = []
        self.action_buttons: list[Any] = []
        self.preview_items: list[Dict[str, Any]] = []
        self.active_thread = None
        self.active_worker = None
        self.preview_dialog_open = False
        self.elapsed_timer = QtCore.QTimer(self.window)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.started_at = 0.0
        self.active_label = ""
        self.worker_class = _make_worker_class(QtCore)
        self.action_bus_class = _make_action_bus_class(QtCore)
        self.preview_model_class = _make_preview_model_class(QtCore, QtGui)
        self.ui_receiver = _make_ui_receiver_class(QtCore)(self)
        self.action_thread = QtCore.QThread(self.window)
        self.action_worker = self.worker_class()
        self.action_bus = self.action_bus_class(self.window)
        self.action_worker.moveToThread(self.action_thread)
        self.action_bus.request.connect(self.action_worker.run_request)
        self.action_worker.started.connect(self.ui_receiver.handle_started)
        self.action_worker.progress.connect(self.ui_receiver.handle_progress)
        self.action_worker.finished.connect(self.ui_receiver.handle_finished)
        self.action_worker.failed.connect(self.ui_receiver.handle_failed)
        self.action_thread.start()
        self._build_ui()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.window, name)

    def shutdown(self) -> None:
        """Stop the single persistent action thread before Qt tears down widgets."""
        thread = getattr(self, "action_thread", None)
        if thread is None or not thread.isRunning():
            return
        self.action_worker.deleteLater()
        thread.quit()
        thread.wait()

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
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([620, 780])
        root.addWidget(splitter, 1)

        root.addLayout(self._build_action_bar())
        root.addWidget(self._build_status_bar())
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
            "Generate scalar RVWMO relation cycles or Vector-memory variants, verify outcomes, and inspect each case on demand."
        )
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _build_flow_panel(self) -> Any:
        panel_class = _make_responsive_flow_panel_class(
            self.QtWidgets, self.QtCore
        )
        self.flow_panel = panel_class(self._flow_step, self._flow_arrow)
        return self.flow_panel

    def _flow_arrow(self, direction: str) -> Any:
        arrow = self.QtWidgets.QLabel()
        arrow.setObjectName("FlowArrow")
        arrow.setAlignment(_align_center(self.QtCore))
        arrow.setPixmap(
            _standard_arrow_icon(
                self.QtWidgets, self.window, direction
            ).pixmap(20, 20)
        )
        arrow.setFixedSize(28, 28)
        return arrow

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
        panel.setMaximumWidth(680)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("Configuration")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)

        self.mode_tabs = QtWidgets.QTabWidget()
        self.scalar_tab = self._build_scalar_tab()
        self.vector_tab = self._build_vector_tab()
        self.mode_tabs.addTab(self.scalar_tab, "Scalar Litmus")
        self.mode_tabs.addTab(self.vector_tab, "Vector Litmus")
        self.mode_tabs.currentChanged.connect(lambda _index: self._update_output_hint())
        layout.addWidget(self.mode_tabs, 1)
        return panel

    def _build_scalar_tab(self) -> Any:
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QScrollArea()
        tab.setObjectName("ScalarConfigScroll")
        tab.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(10)

        form = QtWidgets.QFormLayout()
        _configure_responsive_form(form, QtWidgets)
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
        self.scalar_preview_limit = QtWidgets.QSpinBox()
        self.scalar_preview_limit.setRange(1, 250000)
        self.scalar_preview_limit.setValue(2000)
        self.scalar_judge = QtWidgets.QCheckBox("Verify generated outcomes")
        self.scalar_judge.setChecked(True)
        self.scalar_solver_backend = QtWidgets.QComboBox()
        self.scalar_solver_backend.addItem("Embedded RVWMO (offline)", "embedded")
        self.scalar_solver_backend.addItem("External herd7 + riscv.cat", "herd7")
        self.scalar_solver_backend.addItem("Cross-check embedded and herd7", "crosscheck")
        form.addRow("Generation engine", self.scalar_engine)
        form.addRow("Output directory", self.scalar_out)
        form.addRow("Generation scope", self.scalar_all_cases)
        form.addRow("File cap", self.scalar_limit)
        form.addRow("Maximum preview rows", self.scalar_preview_limit)
        form.addRow("Outcome verification", self.scalar_judge)
        form.addRow("Solver backend", self.scalar_solver_backend)
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
        annotation_tips = {
            "P": "Plain scalar load/store.",
            "AMO": "Relaxed AMO without aq/rl; still atomic.",
            "Aq": "AMO with acquire ordering.",
            "Rl": "AMO with release ordering.",
            "AR": "AMO with acquire and release ordering.",
        }
        for name in self.options["native_scalar"]["annotations"]:
            check = QtWidgets.QCheckBox(name)
            check.setProperty("axis_value", name)
            check.setToolTip(annotation_tips.get(name, name))
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
        _configure_responsive_form(memory_layout, QtWidgets)
        self.scalar_memory_enable = QtWidgets.QCheckBox("Enable extended scalar layouts")
        self.scalar_memory_enable.toggled.connect(self._update_scalar_memory_layout)
        self.scalar_memory_include_aligned = QtWidgets.QCheckBox("Include aligned baseline")
        self.scalar_memory_include_aligned.setChecked(True)

        mode_grid = QtWidgets.QGridLayout()
        for index, (label, value) in enumerate((
            ("Misaligned", "misaligned"),
            ("Mixed-size misaligned", "mixed"),
            ("Fixed-width atomic", "atomic"),
            ("Mixed-size aligned atomic", "atomic_mixed"),
        )):
            check = QtWidgets.QCheckBox(label)
            check.setProperty("axis_value", value)
            check.setChecked(value in {"misaligned", "mixed"})
            self.scalar_memory_mode_checks.append(check)
            mode_grid.addWidget(check, index // 2, index % 2)

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

        atomic_overlap_row = QtWidgets.QHBoxLayout()
        for label, value in (("Same start", "same_start"), ("Partial overlap", "partial_overlap")):
            check = QtWidgets.QCheckBox(label)
            check.setProperty("axis_value", value)
            check.setChecked(True)
            self.scalar_memory_atomic_overlap_checks.append(check)
            atomic_overlap_row.addWidget(check)
        atomic_overlap_row.addStretch(1)

        atomicity = QtWidgets.QLabel("misaligned: byte_level_no_mag; atomic: aligned / mixed-size atomic")
        atomicity.setObjectName("OutputHint")
        atomicity.setWordWrap(True)
        for control in (
            self.scalar_memory_include_aligned,
            *self.scalar_memory_mode_checks,
            *self.scalar_memory_width_checks,
            *self.scalar_memory_boundary_checks,
            *self.scalar_memory_atomic_overlap_checks,
        ):
            control.toggled.connect(
                lambda _checked: self._update_scalar_memory_layout(
                    self.scalar_memory_enable.isChecked()
                )
            )
        memory_layout.addRow("Mode", self.scalar_memory_enable)
        memory_layout.addRow("Corpus", self.scalar_memory_include_aligned)
        memory_layout.addRow("Layouts", mode_grid)
        memory_layout.addRow("Widths", width_row)
        memory_layout.addRow("Boundaries", boundary_row)
        memory_layout.addRow("Atomic overlap", atomic_overlap_row)
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
        tab.setWidget(content)
        return tab

    def _update_scalar_memory_layout(self, enabled: bool) -> None:
        if not hasattr(self, "scalar_memory_group"):
            return
        controls = [
            self.scalar_memory_include_aligned,
            *self.scalar_memory_mode_checks,
            *self.scalar_memory_width_checks,
            *self.scalar_memory_boundary_checks,
            *self.scalar_memory_atomic_overlap_checks,
        ]
        for control in controls:
            control.setEnabled(enabled)
            control.setProperty("choice_state", "on" if enabled and control.isChecked() else "base")
            self._refresh_widget_style(control)
        self._set_group_state(self.scalar_memory_group, "active" if enabled else "inactive")
        selected_modes = {
            str(check.property("axis_value"))
            for check in self.scalar_memory_mode_checks
            if check.isChecked()
        }
        atomic_layout_enabled = bool(selected_modes & {"atomic", "atomic_mixed"})
        for check in self.scalar_annotation_checks:
            annotation = str(check.property("axis_value"))
            if enabled and not atomic_layout_enabled and annotation != "P":
                check.setChecked(False)
                check.setEnabled(False)
            else:
                check.setEnabled(True)
                if enabled and atomic_layout_enabled and annotation != "P":
                    check.setChecked(True)
        if enabled and not atomic_layout_enabled:
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

    def _build_vector_tab(self) -> Any:
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(10)

        basic_group = QtWidgets.QGroupBox("Basic configuration")
        basic_group.setObjectName("VectorBasicConfiguration")
        form = QtWidgets.QFormLayout(basic_group)
        _configure_responsive_form(form, QtWidgets)
        self.vector_complete = QtWidgets.QCheckBox("Use the complete legal fusion domain")
        self.vector_complete.setObjectName("VectorComplete")
        self.vector_complete.setChecked(True)
        self.vector_complete.setToolTip(
            "Include every supported relation cycle, endpoint category, width, Vector form, "
            "AMO variant, and overlap layout. Clear it to edit the axes below."
        )
        self.vector_out = QtWidgets.QLineEdit("out/qt-vector")
        self.vector_out.textChanged.connect(lambda _text: self._update_output_hint())
        self.vector_preview_limit = QtWidgets.QSpinBox()
        self.vector_preview_limit.setRange(1, 100000)
        self.vector_preview_limit.setValue(1000)
        self.vector_preview_sampling = QtWidgets.QComboBox()
        self.vector_preview_sampling.addItem("Balanced skeleton coverage", "balanced")
        self.vector_preview_sampling.addItem("Domain-weighted random", "domain_weighted")
        self.vector_random_seed = QtWidgets.QSpinBox()
        self.vector_random_seed.setRange(0, 2_147_483_647)
        self.vector_random_seed.setValue(1)
        self.vector_generation_mode = QtWidgets.QComboBox()
        self.vector_generation_mode.setObjectName("VectorGenerationMode")
        self.vector_generation_mode.addItem("Balanced coverage sample", "balanced")
        self.vector_generation_mode.addItem("Domain-weighted sample", "domain_weighted")
        self.vector_generation_mode.addItem("All legal combinations", "all")
        self.vector_generate_limit = QtWidgets.QSpinBox()
        self.vector_generate_limit.setRange(1, 1_000_000)
        self.vector_generate_limit.setValue(10_000)
        self.vector_solver_backend = QtWidgets.QComboBox()
        self.vector_solver_backend.addItem("Embedded RVWMO (offline)", "embedded")
        self.vector_solver_backend.addItem(
            "Cross-check with herd7 scalar projection", "crosscheck"
        )
        self.vector_solver_backend.setCurrentIndex(0)
        self.vector_verification_effort = QtWidgets.QComboBox()
        self.vector_verification_effort.addItem("Interactive (fast preview)", "interactive")
        self.vector_verification_effort.addItem("Balanced", "balanced")
        self.vector_verification_effort.addItem("Thorough (deep search)", "thorough")
        self.vector_verification_effort.setCurrentIndex(0)
        self.vector_verification_effort.setToolTip(
            "Controls the per-case search budget and the number of herd7 projections. "
            "A budget limit returns inconclusive, never a guessed forbidden verdict."
        )
        self.vector_solver_workers = QtWidgets.QSpinBox()
        self.vector_solver_workers.setObjectName("VectorSolverWorkers")
        self.vector_solver_workers.setRange(1, 64)
        self.vector_solver_workers.setValue(16)
        self.vector_solver_workers.setSuffix(" processes")
        self.vector_solver_workers.setToolTip(
            "Maximum local processes used by Embedded RVWMO batch verification. "
            "The runtime also respects CPU affinity and LITMUS_LINK_SOLVER_WORKERS."
        )
        form.addRow("Generation scope", self.vector_complete)
        form.addRow("Output directory", self.vector_out)
        form.addRow("Random preview cases", self.vector_preview_limit)
        form.addRow("Preview distribution", self.vector_preview_sampling)
        form.addRow("Random seed", self.vector_random_seed)
        form.addRow("Generation mode", self.vector_generation_mode)
        form.addRow("Maximum generated cases", self.vector_generate_limit)
        form.addRow("Vector solver backend", self.vector_solver_backend)
        form.addRow("Verification effort", self.vector_verification_effort)
        form.addRow("Solver processes", self.vector_solver_workers)
        layout.addWidget(basic_group)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("VectorAxesScroll")
        scroll.setWidgetResizable(True)
        self.vector_filter_widget = QtWidgets.QWidget()
        filter_layout = QtWidgets.QVBoxLayout(self.vector_filter_widget)
        filter_layout.setContentsMargins(0, 0, 8, 0)
        filter_layout.setSpacing(8)
        vector_forms = [
            value for value in self.options["axes"]["vector"] if value != "none"
        ]
        core_header = QtWidgets.QLabel("Core axes")
        core_header.setObjectName("AxisSectionHeader")
        filter_layout.addWidget(core_header)
        core_groups = [
            ("Relation skeletons", "skeletons", self.options["axes"]["skeleton"], 5, _axis_label),
            ("Relation mechanisms", "mechanisms", self.options["vector_native"]["mechanisms"], 3, _axis_label),
            ("Endpoint categories", "endpoint_categories", ("vector", "scalar", "amo"), 3, _endpoint_category_label),
            ("Endpoint composition", "endpoint_compositions", self.options["vector_native"]["endpoint_compositions"], 2, _endpoint_composition_label),
            ("Vector memory forms", "forms", vector_forms, 2, _vector_form_label),
        ]
        for title, key, values, columns, labeler in core_groups:
            filter_layout.addWidget(
                self._build_vector_choice_group(
                    title, key, values, columns, labeler=labeler, role="core"
                )
            )
        parameter_header = QtWidgets.QLabel("Parameter axes")
        parameter_header.setObjectName("AxisSectionHeader")
        filter_layout.addWidget(parameter_header)
        parameter_groups = [
            ("Scalar width", "scalar_widths", self.options["vector_native"]["scalar_widths"], 4, _scalar_width_label),
            ("AMO opcode", "amo_ops", self.options["vector_native"]["amo_ops"], 3, _amo_op_label),
            ("AMO width", "amo_widths", self.options["vector_native"]["amo_widths"], 2, _amo_width_label),
            ("AMO ordering", "amo_orderings", self.options["vector_native"]["amo_orderings"], 2, _amo_ordering_label),
            ("Overlap layout", "overlap_layouts", self.options["vector_native"]["overlap_layouts"], 2, _overlap_layout_label),
            ("Vector element alignment", "alignments", self.options["vector_native"]["alignments"], 2, _vector_alignment_label),
            ("Data SEW", "sew", PARAM_AXIS_VALUES["sew"], 4, _sew_label),
            ("LMUL", "lmul", PARAM_AXIS_VALUES["lmul"], 4, _axis_label),
            ("Indexed offset EEW", "index_eew", PARAM_AXIS_VALUES["index_eew"], 4, _index_eew_label),
            ("Segment NFIELDS", "nf", PARAM_AXIS_VALUES["nf"], 4, _nf_label),
            ("Whole-register NREG", "whole_nreg", PARAM_AXIS_VALUES["whole_nreg"], 4, _whole_nreg_label),
            ("Mask mode", "mask", PARAM_AXIS_VALUES["mask"], 2, _mask_label),
            ("Tail and mask policy", "tail", PARAM_AXIS_VALUES["tail"], 4, _tail_label),
            ("Vector length", "vl", PARAM_AXIS_VALUES["vl"], 4, _vl_label),
        ]
        for title, key, values, columns, labeler in parameter_groups:
            filter_layout.addWidget(
                self._build_vector_choice_group(
                    title, key, values, columns, labeler=labeler, role="parameter"
                )
            )
        filter_layout.addStretch(1)
        scroll.setWidget(self.vector_filter_widget)
        layout.addWidget(scroll, 1)

        self.vector_complete.toggled.connect(self._update_vector_scope)
        self.vector_generation_mode.currentIndexChanged.connect(
            lambda _index: self._update_vector_generation_mode()
        )
        self._update_vector_scope(True)
        self._update_vector_generation_mode()
        return tab

    def _build_vector_choice_group(
        self,
        title: str,
        key: str,
        values: Iterable[str],
        columns: int,
        *,
        labeler: Any = None,
        role: str = "parameter",
    ) -> Any:
        QtWidgets = self.QtWidgets
        group = QtWidgets.QGroupBox(title)
        group.setObjectName("VectorFilterGroup")
        group.setProperty("axis_role", role)
        grid = QtWidgets.QGridLayout(group)
        grid.setContentsMargins(10, 8, 10, 10)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        checks: list[Any] = []
        for index, value in enumerate(values):
            value = str(value)
            check = QtWidgets.QCheckBox(labeler(value) if labeler else value)
            check.setProperty("axis_value", str(value))
            check.setToolTip(value)
            check.setChecked(True)
            grid.addWidget(check, index // columns, index % columns)
            check.toggled.connect(
                lambda checked, control=check: self._update_vector_choice_style(
                    control, checked
                )
            )
            checks.append(check)
        self.vector_checks[key] = checks
        self.vector_filter_groups.append(group)
        self.vector_group_by_key[key] = group
        if role == "core":
            self.vector_core_groups.append(group)
        else:
            self.vector_parameter_groups.append(group)
        if key == "endpoint_categories":
            self.vector_category_checks = checks
            for check in checks:
                if check.property("axis_value") == "vector":
                    check.setChecked(True)
                    check.setEnabled(False)
        elif key == "endpoint_compositions":
            self.vector_composition_checks = checks
        return group

    def _update_vector_choice_style(self, control: Any, checked: bool) -> None:
        control.setProperty("choice_state", "on" if checked else "off")
        self._refresh_widget_style(control)
        if control in self.vector_category_checks:
            self._update_vector_endpoint_scope()
        elif control in self.vector_composition_checks:
            self._ensure_vector_composition()
        elif control in self.vector_checks.get("forms", []):
            self._update_vector_form_scope()

    def _update_vector_endpoint_scope(self) -> None:
        selected = {
            str(check.property("axis_value"))
            for check in self.vector_category_checks
            if check.isChecked()
        }
        for key in ("scalar_widths", "amo_ops", "amo_widths", "amo_orderings"):
            group = self.vector_group_by_key.get(key)
            if group is not None:
                active = key.startswith("scalar") and "scalar" in selected or key.startswith("amo") and "amo" in selected
                group.setProperty("dependency_state", "active" if active else "inactive")
                group.setEnabled(active and self.vector_filter_widget.isEnabled())
                for check in self.vector_checks.get(key, []):
                    check.setEnabled(active and self.vector_filter_widget.isEnabled())
                self._refresh_widget_style(group)
        self._ensure_vector_composition()

    def _ensure_vector_composition(self) -> None:
        selected = {
            str(check.property("axis_value"))
            for check in self.vector_category_checks
            if check.isChecked()
        }
        required = {
            "vector_only": {"vector"},
            "vector_scalar": {"vector", "scalar"},
            "vector_amo": {"vector", "amo"},
            "vector_scalar_amo": {"vector", "scalar", "amo"},
        }
        for check in self.vector_composition_checks:
            name = str(check.property("axis_value"))
            valid = required.get(name, set()) <= selected
            check.setEnabled(valid and self.vector_filter_widget.isEnabled())
            if not valid:
                check.setChecked(False)
            check.setProperty(
                "choice_state", "on" if check.isChecked() else "off"
            )
            self._refresh_widget_style(check)
        if not any(check.isChecked() for check in self.vector_composition_checks):
            for check in self.vector_composition_checks:
                if str(check.property("axis_value")) == "vector_only" and check.isEnabled():
                    check.setChecked(True)
                    break

    def _update_vector_form_scope(self) -> None:
        indexed = any(
            check.isChecked()
            and "indexed" in str(check.property("axis_value"))
            for check in self.vector_checks.get("forms", [])
        )
        segment = any(
            check.isChecked()
            and str(check.property("axis_value")).startswith("segment_")
            for check in self.vector_checks.get("forms", [])
        )
        whole = any(
            check.isChecked()
            and str(check.property("axis_value")).startswith("whole_register_")
            for check in self.vector_checks.get("forms", [])
        )
        ordinary = any(
            check.isChecked()
            and not str(check.property("axis_value")).startswith("whole_register_")
            for check in self.vector_checks.get("forms", [])
        )
        for key, selected in (
            ("index_eew", indexed),
            ("nf", segment),
            ("whole_nreg", whole),
            ("lmul", ordinary),
            ("mask", ordinary),
            ("tail", ordinary),
            ("vl", ordinary),
        ):
            group = self.vector_group_by_key.get(key)
            if group is None:
                continue
            active = selected and self.vector_filter_widget.isEnabled()
            group.setEnabled(active)
            group.setProperty(
                "dependency_state", "active" if selected else "inactive"
            )
            for check in self.vector_checks.get(key, []):
                check.setEnabled(active)
            self._refresh_widget_style(group)

    def _update_vector_scope(self, complete: bool) -> None:
        if not hasattr(self, "vector_filter_widget"):
            return
        if complete:
            for checks in self.vector_checks.values():
                for check in checks:
                    check.setChecked(True)
        self.vector_filter_widget.setEnabled(not complete)
        for group in self.vector_filter_groups:
            group.setProperty("group_state", "inactive" if complete else "active")
            self._refresh_widget_style(group)
        self._update_vector_endpoint_scope()
        self._update_vector_form_scope()

    def _update_vector_generation_mode(self) -> None:
        if not hasattr(self, "vector_generation_mode"):
            return
        exhaustive = str(self.vector_generation_mode.currentData()) == "all"
        self.vector_generate_limit.setEnabled(not exhaustive)
        self.vector_generation_mode.setProperty("exhaustive", exhaustive)
        self._refresh_widget_style(self.vector_generation_mode)

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
        self.summary_view.setPlainText(
            "Choose Scalar Litmus or Vector Litmus, then run Preview Cases, "
            "Run Audit, or Generate Files."
        )
        self.preview_page = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(self.preview_page)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        self.preview_stats_label = QtWidgets.QLabel("Preview classification: no sample loaded")
        self.preview_stats_label.setObjectName("PreviewStatsLabel")

        filter_bar = QtWidgets.QVBoxLayout()
        filter_bar.setSpacing(6)
        search_row = QtWidgets.QHBoxLayout()
        search_row.setSpacing(8)
        choice_row = QtWidgets.QHBoxLayout()
        choice_row.setSpacing(8)
        self.preview_search = QtWidgets.QLineEdit()
        self.preview_search.setObjectName("PreviewSearch")
        self.preview_search.setPlaceholderText("Search case, cycle, family, or verdict")
        self.preview_search.setClearButtonEnabled(True)
        self.preview_status_filter = QtWidgets.QComboBox()
        self.preview_status_filter.addItem("All statuses", "")
        self.preview_skeleton_filter = QtWidgets.QComboBox()
        self.preview_skeleton_filter.addItem("All families", "")
        self.preview_verdict_filter = QtWidgets.QComboBox()
        self.preview_verdict_filter.addItem("All verdicts", "")
        self.preview_filter_count = QtWidgets.QLabel("0 cases")
        self.preview_filter_count.setObjectName("PreviewFilterCount")
        self.preview_stats_toggle = QtWidgets.QPushButton("Statistics")
        self.preview_stats_toggle.setCheckable(True)
        self.preview_stats_toggle.setChecked(False)
        search_row.addWidget(self.preview_search, 1)
        search_row.addWidget(self.preview_filter_count)
        search_row.addWidget(self.preview_stats_toggle)
        choice_row.addWidget(self.preview_skeleton_filter)
        choice_row.addWidget(self.preview_status_filter)
        choice_row.addWidget(self.preview_verdict_filter)
        choice_row.addStretch(1)
        filter_bar.addLayout(search_row)
        filter_bar.addLayout(choice_row)

        browser = QtWidgets.QSplitter(_horizontal(self.QtCore))
        self.preview_stats_panel = QtWidgets.QWidget()
        self.preview_stats_panel.setMinimumWidth(250)
        stats_layout = QtWidgets.QVBoxLayout(self.preview_stats_panel)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_stats = QtWidgets.QTreeWidget()
        self.preview_stats.setObjectName("PreviewStats")
        self.preview_stats.setHeaderLabels(["Classification", "Value", "Count"])
        self.preview_stats.setRootIsDecorated(True)
        self.preview_stats.setAlternatingRowColors(True)
        self.preview_stats.setUniformRowHeights(True)
        stats_layout.addWidget(self.preview_stats)

        table_panel = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_model = self.preview_model_class()
        self.preview_table = QtWidgets.QTableView()
        self.preview_table.setObjectName("PreviewTable")
        self.preview_table.setModel(self.preview_model)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setWordWrap(False)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(_no_edit_triggers(QtWidgets))
        self.preview_table.setSelectionBehavior(_select_rows(QtWidgets))
        self.preview_table.setSelectionMode(_single_selection(QtWidgets))
        self.preview_table.doubleClicked.connect(self._open_preview_detail)
        _configure_preview_header(self.preview_table, QtWidgets)
        table_layout.addWidget(self.preview_table)

        browser.addWidget(self.preview_stats_panel)
        browser.addWidget(table_panel)
        browser.setCollapsible(0, True)
        browser.setStretchFactor(0, 1)
        browser.setStretchFactor(1, 4)
        browser.setSizes([260, 700])

        self.preview_filter_timer = self.QtCore.QTimer(self.window)
        self.preview_filter_timer.setSingleShot(True)
        self.preview_filter_timer.setInterval(180)
        self.preview_filter_timer.timeout.connect(self._apply_preview_filters)
        self.preview_search.textChanged.connect(lambda _text: self.preview_filter_timer.start())
        self.preview_status_filter.currentIndexChanged.connect(lambda _index: self._apply_preview_filters())
        self.preview_skeleton_filter.currentIndexChanged.connect(lambda _index: self._apply_preview_filters())
        self.preview_verdict_filter.currentIndexChanged.connect(lambda _index: self._apply_preview_filters())
        self.preview_stats_toggle.toggled.connect(self.preview_stats_panel.setVisible)
        self.preview_stats_panel.setVisible(False)

        preview_layout.addWidget(self.preview_stats_label)
        preview_layout.addLayout(filter_bar)
        preview_layout.addWidget(browser, 1)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.raw_json = QtWidgets.QPlainTextEdit()
        self.raw_json.setReadOnly(True)
        self.result_tabs.addTab(self.summary_view, "Overview")
        self.result_tabs.addTab(self.preview_page, "Cases")
        self.result_tabs.addTab(self.log_view, "Log")
        self.result_tabs.addTab(self.raw_json, "Raw JSON")
        layout.addWidget(self.result_tabs, 1)
        return panel

    def _build_action_bar(self) -> Any:
        QtWidgets = self.QtWidgets
        controls = QtWidgets.QHBoxLayout()
        self.defer_solver = QtWidgets.QCheckBox("Skip outcome solving")
        self.defer_solver.setToolTip(
            "Generate source and metadata without running the embedded solver or optional herd7 cross-check."
        )
        self.defer_solver.setChecked(False)
        controls.addWidget(self.defer_solver)
        self.output_hint = QtWidgets.QLabel("Output: out/qt-scalar")
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

    def _set_group_state(self, group: Any, state: str) -> None:
        group.setProperty("group_state", state)
        self._refresh_widget_style(group)

    def _refresh_widget_style(self, widget: Any) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

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
                "atomic_overlaps": self._selected(self.scalar_memory_atomic_overlap_checks),
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
        if current is self.vector_tab:
            generation_mode = str(self.vector_generation_mode.currentData())
            return {
                "mode": "vector",
                "name": "vector-native-complete" if self.vector_complete.isChecked() else "vector-native-custom",
                "complete": self.vector_complete.isChecked(),
                "out": self.vector_out.text() or "out/qt-vector",
                "sample_limit": sample_limit,
                "generate_limit": (
                    None if generation_mode == "all" else self.vector_generate_limit.value()
                ),
                "preview_sampling": str(self.vector_preview_sampling.currentData()),
                "generation_mode": generation_mode,
                "random_seed": self.vector_random_seed.value(),
                "solver_backend": str(self.vector_solver_backend.currentData()),
                "verification_effort": str(self.vector_verification_effort.currentData()),
                "solver_workers": self.vector_solver_workers.value(),
                "compute_verdicts": not self.defer_solver.isChecked(),
                **{
                    key: self._selected(checks)
                    for key, checks in self.vector_checks.items()
                },
            }
        raise ValueError("Qt GUI supports only scalar and vector modes")

    def _preview_sample_limit(self) -> int:
        current = self.mode_tabs.currentWidget()
        if current is self.scalar_tab:
            return self.scalar_preview_limit.value()
        if current is self.vector_tab:
            return self.vector_preview_limit.value()
        raise ValueError("Qt GUI supports only scalar and vector modes")

    def _run_action(self, action: str, label: str) -> None:
        if self.active_thread is not None:
            self._append_log("Another action is still running; wait for it to finish.")
            return
        try:
            payload = self._payload()
        except Exception as exc:
            self._show_error("Invalid configuration", str(exc))
            return
        if (
            action == "generate"
            and payload.get("mode") == "vector"
            and payload.get("generation_mode") == "all"
            and not self._confirm_exhaustive_generation()
        ):
            self._append_log("Exhaustive generation cancelled")
            return

        if action in {"preview", "verify", "generate"}:
            self._release_preview_results()
        self.active_thread = self.action_thread
        self.active_worker = self.action_worker
        self.action_bus.request.emit(action, label, payload)

    def _release_preview_results(self) -> None:
        """Drop the previous preview before a memory-intensive action starts."""

        self.preview_filter_timer.stop()
        self.preview_table.clearSelection()
        self.preview_model.set_items(())
        self.preview_items.clear()
        self.preview_stats.clear()
        self.preview_stats_label.setText("Preview classification: loading...")
        self.preview_filter_count.setText("0 cases")
        self.raw_json.clear()
        self.summary_view.clear()
        gc.collect()

    def _confirm_exhaustive_generation(self) -> bool:
        message_box = self.QtWidgets.QMessageBox
        buttons = getattr(message_box, "StandardButton", message_box)
        yes = getattr(buttons, "Yes")
        cancel = getattr(buttons, "Cancel")
        answer = message_box.warning(
            self.window,
            "Generate all legal combinations",
            "This will generate every legal case in the current Vector configuration. "
            "A broad configuration can contain an impractically large number of files. "
            "Run Audit or narrow the filters before continuing.",
            yes | cancel,
            cancel,
        )
        return answer == yes

    def _handle_started(self, label: str) -> None:
        self.started_at = time.monotonic()
        self.active_label = label
        self.status_label.setText(f"{label} running")
        self.elapsed_label.setText("Elapsed: 0.0s")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Preparing...")
        self.log_view.clear()
        self.raw_json.clear()
        self.summary_view.setPlainText(f"{label} is running. Progress messages are shown in the Log tab.")
        self._append_log(f"Started: {label}")
        for button in self.action_buttons:
            button.setEnabled(False)
        self.elapsed_timer.start(250)
        self.result_tabs.setCurrentWidget(self.log_view)

    def _handle_progress(self, payload: object) -> None:
        if isinstance(payload, dict):
            message = str(payload.get("message", "Working"))
            current = max(int(payload.get("current", 0) or 0), 0)
            total = max(int(payload.get("total", 0) or 0), 0)
        else:
            message = str(payload)
            current = 0
            total = 0
        if total > 0:
            percent = min(100, int(current * 100 / total))
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setFormat(f"{percent}%  ({current:,}/{total:,})")
        else:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Working...")
        self.status_label.setText(f"{self.active_label}: {message}")
        self._append_log(message)

    def _handle_finished(self, label: str, result: object) -> None:
        elapsed = time.monotonic() - self.started_at
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("100%  Complete")
        self.status_label.setText(f"{label} finished")
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self.elapsed_timer.stop()
        self._append_log(f"Finished: {label} in {elapsed:.1f}s")
        if isinstance(result, dict):
            self.summary_view.setPlainText(_summary_text(label, result, self._current_out_dir()))
            if label in {"Preview Cases", "Verify Preview"}:
                self._populate_preview_list(result)
            self.raw_json.setPlainText(_raw_result_json(result))
        else:
            self.summary_view.setPlainText(str(result))
            self.raw_json.setPlainText(json.dumps({"result": str(result)}, indent=2, sort_keys=True))
        self.result_tabs.setCurrentWidget(
            self.preview_page
            if label in {"Preview Cases", "Verify Preview"} and isinstance(result, dict)
            else self.summary_view
        )

    def _handle_failed(self, label: str, message: str) -> None:
        elapsed = time.monotonic() - self.started_at if self.started_at else 0.0
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Failed")
        self.status_label.setText(f"{label} failed")
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self.elapsed_timer.stop()
        self._show_error(f"{label} failed", message)

    def _clear_worker(self) -> None:
        self.active_thread = None
        self.active_worker = None
        for button in self.action_buttons:
            button.setEnabled(True)

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
        # Show every sampled case -- generated, hand-required, and illegal --
        # so the count matches the audit summary instead of silently dropping
        # everything without a rendered litmus body.
        self.preview_items = list(result.get("sample", []))
        domain_statistics = result.get("domain_classification_counts", {}) or {}
        sample_statistics = result.get("classification_counts", {}) or {}
        statistics = dict(domain_statistics or sample_statistics)
        if sample_statistics.get("groups"):
            # The complete relation-cycle domain is counted analytically, while
            # per-form/AMO/alignment breakdowns describe the random sample.
            statistics["groups"] = sample_statistics["groups"]
        statistics.setdefault("displayed_cases", len(self.preview_items))
        statistics.setdefault("preview_displayed_cases", len(self.preview_items))
        self._populate_preview_statistics(statistics)
        self.preview_model.set_items(self.preview_items)
        self._set_preview_filter_options()
        self._apply_preview_filters()

    def _set_preview_filter_options(self) -> None:
        skeletons = sorted(
            {
                str((item.get("combination") or {}).get("skeleton", ""))
                for item in self.preview_items
                if (item.get("combination") or {}).get("skeleton")
            }
        )
        statuses = sorted(
            {
                str((item.get("decision") or {}).get("status", ""))
                for item in self.preview_items
                if (item.get("decision") or {}).get("status")
            }
        )
        verdicts = sorted({_preview_filter_verdict(item) for item in self.preview_items})
        _replace_combo_items(self.preview_skeleton_filter, "All families", skeletons)
        _replace_combo_items(self.preview_status_filter, "All statuses", statuses, _status_label)
        _replace_combo_items(self.preview_verdict_filter, "All verdicts", verdicts)

    def _apply_preview_filters(self) -> None:
        if not hasattr(self, "preview_model"):
            return
        self.preview_model.set_filters(
            self.preview_search.text(),
            str(self.preview_status_filter.currentData() or ""),
            str(self.preview_skeleton_filter.currentData() or ""),
            str(self.preview_verdict_filter.currentData() or ""),
        )
        visible = self.preview_model.rowCount()
        total = len(self.preview_items)
        self.preview_filter_count.setText(
            f"{visible:,} / {total:,}" if visible != total else f"{total:,} cases"
        )

    def _populate_preview_statistics(self, statistics: Dict[str, Any]) -> None:
        QtWidgets = self.QtWidgets
        tree = self.preview_stats
        tree.clear()
        displayed = int(
            statistics.get(
                "preview_displayed_cases",
                statistics.get("displayed_cases", len(self.preview_items)),
            )
            or 0
        )
        domain = int(statistics.get("domain_cases", 0) or 0)
        if domain:
            generated = int(statistics.get("generated_cases", domain) or 0)
            self.preview_stats_label.setText(
                f"Full domain: {generated:,} generated cases; preview shows {displayed:,}"
            )
        else:
            self.preview_stats_label.setText(
                f"Preview classification: {displayed} displayed case{'s' if displayed != 1 else ''}"
            )
        labels = {
            "status": "Generation status",
            "solver_status": "Solver status",
            "verdict": "Solver verdict",
            "external_status": "External check",
            "solver_backend": "Solver backend",
            "skeleton": "Skeleton",
            "category": "Category",
            "memory_layout": "Memory layout",
            "attribute": "Memory attribute",
            "memory_event": "Memory event",
            "vector": "Vector axis",
            "cmo": "CMO axis",
            "tlb": "TLB axis",
            "sew": "Vector SEW",
            "lmul": "Vector LMUL",
            "index_eew": "Indexed EEW",
            "nf": "Segment NFIELDS",
            "whole_nreg": "Whole-register NREG",
            "mask": "Vector mask",
            "tail": "Vector tail policy",
            "vl": "Vector VL",
            "endpoint_category": "Endpoint instances",
            "endpoint_composition": "Endpoint composition",
            "scalar_width": "Scalar width",
            "amo_opcode": "AMO opcode",
            "amo_width": "AMO width",
            "amo_ordering": "AMO ordering",
            "overlap_layout": "Overlap layout",
            "vector_event_form": "Vector instruction form",
            "alignment": "Vector alignment",
            "relation_mechanism": "Relation mechanism",
        }
        groups = statistics.get("groups", {}) or {}
        for key, values in groups.items():
            if not isinstance(values, dict):
                continue
            total = sum(int(count) for count in values.values())
            parent = QtWidgets.QTreeWidgetItem([labels.get(key, key), "", str(total)])
            for value, count in sorted(values.items(), key=lambda item: (-int(item[1]), str(item[0]))):
                parent.addChild(QtWidgets.QTreeWidgetItem(["", str(value), str(count)]))
            parent.setExpanded(
                key in {"status", "solver_status", "verdict", "external_status", "skeleton"}
            )
            tree.addTopLevelItem(parent)
        for column in range(3):
            tree.resizeColumnToContents(column)

    def _open_preview_detail(self, item: Any) -> None:
        if self.preview_dialog_open:
            return
        index = item.data(_user_role(self.QtCore))
        if index is None or index < 0 or index >= len(self.preview_items):
            return
        dialog = _LitmusPreviewDialog(self.QtWidgets, self.QtCore, self.QtGui, self.preview_items[index], self.window)
        screen = self.window.screen()
        if screen is None:
            screen = self.QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            dialog.resize(
                max(640, min(1320, int(available.width() * 0.9))),
                max(560, min(980, int(available.height() * 0.9))),
            )
        else:
            dialog.resize(1180, 860)
        self.preview_dialog_open = True
        try:
            _exec_dialog(dialog)
        finally:
            self.preview_dialog_open = False

    def _current_out_dir(self) -> str:
        current = self.mode_tabs.currentWidget()
        if current is self.scalar_tab:
            return self.scalar_out.text() or "out/qt-scalar"
        if current is self.vector_tab:
            return self.vector_out.text() or "out/qt-vector"
        raise ValueError("Qt GUI supports only scalar and vector modes")

    def _update_output_hint(self) -> None:
        if hasattr(self, "output_hint"):
            self.output_hint.setText(f"Output: {self._current_out_dir()}")
        if hasattr(self, "defer_solver") and hasattr(self, "mode_tabs"):
            self.defer_solver.setVisible(self.mode_tabs.currentWidget() is self.vector_tab)


def _summary_text(label: str, result: Dict[str, Any], out_dir: str) -> str:
    counts = result.get("report") if isinstance(result.get("report"), dict) else result
    lines = [f"{label} complete", ""]
    if "profile" in result:
        lines.append(f"Profile: {result['profile']}")
    if "source" in result and result["source"]:
        lines.append(f"Source: {result['source']}")
    audit = result.get("audit") if isinstance(result.get("audit"), dict) else {}
    verification_effort = counts.get(
        "verification_effort",
        audit.get("verification_effort"),
    )
    if verification_effort:
        lines.append(f"Verification effort: {verification_effort}")
    solver_workers = counts.get(
        "solver_workers",
        audit.get("solver_workers"),
    )
    if solver_workers is not None:
        lines.append(f"Solver processes used: {solver_workers}")
    lines.extend(
        [
            f"Output directory: {out_dir}",
            "",
            "Counts:",
            f"  total combinations: {counts.get('total_combinations', counts.get('available_litmus', counts.get('total_cases', '-')))}",
            f"  raw combinations: {counts.get('raw_combinations', '-')}",
            f"  generated: {counts.get('generated', counts.get('generated_litmus', '-'))}",
            f"  excluded illegal: {counts.get('excluded_illegal', '-')}",
            f"  excluded unsupported: {counts.get('excluded_unsupported', '-')}",
            f"  HAND-required: {counts.get('hand_required', '-')}",
            f"  missing: {counts.get('missing', '-')}",
        ]
    )
    classifications = result.get("classification_counts")
    groups = classifications.get("groups", {}) if isinstance(classifications, dict) else {}
    if groups:
        lines.extend(["", "Preview distribution:"])
        for key, title in (
            ("endpoint_composition", "composition"),
            ("scalar_width", "scalar width"),
            ("amo_opcode", "AMO opcode"),
            ("amo_width", "AMO width"),
            ("amo_ordering", "AMO ordering"),
            ("sew", "Vector SEW"),
            ("vector_event_form", "Vector form"),
            ("overlap_layout", "overlap"),
            ("verdict", "verdict"),
            ("external_status", "external check"),
        ):
            values = groups.get(key)
            if isinstance(values, dict) and values:
                rendered = ", ".join(
                    f"{name}={count}" for name, count in sorted(values.items())
                )
                lines.append(f"  {title}: {rendered}")
    if label == "Generate Files":
        out_path = Path(out_dir)
        schema = str(result.get("schema", ""))
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
                f"  generation mode: {result.get('generation_mode', 'configured-domain')}",
                f"  solver results: {solver_files}",
                f"  solver verdicts: {_compact_counts(result.get('solver_verdict'))}",
                f"  external checks: {_compact_counts(result.get('external_status'))}",
                f"  verdict mode: {result.get('verdict_mode', 'computed')}",
                f"  diagram mode: {result.get('diagram_mode', 'on_demand')}",
                f"  generated diagrams: {result.get('generated_diagrams', 0)}",
                f"  @all: {out_path / '@all'}",
            ]
        )
        if schema == "litmus-link.vector-native-generation.v1":
            lines.append(f"  generation report: {out_path / 'generation-report.json'}")
            lines.append(f"  audit report: {out_path / 'audit-report.json'}")
        elif schema.startswith("litmus-link.scalar-"):
            lines.append(f"  generation report: {out_path / 'generation-report.json'}")
        elif not schema.startswith("litmus-link.scalar-"):
            lines.append(f"  audit report: {out_path / 'audit-report.json'}")
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
            file_name = item.get("file_name") or "no file"
            lines.append(f"  {item.get('name', '<unnamed>')}  [{file_name}]")
    return "\n".join(lines)


def _compact_counts(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ", ".join(f"{key}={count}" for key, count in sorted(value.items()))


def _raw_result_json(result: Dict[str, Any]) -> str:
    """Serialize result metadata without duplicating every preview source."""
    compact = dict(result)
    sample = compact.pop("sample", None)
    if isinstance(sample, list):
        compact["sample"] = {
            "omitted_from_raw_json": True,
            "count": len(sample),
            "reason": "Cases remain available in the virtualized Cases tab and per-case inspector.",
        }
    return json.dumps(compact, indent=2, sort_keys=True)


def _vector_solver_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    solver = item.get("solver") or {}
    vector = solver.get("vector") if isinstance(solver, dict) else None
    return vector if isinstance(vector, dict) else {}


def _embedded_solver_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    solver = item.get("solver") or {}
    vector = _vector_solver_payload(item)
    embedded = vector.get("embedded") if vector else None
    if isinstance(embedded, dict):
        return embedded
    embedded = solver.get("embedded") if isinstance(solver, dict) else None
    if isinstance(embedded, dict):
        return embedded
    if isinstance(solver, dict) and solver.get("schema") == "litmus-link.embedded-rvwmo.v1":
        return solver
    return {}


def _solver_external_payload(solver: Dict[str, Any]) -> Dict[str, Any]:
    vector = solver.get("vector")
    if isinstance(vector, dict) and isinstance(vector.get("external"), dict):
        return dict(vector["external"])
    if isinstance(solver.get("external"), dict):
        return dict(solver["external"])
    if isinstance(solver.get("herd7"), dict):
        return dict(solver["herd7"])
    return {}


def _solver_external_status(solver: Dict[str, Any]) -> str:
    external = _solver_external_payload(solver)
    if external:
        return str(external.get("status", "external_unsupported"))
    return str(solver.get("cross_check", "not_run") or "not_run")


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"0x{value:x}"
    return str(value)


def _transaction_detail_text(item: Dict[str, Any]) -> str:
    embedded = _embedded_solver_payload(item)
    events = embedded.get("events") if isinstance(embedded, dict) else None
    case_ir = item.get("case_ir") or {}
    lines = ["Architectural memory transactions", ""]
    if isinstance(events, list):
        visible = [event for event in events if not bool(event.get("initial"))]
        for event in visible:
            footprint = ", ".join(str(value) for value in event.get("footprint", []))
            access = "RMW" if event.get("amo") else "R" if event.get("read") else "W"
            hart = event.get("hart")
            lines.append(
                f"{event.get('event_id', '?')}  P{hart if hart is not None else '-'}  "
                f"{access}  {event.get('transaction_kind', 'scalar_plain')}"
            )
            lines.append(
                f"  bytes={event.get('access_size', '?')}  offset={event.get('byte_offset', '-')}  "
                f"footprint=[{footprint}]"
            )
            if event.get("read"):
                lines.append(f"  read={_format_value(event.get('read_value'))}")
            if event.get("write"):
                lines.append(f"  write={_format_value(event.get('write_value'))}")
            if event.get("amo"):
                lines.append(
                    "  AMO "
                    f"op={event.get('amo_operation')} "
                    f"ordering={event.get('amo_ordering')} "
                    f"old={_format_value(event.get('read_value'))} "
                    f"operand={_format_value(event.get('amo_operand'))} "
                    f"new={_format_value(event.get('write_value'))}"
                )
            lines.append("")
    else:
        for hart in case_ir.get("harts", []) if isinstance(case_ir, dict) else []:
            for event in hart:
                access = event.get("memory_access") if isinstance(event, dict) else None
                if not isinstance(access, dict):
                    continue
                kind = str(access.get("transaction_kind", "scalar_plain"))
                lines.append(f"{event.get('event_id', '?')}  {kind}")
                lines.append(
                    f"  bytes={access.get('size_bytes', '?')}  "
                    f"offset={access.get('offset_bytes', '-')}  "
                    f"footprint={access.get('covered_bytes', [])}"
                )
                if event.get("read_value") is not None:
                    lines.append(f"  read={_format_value(event.get('read_value'))}")
                if event.get("write_value") is not None:
                    lines.append(f"  write={_format_value(event.get('write_value'))}")
                if event.get("kind") == "amo":
                    lines.append(
                        "  AMO "
                        f"op={event.get('amo_op')} "
                        f"ordering={event.get('amo_ordering')} "
                        f"old={_format_value(event.get('read_value'))} "
                        f"operand={_format_value(event.get('amo_operand'))} "
                        f"new={_format_value(event.get('write_value'))}"
                    )
                lines.append("")

    vector = _vector_solver_payload(item)
    vector_ir = vector.get("vector_ir") if isinstance(vector, dict) else None
    instructions = vector_ir.get("instructions") if isinstance(vector_ir, dict) else None
    if isinstance(instructions, list) and instructions:
        lines.extend(["", "Vector element execution", ""])
        preserved = vector_ir.get("preserved_element_order", [])
        preserved_by_parent: Dict[str, list[str]] = {}
        for pair in preserved if isinstance(preserved, list) else []:
            if isinstance(pair, list) and len(pair) == 2:
                parent = str(pair[0]).split(".e", 1)[0]
                preserved_by_parent.setdefault(parent, []).append(
                    f"{pair[0]} -> {pair[1]}"
                )
        for instruction in instructions:
            active = [
                element
                for element in instruction.get("elements", [])
                if element.get("active")
            ]
            offsets = ", ".join(
                (
                    f"e{element.get('index')}.f{element.get('field_index')}"
                    if element.get("field_index") is not None
                    else f"e{element.get('index')}"
                )
                + f"@+{element.get('offset_bytes')}"
                for element in active
            )
            parent = str(instruction.get("event_id", "?"))
            policy = (
                "ordered siblings"
                if preserved_by_parent.get(parent)
                else "unordered siblings"
            )
            lines.append(
                f"{parent}: {instruction.get('form')}  active={len(active)}  {policy}"
            )
            lines.append(f"  {offsets or 'no active elements'}")
            for relation in preserved_by_parent.get(parent, []):
                lines.append(f"  order: {relation}")
    elif isinstance(case_ir, dict):
        metadata = case_ir.get("metadata", {})
        vectors = metadata.get("vectors", {}) if isinstance(metadata, dict) else {}
        if isinstance(vectors, dict) and vectors:
            lines.extend(["", "Vector element execution", ""])
            for parent, config in sorted(vectors.items()):
                if not isinstance(config, dict):
                    continue
                sew_bits = int(config.get("sew_bits", 0) or 0)
                sew = f"e{sew_bits}" if sew_bits else "e32"
                lmul = str(config.get("lmul", "m1"))
                avl = str(config.get("avl", "vl1"))
                active_count = vector_effective_vl(sew, lmul, avl)
                if active_count is not None and config.get("mask") == "masked":
                    active_count = (active_count + 1) // 2
                policy = (
                    "ordered siblings"
                    if config.get("ordered_elements")
                    else "unordered siblings"
                )
                lines.append(
                    f"{parent}: {config.get('form', 'vector')}  "
                    f"active segments={active_count if active_count is not None else '?'}  "
                    f"NFIELDS={config.get('nf', 1)}  {policy}"
                )
                lines.append(
                    f"  SEW={sew_bits or '?'}  LMUL={lmul}  AVL={avl}  "
                    f"mask={config.get('mask', 'unmasked')}"
                )
    return "\n".join(lines).rstrip() or "No transaction metadata is available."


def _relation_detail_text(item: Dict[str, Any]) -> str:
    embedded = _embedded_solver_payload(item)
    execution = embedded.get("execution") if isinstance(embedded, dict) else None
    lines = ["Execution relations", ""]
    if not isinstance(execution, dict):
        lines.append("No complete execution witness is available.")
        reason = embedded.get("reason") if isinstance(embedded, dict) else None
        if reason:
            lines.append(f"Reason: {reason}")
        return "\n".join(lines)

    byte_relations = execution.get("byte_relations") or {}
    for kind in ("rf", "co", "fr"):
        entries = byte_relations.get(kind, []) if isinstance(byte_relations, dict) else []
        lines.append(f"{kind} by byte ({len(entries)}):")
        if entries:
            lines.extend(
                f"  {entry.get('src')} -> {entry.get('dst')}  [{entry.get('byte')}]"
                for entry in entries
                if isinstance(entry, dict)
            )
        else:
            lines.append("  none")
        lines.append("")
    for kind in ("po", "po_loc", "ppo"):
        entries = execution.get(kind, [])
        lines.append(f"{kind} transaction edges ({len(entries)}):")
        lines.extend(
            f"  {pair[0]} -> {pair[1]}"
            for pair in entries
            if isinstance(pair, list) and len(pair) == 2
        )
        if not entries:
            lines.append("  none")
        lines.append("")
    return "\n".join(lines).rstrip()


def _value_detail_text(item: Dict[str, Any]) -> str:
    case_ir = item.get("case_ir") or {}
    metadata = case_ir.get("metadata", {}) if isinstance(case_ir, dict) else {}
    plan = metadata.get("value_plan") if isinstance(metadata, dict) else None
    analysis = item.get("analysis") or {}
    lines = ["Exists reconstruction", "", str(analysis.get("exists", "-")), ""]
    if not isinstance(plan, dict):
        lines.append("No byte-provenance value plan is available for this case.")
        return "\n".join(lines)

    endpoints = plan.get("endpoints", {})
    lines.append("Endpoint values:")
    for vertex, value in sorted(
        endpoints.items(), key=lambda entry: int(entry[0])
    ):
        if not isinstance(value, dict):
            continue
        line = (
            f"  E{vertex} {value.get('category')}/{value.get('direction')} "
            f"{int(value.get('width_bytes', 0) or 0) * 8}-bit"
        )
        details = []
        for key, label in (
            ("read_memory_value", "memory-read"),
            ("read_register_value", "register-read"),
            ("write_value", "write"),
            ("amo_old", "old"),
            ("amo_operand", "operand"),
            ("amo_new", "new"),
        ):
            if value.get(key) is not None:
                details.append(f"{label}={_format_value(value[key])}")
        lines.append(line + ("  " + "  ".join(details) if details else ""))

    lines.extend(["", "Coherence writer order:"])
    for location, order in sorted((plan.get("co_orders") or {}).items()):
        lines.append(f"  {location}: " + " -> ".join(f"E{value}" for value in order))
    lines.extend(["", "Observed final bytes:"])
    final = plan.get("final_bytes") or {}
    for location, offsets in sorted((plan.get("observed_final_bytes") or {}).items()):
        image = final.get(location, {}) if isinstance(final, dict) else {}
        rendered = ", ".join(
            f"{location}[{offset}]={_format_value(image.get(str(offset), image.get(offset)))}"
            for offset in offsets
        )
        lines.append(f"  {rendered}")
    lines.extend(
        [
            "",
            "Outcome:",
            f"  {analysis.get('outcome_interpretation', analysis.get('forbidden_outcome', '-'))}",
        ]
    )
    return "\n".join(lines)


def _external_detail_text(item: Dict[str, Any]) -> str:
    solver = item.get("solver") or {}
    external = _solver_external_payload(solver)
    status = _solver_external_status(solver)
    lines = [f"External check: {status}"]
    if not external:
        lines.extend(
            [
                "",
                "No external check was run. Embedded RVWMO remains the primary result.",
                "Select the Vector cross-check backend to request herd7 scalar projections.",
            ]
        )
        return "\n".join(lines)
    for key in ("reason", "oracle_kind", "exact", "requested_projections", "generated_projections"):
        if external.get(key) is not None:
            lines.append(f"{key.replace('_', ' ').title()}: {external[key]}")
    projection = external.get("projection")
    if isinstance(projection, dict):
        lines.extend(["", "Scalar projection:"])
        for key in (
            "status",
            "oracle_kind",
            "exact",
            "storage_mode",
            "requested_projections",
            "generated_projections",
            "reason",
        ):
            if projection.get(key) is not None:
                lines.append(f"  {key.replace('_', ' ')}: {projection[key]}")
    capabilities = external.get("capabilities")
    if isinstance(capabilities, dict):
        lines.extend(
            [
                "",
                f"herd7 available: {capabilities.get('available', False)}",
                f"herd7 version: {(capabilities.get('tool') or {}).get('version', '-')}",
                f"mixed-size support: {(capabilities.get('mixed_size') or {}).get('supported', False)}",
            ]
        )
    results = external.get("results", external.get("projections"))
    if isinstance(results, list):
        lines.extend(["", f"Projection results ({len(results)}):"])
        for result in results:
            if isinstance(result, dict):
                lines.append(
                    f"  {result.get('name', '?')}: "
                    f"{result.get('status', '?')} / {result.get('verdict', '?')}"
                )
    return "\n".join(lines)


class _LitmusPreviewDialog:
    def __init__(self, QtWidgets: Any, QtCore: Any, QtGui: Any, item: Dict[str, Any], parent: Any) -> None:
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.item = item
        self.diagram_rendering = False
        self.diagram_pixmap = None
        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle(str(item.get("name", "Litmus preview")))
        self._build_ui()
        self.QtCore.QTimer.singleShot(0, self._fit_diagram)
        self.QtCore.QTimer.singleShot(25, self._start_diagram_render)

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
        file_name = QtWidgets.QLabel(
            f"File: {self.item.get('file_name') or '-'}"
        )
        file_name.setObjectName("DialogFileName")
        file_name.setTextInteractionFlags(_text_selectable(self.QtCore))
        layout.addWidget(file_name)

        splitter = QtWidgets.QSplitter(_vertical(self.QtCore))
        splitter.addWidget(self._build_diagram_view())
        splitter.addWidget(self._build_detail_tabs())
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([650, 250])
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
        self.diagram_label = QtWidgets.QLabel()
        self.diagram_label.setAlignment(_align_center(self.QtCore))
        self.diagram_label.setMinimumSize(0, 0)
        size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
        self.diagram_label.setSizePolicy(size_policy.Ignored, size_policy.Ignored)
        self.diagram_status = QtWidgets.QLabel("Diagram will be rendered when this window opens.")
        self.diagram_status.setAlignment(_align_center(self.QtCore))
        self.diagram_progress = QtWidgets.QProgressBar()
        self.diagram_progress.setRange(0, 0)
        self.diagram_progress.setTextVisible(False)
        scroll_class = _make_fit_scroll_area_class(QtWidgets)
        self.diagram_scroll = scroll_class(self._fit_diagram)
        self.diagram_scroll.setWidgetResizable(True)
        self.diagram_scroll.setWidget(self.diagram_label)
        if png and png.exists():
            if self._show_diagram_png(png):
                self.diagram_status.setText("Loaded cached diagram")
                self.diagram_progress.hide()
        else:
            self.diagram_label.setText("Rendering diagram on demand...")
        layout.addWidget(self.diagram_status)
        layout.addWidget(self.diagram_progress)
        layout.addWidget(self.diagram_scroll, 1)
        return container

    def _build_detail_tabs(self) -> Any:
        QtWidgets = self.QtWidgets
        tabs = QtWidgets.QTabWidget()
        self.summary_detail = self._text_view(self._analysis_text())
        tabs.addTab(self.summary_detail, "Summary")
        tabs.addTab(
            self._text_view(_transaction_detail_text(self.item)), "Transactions"
        )
        tabs.addTab(
            self._text_view(_relation_detail_text(self.item)), "Byte Relations"
        )
        tabs.addTab(self._text_view(_value_detail_text(self.item)), "Values")
        tabs.addTab(self._text_view(_external_detail_text(self.item)), "External")
        tabs.addTab(self._text_view(str(self.item.get("litmus", ""))), "Litmus")
        tabs.addTab(self._json_view(self.item.get("solver", {})), "Solver")
        tabs.addTab(self._json_view(self.item.get("case_ir", {})), "IR")
        self.diagram_detail = self._json_view(self.item.get("diagram", {}))
        tabs.addTab(self.diagram_detail, "Diagram")
        return tabs

    def _start_diagram_render(self) -> None:
        if self.diagram_rendering:
            return
        diagram = self.item.get("diagram") or {}
        png = Path(str(diagram.get("png", ""))) if diagram.get("png") else None
        if png and png.exists():
            if self._show_diagram_png(png):
                self.diagram_status.setText("Loaded cached diagram")
                self.diagram_progress.hide()
            return
        if not isinstance(self.item.get("case_ir"), dict):
            self._diagram_failed("this preview case has no case IR to draw")
            return

        self.diagram_rendering = True
        self.diagram_status.setText("Rendering diagram on demand...")
        self.diagram_progress.show()
        try:
            self._diagram_ready(materialize_preview_diagram(self.item))
        except Exception as exc:
            self._diagram_failed(str(exc))
        finally:
            self.diagram_rendering = False

    def _diagram_ready(self, result: object) -> None:
        if not isinstance(result, dict):
            self._diagram_failed("diagram renderer returned an invalid result")
            return
        self.item["diagram"] = result
        png = Path(str(result.get("png", ""))) if result.get("png") else None
        if not png or not png.exists():
            self._diagram_failed(f"diagram renderer did not write {png or '<unknown>'}")
            return
        if not self._show_diagram_png(png):
            return
        self.diagram_status.setText("Loaded cached diagram" if result.get("cached") else "Diagram rendered")
        self.diagram_progress.hide()
        self.summary_detail.setPlainText(self._analysis_text())
        self.diagram_detail.setPlainText(json.dumps(result, indent=2, sort_keys=True))

    def _diagram_failed(self, message: str) -> None:
        self.diagram_pixmap = None
        self.diagram_label.clear()
        self.diagram_label.setText(f"Diagram generation failed:\n{message}")
        self.diagram_status.setText("Diagram unavailable")
        self.diagram_progress.hide()

    def _show_diagram_png(self, png: Path) -> bool:
        pixmap = self.QtGui.QPixmap(str(png))
        if pixmap.isNull():
            self._diagram_failed(f"cannot load diagram PNG: {png}")
            return False
        self.diagram_pixmap = pixmap
        self.diagram_label.clear()
        self._fit_diagram()
        self.QtCore.QTimer.singleShot(0, self._fit_diagram)
        return True

    def _fit_diagram(self) -> None:
        if self.diagram_pixmap is None or not hasattr(self, "diagram_scroll"):
            return
        viewport = self.diagram_scroll.viewport().size()
        width = max(viewport.width() - 12, 1)
        height = max(viewport.height() - 12, 1)
        scaled = self.diagram_pixmap.scaled(
            width,
            height,
            _keep_aspect_ratio(self.QtCore),
            _smooth_transformation(self.QtCore),
        )
        self.diagram_label.setPixmap(scaled)

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
        axes = self.item.get("name", combination.get("name", ""))
        lines = [
            f"Case: {axes}",
            f"File name: {self.item.get('file_name') or '-'}",
            f"Case ID: {self.item.get('case_id') or '-'}",
            f"Status: {decision.get('status', '-')}",
            f"RVWMO class: {decision.get('rvwmo_class', '-')}",
            f"Expected kind: {decision.get('expected_kind', '-')}",
            f"Solver: {solver.get('status', '-')} / {solver.get('verdict', '-')}",
            f"External check: {_solver_external_status(solver)}",
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
        formal_scope = (
            (self.item.get("case_ir") or {}).get("metadata", {}).get("formal_scope")
            if isinstance(self.item.get("case_ir"), dict)
            else None
        )
        if formal_scope:
            lines.extend(["", f"Formal scope: {formal_scope}"])
        return "\n".join(lines)


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


def _text_selectable(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    flags = getattr(qt, "TextInteractionFlag", qt)
    return flags.TextSelectableByMouse


def _keep_aspect_ratio(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    modes = getattr(qt, "AspectRatioMode", qt)
    return modes.KeepAspectRatio


def _smooth_transformation(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    modes = getattr(qt, "TransformationMode", qt)
    return modes.SmoothTransformation


def _user_role(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    if hasattr(qt, "ItemDataRole"):
        return qt.ItemDataRole.UserRole
    return qt.UserRole


def _display_role(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    if hasattr(qt, "ItemDataRole"):
        return qt.ItemDataRole.DisplayRole
    return qt.DisplayRole


def _tooltip_role(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    if hasattr(qt, "ItemDataRole"):
        return qt.ItemDataRole.ToolTipRole
    return qt.ToolTipRole


def _foreground_role(QtCore: Any) -> Any:
    qt = getattr(QtCore, "Qt")
    if hasattr(qt, "ItemDataRole"):
        return qt.ItemDataRole.ForegroundRole
    return qt.ForegroundRole


def _horizontal_orientation(QtCore: Any) -> Any:
    return _horizontal(QtCore)


def _standard_arrow_icon(QtWidgets: Any, widget: Any, direction: str = "right") -> Any:
    style = getattr(QtWidgets, "QStyle")
    standard = style.StandardPixmap if hasattr(style, "StandardPixmap") else style
    names = {
        "right": "SP_ArrowRight",
        "left": "SP_ArrowLeft",
        "down": "SP_ArrowDown",
    }
    try:
        arrow = getattr(standard, names[direction])
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unsupported flow-arrow direction: {direction}") from exc
    return widget.style().standardIcon(arrow)


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


def _configure_responsive_form(form: Any, QtWidgets: Any) -> None:
    form_layout = QtWidgets.QFormLayout
    wrap_policy = getattr(form_layout, "RowWrapPolicy", form_layout)
    growth_policy = getattr(form_layout, "FieldGrowthPolicy", form_layout)
    form.setRowWrapPolicy(wrap_policy.WrapLongRows)
    form.setFieldGrowthPolicy(growth_policy.AllNonFixedFieldsGrow)


def _configure_preview_header(table: Any, QtWidgets: Any) -> None:
    header = table.horizontalHeader()
    resize = getattr(QtWidgets.QHeaderView, "ResizeMode", QtWidgets.QHeaderView)
    # Fixed/interactive widths avoid ResizeToContents scanning every row in a
    # 100k-case model.  The final cycle column consumes remaining space.
    column_count = table.model().columnCount()
    header.setStretchLastSection(True)
    for column in range(column_count):
        header.setSectionResizeMode(column, resize.Interactive)
    for column, width in enumerate((44, 72, 68, 150, 190)):
        table.setColumnWidth(column, width)
    header.setSectionResizeMode(column_count - 1, resize.Stretch)


_STATUS_LABELS = {
    "generated": "GEN",
    "hand_required": "HAND",
    "excluded_illegal": "ILLEGAL",
    "excluded_unsupported": "UNSUPP",
    "verified": "VERIFIED",
    "inconclusive": "INCONCLUSIVE",
    "conflict": "CONFLICT",
    "external_unsupported": "EXT UNSUPP",
}
_STATUS_COLORS = {
    "generated": "#16a34a",
    "hand_required": "#d97706",
    "excluded_illegal": "#dc2626",
    "excluded_unsupported": "#64748b",
    "verified": "#0f766e",
    "inconclusive": "#b45309",
    "conflict": "#dc2626",
    "external_unsupported": "#64748b",
}


def _status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status.upper())


def _axis_label(value: str) -> str:
    return value


def _endpoint_category_label(value: str) -> str:
    return {"vector": "Vector", "scalar": "Scalar", "amo": "AMO"}.get(value, value)


def _endpoint_composition_label(value: str) -> str:
    return {
        "vector_only": "V only",
        "vector_scalar": "V + S",
        "vector_amo": "V + A",
        "vector_scalar_amo": "V + S + A",
    }.get(value, value)


def _vector_form_label(value: str) -> str:
    return {
        "unit_load": "Unit-stride load",
        "unit_store": "Unit-stride store",
        "strided_load": "Strided load",
        "strided_store": "Strided store",
        "indexed_unordered_load": "Indexed-unordered load",
        "indexed_unordered_store": "Indexed-unordered store",
        "indexed_ordered_load": "Indexed-ordered load",
        "indexed_ordered_store": "Indexed-ordered store",
        "segment_unit_load": "Segment unit-stride load",
        "segment_unit_store": "Segment unit-stride store",
        "segment_strided_load": "Segment strided load",
        "segment_strided_store": "Segment strided store",
        "segment_indexed_unordered_load": "Segment indexed-unordered load",
        "segment_indexed_unordered_store": "Segment indexed-unordered store",
        "segment_indexed_ordered_load": "Segment indexed-ordered load",
        "segment_indexed_ordered_store": "Segment indexed-ordered store",
        "whole_register_load": "Whole-register load",
        "whole_register_store": "Whole-register store",
    }.get(value, value)


def _scalar_width_label(value: str) -> str:
    return {"b": "B (8-bit)", "h": "H (16-bit)", "w": "W (32-bit)", "d": "D (64-bit)"}.get(value, value)


def _amo_op_label(value: str) -> str:
    return f"amo{value}"


def _amo_width_label(value: str) -> str:
    return {"w": "W (32-bit)", "d": "D (64-bit)"}.get(value, value)


def _amo_ordering_label(value: str) -> str:
    return {
        "relaxed": "Relaxed",
        "aq": ".aq",
        "rl": ".rl",
        "aqrl": ".aqrl",
    }.get(value, value)


def _overlap_layout_label(value: str) -> str:
    return {
        "same_start": "Same start",
        "contained": "Contained overlap",
        "low_partial": "Low-side overlap",
        "high_partial": "High-side overlap",
        "disjoint_control": "Disjoint control",
    }.get(value, value)


def _vector_alignment_label(value: str) -> str:
    return {
        "aligned": "Naturally aligned",
        "misalign_same16": "Misaligned within 16B",
        "misalign_cross16": "Misaligned crossing 16B",
        "misalign_cross64": "Misaligned crossing 64B",
    }.get(value, value)


def _sew_label(value: str) -> str:
    return value.upper()


def _whole_nreg_label(value: str) -> str:
    return value.replace("nreg", "NREG=")


def _index_eew_label(value: str) -> str:
    return value.upper()


def _nf_label(value: str) -> str:
    return value.removeprefix("nf") + " fields"


def _mask_label(value: str) -> str:
    return {"unmasked": "Unmasked", "masked": "Masked (even elements)"}.get(value, value)


def _tail_label(value: str) -> str:
    return value.replace("_", ",")


def _vl_label(value: str) -> str:
    return "VLMAX" if value == "vlmax" else value.upper()


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


def _preview_table_values(item: Dict[str, Any], source_index: int) -> list[str]:
    combination = item.get("combination", {}) or {}
    decision = item.get("decision", {}) or {}
    case_ir = item.get("case_ir", {}) or {}
    status = str(decision.get("status", "unknown"))
    skeleton = str(combination.get("skeleton", case_ir.get("skeleton", "?")))
    cycle = str(case_ir.get("cycle", (item.get("analysis") or {}).get("cycle", "")))
    name = str(case_ir.get("display_name", item.get("name", f"case-{source_index + 1}")))
    file_name = str(item.get("file_name", ""))
    return [
        str(source_index + 1),
        _status_label(status),
        skeleton,
        _verdict_label(item.get("solver"), decision),
        _short_file_name(file_name) if file_name else "-",
        name,
        cycle,
    ]


def _short_file_name(file_name: str) -> str:
    match = re.fullmatch(r"(LLV-[A-Za-z0-9_.-]+-)([0-9a-f]{64})(\.litmus)", file_name)
    if not match:
        return file_name
    prefix, digest, suffix = match.groups()
    return f"{prefix}{digest[:10]}...{digest[-8:]}{suffix}"


def _preview_filter_verdict(item: Dict[str, Any]) -> str:
    solver = item.get("solver") or {}
    status = str(solver.get("status", ""))
    verdict = str(solver.get("verdict", ""))
    if status == "conflict" or verdict == "conflict":
        return "conflict"
    if status == "inconclusive":
        return "inconclusive"
    if status == "verified" and verdict in {"allowed", "observable"}:
        return "allowed"
    if status == "verified" and verdict:
        return verdict
    if status:
        return status
    decision = item.get("decision") or {}
    return str(decision.get("status", "unknown"))


def _replace_combo_items(
    combo: Any,
    all_label: str,
    values: Iterable[str],
    labeler: Any = None,
) -> None:
    selected = str(combo.currentData() or "")
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(all_label, "")
    for value in values:
        combo.addItem(labeler(value) if labeler else value, value)
    index = combo.findData(selected)
    combo.setCurrentIndex(index if index >= 0 else 0)
    combo.blockSignals(False)


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
        external = _solver_external_status(solver)
        if status == "verified":
            verdict = solver.get("verdict", "verified")
            verdict = "allowed" if verdict == "observable" else verdict
            return (
                f"{verdict} / ext unsupported"
                if external == "external_unsupported"
                else str(verdict)
            )
        if status == "conflict":
            return "conflict"
        if status == "inconclusive":
            return "inconclusive"
        if status in {"unchecked", "not_applicable", "unavailable"}:
            return solver.get("verdict") or status
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
    QLabel#AxisSectionHeader { color: #0f4c5c; font-size: 14px; font-weight: 700; padding: 7px 2px 2px 2px; }
    QLabel#FlowArrow { color: #64748b; font-size: 18px; font-weight: 700; }
    QLabel#SectionTitle { color: #111827; font-size: 17px; font-weight: 700; }
    QLabel#DialogFileName { color: #475569; font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 12px; }
    QTabWidget::pane { border: 1px solid #cfd9e6; border-radius: 7px; background: #ffffff; }
    QTabBar::tab { background: #e7edf5; color: #475569; padding: 8px 14px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
    QTabBar::tab:selected { background: #ffffff; color: #0f766e; font-weight: 700; border-top: 3px solid #0f766e; }
    QGroupBox { border: 1px solid #d9e2ec; border-radius: 6px; margin-top: 10px; padding-top: 10px; font-weight: 700; background: #ffffff; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QGroupBox#AxisGroup[axis_role="parameter"] { border: 1px solid #d7dee8; background: #ffffff; }
    QGroupBox#AxisGroup[axis_role="parameter"][group_state="active"] { border: 2px solid #0f766e; background: #ecfdf5; }
    QGroupBox#AxisGroup[axis_role="parameter"][group_state="inactive"] { border: 1px solid #d7dee8; background: #f8fafc; }
    QGroupBox#VectorFilterGroup[group_state="active"] { border: 2px solid #0f766e; background: #ecfdf5; }
    QGroupBox#VectorFilterGroup[group_state="inactive"] { border: 1px solid #d7dee8; background: #f8fafc; }
    QGroupBox#VectorFilterGroup[axis_role="core"][group_state="active"] { border: 2px solid #0f766e; background: #ecfdf5; }
    QGroupBox#VectorFilterGroup[axis_role="parameter"][group_state="active"] { border: 1px solid #2563eb; background: #eff6ff; }
    QGroupBox#VectorFilterGroup[dependency_state="inactive"] { border: 1px solid #d7dee8; background: #f8fafc; color: #94a3b8; }
    QLineEdit, QComboBox, QPlainTextEdit { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px; }
    QComboBox#VectorGenerationMode[exhaustive="true"] { border: 2px solid #b45309; background: #fff7ed; color: #9a3412; font-weight: 700; }
    QPlainTextEdit { font-family: monospace; font-size: 12px; }
    QTableView#PreviewTable { background: #ffffff; alternate-background-color: #f8fafc; border: 1px solid #cfd9e6; border-radius: 7px; gridline-color: #edf2f7; font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 12px; selection-background-color: #dbeafe; selection-color: #0c4a6e; }
    QLineEdit#PreviewSearch { min-width: 260px; }
    QLabel#PreviewFilterCount { color: #334155; font-weight: 700; min-width: 90px; }
    QLabel#PreviewStatsLabel { color: #0f766e; font-weight: 700; padding: 2px 4px; }
    QTreeWidget#PreviewStats { background: #f8fafc; border: 1px solid #cfd9e6; border-radius: 6px; alternate-background-color: #ffffff; }
    QTreeWidget#PreviewStats::item { padding: 3px 6px; }
    QHeaderView::section { background: #172033; color: #ffffff; padding: 6px 8px; border: none; border-right: 1px solid #2a3650; font-weight: 700; }
    QCheckBox { spacing: 7px; padding: 3px 6px; border-radius: 5px; }
    QCheckBox[choice_state="on"] { background: #d1fae5; color: #064e3b; font-weight: 700; }
    QCheckBox[choice_state="off"] { background: #fee2e2; color: #7f1d1d; font-weight: 700; }
    QCheckBox#VectorComplete:checked { background: #d1fae5; color: #064e3b; font-weight: 700; }
    QCheckBox:disabled { color: #94a3b8; background: transparent; }
    QPushButton { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 12px; font-weight: 700; }
    QPushButton:hover { background: #eef6ff; }
    QPushButton:disabled { color: #94a3b8; background: #f1f5f9; }
    QPushButton#GenerateButton { background: #0f766e; color: #ffffff; border-color: #0f766e; }
    QProgressBar { background: #e8eef6; border: 1px solid #cbd5e1; border-radius: 6px; text-align: center; min-height: 18px; }
    QProgressBar::chunk { background: #0f766e; border-radius: 5px; }
    """
