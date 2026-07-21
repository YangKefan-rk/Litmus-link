from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from litmus_link.qt_gui import _LitmusLinkQtWindow, _display_role, _load_qt_modules
from litmus_link.workflow import materialize_preview_diagram, preview_payload


@pytest.fixture(scope="module")
def qt_app():  # type: ignore[no-untyped-def]
    try:
        QtWidgets, QtCore, QtGui, binding = _load_qt_modules()
    except Exception as exc:
        pytest.skip(f"Qt binding unavailable: {exc}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app, QtWidgets, QtCore, QtGui, binding


def _process_until(app, predicate, timeout: float = 20.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("Qt action did not finish")
        time.sleep(0.005)
    app.processEvents()


def test_preview_action_can_run_twice(qt_app) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, binding = qt_app
    ui = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    ui.mode_tabs.setCurrentWidget(ui.profile_tab)
    ui.window.show()
    try:
        for _ in range(2):
            ui.preview_button.click()
            # A second click can arrive before the worker's started signal reaches
            # the GUI thread; it must be rejected without creating another thread.
            ui.preview_button.click()
            _process_until(app, lambda: ui.active_thread is None)
            assert ui.preview_items
            assert all(item["solver"]["status"] == "unchecked" for item in ui.preview_items)
    finally:
        ui.shutdown()
        ui.window.close()
        app.processEvents()


def test_preview_detail_rejects_reentrant_open(qt_app) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, binding = qt_app
    ui = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    try:
        ui.window.show()
        result = preview_payload({"mode": "profile", "profile": "smoke", "sample_limit": 1})
        ui._populate_preview_list(result)
        table_item = ui.preview_model.index(0, 0)
        preview_item = ui.preview_items[0]
        png = Path(str(preview_item["diagram"]["png"]))
        shutil.rmtree(png.parent, ignore_errors=True)

        QtCore.QTimer.singleShot(30, lambda: ui._open_preview_detail(table_item))
        QtCore.QTimer.singleShot(
            250,
            lambda: app.activeModalWidget() and app.activeModalWidget().accept(),
        )
        ui._open_preview_detail(table_item)

        assert ui.preview_dialog_open is False
        assert png.exists()
    finally:
        ui.shutdown()
        ui.window.close()
        app.processEvents()


def test_preview_only_describes_diagrams_until_a_case_is_opened() -> None:
    initial = preview_payload({"mode": "profile", "profile": "smoke", "sample_limit": 1})
    initial_item = initial["sample"][0]
    initial_png = Path(str(initial_item["diagram"]["png"]))
    shutil.rmtree(initial_png.parent, ignore_errors=True)

    preview = preview_payload({"mode": "profile", "profile": "smoke", "sample_limit": 1})
    item = preview["sample"][0]
    png = Path(str(item["diagram"]["png"]))
    diagram_json = png.with_suffix("").with_suffix(".diagram.json")

    assert item["diagram"]["status"] == "deferred"
    assert not png.exists()
    assert not diagram_json.exists()

    materialized = materialize_preview_diagram(item)
    assert materialized["status"] == "ready"
    assert png.exists()
    assert diagram_json.exists()


def test_preview_model_virtualizes_and_filters_large_case_sets(qt_app) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, binding = qt_app
    ui = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    try:
        items = [
            {
                "name": f"MP+po+case.{index}",
                "combination": {"skeleton": "MP"},
                "decision": {"status": "generated"},
                "solver": {"status": "unchecked", "verdict": "unchecked"},
                "case_ir": {
                    "display_name": f"MP+po+case.{index}",
                    "cycle": "PodWW -> Rfe -> PodRR -> Fre",
                },
            }
            for index in range(50_000)
        ]
        ui.preview_model.set_items(items)
        assert ui.preview_model.rowCount() == 50_000
        assert ui.preview_model.data(ui.preview_model.index(49_999, 4), _display_role(QtCore)) == "MP+po+case.49999"

        ui.preview_model.set_filters(search="case.49999")
        assert ui.preview_model.rowCount() == 1
        assert ui.preview_model.data(ui.preview_model.index(0, 4), _display_role(QtCore)) == "MP+po+case.49999"
    finally:
        ui.shutdown()
        ui.window.close()
        app.processEvents()


def test_vector_tab_exposes_complete_and_filtered_generation(qt_app) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, binding = qt_app
    ui = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    try:
        labels = [ui.mode_tabs.tabText(index) for index in range(ui.mode_tabs.count())]
        assert "Vector Litmus" in labels
        ui.mode_tabs.setCurrentWidget(ui.vector_tab)
        complete = ui._payload()
        assert complete["mode"] == "vector"
        assert complete["complete"] is True
        assert complete["out"] == "out/qt-vector"
        assert set(complete["forms"]) == {
            "unit_load", "unit_store", "strided_load", "strided_store",
            "indexed_ordered_load", "indexed_ordered_store",
            "indexed_unordered_load", "indexed_unordered_store",
        }

        ui.vector_complete.setChecked(False)
        assert ui.vector_filter_widget.isEnabled()
        filtered = ui._payload()
        assert filtered["complete"] is False
        assert filtered["random_seed"] == 1
        assert filtered["generate_limit"] == 10_000
        assert set(filtered["endpoint_modes"]) == {"P", "AMO", "Aq", "Rl", "AR"}
        assert set(filtered["mechanisms"]) == {"po", "fence", "dependency"}
        assert set(filtered["alignments"]) == {
            "aligned", "misalign_same16", "misalign_cross16", "misalign_cross64"
        }

        ui.mode_tabs.setCurrentWidget(ui.custom_tab)
        ui._select_supported_vector_matrix()
        assert ui.mode_tabs.currentWidget() is ui.vector_tab
        assert ui.vector_complete.isChecked()
        assert "relation-cycle Vector domain" in ui.status_label.text()
    finally:
        ui.shutdown()
        ui.window.close()
        app.processEvents()
