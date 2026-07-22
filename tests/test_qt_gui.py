from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from litmus_link.qt_gui import (
    _LitmusLinkQtWindow,
    _LitmusPreviewDialog,
    _display_role,
    _load_qt_modules,
    _short_file_name,
)
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


def _scalar_preview_payload(sample_limit: int = 1) -> dict[str, object]:
    return {
        "mode": "scalar",
        "engine": "native_templates",
        "skeletons": ["MP"],
        "mechanisms": ["po"],
        "annotations": ["P"],
        "include_same": False,
        "sample_limit": sample_limit,
        "judge": False,
    }


def test_preview_action_can_run_twice(qt_app) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, binding = qt_app
    ui = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    ui.mode_tabs.setCurrentWidget(ui.scalar_tab)
    ui.scalar_preview_limit.setValue(1)
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
        result = preview_payload(_scalar_preview_payload())
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
    initial = preview_payload(_scalar_preview_payload())
    initial_item = initial["sample"][0]
    initial_png = Path(str(initial_item["diagram"]["png"]))
    shutil.rmtree(initial_png.parent, ignore_errors=True)

    preview = preview_payload(_scalar_preview_payload())
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
                "file_name": f"LLV-MP-{index:064x}.litmus",
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
        assert ui.preview_model.HEADERS == [
            "#", "Status", "Family", "Verdict", "File name", "Case"
        ]
        assert ui.preview_model.data(
            ui.preview_model.index(49_999, 4), _display_role(QtCore)
        ) == _short_file_name(f"LLV-MP-{49_999:064x}.litmus")
        assert ui.preview_model.data(
            ui.preview_model.index(49_999, 5), _display_role(QtCore)
        ) == "MP+po+case.49999"

        ui.preview_model.set_filters(search="case.49999")
        assert ui.preview_model.rowCount() == 1
        assert ui.preview_model.data(
            ui.preview_model.index(0, 5), _display_role(QtCore)
        ) == "MP+po+case.49999"
    finally:
        ui.shutdown()
        ui.window.close()
        app.processEvents()


def test_vector_tab_exposes_complete_and_filtered_generation(qt_app) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, binding = qt_app
    ui = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    try:
        labels = [ui.mode_tabs.tabText(index) for index in range(ui.mode_tabs.count())]
        assert labels == ["Scalar Litmus", "Vector Litmus"]
        result_labels = [
            ui.result_tabs.tabText(index) for index in range(ui.result_tabs.count())
        ]
        assert result_labels == ["Overview", "Cases", "Log", "Raw JSON"]
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
        assert filtered["preview_sampling"] == "balanced"
        assert filtered["generation_mode"] == "balanced"
        assert set(filtered["endpoint_modes"]) == {"P", "AMO", "Aq", "Rl", "AR"}
        assert set(filtered["mechanisms"]) == {"po", "fence", "dependency"}
        assert filtered["alignments"] == ["aligned"]

        all_index = ui.vector_generation_mode.findData("all")
        ui.vector_generation_mode.setCurrentIndex(all_index)
        assert not ui.vector_generate_limit.isEnabled()
        exhaustive = ui._payload()
        assert exhaustive["generation_mode"] == "all"
        assert exhaustive["generate_limit"] is None

        weighted_index = ui.vector_generation_mode.findData("domain_weighted")
        ui.vector_generation_mode.setCurrentIndex(weighted_index)
        assert ui.vector_generate_limit.isEnabled()
        assert ui._payload()["generation_mode"] == "domain_weighted"

    finally:
        ui.shutdown()
        ui.window.close()
        app.processEvents()


def test_flow_panel_switches_layout_without_overlapping_cards(qt_app) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, binding = qt_app
    ui = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    ui.window.show()
    try:
        ui.window.resize(1440, 850)
        _process_until(app, lambda: ui.flow_panel.layout_mode == "wide")

        ui.window.resize(760, 850)
        _process_until(app, lambda: ui.flow_panel.layout_mode == "compact")
        visible_cards = [card for card in ui.flow_panel.compact_cards if card.isVisible()]
        assert len(visible_cards) == 4
        for index, card in enumerate(visible_cards):
            for other in visible_cards[index + 1 :]:
                assert not card.geometry().intersects(other.geometry())
    finally:
        ui.shutdown()
        ui.window.close()
        app.processEvents()


def test_preview_dialog_fits_full_png_in_first_view(qt_app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, _binding = qt_app
    png = tmp_path / "large-diagram.png"
    source = QtGui.QPixmap(1600, 1100)
    source.fill(QtGui.QColor("white"))
    assert source.save(str(png), "PNG")
    item = {
        "name": "MP+{PodWW>Rfe>PodRR>Fre}",
        "file_name": "LL-MP-test.litmus",
        "diagram": {"png": str(png), "status": "ready"},
        "analysis": {},
        "decision": {},
        "solver": {},
    }
    parent = QtWidgets.QWidget()
    dialog = _LitmusPreviewDialog(QtWidgets, QtCore, QtGui, item, parent)
    dialog.resize(900, 700)
    dialog.show()
    try:
        _process_until(
            app,
            lambda: dialog.diagram_label.pixmap() is not None
            and not dialog.diagram_label.pixmap().isNull(),
        )
        displayed = dialog.diagram_label.pixmap().size()
        viewport = dialog.diagram_scroll.viewport().size()
        assert displayed.width() <= viewport.width()
        assert displayed.height() <= viewport.height()
        assert dialog.diagram_scroll.horizontalScrollBar().maximum() == 0
        assert dialog.diagram_scroll.verticalScrollBar().maximum() == 0
    finally:
        dialog.close()
        parent.close()
        app.processEvents()
