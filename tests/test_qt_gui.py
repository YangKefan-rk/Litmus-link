from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from litmus_link.qt_gui import _LitmusLinkQtWindow, _load_qt_modules
from litmus_link.workflow import preview_payload


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
    for _ in range(2):
        ui.preview_button.click()
        _process_until(app, lambda: ui.active_thread is None)
        assert ui.preview_items
    ui.window.close()
    app.processEvents()


def test_preview_detail_rejects_reentrant_open(qt_app) -> None:  # type: ignore[no-untyped-def]
    app, QtWidgets, QtCore, QtGui, binding = qt_app
    ui = _LitmusLinkQtWindow(QtWidgets, QtCore, QtGui, binding)
    ui.window.show()
    result = preview_payload({"mode": "profile", "profile": "smoke", "sample_limit": 1})
    ui._populate_preview_list(result)
    table_item = ui.preview_table.item(0, 0)
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
    ui.window.close()
    app.processEvents()
