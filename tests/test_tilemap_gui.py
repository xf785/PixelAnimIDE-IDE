"""瓦片地图 UI 冒烟测试（QT_QPA_PLATFORM=offscreen）。"""
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from config.api_config import APIConfig, APIConfigManager
from core.storage.keyring import Keyring
from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow
from ui.app_context import AppContext, UISettings
from ui.main_window import MainWindow
from ui.pages.tilemap_page import TilemapPage
from ui.widgets.tile_editor import TileEditorDialog, base_set_with_edits
from ui.widgets.tilemap_view import TilemapView
from ui.workers import TilemapWorker


@pytest.fixture()
def ctx(tmp_path):
    """带默认 mock 配置的应用上下文。"""
    api = APIConfigManager(
        config_file=tmp_path / "api_config.json",
        keyring=Keyring(tmp_path / ".keyring"),
    )
    for kind in ("llm", "image", "video"):
        api.add(
            APIConfig(
                kind=kind,
                name=f"mock-{kind}",
                base_url="mock",
                model="mock-model",
                params={"mock": True, "frames": 8, "fps": 8},
            )
        )
    return AppContext(api=api, ui_settings=UISettings(tmp_path / "ui_settings.json"))


def _make_result(tmp_path, mode="47"):
    from core.api.mock_clients import MockImageAPI

    params = TilemapParams(
        description="grass", tile_size=32, sheet_size=768, atlas_mode=mode,
        map_width=10, map_height=8, output_dir=tmp_path / "out",
    )
    return TilemapWorkflow(image_api=MockImageAPI()).run(params)


def test_main_window_tilemap_mode(qtbot, ctx):
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    window.switch_page("tilemap")
    assert window._stack.currentIndex() == 4
    assert window._mode == "tilemap"
    assert window._mode_tilemap_btn.isChecked()
    assert window.statusBar().currentMessage()  # 模式提示文案


def test_page_preview_and_tile_editor(qtbot, ctx, tmp_path):
    page = TilemapPage(ctx)
    qtbot.addWidget(page)
    page.show()
    result = _make_result(tmp_path)
    page._session = result.session
    page._show_atlas()
    assert page._preview_label.pixmap() is not None and not page._preview_label.pixmap().isNull()

    dialog = TileEditorDialog(result.session.base)
    qtbot.addWidget(dialog)
    edited = dialog.result()
    assert set(edited) == {"tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"}
    new_base = base_set_with_edits(result.session.base, edited)
    assert new_base.size == result.session.base.size


def test_tilemap_view_paint_and_zoom(qtbot, ctx, tmp_path):
    result = _make_result(tmp_path, mode="dual")
    model = result.session.map_model
    model.clear()
    view = TilemapView(
        model,
        result.session.processed.center,
        line_color=result.session.processed.line_color,
        line_width=1,
        atlas_mode="dual",
    )
    qtbot.addWidget(view)
    view.show()
    assert view._canvas.pixmap() is not None
    view._paint_cell(QPoint(5 * 32 * view._zoom + 2, 5 * 32 * view._zoom + 2))
    assert view.model().cell(5, 5) == 1
    view.set_zoom(5)
    assert view.zoom() == 5
    view.clear()
    assert view.model().cell(5, 5) == 0


def test_tilemap_worker_thread(qtbot, ctx, tmp_path):
    params = TilemapParams(
        description="grass", tile_size=32, sheet_size=768, atlas_mode="47",
        map_width=10, map_height=8, output_dir=tmp_path / "out",
    )
    worker = TilemapWorker(ctx.api, params)
    with qtbot.waitSignal(worker.succeeded, timeout=60000) as blocker:
        worker.start()
    result = blocker.args[0]
    assert result.atlas_path.exists()
    assert result.map_preview_path.exists()
