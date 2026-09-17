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
    page._show_preview()
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


def test_page_two_stage_accept_flow(qtbot, ctx, tmp_path):
    """地块生态两段式：生成底图 → 展示中间结果 → 用户接受 → 本地继续。"""
    from core.api.mock_clients import MockImageAPI
    from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow

    page = TilemapPage(ctx)
    qtbot.addWidget(page)
    page.show()
    page._category_combo.setCurrentIndex(page._category_combo.findData("ground"))
    params = TilemapParams(
        description="草地", category="ground", features={"水塘": "pond"},
        tile_size=32, map_width=10, map_height=8, output_dir=tmp_path / "out",
    )
    session = TilemapWorkflow(image_api=MockImageAPI()).run_to_base(params)
    page._stage = "base"
    page._on_worker_done(session)
    # 中间底图已保留并展示，等待确认
    assert session.sheet_path.exists()
    assert page._accept_btn.isVisible() and page._accept_btn.isEnabled()
    assert page._preview_label.pixmap() is not None
    assert page._edit_btn.isEnabled() is False
    # 用户接受 → 本地继续处理 → 全部完成
    # 确认时可选「基础块位置」：改选手动位置应影响到裁切（无需重新生图）
    page._base_pos_combo.setCurrentIndex(page._base_pos_combo.findData("br"))
    page._on_accept()
    assert page._session.result is not None
    assert page._session.ecosystem.base_pos == "br"
    assert page._map_btn.isEnabled()
    assert page._accept_btn.isVisible() is False


def test_page_category_and_multiterrain_view(qtbot, ctx, tmp_path):
    """地块生态页：类别切换、多地形画笔、预览渲染。"""
    from core.workflow.tilemap_workflow import TilemapWorkflow
    from core.api.mock_clients import MockImageAPI

    page = TilemapPage(ctx)
    qtbot.addWidget(page)
    page.show()
    # 切换到地块生态类别并生成（直接走工作流，绕过线程）
    page._category_combo.setCurrentIndex(page._category_combo.findData("ground"))
    assert page._feature_rows[0][0].isVisible()  # 特征槽可见
    # 基础块位置选择器：仅地块生态类别可见，默认自动识别
    assert page._base_pos_combo.isVisible()
    assert page._base_pos_combo.currentData() == "auto"
    params = TilemapParams(
        description="草地", category="ground", features={"水塘": "pond"},
        tile_size=32, map_width=10, map_height=8, output_dir=tmp_path / "out",
    )
    result = TilemapWorkflow(image_api=MockImageAPI()).run(params)
    page._session = result.session
    page._show_preview()
    assert page._preview_label.pixmap() is not None
    # 多地形视图：地形画笔切换 + 铺设
    view = TilemapView(
        result.session.map_model,
        center=None,
        terrain_labels={1: "基础", 2: "水塘"},
    )
    qtbot.addWidget(view)
    view.show()
    assert view._terrain_combo.count() == 2
    view._terrain_combo.setCurrentIndex(1)  # 水塘
    view._paint_cell(QPoint(3 * 32 * view._zoom + 2, 3 * 32 * view._zoom + 2))
    assert view.model().cell(3, 3) == 2
    # 建筑拼件视图
    bparams = TilemapParams(
        description="stone wall", category="building",
        tile_size=32, map_width=8, map_height=6, output_dir=tmp_path / "bout",
    )
    bres = TilemapWorkflow(image_api=MockImageAPI()).run(bparams)
    bview = TilemapView(
        bres.session.map_model,
        center=None,
        pieces=bres.session.pieces["pieces"],
    )
    qtbot.addWidget(bview)
    bview.show()
    assert bview._piece_combo.count() == 21  # （无拼件）+ 墙体 16 族 16 件 + 实心 + 门×2 + 立柱
    bview._piece_combo.setCurrentIndex(1)   # straight
    bview._paint_cell(QPoint(2 * 32 * bview._zoom + 2, 2 * 32 * bview._zoom + 2))
    assert (2, 2) in bview.model().overlay


def test_view_accepts_pack_added_later_with_scale(qtbot, ctx, tmp_path):
    """预览里后加的瓦片包立即可用（拼件画笔/缩放/自动墙），缩放在放置时生效。"""
    from PIL import Image

    from core.tilemap import TileMapModel
    from core.tilemap.pack import TilePack
    from core.tilemap.walls import build_piece_set, wall_art_from_sheet
    from core.tilemap.tiles import BaseTileSet, BuildingSheet
    from ui.widgets.tilemap_view import TilemapView

    model = TileMapModel(6, 4, tile_size=32)
    model.set_cell(0, 0, 1)
    view = TilemapView(model)
    qtbot.addWidget(view)
    assert not view._piece_combo.isEnabled()

    wall = BaseTileSet(size=32, center=Image.new("RGBA", (32, 32), (150, 140, 128, 255)),
                       edges={}, corners={})
    art = wall_art_from_sheet(BuildingSheet(wall=wall, top=wall, opening=wall, pillar=wall))
    pack = TilePack(name="墙包", category="building", tile_size=32, pieces=build_piece_set(art),
                    wall_art=art)
    view.add_pack(pack)
    assert view._piece_combo.isEnabled() and view._piece_combo.count() > 1
    assert view._auto_wall_check.isEnabled()
    view._scale_spin.setValue(200)
    view._piece_combo.setCurrentIndex(1)
    view._paint_cell(__import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(32, 0))
    item = next(iter(model.overlay.values()))
    assert item[3] == 2.0, item


def test_map_preview_available_without_generating(qtbot, ctx, tmp_path):
    """预览无需先生成：按钮一开始可用；空白模型可直接加载瓦片包并铺设。"""
    import numpy as np
    from PIL import Image

    from core.tilemap import TileMapModel
    from core.tilemap.pack import TilePack
    from core.tilemap.tiles import EDGE_NAMES, BaseTileSet
    from ui.pages.tilemap_page import TilemapPage
    from ui.widgets.tilemap_view import TilemapView

    page = TilemapPage(ctx)
    qtbot.addWidget(page)
    assert page._map_btn.isEnabled(), "未生成时也应能进入地图预览"
    model = page.scratch_map_model()
    assert model.width == page._map_w_spin.value() and model.height == page._map_h_spin.value()
    assert not np.asarray(model.grid).any()

    tex = Image.new("RGBA", (32, 32), (90, 140, 80, 255))
    ground = Image.new("RGBA", (32, 32), (236, 240, 246, 255))
    art = BaseTileSet(size=32, center=tex, edges={n: tex for n in EDGE_NAMES},
                      corners={n: tex for n in ("tl", "tr", "bl", "br")},
                      line_color=(0, 0, 0), line_width=1, band=8, radius=8, base_texture=ground)
    pack = TilePack(name="草地包", category="ground", tile_size=32, terrains={1: art},
                    terrain_names={1: "草地"}, base_terrain=1)
    view = TilemapView(model)
    qtbot.addWidget(view)
    view.add_pack(pack)
    # 空地图铺上该包的基础地形，画笔切到它，且可以直接画
    assert np.asarray(model.grid).any(), "加载地块包后应自动铺满基础地形"
    assert view._paint_terrain == 1
    view._paint_cell(__import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(0, 0))
    assert int(model.grid[0, 0]) == 1
