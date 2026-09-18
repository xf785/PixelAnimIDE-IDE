"""像素页瓦片包/素材包支持 + 左栏切换器的回归测试。"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.tilemap.pack import TilePack, export_tileset_dir
from core.tilemap.tiles import EDGE_NAMES, BaseTileSet
from ui.widgets.pack_browser import CATEGORIES, PackBrowser

S = 32


def _terrain(col=(80, 150, 90)) -> BaseTileSet:
    tex = Image.new("RGBA", (S, S), col + (255,))
    ground = Image.new("RGBA", (S, S), (60, 120, 70, 255))
    return BaseTileSet(size=S, center=tex, edges={n: tex for n in EDGE_NAMES},
                       corners={n: tex for n in ("tl", "tr", "bl", "br")},
                       line_color=(24, 22, 24), line_width=1, band=8, radius=8, base_texture=ground,
                       art_meta={"outline": [[24, 22, 24]], "outline_px": 1, "bevel": [],
                                 "bevel_px": 0, "edge_noise_px": 0})


def _prop(name="tree") -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((12, 6, 20, 28), fill=(90, 140, 70, 255))
    return img


def _make_packs(tmp_path):
    """造一个地块包 + 一个素材包（都导出成目录/zip，走真实导入路径）。"""
    ground = TilePack(name="草地包", category="ground", tile_size=S,
                      terrains={1: _terrain(), 2: _terrain((70, 120, 190))},
                      terrain_names={1: "草地", 2: "水"}, base_terrain=1,
                      sheets={"ai_sheet": Image.new("RGBA", (64, 64), (255, 255, 255, 255))})
    props = TilePack(name="树木包", category="prop", tile_size=S,
                     pieces={"tree_1": _prop(), "tree_2": _prop()})
    g = export_tileset_dir(tmp_path / "ground", ground)
    p = export_tileset_dir(tmp_path / "props", props)
    return g, p


# --------------------------------------------------------------------------- #
def test_browser_loads_packs_and_switches_categories(qtbot, tmp_path):
    g, p = _make_packs(tmp_path)
    browser = PackBrowser()
    qtbot.addWidget(browser)
    assert browser.stats()["packs"] == 0

    assert browser.add_pack_from_path(g["dir"]) == (2, 0)      # 目录导入
    assert browser.add_pack_from_path(p["zip"]) == (0, 2)      # zip 导入
    st = browser.stats()
    assert st == {"packs": 2, "terrains": 2, "pieces": 2, "props": 2, "sheets": 1}

    def count(cat: str) -> int:
        for btn in browser._group.buttons():
            if btn.property("category") == cat:
                btn.setChecked(True)
        browser._refresh_assets()
        return browser._grid.count()

    assert count("all") == 2 + 2 + 1          # 地形 2 + 拼件 2 + 底图 1
    assert count("terrain") == 2
    assert count("prop") == 2
    assert count("building") == 0
    assert count("sheet") == 1
    # 搜索过滤（先切回"全部"，否则会叠加分类过滤）
    for btn in browser._group.buttons():
        if btn.property("category") == "all":
            btn.setChecked(True)
    browser._search.setText("tree")
    browser._refresh_assets()
    assert browser._grid.count() == 2


def test_browser_asset_choice_and_signals(qtbot, tmp_path):
    g, p = _make_packs(tmp_path)
    browser = PackBrowser()
    qtbot.addWidget(browser)
    browser.add_pack_from_path(p["dir"])
    chosen, replaced = [], []
    browser.assetChosen.connect(lambda img, name: chosen.append((img, name)))
    browser.assetReplaceRequested.connect(lambda img, name: replaced.append((img, name)))
    changed = []
    browser.packChanged.connect(lambda: changed.append(1))

    browser._grid.setCurrentRow(0)
    browser._btn_put.click()
    assert chosen and isinstance(chosen[0][0], Image.Image)
    browser._btn_replace.click()
    assert replaced and replaced[0][1] == chosen[0][1]
    # 移除包 -> 资源清空 + 发信号
    browser.remove_current_pack()
    assert browser.stats()["packs"] == 0 and browser._grid.count() == 0
    browser.add_pack_from_path(g["dir"])
    assert changed, "包列表变化应发出 packChanged（用于持久化）"
    browser.clear()
    assert browser.pack_paths() == []


def test_browser_checkbox_filters_visible_packs(qtbot, tmp_path):
    g, p = _make_packs(tmp_path)
    browser = PackBrowser()
    qtbot.addWidget(browser)
    browser.add_pack_from_path(g["dir"])
    browser.add_pack_from_path(p["dir"])
    assert len(browser.visible_packs()) == 2
    from PySide6.QtCore import Qt

    # 取消勾选「素材包」（第 1 行），只留地块包可见
    browser._pack_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert len(browser.visible_packs()) == 1
    browser._refresh()
    st = browser.stats()
    assert st["terrains"] == 2 and st["props"] == 0, st   # 只统计可见包


def test_pixel_page_restores_packs_and_places_asset(qtbot, tmp_path):
    """像素页：左栏有包浏览器；放入画布 = 居中合成且保持画布尺寸；路径可持久化。"""
    from PySide6.QtWidgets import QGroupBox

    from config.api_config import APIConfigManager
    from core.storage.keyring import Keyring
    from ui.app_context import AppContext, UISettings
    from ui.pages.pixel_page import PixelPage

    ctx = AppContext(
        api=APIConfigManager(config_file=tmp_path / "api.json", keyring=Keyring(tmp_path / ".keyring")),
        ui_settings=UISettings(tmp_path / "ui.json"),
    )

    g, p = _make_packs(tmp_path)
    page = PixelPage(ctx)
    qtbot.addWidget(page)
    assert page._pack_browser is not None
    groups = [w for w in page._settings_panel.findChildren(QGroupBox)]
    assert any(w is page._pack_box for w in groups), "左栏应有「瓦片包 / 素材包」分组"

    page._pack_browser.add_pack_from_path(p["dir"])
    page._remember_packs()
    assert ctx.ui_settings.get("pixel_pack_paths"), "包路径应写入 UI 设置以便下次恢复"

    before = page._editor.frame().size
    page._editor.set_frame(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
    page._pack_browser._grid.setCurrentRow(0)
    page._pack_browser._btn_put.click()
    assert page._editor.frame().size == (64, 64), "放入画布不应改变画布尺寸"
    assert np.asarray(page._editor.frame())[..., 3].max() == 255, "资源应被合成进画布"

    # 替换画布：尺寸随资源变化
    page._pack_browser._btn_replace.click()
    assert page._editor.frame().size == (S, S)
    assert before is not None

    # 新页面应能自动恢复（模拟重开）
    page2 = PixelPage(ctx)
    qtbot.addWidget(page2)
    assert page2._pack_browser.stats()["packs"] == 1


def test_category_switcher_labels_are_translated():
    from ui.i18n import LANG_PACKS

    for _key, label in CATEGORIES:
        assert label in LANG_PACKS["en"], f"切换器标签缺少英文: {label}"
