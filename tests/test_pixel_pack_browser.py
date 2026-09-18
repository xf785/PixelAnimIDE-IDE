"""像素页瓦片包/素材包支持 + 左栏切换器的回归测试。"""
import numpy as np
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
    """像素页：左停靠栏有包浏览器；放入画布 = 居中合成且保持画布尺寸；路径可持久化。"""
    from config.api_config import APIConfigManager
    from core.storage.keyring import Keyring
    from ui.app_context import AppContext, UISettings
    from ui.pages.pixel_page import PixelPage
    from ui.widgets.dock import Docker

    ctx = AppContext(
        api=APIConfigManager(config_file=tmp_path / "api.json", keyring=Keyring(tmp_path / ".keyring")),
        ui_settings=UISettings(tmp_path / "ui.json"),
    )

    g, p = _make_packs(tmp_path)
    page = PixelPage(ctx)
    qtbot.addWidget(page)
    assert page._pack_browser is not None
    # 左停靠栏里应有 PackBrowser 提供的「瓦片包」「资源浏览」两个 docker
    titles = [d.title() for d in page._left_dock.panels()]
    assert titles == ["瓦片包", "资源浏览"], titles
    assert all(isinstance(d, Docker) for d in page._left_dock.panels())
    # 三栏工作区（左栏 | 画布 | 右栏），每条分隔线都能拖
    assert page._splitter.count() == 3
    assert page._left_dock.minimumWidth() == 0 and page._right_dock.minimumWidth() == 0

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


# --------------------------------------------------------------------------- #
# 包内「逐级目录」浏览
# --------------------------------------------------------------------------- #
def test_browser_browses_every_folder_level(qtbot, tmp_path):
    """导入的包要能逐级浏览内部目录：atlas / tiles/terrain_1 / source / textures。"""
    g, _p = _make_packs(tmp_path)
    browser = PackBrowser()
    qtbot.addWidget(browser)
    browser.add_pack_from_path(g["dir"])

    # 目录树：顶层 = 「全部包·逻辑资源」 + 每个可见包
    assert browser._tree.topLevelItemCount() == 2
    pack_node = browser._tree.topLevelItem(1)
    dirs = {pack_node.child(i).text(0).split("  ")[0] for i in range(pack_node.childCount())}
    assert {"逻辑资源", "atlas", "textures", "tiles", "source"} <= dirs, dirs

    # 展开到最深的 tiles/terrain_1：应能看到 47 张单瓦片
    browser.set_scope(0, "tiles/terrain_1")
    assert browser._grid.count() == 47, browser._grid.count()
    got = browser.asset_of(browser._grid.item(0))
    assert got is not None and got[1].size == (S, S)  # 逐张瓦片是可送入画布的真实图片

    # 上一层 tiles/<id> 目录 = 所有地形的瓦片（递归统计）
    browser.set_scope(0, "tiles")
    assert browser._grid.count() == 47 * 2

    # atlas 目录 = 47 图集大图；source 目录 = 文生图底图
    browser.set_scope(0, "atlas")
    assert browser._grid.count() == 2
    browser.set_scope(0, "source")
    assert browser._grid.count() == 1

    # 整个包（rel_dir 为空）= 包内全部图片
    browser.set_scope(0, "")
    assert browser._grid.count() == browser.file_count(0)
    assert browser._grid.count() > 47 * 2, "整包应包含图集/纹理/底图等全部资源"
    assert "草地包" in browser.scope_text()

    # 分类切换器在目录范围内同样生效（单瓦片 / 图集 / 纹理）
    def count(cat: str) -> int:
        for btn in browser._group.buttons():
            if btn.property("category") == cat:
                btn.setChecked(True)
        browser._refresh_assets()
        return browser._grid.count()

    browser.set_scope(0, "")
    assert count("tile") == 47 * 2
    assert count("atlas") == 2
    assert count("texture") == 4
    assert count("sheet") == 1


def test_browser_imports_plain_image_folder(qtbot, tmp_path):
    """没有 manifest 的普通图片文件夹也能导入并逐级浏览（按图片文件夹）。"""
    folder = tmp_path / "loose"
    (folder / "chars").mkdir(parents=True)
    (folder / "tiles").mkdir(parents=True)
    Image.new("RGBA", (16, 16), (200, 30, 30, 255)).save(folder / "hero.png")
    Image.new("RGBA", (16, 16), (30, 200, 30, 255)).save(folder / "chars" / "walk.png")
    Image.new("RGBA", (16, 16), (30, 30, 200, 255)).save(folder / "tiles" / "grass.png")

    browser = PackBrowser()
    qtbot.addWidget(browser)
    assert browser.add_pack_from_path(folder) == (0, 0)   # 无地形/拼件
    assert browser.stats()["packs"] == 1
    assert browser.file_count(0) == 3
    assert browser.file_count(0, "chars") == 1
    stats = browser.stats()
    assert stats["terrains"] == 0 and stats["pieces"] == 0

    browser.set_scope(0, "chars")
    assert browser._grid.count() == 1
    browser.set_scope(0, "")
    assert browser._grid.count() == 3
    # 图片资源可直接取出（送入画布）
    got = browser.asset_of(browser._grid.item(0))
    assert got is not None and isinstance(got[1], Image.Image) and got[1].size == (16, 16)


def test_category_switcher_labels_are_translated():
    from ui.i18n import LANG_PACKS

    for _key, label in CATEGORIES:
        assert label in LANG_PACKS["en"], f"切换器标签缺少英文: {label}"
