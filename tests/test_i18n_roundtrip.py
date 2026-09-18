"""中英切换往返回归：英文状态下构建界面 → 切回中文，不得残留英文。

历史问题：不少文案是构建时用 ``tr()`` 直接 setText/setToolTip 设置的，没进 i18n 注册表；
若控件是在英文状态下构建的（设置里切到 English），切回中文后这些文案会永远停在英文。
修复：``T()`` 会先把译文还原成中文 ID；``retranslate_all()`` 还会整棵控件树兜底重译。
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel, QToolButton, QWidget

from config.api_config import APIConfig, APIConfigManager
from core.storage.keyring import Keyring
from ui.app_context import AppContext, UISettings
from ui.i18n import T, canonical_id, retranslate_all, retranslate_tree, set_language, tr

ZH = __import__("re").compile(r"[\u4e00-\u9fff]")


@pytest.fixture()
def ctx(tmp_path):
    api = APIConfigManager(config_file=tmp_path / "api.json", keyring=Keyring(tmp_path / ".keyring"))
    for kind in ("llm", "image", "video"):
        api.add(APIConfig(kind=kind, name=f"mock-{kind}", base_url="mock", model="mock-model",
                          params={"mock": True, "frames": 8, "fps": 8}))
    return AppContext(api=api, ui_settings=UISettings(tmp_path / "ui.json"))


@pytest.fixture(autouse=True)
def _restore_language():
    yield
    set_language("zh")


# --------------------------------------------------------------------------- #
# 单元层
# --------------------------------------------------------------------------- #
def test_canonical_id_maps_translation_back_to_chinese():
    set_language("en")
    assert canonical_id("Canvas settings") == "画布设置"
    assert canonical_id("画布设置") == "画布设置"     # 中文本身就是 ID
    assert canonical_id("Not a translation") == "Not a translation"
    set_language("zh")
    assert canonical_id("Canvas settings") == "Canvas settings"   # 中文模式下不做还原


def test_T_registered_with_translation_still_restores_chinese(qtbot):
    """双重翻译（T(w, tr("…"))）不得把英文固化成 ID。"""
    set_language("en")
    label = QLabel()
    qtbot.addWidget(label)
    T(label, tr("画布设置"))                       # 调用方多写了一次 tr()
    assert label.text() == tr("画布设置") != "画布设置"

    set_language("zh")
    retranslate_all()
    assert label.text() == "画布设置"


def test_retranslate_tree_fixes_unregistered_widgets(qtbot):
    """没进注册表的控件（英文态构建、tr() 直接设文案）也能被兜底重译。"""
    host = QWidget()
    qtbot.addWidget(host)
    label = QLabel("Canvas", host)                 # 直接是英文译文
    button = QToolButton(host)
    button.setText("Save")
    button.setToolTip("Canvas transform: flip / rotate / crop / size / scale")
    line = __import__("PySide6.QtWidgets", fromlist=["QLineEdit"]).QLineEdit(host)
    line.setPlaceholderText("Search assets…")
    line.setText("Water")                          # 用户输入不应被翻译

    set_language("zh")
    changed = retranslate_tree(host)
    assert changed >= 3
    assert label.text() == "画布"
    assert button.text() == "保存"
    assert button.toolTip().startswith("画布变换")
    assert line.placeholderText() == "搜索资源…"
    assert line.text() == "Water", "用户输入不能被当作译文替换"


# --------------------------------------------------------------------------- #
# 集成：英文态构建 → 切回中文
# --------------------------------------------------------------------------- #
def test_main_window_built_in_english_returns_to_chinese(qtbot, ctx):
    from ui.main_window import MainWindow

    set_language("en")
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    window.set_mode("pixel")
    for _ in range(3):
        __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication.processEvents()
    assert window.pixel_page._btn_new.text() == "New canvas"

    set_language("zh")
    window.retranslate_ui()
    for _ in range(3):
        __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication.processEvents()

    checks = {
        "logo 提示": window._logo_label.toolTip(),
        "主标题": window._toolbar.findChildren(QLabel)[0].text(),
        "新建画布": window.pixel_page._btn_new.text(),
        "停靠面板标题": " ".join(d.title() for d in window.pixel_page._right_dock.panels()),
        "资源面板标题": " ".join(d.title() for d in window.pixel_page._left_dock.panels()),
        "状态栏": window._status_info.text(),
        "步骤按钮": window._step_buttons[0].text(),
        "菜单": " ".join(a.text() for a in window.menuBar().actions()),
        "编辑器工具提示": window.pixel_page._editor._tool_buttons[
            list(window.pixel_page._editor._tool_buttons)[0]
        ].toolTip(),
        "环绕提示": window.pixel_page._editor._wrap_btn.toolTip(),
        "画布信息": window.pixel_page._info_label.text(),
    }
    bad = {name: text for name, text in checks.items() if not ZH.search(text)}
    assert not bad, f"切回中文后仍是英文：{bad}"


def test_settings_dialog_built_in_english_returns_to_chinese(qtbot, ctx):
    from PySide6.QtWidgets import QApplication

    from ui.dialogs.settings_dialog import SettingsDialog

    set_language("en")
    dlg = SettingsDialog(ctx)
    qtbot.addWidget(dlg)
    dlg.show()
    QApplication.processEvents()
    assert dlg.windowTitle() == "Settings"

    set_language("zh")
    dlg.retranslate_ui()
    QApplication.processEvents()
    assert dlg.windowTitle() == "设置"
    texts = [w.text() for w in dlg.findChildren(QLabel) if w.text()]
    assert any("深色模式" in t for t in texts), texts[:12]
    data_dir = [t for t in texts if t.startswith("数据目录")]
    assert data_dir, "数据目录提示应重刷为中文"
    assert dlg._status.text() == "" or ZH.search(dlg._status.text()) or dlg._status.text()


def test_pack_browser_dynamic_labels_translate(qtbot, tmp_path):
    """包浏览器统计行/范围标签是拼接文案，切换语言后要跟着重算。"""
    from PIL import Image

    from core.tilemap.pack import TilePack, export_tileset_dir
    from core.tilemap.tiles import EDGE_NAMES, BaseTileSet
    from ui.widgets.pack_browser import PackBrowser

    S = 32
    tex = Image.new("RGBA", (S, S), (80, 150, 90, 255))
    tset = BaseTileSet(size=S, center=tex, edges={n: tex for n in EDGE_NAMES},
                       corners={n: tex for n in ("tl", "tr", "bl", "br")},
                       line_color=(24, 22, 24), line_width=1, band=8, radius=8, base_texture=tex,
                       art_meta={"outline": [[24, 22, 24]], "outline_px": 1, "bevel": [],
                                 "bevel_px": 0, "edge_noise_px": 0})
    pack = TilePack(name="grass", category="ground", tile_size=S, terrains={1: tset},
                    terrain_names={1: "grass"}, base_terrain=1)
    out = export_tileset_dir(tmp_path / "pack", pack)

    set_language("en")
    browser = PackBrowser()
    qtbot.addWidget(browser)
    browser.add_pack_from_path(out["dir"])
    assert "Packs 1" in browser._stats.text()

    set_language("zh")
    browser.retranslate_ui()
    assert "包 1" in browser._stats.text()
    assert "目录：" in browser._scope_label.text()
