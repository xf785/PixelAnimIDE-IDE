"""Krita 式外壳与停靠面板（docker）布局回归。

覆盖：
- `Docker` 折叠/展开、标题 i18n；
- `DockerColumn` 垂直可拖拽分栏；
- `SideDock` 整栏收起成竖标签 + 宽度记忆；
- 主窗口菜单栏 / 上下文工具条 / 状态栏；
- 像素页三栏工作区（左右停靠栏 + 画布）与布局持久化。
"""
from __future__ import annotations

import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QWidget

from config.api_config import APIConfig, APIConfigManager
from core.storage.keyring import Keyring
from ui.app_context import AppContext, UISettings
from ui.widgets.dock import Docker, DockerColumn, SideDock


@pytest.fixture()
def ctx(tmp_path):
    api = APIConfigManager(config_file=tmp_path / "api.json", keyring=Keyring(tmp_path / ".keyring"))
    for kind in ("llm", "image", "video"):
        api.add(APIConfig(kind=kind, name=f"mock-{kind}", base_url="mock", model="mock-model",
                          params={"mock": True, "frames": 8, "fps": 8}))
    return AppContext(api=api, ui_settings=UISettings(tmp_path / "ui.json"))


# --------------------------------------------------------------------------- #
# Docker
# --------------------------------------------------------------------------- #
def test_docker_collapses_and_expands(qtbot):
    content = QLabel("内容")
    docker = Docker("画布设置", content)
    qtbot.addWidget(docker)
    docker.show()
    assert docker.title() == "画布设置"
    assert not docker.is_collapsed()
    assert content.isVisible()

    changes = []
    docker.toggled.connect(changes.append)
    docker.toggle()
    assert docker.is_collapsed() and not content.isVisible()
    assert changes == [True]
    assert docker._chevron.text() == "▸"
    docker.set_collapsed(False)
    assert not docker.is_collapsed() and content.isVisible()
    assert docker._chevron.text() == "▾"


def test_docker_header_widgets_and_content_swap(qtbot):
    docker = Docker("包", QLabel("a"))
    qtbot.addWidget(docker)
    btn = docker.add_header_widget(QWidget())
    assert btn in docker._header_widgets
    replacement = QLabel("b")
    docker.set_content(replacement)
    assert docker.content() is replacement


def test_docker_column_is_a_vertical_splitter(qtbot):
    column = DockerColumn()
    qtbot.addWidget(column)
    column.show()
    p1 = column.add_docker("上", QLabel("1"))
    p2 = column.add_docker("下", QLabel("2"), stretch=1)
    assert column.panels() == [p1, p2]
    splitters = column.findChildren(QSplitter)
    assert splitters and splitters[0].orientation() == Qt.Orientation.Vertical
    assert splitters[0].count() == 2


def test_docker_column_collapse_fills_other_panels(qtbot):
    """折叠一个 docker 后，它只占标题条，腾出的空间由其余展开面板填满（堆叠填充）。"""
    from PySide6.QtWidgets import QApplication, QVBoxLayout

    host = QWidget()
    qtbot.addWidget(host)
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    column = DockerColumn()
    lay.addWidget(column)
    a = column.add_docker("A", QLabel("a"), stretch=1)
    b = column.add_docker("B", QLabel("b"))
    c = column.add_docker("C", QLabel("c"))
    host.resize(300, 700)
    host.show()
    for _ in range(4):
        QApplication.processEvents()

    sizes = column._splitter.sizes()
    total = sum(sizes)
    assert total > 400 and all(s > 40 for s in sizes), sizes

    b.set_collapsed(True)
    for _ in range(3):
        QApplication.processEvents()
    sizes = column._splitter.sizes()
    assert sizes[1] <= 40, f"折叠后应只剩标题条：{sizes}"
    assert sum(sizes) == total or abs(sum(sizes) - total) <= 8, sizes
    assert sizes[0] + sizes[2] >= total - 45, f"腾出的空间应给其余面板：{sizes}"

    b.set_collapsed(False)
    for _ in range(3):
        QApplication.processEvents()
    assert column._splitter.sizes()[1] > 40, "展开后应拿回高度"

    # 只留一个展开面板时，它吃掉整列
    a.set_collapsed(True)
    c.set_collapsed(True)
    for _ in range(3):
        QApplication.processEvents()
    sizes = column._splitter.sizes()
    assert sizes[0] <= 40 and sizes[2] <= 40
    assert sizes[1] >= total - 90, f"唯一展开的面板应填满：{sizes}"


# --------------------------------------------------------------------------- #
# SideDock
# --------------------------------------------------------------------------- #
def test_side_dock_collapses_to_rail(qtbot):
    dock = SideDock("资源", side="left", default_width=240)
    qtbot.addWidget(dock)
    dock.add_docker("资源浏览", QLabel("x"), stretch=1)
    dock.show()
    assert dock.minimumWidth() <= 30           # 允许被 splitter 拖到很窄（最多留竖标签宽）
    assert not dock.is_collapsed()

    events = []
    dock.collapsedChanged.connect(events.append)
    dock.set_collapsed(True)
    assert dock.is_collapsed() and dock.width() <= 30
    assert dock._stack.currentWidget() is dock._rail
    assert dock._rail.toolTip() == "资源"
    dock.set_collapsed(False)
    assert not dock.is_collapsed()
    assert dock._stack.currentWidget() is dock._column
    assert events == [True, False]


def test_side_dock_remembers_width_in_splitter(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    from PySide6.QtWidgets import QHBoxLayout

    lay = QHBoxLayout(host)
    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setObjectName("Workspace")
    dock = SideDock("参数", side="left", default_width=200)
    dock.add_docker("参数", QLabel("x"), stretch=1)
    splitter.addWidget(dock)
    splitter.addWidget(QWidget())
    splitter.setSizes([200, 600])
    lay.addWidget(splitter)
    dock.bind_splitter(splitter, 0, default_width=200)
    host.resize(800, 400)
    host.show()
    assert abs(splitter.sizes()[0] - 200) <= 20

    dock.set_collapsed(True)
    assert dock.width() <= 30
    dock.set_collapsed(False)
    # 展开后恢复到收起前的宽度（而不是 0）
    assert splitter.sizes()[0] >= 100


# --------------------------------------------------------------------------- #
# 主窗口外壳
# --------------------------------------------------------------------------- #
def test_main_window_has_menu_bar_and_workspace_menu(qtbot, ctx):
    from ui.main_window import MainWindow

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    menus = [a.text() for a in window.menuBar().actions() if a.menu() is not None]
    assert menus == ["文件", "编辑", "视图", "工作区", "帮助"], menus
    # 工作区菜单与侧栏模式开关联动
    assert window._ws_actions["ide"].isCheckable()
    window._ws_actions["ide"].trigger()
    assert window._mode == "ide"
    assert window._mode_ide_btn.isChecked()
    window._ws_actions["pixel"].trigger()
    assert window._mode == "pixel" and window._stack.currentIndex() == 3
    # 视图菜单：界面比例
    assert set(window._scale_actions) == {0.8, 0.9, 1.0, 1.1, 1.25, 1.5}
    window._scale_actions[1.25].trigger()
    assert ctx.ui_settings.get("ui_scale") == 1.25
    window._scale_actions[1.0].trigger()


def test_toolbar_follows_workspace(qtbot, ctx):
    """工具条右侧动作随工作区切换（页面 toolbar_actions 协议）。"""
    from ui.main_window import MainWindow

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    for mode in ("solo", "ide", "sprite", "pixel", "tilemap"):
        window.set_mode(mode)
        page = window._current_page()
        actions = page.toolbar_actions() if hasattr(page, "toolbar_actions") else []
        texts = _toolbar_texts(window)
        for item in actions:
            assert item[1] in texts, f"{mode} 模式工具条缺少动作：{item[1]}"
        status = page.workspace_status() if hasattr(page, "workspace_status") else ""
        if status:
            assert window._toolbar_status.text() == status


def _toolbar_texts(window) -> list:
    from PySide6.QtWidgets import QToolButton

    return [w.text() for w in window._toolbar.findChildren(QToolButton) if w.text()]


def test_status_bar_shows_scale_and_theme(qtbot, ctx):
    from ui.main_window import MainWindow

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    assert "界面" in window._status_info.text()
    assert "深色" in window._status_info.text()
    window._on_toggle_theme()
    assert "浅色" in window._status_info.text()
    window._on_toggle_theme()


def test_edit_menu_defers_to_focused_text_widget(qtbot, ctx):
    """编辑菜单快捷键不应抢走文本框的 Ctrl+C / Ctrl+V。"""
    from PySide6.QtWidgets import QLineEdit

    from ui.main_window import MainWindow

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    window.set_mode("pixel")
    edit = QLineEdit()
    window.pixel_page.layout().addWidget(edit)
    edit.setText("hello")
    edit.setFocus()
    edit.selectAll()
    window._menu_copy()                       # 应走文本框自己的复制
    assert edit.selectedText() == "hello"
    window._menu_merge()                      # 文本框有焦点时空操作，不应报错
    assert window._active_editor() is not None


def test_no_stray_top_level_widgets_on_mode_switch(qtbot, ctx):
    """切换工作区不得弹出「空白窗口」。

    回归：工具条重建时若控件没有父控件，``QToolBar.addWidget()`` 会先把它当顶层窗口
    show() 一帧（Windows 上表现为空白窗口闪现），并且旧控件会不断泄漏成游离顶层控件。
    """
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QApplication

    from ui.main_window import MainWindow

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()

    def pump():
        for _ in range(3):
            QApplication.processEvents()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QApplication.processEvents()

    pump()
    base = {id(w) for w in QApplication.topLevelWidgets() if w.parent() is None}
    shown_windows = []

    class _Spy(QObject):
        def eventFilter(self, obj, event):  # noqa: N802
            if event.type() == QEvent.Type.Show and getattr(obj, "isWindow", lambda: False)():
                if obj is not window:
                    shown_windows.append(obj)
            return False

    spy = _Spy()
    QApplication.instance().installEventFilter(spy)
    try:
        for mode in ("ide", "sprite", "pixel", "tilemap", "solo", "pixel"):
            window.set_mode(mode)
            pump()
        current = {id(w) for w in QApplication.topLevelWidgets() if w.parent() is None}
    finally:
        QApplication.instance().removeEventFilter(spy)

    assert not shown_windows, f"切换页面时弹出了顶层窗口：{[type(w).__name__ for w in shown_windows]}"
    assert current <= base, "切换页面不断新增游离顶层控件（控件泄漏）"
    # 工具条自建控件必须挂在工具条上，不得游离
    assert window._toolbar_widgets and all(w.parent() is not None for w in window._toolbar_widgets)


def test_theme_and_language_switch_do_not_leak_toolbar_widgets(qtbot, ctx):
    """主题 / 语言切换同样会重建工具条与菜单，控件数不得持续增长。"""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    from ui.i18n import set_language
    from ui.main_window import MainWindow

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()

    def pump():
        for _ in range(3):
            QApplication.processEvents()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QApplication.processEvents()

    pump()
    base = len([w for w in QApplication.topLevelWidgets() if w.parent() is None])
    for _ in range(3):
        window._on_toggle_theme()
        set_language("en")
        window.retranslate_ui()
        set_language("zh")
        window.retranslate_ui()
        pump()
    assert len([w for w in QApplication.topLevelWidgets() if w.parent() is None]) == base
    assert len(window.menuBar().actions()) == 5
    assert len(window._menus) == 5


def test_pixel_page_three_column_workspace_is_resizable(qtbot, ctx):
    """像素页三栏工作区：分隔线可拖、宽度可持久化、可一键重置。"""
    from ui.main_window import MainWindow

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    window.set_mode("pixel")
    page = window.pixel_page
    assert page._splitter.objectName() == "Workspace"
    assert page._splitter.count() == 3
    assert page._splitter.handleWidth() >= 3

    # 拖动分隔线（模拟 setSizes）后写回设置，下次打开恢复
    page._splitter.setSizes([300, 600, 260])
    page._remember_layout()
    sizes = ctx.ui_settings.get("pixel_dock_sizes")
    assert isinstance(sizes, list) and len(sizes) == 3 and sizes[0] == 300

    # 「重置面板布局」恢复默认
    page._splitter.setSizes([120, 600, 120])
    window._menu_reset_layout()
    assert page._splitter.sizes()[0] > 150


def test_pixel_page_canvas_starts_fitted(qtbot, ctx):
    """像素页默认 64×64 画布，并按视图自适应整数倍缩放。"""
    from ui.pages.pixel_page import PixelPage

    page = PixelPage(ctx)
    qtbot.addWidget(page)
    page.resize(1200, 700)
    page.show()
    assert page.image().size == (64, 64)
    assert page.image().getpixel((0, 0))[3] == 0
    page._fit_view_now()
    assert page._editor.zoom() >= 1
    assert page.workspace_status() == "64 × 64"


def test_pixel_page_replaces_canvas_from_asset_and_keeps_signals(qtbot, ctx):
    """放入/替换画布的信号连接在重构后仍工作。"""
    from ui.pages.pixel_page import PixelPage

    page = PixelPage(ctx)
    qtbot.addWidget(page)
    page._editor.set_frame(Image.new("RGBA", (32, 32), (0, 0, 0, 0)))
    art = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    page._on_pack_asset(art, "a")
    assert page.image().size == (32, 32)
    page._on_pack_asset_replace(art, "a")
    assert page.image().size == (16, 16)
