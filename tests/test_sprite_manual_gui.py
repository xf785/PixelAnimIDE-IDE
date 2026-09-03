"""精灵图手动模式 GUI 测试（逐步执行：每步完成后可重跑或继续）。

用 QApplication.processEvents 循环驱动真实 QThread 步骤线程
（与 debug 脚本同模式，避免 pytest-qt 事件循环差异）。
"""
import time

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from config.api_config import APIConfig, APIConfigManager
from core.storage.keyring import Keyring
from core.workflow import STEP_ORDER
from ui.app_context import AppContext, UISettings
from ui.pages.sprite_page import SpritePage


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    """确保 QApplication 存在（本模块不用 qtbot，自行管理）。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


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
    ui = UISettings(tmp_path / "ui_settings.json")
    return AppContext(api=api, ui_settings=ui)


def _setup_manual(page):
    """切到手动模式并填好参数。"""
    page.set_manual_mode(True)
    page._desc_edit.setPlainText("一只小猫")
    page._frames_spin.setValue(4)
    page._rows_spin.setValue(2)
    page._cols_spin.setValue(2)
    page._size_combo.setCurrentText("64")


def _pump(seconds: float = 30.0) -> None:
    """跑事件循环直到超时（线程信号经 queued connection 派发）。"""
    app = QApplication.instance()
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


def _drive_until(page, cond, timeout: float = 60.0) -> bool:
    """processEvents 循环等待条件成立，超时返回 False。"""
    app = QApplication.instance()
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    app.processEvents()
    return cond()


def test_sprite_page_manual_mode_controls(ctx, monkeypatch):
    """执行方式由主窗口侧栏开关驱动：默认自动，set_manual_mode 切换；重跑/继续按钮初始隐藏。"""
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    )
    page = SpritePage(ctx)
    assert not page._manual_mode          # 默认自动
    page.set_manual_mode(True)
    assert page._manual_mode              # 手动
    page.set_manual_mode(False)
    assert not page._manual_mode
    assert not page._btn_rerun.isVisible()
    assert not page._btn_next.isVisible()
    page.close()


def test_sprite_mode_switch_in_sidebar(ctx, monkeypatch):
    """侧栏执行方式开关（A/M 点击切换）：仅精灵图模式显示；切换 sprite_page 自动/手动。"""
    from ui.main_window import MainWindow

    window = MainWindow(ctx)
    assert window._sprite_switch.isHidden()           # 默认 solo 模式隐藏
    window.set_mode("sprite")
    assert not window._sprite_switch.isHidden()       # 精灵图模式显示
    assert window._sprite_toggle.isChecked() is False  # 默认自动（A 段）
    assert not window.sprite_page._manual_mode
    window._sprite_toggle.setChecked(True)            # 切到手动（M 段）
    assert window.sprite_page._manual_mode
    window._sprite_toggle.setChecked(False)           # 切回自动
    assert not window.sprite_page._manual_mode
    window.set_mode("ide")
    assert window._sprite_switch.isHidden()           # IDE 模式隐藏
    window.close()


def test_segmented_toggle_hover_tooltip(ctx, monkeypatch):
    """执行方式开关：悬停显示信息（左半 A=自动说明 / 右半 M=手动说明）。"""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    from ui.widgets.segmented_toggle import SegmentedToggle

    def _move(x: int, y: int):
        return QMouseEvent(
            QEvent.Type.MouseMove, QPointF(x, y),
            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )

    toggle = SegmentedToggle()
    assert "自动" in toggle.toolTip()                  # 默认提示自动说明
    # 悬停右半区 -> 手动说明
    QApplication.sendEvent(toggle, _move(toggle.width() - 5, toggle.height() // 2))
    assert "手动" in toggle.toolTip()
    # 悬停左半区 -> 自动说明
    QApplication.sendEvent(toggle, _move(5, toggle.height() // 2))
    assert "自动" in toggle.toolTip()
    toggle.close()


def test_pixel_page_settings_collapsible(ctx):
    """像素页画布设置栏：可收起（画布更大），可重新展开；分栏可拖拽调宽。"""
    from ui.pages.pixel_page import PixelPage

    page = PixelPage(ctx)
    assert not page._settings_panel.isHidden()
    assert page._splitter is not None
    page._on_collapse_settings()
    assert page._settings_panel.isHidden()            # 收起后隐藏
    assert not page._btn_expand.isHidden()            # 仅剩展开按钮
    page._on_expand_settings()
    assert not page._settings_panel.isHidden()        # 重新展开
    assert page._btn_expand.isHidden()
    page.close()


def test_ide_page_shortcut_bindings(ctx, monkeypatch):
    """IDE 页键盘绑定（设置 → 快捷键 → IDE）：Space 播放/暂停，I 插入帧。"""
    from PIL import Image
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    from ui import shortcuts as sc
    from ui.pages.ide_page import IdePage

    sc.set_shortcuts(None)
    sc.set_active_mode("ide")
    page = IdePage(ctx)
    page._session.frames = [Image.new("RGBA", (16, 16), (255, 255, 255, 255))]

    space = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ")
    page.keyPressEvent(space)
    assert page._playing is True
    page.keyPressEvent(space)
    assert page._playing is False

    ins = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_I, Qt.KeyboardModifier.NoModifier, "i")
    page.keyPressEvent(ins)
    assert len(page._session.frames) == 2  # 插入空白帧

    # 未配置的模式不影响（当前生效模式为 ide，像素的 Ctrl+Z 不触发 IDE 动作）
    z = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier, "z")
    page.keyPressEvent(z)  # 不应崩溃、不应触发任何 IDE 绑定
    page.close()
    sc.set_shortcuts(None)


def test_sprite_page_manual_mode_full_run(ctx, monkeypatch):
    """手动模式全流程：7 步逐步执行，每步完成后可继续，最后导出完成。"""
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    )
    page = SpritePage(ctx)
    _setup_manual(page)
    page._on_start()
    assert page._manual_active

    # 逐步推进：每步完成后「继续下一步」可用，点击后执行下一步
    for _ in range(len(STEP_ORDER)):
        assert _drive_until(page, lambda: page._btn_next.isEnabled() or not page._manual_active), (
            "步骤未在超时内完成"
        )
        if not page._manual_active:
            break
        page._on_next()
    assert _drive_until(page, lambda: not page._manual_active)
    assert page._result is not None
    assert page._result.frame_count == 4
    assert page._result.frames_dir and page._result.frames_dir.exists()
    # 完成后按钮复位
    assert page._btn_start.isEnabled()
    assert not page._btn_next.isVisible()
    page.close()


def test_sprite_page_manual_mode_rerun(ctx, monkeypatch):
    """手动模式重跑：步骤 3（生成网格精灵图）完成后「重跑本步」重新执行该步。"""
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    )
    page = SpritePage(ctx)
    _setup_manual(page)
    page._on_start()
    assert page._manual_active

    # 完成前 2 步（prompts / base），进入 sheet（第 3 步）
    for _ in range(2):
        assert _drive_until(page, lambda: page._btn_next.isEnabled())
        page._on_next()
    assert _drive_until(page, lambda: page._btn_rerun.isEnabled())
    assert page._manual_step_idx == 2  # 停在 sheet
    # 重跑本步：再次执行 sheet，仍停在该步
    page._on_rerun()
    assert _drive_until(page, lambda: page._btn_next.isEnabled())
    assert page._manual_step_idx == 2
    # 继续走完剩余步骤
    for _ in range(4):
        assert _drive_until(page, lambda: page._btn_next.isEnabled() or not page._manual_active)
        if not page._manual_active:
            break
        page._on_next()
    assert _drive_until(page, lambda: not page._manual_active)
    assert page._result is not None
    assert page._result.frame_count == 4
    page.close()
