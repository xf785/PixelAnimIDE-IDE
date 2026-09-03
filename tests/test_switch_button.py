"""iOS 风格开关控件测试（点击切换、150ms 动画、主题配色）。"""
import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ui.widgets.switch_button import SwitchButton


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _pump_until(cond, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    QApplication.processEvents()
    return cond()


def test_switch_default_off():
    sw = SwitchButton()
    assert not sw.isChecked()
    assert sw._pos == 0.0
    assert sw._w > sw._h  # 长圆形（胶囊）


def test_switch_click_toggles_and_signals():
    sw = SwitchButton()
    toggles = []
    clicks = []
    sw.toggled.connect(toggles.append)
    sw.clicked.connect(lambda: clicks.append(1))
    QTest.mouseClick(sw, Qt.MouseButton.LeftButton)
    assert sw.isChecked()
    assert toggles == [True] and len(clicks) == 1
    QTest.mouseClick(sw, Qt.MouseButton.LeftButton)
    assert not sw.isChecked()
    assert toggles == [True, False]


def test_switch_set_checked_immediate():
    sw = SwitchButton()
    sw.setChecked(True, animate=False)
    assert sw.isChecked() and sw._pos == 1.0
    sw.setChecked(False, animate=False)
    assert not sw.isChecked() and sw._pos == 0.0


def test_switch_animation_completes():
    """点击后按钮 150ms 内平滑滑动到位（缓动动画）。"""
    sw = SwitchButton()
    sw.show()
    sw.setChecked(True)  # 动画切换
    assert _pump_until(lambda: sw._pos >= 1.0), "动画未在超时内完成"
    sw.setChecked(False)
    assert _pump_until(lambda: sw._pos <= 0.0), "动画未在超时内完成"
    sw.close()


def test_switch_theme_variants():
    sw = SwitchButton(dark=True)
    assert sw.isDark()
    sw.setDark(False)
    assert not sw.isDark()
    # 暗色主题开关尺寸与浅色一致
    sw2 = SwitchButton(dark=False)
    assert sw.size() == sw2.size()


def test_segmented_toggle_click_switches():
    """分段式执行方式开关：点击切换（左=自动 / 右=手动），信号正确。"""
    from PySide6.QtWidgets import QWidget

    from ui.widgets.segmented_toggle import SegmentedToggle

    toggle = SegmentedToggle()
    toggles = []
    clicks = []
    toggle.toggled.connect(toggles.append)
    toggle.clicked.connect(lambda: clicks.append(1))
    assert toggle.isChecked() is False               # 默认左段（自动）选中
    QTest.mouseClick(toggle, Qt.MouseButton.LeftButton)
    assert toggle.isChecked() is True                # 点击 -> 右段（手动）
    assert toggles == [True] and len(clicks) == 1
    QTest.mouseClick(toggle, Qt.MouseButton.LeftButton)
    assert toggle.isChecked() is False
    assert toggles == [True, False]
    # setChecked 幂等 + 主题切换不改变状态
    toggle.setChecked(True)
    assert toggle.isChecked() is True
    assert toggles == [True, False, True]
    toggle.setDark(True)
    assert toggle.isChecked() is True
    toggle.setChecked(True)                          # 同值不重复发信号
    assert toggles == [True, False, True]
    # 控件宽高比：两段式胶囊
    assert toggle.width() > toggle.height()
