"""iOS 风格开关控件：长圆形基座 + 白色圆形按钮，点击切换、150ms 缓动动画。

规格（与 Trae 左上角开关一致的简洁扁平风格）：
- 基座胶囊形：关闭态浅灰（#D0D0D0），开启态强调蓝（#007AFF，暗色主题 #0A84FF）；
- 按钮纯白（暗色主题用浅灰+细边框），带轻微阴影，与基座等高；
- 只响应点击（不响应拖动）：点击基座任意位置切换，按钮 150ms OutCubic 平滑滑动；
- 暗色主题自动切换配色。
"""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPointF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.layout import scaled

ON_COLOR = QColor("#007AFF")        # iOS 蓝（浅色主题开启）
ON_COLOR_DARK = QColor("#0A84FF")   # iOS 暗色模式蓝
OFF_TRACK_LIGHT = QColor("#D0D0D0")
OFF_TRACK_DARK = QColor("#3A3F47")
KNOB_WHITE = QColor("#FFFFFF")
KNOB_DARK = QColor("#E4E4E4")
KNOB_BORDER_DARK = QColor("#6A6A6A")

ANIM_MS = 150


class SwitchButton(QWidget):
    """iOS 风格开关。默认尺寸 49×28（可按 height 缩放，随 UI 布局比例）。"""

    toggled = Signal(bool)
    clicked = Signal()

    def __init__(self, parent=None, dark: bool = False, height: int = 28):
        super().__init__(parent)
        self._checked = False
        self._dark = bool(dark)
        h = max(16, scaled(height))
        self._h = h
        self._w = int(h * 1.75)
        self.setFixedSize(self._w, h)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0  # 0.0 = 关（按钮在左），1.0 = 开（按钮在右）
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_value)
        self._anim.finished.connect(self._on_anim_finished)

    # ------------------------------------------------------------------ #
    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, animate: bool = True) -> None:
        """设置开关状态；animate=False 时立即跳变（初始化用）。"""
        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        target = 1.0 if checked else 0.0
        if animate:
            self._anim.stop()
            self._anim.setStartValue(self._pos)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._pos = target
            self.update()
        self.toggled.emit(checked)

    def setDark(self, dark: bool) -> None:
        """切换主题配色（浅色/暗色）。"""
        self._dark = bool(dark)
        self.update()

    def isDark(self) -> bool:
        return self._dark

    # ------------------------------------------------------------------ #
    def _on_anim_value(self, value) -> None:
        self._pos = float(value)
        self.update()

    def _on_anim_finished(self) -> None:
        self._pos = 1.0 if self._checked else 0.0
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            self.setChecked(not self._checked)
            event.accept()
            return
        super().mousePressEvent(event)

    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self._w, self._h
        radius = h / 2.0
        # 基座（胶囊）
        if self._checked:
            track = ON_COLOR_DARK if self._dark else ON_COLOR
        else:
            track = OFF_TRACK_DARK if self._dark else OFF_TRACK_LIGHT
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(0, 0, w, h, radius, radius)
        # 圆形按钮（与基座等高、留 2px 边距；轻微阴影）
        d = h - 4
        x = 2 + (w - d - 4) * self._pos
        y = 2
        p.setBrush(QColor(0, 0, 0, 28))
        p.drawEllipse(QPointF(x + d / 2.0, y + 1 + d / 2.0), d / 2.0, d / 2.0)
        if self._dark:
            p.setBrush(KNOB_DARK)
            p.setPen(QPen(KNOB_BORDER_DARK, 1))
        else:
            p.setBrush(KNOB_WHITE)
            p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(x + d / 2.0, y + d / 2.0), d / 2.0, d / 2.0)
        p.end()
