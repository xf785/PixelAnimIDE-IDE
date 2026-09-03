"""两段式点击切换开关：A=Auto / M=Manual（精灵图执行方式）。

设计：
- 点击切换（不响应拖动）；白底胶囊 + 细边框，中间浅灰「/」斜线分界；
- 左段字母 A（自动）、右段字母 M（手动）；
- 选中段为项目强调蓝（#4176e6，与 PrimaryButton 一致）底 + 白色字母；
  未选中段白底 + 辅助灰字母（保证白底上清晰）；
- 悬停显示信息：鼠标在左半区提示自动说明，右半区提示手动说明。
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.i18n import tr
from ui.layout import scaled

ACCENT = QColor("#4176e6")       # 项目主蓝（与 PrimaryButton 一致）
ACCENT_HOVER = QColor("#5686fe")
WHITE = QColor("#FFFFFF")
TRACK = QColor("#FFFFFF")        # 白底
BORDER = QColor("#D9DCE1")
TEXT_IDLE = QColor("#81858c")    # 未选中字母（辅助灰，白底上清晰）
DIVIDER = QColor("#E0E2E6")      # 中间斜线


class SegmentedToggle(QWidget):
    """两段式点击切换开关（checked=False -> 左段 A/自动，True -> 右段 M/手动）。"""

    toggled = Signal(bool)
    clicked = Signal()

    def __init__(self, parent=None, height: int = 30):
        super().__init__(parent)
        self._checked = False
        self._hover = False
        h = max(18, scaled(height))
        self._h = h
        self._w = int(h * 3.0)
        self.setFixedSize(self._w, h)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setToolTip(tr("自动：无干涉跑完全流程"))

    # ------------------------------------------------------------------ #
    def isChecked(self) -> bool:
        """True = 右段（M/手动）选中。"""
        return self._checked

    def setChecked(self, checked: bool) -> None:
        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        self.update()
        self.toggled.emit(checked)

    def setDark(self, dark: bool) -> None:
        """主题适配（白底两主题通用；保留接口，供主题切换调用）。"""
        self.update()

    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            self.setChecked(not self._checked)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        left = event.position().x() < self._w / 2.0
        self.setToolTip(tr("自动：无干涉跑完全流程") if left else tr("手动：逐步执行，每步完成后可重跑或继续"))
        super().mouseMoveEvent(event)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.setToolTip(tr("自动") + " / " + tr("手动"))
        super().leaveEvent(event)

    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self._w, self._h
        r = h / 2.0

        # 白底胶囊 + 细边框
        p.setPen(QPen(BORDER, 1))
        p.setBrush(TRACK)
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), r, r)

        # 中间「/」斜线分界（选中段蓝块会盖住自己那一半）
        pen = QPen(DIVIDER, max(1.2, h * 0.05))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        x = w / 2.0
        p.drawLine(QPointF(x - 1.5, h * 0.16), QPointF(x + 1.5, h * 0.84))

        # 选中段：蓝底圆角块（与 PrimaryButton 同蓝，悬停提亮）+ 白色字母
        accent = ACCENT_HOVER if self._hover else ACCENT
        sel_left = not self._checked
        if sel_left:
            seg = QRectF(2, 2, w / 2.0 - 3, h - 4)
        else:
            seg = QRectF(w / 2.0 + 1, 2, w / 2.0 - 3, h - 4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(accent)
        p.drawRoundedRect(seg, r - 2, r - 2)

        # 字母 A / M（选中侧白色，未选中侧辅助灰）
        font = QFont()
        font.setBold(True)
        font.setPixelSize(max(10, int(h * 0.52)))
        p.setFont(font)
        left_rect = QRectF(0, 0, w / 2.0, h)
        right_rect = QRectF(w / 2.0, 0, w / 2.0, h)
        if sel_left:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(WHITE)
            p.drawText(left_rect, Qt.AlignmentFlag.AlignCenter, "A")
            p.setPen(TEXT_IDLE)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawText(right_rect, Qt.AlignmentFlag.AlignCenter, "M")
        else:
            p.setPen(TEXT_IDLE)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawText(left_rect, Qt.AlignmentFlag.AlignCenter, "A")
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(WHITE)
            p.drawText(right_rect, Qt.AlignmentFlag.AlignCenter, "M")
        p.end()
