"""Krita 风格右键取色圆盘：按住右键弹出 HSV 色环 + 饱和度/明度方块 + 最近色。

交互：
- 按住鼠标在色环（外环 = 色相）或方块（x = 饱和度，y = 明度）上移动实时预览；
- 松开提交所选颜色；Esc 取消；
- 底部横条为最近使用色，点击直接选中。
数值算法（numpy 向量化）生成色环/方块像素图并缓存，拖拽流畅。
"""
from __future__ import annotations

import colorsys
import math
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

RGBA = Tuple[int, int, int, int]


def hsv_to_rgb_tuple(h: float, s: float, v: float) -> RGBA:
    """h: 0-360, s/v: 0-1 -> (r, g, b, 255)。"""
    r, g, b = colorsys.hsv_to_rgb((h % 360.0) / 360.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), 255


def _hsv_to_rgb_np(h, s, v) -> np.ndarray:
    """向量化 HSV(0..1 数组) -> RGB(0..1)，形状与输入相同，末维为通道。"""
    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    cond = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(cond, [v, q, p, p, t, v])
    g = np.select(cond, [t, v, v, q, p, p])
    b = np.select(cond, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


class ColorWheelPopup(QWidget):
    """右键取色圆盘弹窗（Krita 式）。"""

    color_preview = Signal(object)   # 拖动实时预览
    color_selected = Signal(object)  # 松开提交
    cancelled = Signal()             # Esc 取消

    SIZE = 240        # 色环区域边长
    R_OUT = 112.0     # 外环半径
    R_IN = 88.0       # 内环半径（环宽 24）
    SQUARE = 124      # 中央饱和度/明度方块边长
    STRIP_H = 22      # 最近色条高度
    PAD = 8

    def __init__(self, initial: RGBA = (0, 0, 0, 255), recent: Optional[List[RGBA]] = None, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMouseTracking(True)
        self.setFixedSize(self.SIZE, self.SIZE + self.PAD + self.STRIP_H + self.PAD)
        r, g, b, a = initial
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        self._hue = h * 360.0
        self._s = float(s)
        self._v = float(v)
        self._color: RGBA = initial
        self._recent: List[RGBA] = list(recent or [])
        self._ring_pixmap: Optional[QPixmap] = None
        self._square_cache: dict = {}

    # ------------------------------------------------------------------ #
    # 几何
    # ------------------------------------------------------------------ #
    @property
    def _cx(self) -> float:
        return self.SIZE / 2.0

    @property
    def _sq0(self) -> float:
        return (self.SIZE - self.SQUARE) / 2.0

    # ------------------------------------------------------------------ #
    # 显示
    # ------------------------------------------------------------------ #
    def show_at(self, global_pos: QPoint) -> None:
        """在屏幕坐标 global_pos 附近居中显示（自动避开屏幕边缘）。"""
        screen = QApplication.primaryScreen().availableGeometry()
        x = global_pos.x() - self.width() // 2
        y = global_pos.y() - self.height() // 2
        x = max(screen.left() + 4, min(x, screen.right() - self.width() - 4))
        y = max(screen.top() + 4, min(y, screen.bottom() - self.height() - 4))
        self.move(x, y)
        self.show()
        self.activateWindow()
        self.setFocus()

    # ------------------------------------------------------------------ #
    # 取色计算
    # ------------------------------------------------------------------ #
    def _color_at(self, pos: QPoint) -> Optional[RGBA]:
        """按弹窗内坐标计算颜色：外环=色相，方块=饱和度/明度，底条=最近色。"""
        dx = pos.x() - self._cx
        dy = pos.y() - self._cx
        dist = math.hypot(dx, dy)
        if self.R_IN <= dist <= self.R_OUT:
            hue = math.degrees(math.atan2(dy, dx)) % 360.0
            return hsv_to_rgb_tuple(hue, 1.0, 1.0)
        sq0 = self._sq0
        if sq0 <= pos.x() <= sq0 + self.SQUARE and sq0 <= pos.y() <= sq0 + self.SQUARE:
            s = max(0.0, min(1.0, (pos.x() - sq0) / self.SQUARE))
            v = max(0.0, min(1.0, 1.0 - (pos.y() - sq0) / self.SQUARE))
            return hsv_to_rgb_tuple(self._hue, s, v)
        if pos.y() >= self.SIZE + self.PAD and self._recent:
            n = min(len(self._recent), 8)
            cell = (self.SIZE - 20) / 8
            idx = int((pos.x() - 10) // cell)
            if 0 <= idx < n:
                return self._recent[idx]
        return None

    def _apply(self, color: RGBA) -> None:
        """记录当前色（色环拖动时同步更新色相）。"""
        self._color = color
        r, g, b, _a = color
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        self._hue = h * 360.0
        self._s = float(s)
        self._v = float(v)

    # ------------------------------------------------------------------ #
    # 事件
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._on_pick(event.position().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._on_pick(event.position().toPoint())

    def _on_pick(self, pos: QPoint) -> None:
        c = self._color_at(pos)
        if c is not None:
            self._apply(c)
            self.color_preview.emit(self._color)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._on_pick(event.position().toPoint())
        self.color_selected.emit(self._color)
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    # 绘制
    # ------------------------------------------------------------------ #
    def _ring(self) -> QPixmap:
        if self._ring_pixmap is None:
            size = self.SIZE
            arr = np.zeros((size, size, 4), dtype=np.uint8)
            ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
            dx = xs - self._cx
            dy = ys - self._cx
            dist = np.sqrt(dx * dx + dy * dy)
            ang = (np.degrees(np.arctan2(dy, dx)) % 360.0) / 360.0
            mask = (dist >= self.R_IN) & (dist <= self.R_OUT)
            h = ang[mask]
            rgb = _hsv_to_rgb_np(h, np.ones_like(h), np.ones_like(h))
            arr[mask, 0] = (rgb[:, 0] * 255).astype(np.uint8)
            arr[mask, 1] = (rgb[:, 1] * 255).astype(np.uint8)
            arr[mask, 2] = (rgb[:, 2] * 255).astype(np.uint8)
            arr[mask, 3] = 255
            img = QImage(arr.data, size, size, size * 4, QImage.Format.Format_RGBA8888).copy()
            self._ring_pixmap = QPixmap.fromImage(img)
        return self._ring_pixmap

    def _square(self, hue: float) -> QPixmap:
        key = int(round(hue))
        pm = self._square_cache.get(key)
        if pm is None:
            n = self.SQUARE
            xs = (np.arange(n, dtype=np.float64) + 0.5) / n          # 饱和度
            ys = 1.0 - (np.arange(n, dtype=np.float64) + 0.5) / n    # 明度
            X, Y = np.meshgrid(xs, ys)
            rgb = _hsv_to_rgb_np(np.full((n, n), key / 360.0), X, Y)
            arr = np.zeros((n, n, 4), dtype=np.uint8)
            arr[..., 0] = (rgb[..., 0] * 255).astype(np.uint8)
            arr[..., 1] = (rgb[..., 1] * 255).astype(np.uint8)
            arr[..., 2] = (rgb[..., 2] * 255).astype(np.uint8)
            arr[..., 3] = 255
            img = QImage(arr.data, n, n, n * 4, QImage.Format.Format_RGBA8888).copy()
            pm = QPixmap.fromImage(img)
            self._square_cache[key] = pm
        return pm

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 底色圆角面板
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(26, 26, 30, 246))
        p.drawRoundedRect(0, 0, self.width(), self.height(), 10, 10)

        # 色环 + 方块
        p.drawPixmap(0, 0, self._ring())
        sq0 = self._sq0
        p.drawPixmap(int(sq0), int(sq0), self._square(self._hue))

        # 环指示器
        a = math.radians(self._hue)
        mr = (self.R_OUT + self.R_IN) / 2.0
        ix = self._cx + mr * math.cos(a)
        iy = self._cx + mr * math.sin(a)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(ix, iy), 5, 5)

        # 方块指示器
        ix2 = sq0 + self._s * self.SQUARE
        iy2 = sq0 + (1.0 - self._v) * self.SQUARE
        p.drawEllipse(QPointF(ix2, iy2), 4, 4)

        # 最近色条
        strip_y = self.SIZE + self.PAD
        n = min(len(self._recent), 8)
        cell = (self.SIZE - 20) / 8
        for i in range(n):
            c = self._recent[i]
            rect = (10 + i * cell, strip_y, cell - 2, self.STRIP_H)
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            p.setBrush(QColor(*c))
            p.drawRect(*rect)

        # 当前色 HEX
        hex_text = "#{:02x}{:02x}{:02x}".format(*self._color[:3])
        p.setPen(QColor(200, 205, 212))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(8, 6, 90, 14, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, hex_text)

        p.end()
