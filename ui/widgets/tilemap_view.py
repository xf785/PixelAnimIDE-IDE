"""瓦片地图预览与铺设控件 + PIL/QPixmap 转换工具。

- TilemapView：大网格地图预览，左键画笔铺设 / 右键橡皮擦除，滚轮缩放
  （最近邻放大，像素不模糊），网格线显示；
- pil_to_qpixmap / pil_to_qimage：PIL RGBA -> Qt 图像（供缩略图/预览复用）。
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from PIL import Image
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

from core.tilemap import TileMapModel
from ui.i18n import T, tr

logger = logging.getLogger("PixelAnimIDE.ui.tilemap_view")

GRID_LINE = QColor(0, 0, 0, 46)
GRID_LINE_LIGHT = QColor(255, 255, 255, 60)


def pil_to_qimage(img: Image.Image) -> QImage:
    """PIL RGBA -> QImage（Format_RGBA8888）。"""
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return qimg.copy()


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    return QPixmap.fromImage(pil_to_qimage(img))


class TilemapView(QWidget):
    """地图预览画布：左键画笔（当前值）/ 右键橡皮，滚轮缩放，Ctrl 无效则显示网格。"""

    changed = Signal()

    def __init__(
        self,
        model: TileMapModel,
        center: Image.Image,
        line_color: Tuple[int, int, int] = (0, 0, 0),
        line_width: int = 1,
        atlas_mode: str = "47",
        parent=None,
    ):
        super().__init__(parent)
        self._model = model
        self._center = center
        self._line_color = line_color
        self._line_width = line_width
        self._atlas_mode = atlas_mode
        self._zoom = 3
        self._erase = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(220, 180)
        self._toolbar = QHBoxLayout()
        self._paint_btn = T(QToolButton(), "画笔")
        self._paint_btn.setCheckable(True)
        self._paint_btn.setChecked(True)
        self._erase_btn = T(QToolButton(), "橡皮")
        self._erase_btn.setCheckable(True)
        self._paint_btn.clicked.connect(lambda: self._set_erase(False))
        self._erase_btn.clicked.connect(lambda: self._set_erase(True))
        self._clear_btn = T(QPushButton(), "清空")
        self._clear_btn.clicked.connect(self.clear)
        self._zoom_label = QLabel()
        self._toolbar.addWidget(self._paint_btn)
        self._toolbar.addWidget(self._erase_btn)
        self._toolbar.addWidget(self._clear_btn)
        self._toolbar.addStretch(1)
        self._toolbar.addWidget(self._zoom_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addLayout(self._toolbar)
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._canvas.setMinimumSize(220, 180)
        outer.addWidget(self._canvas, 1)
        self._rebuild()

    # ------------------------------------------------------------------ #
    def _set_erase(self, erase: bool) -> None:
        self._erase = erase
        self._paint_btn.setChecked(not erase)
        self._erase_btn.setChecked(erase)

    def clear(self) -> None:
        self._model.clear()
        self._rebuild()
        self.changed.emit()

    def model(self) -> TileMapModel:
        return self._model

    def set_zoom(self, zoom: int) -> None:
        self._zoom = max(1, min(10, int(zoom)))
        self._rebuild()

    def zoom(self) -> int:
        return self._zoom

    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        img = self._model.render(
            self._center, self._line_color, self._line_width, mode=self._atlas_mode
        )
        s = self._model.tile_size
        pix = pil_to_qpixmap(img).scaled(
            img.width * self._zoom,
            img.height * self._zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        # 叠加网格线
        painter = QPainter(pix)
        pen = QPen(GRID_LINE)
        pen.setWidth(1)
        painter.setPen(pen)
        step = s * self._zoom
        for x in range(0, pix.width() + 1, step):
            painter.drawLine(x, 0, x, pix.height())
        for y in range(0, pix.height() + 1, step):
            painter.drawLine(0, y, pix.width(), y)
        painter.end()
        self._canvas.setPixmap(pix)
        self._canvas.setFixedSize(pix.size())
        self._zoom_label.setText(f"{self._zoom}x")
        self.update()

    def _cell_at(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        s = self._model.tile_size * self._zoom
        if pos.x() < 0 or pos.y() < 0:
            return None
        x, y = pos.x() // s, pos.y() // s
        if 0 <= x < self._model.width and 0 <= y < self._model.height:
            return x, y
        return None

    def _paint_cell(self, pos: QPoint) -> None:
        cell = self._cell_at(pos)
        if cell is None:
            return
        self._model.set_cell(cell[0], cell[1], 0 if self._erase else 1)
        self._rebuild()
        self.changed.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._erase = event.button() == Qt.MouseButton.RightButton
            self._paint_btn.setChecked(not self._erase)
            self._erase_btn.setChecked(self._erase)
            self._paint_cell(event.position().toPoint())
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            self._paint_cell(event.position().toPoint())

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.set_zoom(self._zoom + 1)
        elif delta < 0:
            self.set_zoom(self._zoom - 1)
        event.accept()
