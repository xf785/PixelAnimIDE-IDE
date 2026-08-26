"""参考图上传卡片：豆包/即梦风格的紧凑小卡片。

- 空状态：虚线边框 + 「＋ 参考图」，点击打开文件选择；
- 已加载：显示缩略图 + 右上角 × 移除按钮，点击缩略图可更换；
- 通过 changed 信号通知外部（PIL Image 或 None）。
"""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QLabel, QToolButton, QWidget

from ui.i18n import T

logger = logging.getLogger("PixelAnimIDE.ui.reference_box")

# 深/浅色主题下均可见的中性色
_EMPTY_STYLE = (
    "border: 1.5px dashed rgba(140, 140, 150, 0.5); border-radius: 10px;"
    "color: #81858c; font-size: 12px; background: transparent;"
)
_LOADED_STYLE = (
    "border: 1px solid rgba(140, 140, 150, 0.35); border-radius: 10px; background: transparent;"
)


def _pil_to_pixmap(img: Image.Image, size: int) -> QPixmap:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    from PySide6.QtGui import QImage

    qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    pm = QPixmap.fromImage(qimg)
    return pm.scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,  # NEAREST，保持像素边缘
    )


class ReferenceImageBox(QWidget):
    """点击上传参考图的小卡片（可移除/更换）。"""

    changed = Signal(object)  # 新的参考图（PIL Image）或 None（移除）

    def __init__(self, size: int = 88, parent=None):
        super().__init__(parent)
        self._size = int(size)
        self._image: Optional[Image.Image] = None
        self.setFixedSize(self._size, self._size)

        self._label = QLabel(self)
        self._label.setGeometry(0, 0, self._size, self._size)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._label.setToolTip(T(None, "点击添加参考图（图生图）；已有图时点击可更换"))
        self._label.mousePressEvent = self._on_click

        self._remove_btn = QToolButton(self)
        self._remove_btn.setText("×")
        self._remove_btn.setToolTip(T(None, "移除参考图"))
        self._remove_btn.setGeometry(self._size - 22, 2, 20, 20)
        self._remove_btn.setVisible(False)
        self._remove_btn.clicked.connect(self.clear)

        self._render()

    # ------------------------------------------------------------------ #
    def image(self) -> Optional[Image.Image]:
        return self._image

    def set_image(self, img: Optional[Image.Image]) -> None:
        """程序化设置图片（不触发 changed，用于同步外部状态）。"""
        self._image = img
        self._render()

    def clear(self) -> None:
        self.set_image(None)
        self.changed.emit(None)

    # ------------------------------------------------------------------ #
    def _on_click(self, event) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, T(None, "选择参考图"), "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp);;所有文件 (*)"
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as exc:  # noqa: BLE001
            logger.warning("参考图读取失败: %s", exc)
            return
        self.set_image(img)
        self.changed.emit(img)

    def _render(self) -> None:
        if self._image is not None:
            self._label.setPixmap(_pil_to_pixmap(self._image, self._size))
            self._label.setText("")
            self._label.setStyleSheet(_LOADED_STYLE)
            self._remove_btn.setVisible(True)
        else:
            self._label.setPixmap(QPixmap())
            self._label.setText(T(None, "＋\n参考图"))
            self._label.setStyleSheet(_EMPTY_STYLE)
            self._remove_btn.setVisible(False)
