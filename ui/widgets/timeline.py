"""时间轴 / 帧列表控件：缩略图预览、点击选中、拖动排序、插入/复制/删除/追加。

本控件只负责展示与交互（发信号），帧数据的实际增删改由页面层操作
IdeSession.frames 完成后再调用 set_frames 刷新。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from PIL import Image
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.i18n import T, tr

logger = logging.getLogger("PixelAnimIDE.ui.timeline")

THUMB_SIZE = 56


def _pil_to_qpixmap(img: Image.Image, size: int) -> QPixmap:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    pm = QPixmap.fromImage(qimg)
    return pm.scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,  # NEAREST，保持像素边缘
    )


class _FrameList(QListWidget):
    """支持内部拖拽排序的帧缩略图列表。"""

    reordered = Signal(list)  # 拖拽后按新顺序返回 tag 列表（原始索引）

    def dropEvent(self, event) -> None:  # noqa: N802
        super().dropEvent(event)
        order = [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]
        self.reordered.emit(order)


class TimelineWidget(QWidget):
    """帧时间轴。"""

    frame_selected = Signal(int)          # 选中的帧索引
    reordered = Signal(list)              # 拖拽后的新顺序（原始索引列表）
    insert_requested = Signal()           # 在选中帧前插入
    duplicate_requested = Signal()        # 复制选中帧
    delete_requested = Signal()           # 删除选中帧
    add_requested = Signal()              # 追加空白帧

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        bar = QHBoxLayout()
        bar.setSpacing(4)
        self._insert_btn = T(QPushButton(), "插入")
        T(self._insert_btn, "在当前帧后插入一帧（复制当前帧）", attr="tooltip")
        self._insert_btn.clicked.connect(self.insert_requested.emit)
        bar.addWidget(self._insert_btn)
        self._dup_btn = T(QPushButton(), "复制")
        T(self._dup_btn, "复制当前帧", attr="tooltip")
        self._dup_btn.clicked.connect(self.duplicate_requested.emit)
        bar.addWidget(self._dup_btn)
        self._del_btn = T(QPushButton(), "删除")
        T(self._del_btn, "删除当前帧", attr="tooltip")
        self._del_btn.clicked.connect(self.delete_requested.emit)
        bar.addWidget(self._del_btn)
        self._add_btn = T(QPushButton(), "+ 空白帧")
        T(self._add_btn, "追加一个空白帧", attr="tooltip")
        self._add_btn.clicked.connect(self.add_requested.emit)
        bar.addWidget(self._add_btn)
        bar.addStretch(1)
        hint = T(QLabel(), "拖动缩略图可调整帧顺序")
        hint.setObjectName("HintLabel")
        bar.addWidget(hint)
        layout.addLayout(bar)

        self._list = _FrameList()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setFlow(QListWidget.Flow.LeftToRight)
        self._list.setWrapping(False)
        self._list.setMovement(QListWidget.Movement.Snap)
        self._list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self._list.setSpacing(4)
        self._list.setFixedHeight(THUMB_SIZE + 40)
        self._list.currentRowChanged.connect(self.frame_selected.emit)
        self._list.reordered.connect(self.reordered.emit)
        layout.addWidget(self._list)

    # ------------------------------------------------------------------ #
    def set_frames(self, frames: List[Image.Image], select: Optional[int] = None) -> None:
        """刷新缩略图列表（重建 item，tag = 原始索引）。"""
        self._list.clear()
        for i, frame in enumerate(frames):
            item = QListWidgetItem()
            item.setIcon(QIcon(_pil_to_qpixmap(frame, THUMB_SIZE)))
            item.setText(str(i + 1))
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setToolTip(tr("帧 {i}").format(i=i + 1))
            self._list.addItem(item)
        if select is not None and 0 <= select < self._list.count():
            self._list.setCurrentRow(select)

    def current_index(self) -> int:
        return self._list.currentRow()

    def update_thumbnail(self, index: int, image: Image.Image) -> None:
        """只刷新单帧缩略图（编辑时避免整条重建）。"""
        if 0 <= index < self._list.count():
            self._list.item(index).setIcon(QIcon(_pil_to_qpixmap(image, THUMB_SIZE)))

    def select(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)
            self._list.scrollToItem(self._list.item(index))

    def frame_count(self) -> int:
        return self._list.count()
