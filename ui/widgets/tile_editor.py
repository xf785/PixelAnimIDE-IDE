"""瓦片逐张重绘编辑器（对话框）：复用 PixelEditorWidget 编辑 9 张基础瓦片。

流程：选择瓦片（左侧缩略图）→ 像素编辑器重绘 → 「应用修改」写回当前瓦片；
「完成」返回全部修改（调用方随后重跑 无缝化/瓦片集/导出 步骤）。
"""
from __future__ import annotations

import logging
from typing import Dict, List

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.tilemap import BaseTileSet
from ui.i18n import T, tr
from ui.widgets.pixel_editor import PixelEditorWidget
from ui.widgets.tilemap_view import pil_to_qpixmap

logger = logging.getLogger("PixelAnimIDE.ui.tile_editor")

TILE_NAMES = [
    ("tl", "左上角"), ("top", "上边"), ("tr", "右上角"),
    ("left", "左边"), ("center", "中心"), ("right", "右边"),
    ("bl", "左下角"), ("bottom", "下边"), ("br", "右下角"),
]


class TileEditorDialog(QDialog):
    """9 张瓦片编辑器（基础九宫格，编辑后由调用方重新无缝化/生成瓦片集）。"""

    def __init__(self, base: BaseTileSet, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("编辑瓦片（重绘后需重新生成瓦片集）"))
        self.resize(1080, 720)
        self._tiles: Dict[str, Image.Image] = {n: base.tile(n).copy() for n, _ in TILE_NAMES}
        self._current = TILE_NAMES[0][0]

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 左侧瓦片选择九宫格
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(T(QLabel(), "选择瓦片（点击切换）"))
        grid = QGridLayout()
        grid.setSpacing(4)
        self._buttons: Dict[str, QToolButton] = {}
        for i, (name, zh) in enumerate(TILE_NAMES):
            btn = QToolButton()
            btn.setFixedSize(84, 84)
            btn.setIconSize(btn.size() - btn.size() / 4)
            T(btn, zh, attr="tooltip")
            btn.clicked.connect(lambda _=False, n=name: self._select(n))
            self._buttons[name] = btn
            grid.addWidget(btn, i // 3, i % 3)
        lv.addLayout(grid)
        lv.addStretch(1)
        root.addWidget(left)

        # 右侧像素编辑器
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self._editor = PixelEditorWidget()
        rv.addWidget(self._editor, 1)
        actions = QHBoxLayout()
        self._apply_btn = T(QPushButton(), "应用修改")
        self._apply_btn.clicked.connect(self._apply)
        actions.addStretch(1)
        actions.addWidget(self._apply_btn)
        self._ok_btn = T(QPushButton(), "完成")
        self._ok_btn.clicked.connect(self.accept)
        actions.addWidget(self._ok_btn)
        rv.addLayout(actions)
        root.addWidget(right, 1)

        self._select(self._current)
        self._refresh_thumbnails()

    # ------------------------------------------------------------------ #
    def _select(self, name: str) -> None:
        self._current = name
        self._editor.set_frame(self._tiles[name])
        for n, btn in self._buttons.items():
            btn.setStyleSheet(
                "border: 2px solid #3b82f6;" if n == name else "border: 1px solid #3a3f47;"
            )

    def _apply(self) -> None:
        self._tiles[self._current] = self._editor.frame().copy()
        self._refresh_thumbnails()

    def _refresh_thumbnails(self) -> None:
        for name, btn in self._buttons.items():
            tile = self._tiles[name]
            pix = pil_to_qpixmap(tile).scaled(
                80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
            )
            btn.setIcon(pix)
            btn.setIconSize(pix.size())

    # ------------------------------------------------------------------ #
    def result(self) -> Dict[str, Image.Image]:
        """返回编辑后的 9 张瓦片（name -> RGBA）。"""
        return {n: t.copy() for n, t in self._tiles.items()}


def base_set_with_edits(base: BaseTileSet, tiles: Dict[str, Image.Image]) -> BaseTileSet:
    """把编辑结果套回 BaseTileSet（保持原尺寸/线色元数据）。"""
    return BaseTileSet(
        size=base.size,
        center=tiles["center"],
        edges={n: tiles[n] for n in ("top", "bottom", "left", "right")},
        corners={n: tiles[n] for n in ("tl", "tr", "bl", "br")},
        line_color=base.line_color,
        line_width=base.line_width,
    )
