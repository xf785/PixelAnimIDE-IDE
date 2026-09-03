"""背景抠图预览对话框：实时显示扣除效果，调整 容差 / 内缩(去白边) / 羽化。

- 左侧实时预览（棋盘格底显示透明区域），参数变化即时重算；
- 可切换「强制纯色背景」预归一化与「显示原图」对照；
- 确定后返回参数 dict，由调用方应用到全部帧。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.processing import background as bg_mod
from ui.i18n import tr
from ui.widgets.image_viewer import ImageViewer


def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


def _checkerboard(size, cell: int = 8) -> Image.Image:
    """浅灰棋盘格底（显示透明区域）。"""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    chess = (xx // cell + yy // cell) % 2 == 0
    light = np.array([210, 210, 210, 255], dtype=np.uint8)
    dark = np.array([238, 238, 238, 255], dtype=np.uint8)
    arr = np.where(chess[..., None], light, dark)
    return Image.fromarray(arr, "RGBA")


class BackgroundKeyDialog(QDialog):
    """背景扣除参数预览弹窗。"""

    def __init__(
        self,
        image: Image.Image,
        tolerance: int = 30,
        feather: int = 8,
        erode: int = 0,
        force_pure_bg: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("背景扣除预览"))
        self.setMinimumSize(700, 460)
        self._original = image.convert("RGBA")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 左：预览
        self._viewer = ImageViewer()
        layout.addWidget(self._viewer, 1)

        # 右：参数
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setSpacing(10)

        rl.addWidget(QLabel(tr("参数调整（实时预览）")))

        def row(label: str, spin: QSpinBox) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(label))
            h.addWidget(spin, 1)
            return w

        self._tol_spin = QSpinBox()
        self._tol_spin.setRange(0, 200)
        self._tol_spin.setValue(int(tolerance))
        rl.addWidget(row(tr("背景容差"), self._tol_spin))

        self._erode_spin = QSpinBox()
        self._erode_spin.setRange(0, 12)
        self._erode_spin.setValue(int(erode))
        self._erode_spin.setToolTip(tr("前景内缩像素：消掉对象边缘残留的白边/白晕"))
        rl.addWidget(row(tr("内缩(px)"), self._erode_spin))

        self._feather_spin = QSpinBox()
        self._feather_spin.setRange(0, 30)
        self._feather_spin.setValue(int(feather))
        rl.addWidget(row(tr("羽化(px)"), self._feather_spin))

        self._force_chk = QCheckBox(tr("先强制纯色背景（自适应归一化）"))
        self._force_chk.setChecked(bool(force_pure_bg))
        rl.addWidget(self._force_chk)

        self._orig_chk = QCheckBox(tr("显示原图对照"))
        rl.addWidget(self._orig_chk)

        for w in (self._tol_spin, self._erode_spin, self._feather_spin):
            w.valueChanged.connect(self._update_preview)
        self._force_chk.toggled.connect(self._update_preview)
        self._orig_chk.toggled.connect(self._update_preview)

        rl.addStretch(1)

        btns = QHBoxLayout()
        apply_btn = QPushButton(tr("应用到全部帧"))
        apply_btn.setObjectName("PrimaryButton")
        apply_btn.clicked.connect(self.accept)
        btns.addWidget(apply_btn)
        cancel_btn = QPushButton(tr("取消"))
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        rl.addLayout(btns)

        layout.addWidget(right)

        self._update_preview()

    # ------------------------------------------------------------------ #
    def params(self) -> dict:
        return {
            "tolerance": self._tol_spin.value(),
            "erode": self._erode_spin.value(),
            "feather": self._feather_spin.value(),
            "force_pure_bg": self._force_chk.isChecked(),
        }

    def _update_preview(self) -> None:
        if self._orig_chk.isChecked():
            self._viewer.show_image(_pil_to_qpixmap(self._original))
            return
        tol = self._tol_spin.value()
        erode = self._erode_spin.value()
        feather = self._feather_spin.value()
        img: Optional[Image.Image] = None
        if self._force_chk.isChecked():
            img, _fill, mask = bg_mod.normalize_background(self._original)
            if mask is not None:
                img = bg_mod.apply_background_mask(img, mask, feather=feather, erode=erode)
            else:
                img = bg_mod.remove_background(
                    img, key_color=(255, 255, 255), tolerance=tol, feather=feather, erode=erode
                )
        else:
            img = bg_mod.remove_background(
                self._original, key_color=(255, 255, 255), tolerance=tol, feather=feather, erode=erode
            )
        if img is None:
            return
        composed = _checkerboard(img.size)
        composed.alpha_composite(img)
        self._viewer.show_image(_pil_to_qpixmap(composed))
