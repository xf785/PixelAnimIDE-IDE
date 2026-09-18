"""PIL ↔ Qt 图像转换（供预览/缩略图统一复用）。

原先 `ui/pages/ide_page.py`、`ui/dialogs/background_key_dialog.py`、
`ui/widgets/timeline.py`、`ui/widgets/reference_box.py` 各写了一份同样的
「RGBA 字节 -> QImage -> QPixmap」代码，这里统一成一份。
"""
from __future__ import annotations

from PIL import Image
from PySide6.QtGui import QImage, QPixmap


def pil_to_qimage(img: Image.Image) -> QImage:
    """PIL RGBA -> QImage（Format_RGBA8888）。"""
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    return QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """PIL RGBA -> QPixmap（1:1，不做缩放）。"""
    return QPixmap.fromImage(pil_to_qimage(img))


def pil_to_thumbnail(img: Image.Image, size: int) -> QPixmap:
    """等比缩放到 size×size 的 NEAREST 缩略图（保持像素边缘）。"""
    from PySide6.QtCore import Qt

    return pil_to_qpixmap(img).scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
