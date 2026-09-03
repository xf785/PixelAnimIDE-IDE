"""图片查看器：自适应缩放显示图片，支持 GIF 动画播放（QMovie）与滚轮缩放。

静态图缩放：
- 滚轮（或按钮）缩放，以鼠标位置为焦点（光标下的图像点保持不动）；
- zoom() 返回相对倍数，1.0 = 适应容器（相对「适应」尺寸的倍数，0.2x~8x）；
- 缩放变化通过 zoomChanged 信号通知外部（用于同步显示百分比）。
GIF：QMovie 播放；播放倍速用 set_speed 调整。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMovie, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui.i18n import tr

MIN_ZOOM = 0.2
MAX_ZOOM = 8.0


class ImageViewer(QWidget):
    """显示静态图片或 GIF 动画的容器，支持以光标为焦点的缩放。"""

    zoomChanged = Signal(float)  # 相对缩放倍数（1.0 = 适应容器）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 120)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(tr("暂无预览"))
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._label.hide()  # 仅 GIF 动画时显示
        layout.addWidget(self._label)
        self._movie: Optional[QMovie] = None
        self._source_pixmap: Optional[QPixmap] = None
        self._fit = True                 # True = 适应容器
        self._rel = 1.0                  # 相对适应尺寸的倍数
        self._ox = 0.0                   # 图像左上角在控件中的坐标
        self._oy = 0.0
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ #
    # 缩放
    # ------------------------------------------------------------------ #
    def zoom_in(self) -> None:
        self._zoom_at(self.rect().center(), 1.25)

    def zoom_out(self) -> None:
        self._zoom_at(self.rect().center(), 1 / 1.25)

    def reset_zoom(self) -> None:
        self._fit = True
        self._rel = 1.0
        self._clamp_offset()
        self.update()
        self.zoomChanged.emit(self.zoom())

    def set_zoom(self, factor: float) -> None:
        self._fit = False
        self._rel = max(MIN_ZOOM, min(MAX_ZOOM, float(factor)))
        self._clamp_offset()
        self.update()
        self.zoomChanged.emit(self.zoom())

    def zoom(self) -> float:
        """相对缩放倍数（1.0 = 适应容器）。"""
        return 1.0 if self._fit else self._rel

    # ------------------------------------------------------------------ #
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """滚轮缩放，以鼠标位置为焦点。"""
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self._zoom_at(event.position(), 1.25 if delta > 0 else 1 / 1.25)
        event.accept()

    def _zoom_at(self, pos, factor: float) -> None:
        """在 pos（控件坐标）处缩放 factor 倍，保持光标下的图像点不动。"""
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        if self._fit:
            self._fit = False
            self._rel = 1.0
        old_abs = self._current_scale()
        new_rel = max(MIN_ZOOM, min(MAX_ZOOM, self._rel * factor))
        sw, sh = self._source_pixmap.width(), self._source_pixmap.height()
        if old_abs > 0 and sw > 0 and sh > 0:
            # 光标处的图像相对坐标
            fx = (pos.x() - self._ox) / (sw * old_abs)
            fy = (pos.y() - self._oy) / (sh * old_abs)
            self._rel = new_rel
            new_abs = self._current_scale()
            self._ox = pos.x() - fx * sw * new_abs
            self._oy = pos.y() - fy * sh * new_abs
        else:
            self._rel = new_rel
        self._clamp_offset()
        self.update()
        self.zoomChanged.emit(self.zoom())

    def _current_scale(self) -> float:
        """当前绝对缩放倍数（相对图像原生像素）。"""
        pm = self._source_pixmap
        if pm is None or pm.isNull() or pm.width() <= 0 or pm.height() <= 0:
            return 1.0
        fit = min(self.width() / pm.width(), self.height() / pm.height())
        return fit if self._fit else fit * self._rel

    def _clamp_offset(self) -> None:
        """图像小于视口时居中，大于视口时限制偏移不出现空白。"""
        pm = self._source_pixmap
        if pm is None or pm.isNull():
            return
        scale = self._current_scale()
        sw, sh = pm.width() * scale, pm.height() * scale
        vw, vh = self.width(), self.height()
        if sw <= vw:
            self._ox = (vw - sw) / 2
        else:
            self._ox = max(vw - sw, min(0.0, self._ox))
        if sh <= vh:
            self._oy = (vh - sh) / 2
        else:
            self._oy = max(vh - sh, min(0.0, self._oy))

    # ------------------------------------------------------------------ #
    def show_image(self, pixmap: QPixmap) -> None:
        self._clear_movie()
        self._source_pixmap = pixmap
        self._fit = True
        self._rel = 1.0
        self._clamp_offset()
        self.update()

    def show_path(self, path: Path | str) -> None:
        """按扩展名自动选择静态图或 GIF 播放。"""
        path = Path(path)
        if path.suffix.lower() == ".gif":
            self.show_gif(str(path))
        else:
            self.show_image(QPixmap(str(path)))

    def show_gif(self, path: Path | str, speed: float = 1.0) -> None:
        self._clear_movie()
        self._source_pixmap = None
        self._movie = QMovie(str(path))
        self._movie.setCacheMode(QMovie.CacheMode.CacheAll)
        self._movie.setSpeed(max(10, int(float(speed) * 100)))
        self._label.setMovie(self._movie)
        self._label.setText("")
        self._label.show()
        self._movie.start()
        self.update()

    def set_speed(self, speed: float) -> None:
        """调整 GIF 播放倍速（0.5x / 1x / 2x …）。"""
        if self._movie is not None:
            self._movie.setSpeed(max(10, int(float(speed) * 100)))

    def toggle_play(self) -> bool:
        """播放/暂停 GIF（无动画时返回 False）。"""
        if self._movie is None:
            return False
        if self._movie.state() == QMovie.MovieState.Running:
            self._movie.setPaused(True)
        else:
            self._movie.setPaused(False)
        return True

    def clear(self) -> None:
        self._clear_movie()
        self._source_pixmap = None
        self.update()

    def _clear_movie(self) -> None:
        if self._movie is not None:
            self._movie.stop()
            self._movie.deleteLater()
            self._movie = None
        self._label.hide()

    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        if self._movie is not None:
            return  # GIF 由 QLabel 播放
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window().color())
        pm = self._source_pixmap
        if pm is None or pm.isNull():
            painter.setPen(self.palette().windowText().color())
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr("暂无预览"))
            painter.end()
            return
        scale = self._current_scale()
        scaled = pm.scaled(
            max(1, int(pm.width() * scale)),
            max(1, int(pm.height() * scale)),
            Qt.AspectRatioMode.KeepAspectRatio,
            # 像素风图片用最近邻，避免放大后边缘发糊（完美像素预览）
            Qt.TransformationMode.FastTransformation,
        )
        painter.drawPixmap(int(round(self._ox)), int(round(self._oy)), scaled)
        painter.end()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._clamp_offset()
        self.update()
