"""导出对话框：对已有生成结果重新导出（GIF / PNG 序列 / 雪碧图），可调参数。"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from PIL import Image

from core.processing import frame_utils as fu
from core.workflow import SoloResult
from ui.i18n import tr

logger = logging.getLogger("PixelAnimIDE.ui.export_dialog")


class ExportDialog(QDialog):
    """从 SoloResult 的帧目录重新导出交付物。"""

    def __init__(self, result: SoloResult, parent=None):
        super().__init__(parent)
        self._result = result
        self.setWindowTitle(tr("导出"))
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._out_edit = QLineEdit(str(self._result.output_dir / "re-export"))
        row = QHBoxLayout()
        row.addWidget(self._out_edit, 1)
        btn = QPushButton(tr("浏览…"))
        btn.clicked.connect(self._on_browse)
        row.addWidget(btn)
        form.addRow(tr("输出目录"), row)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(self._result.fps or 8)
        form.addRow(tr("帧率(fps)"), self._fps_spin)

        self._loop_spin = QSpinBox()
        self._loop_spin.setRange(0, 100)
        self._loop_spin.setValue(0)
        form.addRow(tr("循环次数(0=无限)"), self._loop_spin)

        self._scale_combo = QComboBox()
        for scale, label in [(1, tr("1x（原始）")), (2, "2x"), (4, "4x"), (8, "8x")]:
            self._scale_combo.addItem(label, userData=scale)
        form.addRow(tr("缩放比例"), self._scale_combo)

        self._gif_chk = QCheckBox(tr("GIF 动画"))
        self._gif_chk.setChecked(True)
        form.addRow(self._gif_chk)
        self._png_chk = QCheckBox(tr("PNG 序列帧"))
        self._png_chk.setChecked(True)
        form.addRow(self._png_chk)
        self._sheet_chk = QCheckBox(tr("雪碧图 (Sprite Sheet)"))
        form.addRow(self._sheet_chk)
        self._apng_chk = QCheckBox(tr("APNG 动画"))
        form.addRow(self._apng_chk)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("导出"))
        buttons.accepted.connect(self._on_export)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ #
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr("选择输出目录"), self._out_edit.text())
        if path:
            self._out_edit.setText(path)

    def _on_export(self) -> None:
        try:
            frames = self._load_frames()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tr("导出失败"), tr("读取帧失败: {0}").format(exc))
            return
        if not frames:
            QMessageBox.warning(self, tr("提示"), tr("没有可导出的帧"))
            return

        scale = self._scale_combo.currentData()
        if scale != 1:
            frames = [f.resize((f.width * scale, f.height * scale), Image.Resampling.NEAREST) for f in frames]

        out = Path(self._out_edit.text().strip())
        out.mkdir(parents=True, exist_ok=True)
        results = []
        try:
            if self._gif_chk.isChecked():
                path = fu.frames_to_gif(frames, out / "pixel_anim.gif", fps=self._fps_spin.value(), loop=self._loop_spin.value())
                results.append(str(path))
            if self._apng_chk.isChecked():
                path = fu.frames_to_apng(frames, out / "pixel_anim.apng", fps=self._fps_spin.value(), loop=self._loop_spin.value())
                results.append(str(path))
            if self._png_chk.isChecked():
                paths = fu.save_png_sequence(frames, out / "png", prefix="frame")
                results.append(tr("{0} 张 PNG → {1}").format(len(paths), out / "png"))
            if self._sheet_chk.isChecked():
                sheet = fu.frames_to_sprite_sheet(frames)
                path = fu.save_image(sheet, out / "sprite_sheet.png")
                results.append(str(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tr("导出失败"), str(exc))
            return
        QMessageBox.information(self, tr("导出完成"), "\n".join(results))
        self.accept()

    def _load_frames(self):
        frames_dir = self._result.frames_dir
        if not frames_dir or not Path(frames_dir).exists():
            raise FileNotFoundError(tr("帧目录不存在: {0}").format(frames_dir))
        paths = sorted(Path(frames_dir).glob("*.png"))
        return [fu.load_image(p) for p in paths]
