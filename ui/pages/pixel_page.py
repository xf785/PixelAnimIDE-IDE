"""像素编辑板块：独立像素画布（可设分辨率），与 IDE 双向同步，可作图生视频首帧。

- 新建画布：预设/自定义分辨率 + 背景（透明/白/黑）；
- 从 IDE 同步：拉取 IDE 当前帧/首帧进画布精细编辑；
- 同步到 IDE：把画布图作为首帧 + 图生图参考导入 IDE；
- 用作图生视频首帧：把画布图作为参考/首帧发给 Solo，直接走图生视频；
- 导出 PNG。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config.settings import DEFAULT_OUTPUT_DIR
from ui.app_context import AppContext
from ui.i18n import T, tr
from ui.widgets.pixel_editor import PixelEditorWidget

logger = logging.getLogger("PixelAnimIDE.ui.pixel_page")

RESOLUTION_PRESETS = [16, 32, 64, 128, 256, 512]
_BG_FILLS = {"透明": (0, 0, 0, 0), "白色": (255, 255, 255, 255), "黑色": (0, 0, 0, 255)}


class PixelPage(QWidget):
    """独立像素编辑板块。"""

    sync_from_ide_requested = Signal()            # 请求主窗口从 IDE 拉当前帧进来
    sync_to_ide = Signal(object)                  # 画布图 -> IDE 首帧/参考图
    use_as_video_first_frame = Signal(object)     # 画布图 -> Solo 图生视频首帧

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(0)

        # ---------- 展开按钮条（画布设置收起时显示） ----------
        self._btn_expand = QPushButton("▶")
        self._btn_expand.setObjectName("SidebarCollapseBtn")
        self._btn_expand.setFixedWidth(24)
        self._btn_expand.setVisible(False)
        T(self._btn_expand, "展开画布设置", attr="tooltip")
        self._btn_expand.clicked.connect(self._on_expand_settings)
        root.addWidget(self._btn_expand)

        # ---------- 左：设置与操作（可收起；分栏可拖拽调宽） ----------
        self._settings_panel = QWidget()
        lp = QVBoxLayout(self._settings_panel)
        lp.setContentsMargins(0, 0, 0, 0)
        lp.setSpacing(10)

        # 「画布设置」框：标题行右侧带收起按钮
        box = QWidget()
        bv = QVBoxLayout(box)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = T(QLabel(), "画布设置")
        title.setObjectName("GroupTitle")
        header.addWidget(title)
        header.addStretch(1)
        self._btn_collapse = QPushButton("◀")
        self._btn_collapse.setObjectName("SidebarCollapseBtn")
        self._btn_collapse.setFixedSize(22, 22)
        T(self._btn_collapse, "收起画布设置，画布更大", attr="tooltip")
        self._btn_collapse.clicked.connect(self._on_collapse_settings)
        header.addWidget(self._btn_collapse)
        bv.addLayout(header)
        f = QFormLayout()
        f.setContentsMargins(12, 8, 12, 12)
        f.setVerticalSpacing(8)

        self._preset_combo = QComboBox()
        for s in RESOLUTION_PRESETS:
            self._preset_combo.addItem(f"{s}×{s}", userData=s)
        self._preset_combo.setCurrentIndex(2)  # 64×64
        self._preset_combo.currentIndexChanged.connect(self._on_preset)
        f.addRow(T(QLabel(), "预设"), self._preset_combo)

        custom_row = QHBoxLayout()
        self._custom_w = QSpinBox()
        self._custom_w.setRange(8, 1024)
        self._custom_w.setValue(64)
        custom_row.addWidget(self._custom_w, 1)
        custom_row.addWidget(QLabel("×"))
        self._custom_h = QSpinBox()
        self._custom_h.setRange(8, 1024)
        self._custom_h.setValue(64)
        custom_row.addWidget(self._custom_h, 1)
        f.addRow(T(QLabel(), "分辨率(宽×高)"), custom_row)

        self._bg_combo = QComboBox()
        for key, fill in _BG_FILLS.items():
            self._bg_combo.addItem(T(None, key), userData=key)
        f.addRow(T(QLabel(), "背景"), self._bg_combo)

        self._btn_new = T(QPushButton(), "新建画布")
        self._btn_new.setObjectName("PrimaryButton")
        self._btn_new.clicked.connect(self._on_new)
        f.addRow("", self._btn_new)
        bv.addLayout(f)
        lp.addWidget(box)

        act_box = T(QGroupBox(), "操作")
        av = QVBoxLayout(act_box)
        av.setContentsMargins(12, 18, 12, 12)
        av.setSpacing(8)
        self._btn_import = T(QPushButton(), "导入图片…")
        T(self._btn_import, "从本地导入图片替换当前帧", attr="tooltip")
        self._btn_import.clicked.connect(self._on_import)
        av.addWidget(self._btn_import)
        self._btn_from_ide = T(QPushButton(), "从 IDE 同步")
        T(self._btn_from_ide, "把 IDE 当前帧/首帧拉进画布精细编辑", attr="tooltip")
        self._btn_from_ide.clicked.connect(self._on_sync_from_ide)
        av.addWidget(self._btn_from_ide)
        self._btn_to_ide = T(QPushButton(), "同步到 IDE")
        T(self._btn_to_ide, "把画布图作为首帧 + 图生图参考导入 IDE", attr="tooltip")
        self._btn_to_ide.clicked.connect(self._on_sync_to_ide)
        av.addWidget(self._btn_to_ide)
        self._btn_video = T(QPushButton(), "用作图生视频首帧")
        T(self._btn_video, "把画布图作为首帧走图生视频（Solo）；过小会自动最近邻放大到 API 最低要求", attr="tooltip")
        self._btn_video.clicked.connect(self._on_use_as_video)
        av.addWidget(self._btn_video)
        self._btn_export = T(QPushButton(), "导出 PNG")
        self._btn_export.clicked.connect(self._on_export)
        av.addWidget(self._btn_export)
        lp.addWidget(act_box)
        lp.addStretch(1)

        # ---------- 右：画布编辑器（分栏可拖拽调宽） ----------
        self._editor = PixelEditorWidget()
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._settings_panel)
        self._splitter.addWidget(self._editor)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([260, 900])
        root.addWidget(self._splitter, 1)

    # ------------------------------------------------------------------ #
    def _on_collapse_settings(self) -> None:
        """收起画布设置栏：画布占满，仅留展开按钮。"""
        self._settings_panel.setVisible(False)
        self._btn_expand.setVisible(True)

    def _on_expand_settings(self) -> None:
        """重新展开画布设置栏。"""
        self._settings_panel.setVisible(True)
        self._btn_expand.setVisible(False)
        self._splitter.setSizes([260, max(200, self._splitter.width() - 260)])

    # ------------------------------------------------------------------ #
    def _on_preset(self) -> None:
        data = self._preset_combo.currentData()
        if data:
            self._custom_w.setValue(int(data))
            self._custom_h.setValue(int(data))

    def _on_new(self) -> None:
        w, h = self._custom_w.value(), self._custom_h.value()
        fill = _BG_FILLS.get(self._bg_combo.currentData(), (0, 0, 0, 0))
        self._editor.set_frame(Image.new("RGBA", (w, h), fill))
        self._status(tr("已新建 {w}×{h} 画布").format(w=w, h=h))

    def set_image(self, img) -> None:
        """外部（主窗口/IDE）导入图片到画布。"""
        self._editor.set_frame(img.convert("RGBA"))
        self._status(tr("已载入 {w}×{h} 图片").format(w=img.width, h=img.height))

    def image(self):
        return self._editor.frame()

    def apply_ui_scale(self, scale: float) -> None:
        """按界面比例同步缩放内部像素编辑器控件。"""
        self._editor.apply_ui_scale(scale)

    # ------------------------------------------------------------------ #
    def _on_import(self) -> None:
        """从本地导入图片（复用编辑器导入逻辑）。"""
        self._editor.import_image()

    def _on_sync_from_ide(self) -> None:
        self.sync_from_ide_requested.emit()

    def _on_sync_to_ide(self) -> None:
        self.sync_to_ide.emit(self.image())

    def _on_use_as_video(self) -> None:
        self.use_as_video_first_frame.emit(self.image())

    def _on_export(self) -> None:
        base = Path(self._ctx.ui_settings.get("output_dir") or str(DEFAULT_OUTPUT_DIR))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, tr("导出 PNG"), str(base / f"pixel_{ts}.png"), "PNG 图片 (*.png)"
        )
        if not path:
            return
        self.image().save(path, format="PNG")
        self._status(f"{tr('已导出：')}{path}")

    # ------------------------------------------------------------------ #
    def _status(self, message: str) -> None:
        try:
            from PySide6.QtWidgets import QApplication

            win = self.window()
            if win is not None and hasattr(win, "statusBar"):
                win.statusBar().showMessage(message)
        except Exception:  # noqa: BLE001
            pass
