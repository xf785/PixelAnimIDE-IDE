"""瓦片地图模式页（第 5 模式）。

链路：文本描述（纹理/风格）→ 内置严格瓦片集提示词 → 文生 3×3 瓦片集底图
      → 自适应裁切 9 张瓦片 → 逐瓦片重绘（像素编辑器）→ 无缝化
      → 47-tile 瓦片集 / 双网格（用户可选）→ 大网格地图预览与铺设 → 导出。
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.settings import DEFAULT_OUTPUT_DIR
from core.tilemap import TileMapModel
from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow
from core.workflow.solo_workflow import WorkflowError
from ui.app_context import AppContext
from ui.i18n import T, tr
from ui.widgets.tile_editor import TileEditorDialog, base_set_with_edits
from ui.widgets.tilemap_view import TilemapView, pil_to_qpixmap
from ui.workers import TilemapWorker

logger = logging.getLogger("PixelAnimIDE.ui.tilemap_page")

FORM_WIDTH = 360
STYLE_PRESETS = ["game sprite", "retro", "pixel", "top-down RPG", "platformer", "16-bit"]


class TilemapPage(QWidget):
    """瓦片地图模式页。"""

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._worker: TilemapWorker | None = None
        self._result = None
        self._session = None
        self._params: TilemapParams | None = None
        self._local_wf = TilemapWorkflow(image_api=None)  # 编辑后本地重跑（无需 API）
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(14)

        # ---------- 左：参数表单 ----------
        left = QWidget()
        lp = QVBoxLayout(left)
        lp.setContentsMargins(0, 0, 0, 0)
        lp.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setFixedWidth(FORM_WIDTH)
        host = QWidget()
        fl = QVBoxLayout(host)
        fl.setContentsMargins(0, 0, 6, 0)
        fl.setSpacing(10)

        input_box = T(QGroupBox(), "瓦片集参数")
        f = QFormLayout(input_box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(10)

        self._desc_edit = QTextEdit()
        T(self._desc_edit, "例如：草地、石砖墙、熔岩地面、水面……", attr="placeholder")
        self._desc_edit.setMaximumHeight(72)
        f.addRow(T(QLabel(), "纹理描述"), self._desc_edit)

        self._style_combo = QComboBox()
        self._style_combo.setEditable(True)
        self._style_combo.addItems(STYLE_PRESETS)
        self._style_combo.setCurrentText("game sprite")
        f.addRow(T(QLabel(), "风格"), self._style_combo)

        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(8, 128)
        self._tile_spin.setSingleStep(8)
        self._tile_spin.setValue(32)
        f.addRow(T(QLabel(), "瓦片尺寸"), self._tile_spin)

        self._sheet_spin = QSpinBox()
        self._sheet_spin.setRange(256, 1536)
        self._sheet_spin.setSingleStep(128)
        self._sheet_spin.setValue(768)
        f.addRow(T(QLabel(), "生图边长"), self._sheet_spin)

        self._line_spin = QSpinBox()
        self._line_spin.setRange(1, 4)
        self._line_spin.setValue(1)
        f.addRow(T(QLabel(), "边界线宽"), self._line_spin)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem(T(None, "47-tile 瓦片集"), "47")
        self._mode_combo.addItem(T(None, "双网格地图"), "dual")
        f.addRow(T(QLabel(), "瓦片集模式"), self._mode_combo)

        size_row = QHBoxLayout()
        self._map_w_spin = QSpinBox()
        self._map_w_spin.setRange(4, 64)
        self._map_w_spin.setValue(14)
        size_row.addWidget(T(QLabel(), "地图宽"), 1)
        size_row.addWidget(self._map_w_spin, 1)
        self._map_h_spin = QSpinBox()
        self._map_h_spin.setRange(4, 64)
        self._map_h_spin.setValue(10)
        size_row.addWidget(T(QLabel(), "高"), 1)
        size_row.addWidget(self._map_h_spin, 1)
        f.addRow(T(QLabel(), "演示地图"), size_row)

        fl.addWidget(input_box)

        actions = QHBoxLayout()
        self._gen_btn = T(QPushButton(), "生成瓦片集")
        self._gen_btn.setObjectName("PrimaryButton")
        self._gen_btn.clicked.connect(self._on_generate)
        actions.addWidget(self._gen_btn, 1)
        fl.addLayout(actions)

        actions2 = QHBoxLayout()
        self._edit_btn = T(QPushButton(), "编辑瓦片")
        self._edit_btn.clicked.connect(self._on_edit_tiles)
        self._edit_btn.setEnabled(False)
        self._map_btn = T(QPushButton(), "地图预览")
        self._map_btn.clicked.connect(self._on_map_preview)
        self._map_btn.setEnabled(False)
        actions2.addWidget(self._edit_btn)
        actions2.addWidget(self._map_btn)
        fl.addLayout(actions2)

        self._status = QLabel(tr("就绪"))
        self._status.setWordWrap(True)
        fl.addWidget(self._status)
        fl.addStretch(1)
        scroll.setWidget(host)
        lp.addWidget(scroll)
        top.addWidget(left)

        # ---------- 右：预览 ----------
        right = QWidget()
        rp = QVBoxLayout(right)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(T(QLabel(), "瓦片集预览"))
        self._preview_caption = QLabel("")
        head.addWidget(self._preview_caption)
        head.addStretch(1)
        rp.addLayout(head)
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._preview_label.setMinimumSize(320, 320)
        self._preview_label.setStyleSheet("background: rgba(0,0,0,0.12); border-radius: 8px;")
        rp.addWidget(self._preview_label, 1)
        top.addWidget(right, 1)

        root.addLayout(top, 1)

    # ------------------------------------------------------------------ #
    def _collect_params(self) -> TilemapParams:
        desc = self._desc_edit.toPlainText().strip()
        if not desc:
            raise WorkflowError(tr("请先填写纹理描述"), step="瓦片提示词")
        return TilemapParams(
            description=desc,
            style=self._style_combo.currentText().strip() or "game sprite",
            tile_size=self._tile_spin.value(),
            sheet_size=self._sheet_spin.value(),
            atlas_mode=self._mode_combo.currentData(),
            line_width=self._line_spin.value(),
            map_width=self._map_w_spin.value(),
            map_height=self._map_h_spin.value(),
            output_dir=Path(DEFAULT_OUTPUT_DIR) / "tilemap",
        )

    # ------------------------------------------------------------------ #
    def _on_generate(self) -> None:
        try:
            params = self._collect_params()
        except WorkflowError as exc:
            self._status.setText(exc.message)
            return
        if self._worker and self._worker.isRunning():
            return
        self._params = params
        self._gen_btn.setEnabled(False)
        self._edit_btn.setEnabled(False)
        self._map_btn.setEnabled(False)
        self._status.setText(tr("生成中…"))
        self._worker = TilemapWorker(self._ctx.api, params, parent=self)
        self._worker.succeeded.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result) -> None:
        self._result = result
        self._session = result.session
        self._gen_btn.setEnabled(True)
        self._edit_btn.setEnabled(True)
        self._map_btn.setEnabled(True)
        self._show_atlas()
        self._status.setText(tr("瓦片集已生成: {0}").format(result.atlas_path))

    def _on_failed(self, message: str) -> None:
        self._gen_btn.setEnabled(True)
        self._status.setText(tr("生成失败: {0}").format(message))
        QMessageBox.warning(self, tr("瓦片集生成失败"), message)

    def _show_atlas(self) -> None:
        if self._session and self._session.atlas_sheet is not None:
            sheet = self._session.atlas_sheet
            zoom = max(1, min(4, 1024 // max(sheet.size)))
            pix = pil_to_qpixmap(sheet).scaled(
                sheet.width * zoom,
                sheet.height * zoom,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self._preview_label.setPixmap(pix)
            mode = tr("47-tile（8×6）") if self._session.params.atlas_mode == "47" else tr("双网格（16 块）")
            self._preview_caption.setText(f"{mode}  ·  {sheet.size[0]}×{sheet.size[1]}")

    # ------------------------------------------------------------------ #
    def _rerun_local_steps(self) -> None:
        """编辑瓦片后重跑 无缝化/瓦片集/导出（纯本地，无 API）。"""
        if self._session is None:
            return
        try:
            for name in ("seamless", "atlas", "export"):
                self._local_wf.step(name, self._session.params, self._session)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tr("重新生成失败"), str(exc))
            return
        self._result = self._session.result
        self._show_atlas()
        self._status.setText(tr("瓦片已更新并重新生成瓦片集"))

    def _on_edit_tiles(self) -> None:
        if self._session is None or self._session.base is None:
            QMessageBox.information(self, tr("编辑瓦片"), tr("请先生成瓦片集"))
            return
        dialog = TileEditorDialog(self._session.base, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            edited = dialog.result()
            self._session.base = base_set_with_edits(self._session.base, edited)
            self._rerun_local_steps()

    def _on_map_preview(self) -> None:
        if self._session is None or self._session.map_model is None:
            QMessageBox.information(self, tr("地图预览"), tr("请先生成瓦片集"))
            return
        session = self._session
        model = TileMapModel.from_dict(session.map_model.to_dict())
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("地图预览（左键铺设 / 右键擦除 / 滚轮缩放）"))
        dialog.resize(880, 640)
        layout = QVBoxLayout(dialog)
        view = TilemapView(
            model,
            session.processed.center,
            line_color=session.processed.line_color,
            line_width=session.processed.line_width,
            atlas_mode=session.params.atlas_mode,
        )
        layout.addWidget(view, 1)
        close_btn = T(QPushButton(), "应用到会话")
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_btn)
        layout.addLayout(row)
        close_btn.clicked.connect(dialog.accept)
        dialog.exec()
        # 应用回会话并重新导出演示图
        session.map_model = model
        try:
            self._local_wf.step("export", session.params, session)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tr("导出失败"), str(exc))
            return
        self._status.setText(tr("地图已更新并重新导出预览"))
