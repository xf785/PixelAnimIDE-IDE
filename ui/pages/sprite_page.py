"""精灵图模式页：仅用文生图生成网格精灵图（不涉及视频抽帧）。

链路：文生对象底图 → 以底图为参考图生成 i×j 网格精灵图 → 算法裁切帧序列
      → 一键抠除纯色背景 → 导出 GIF / PNG 序列 / 抠图后精灵图。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.settings import DEFAULT_FPS, DEFAULT_MAX_COLORS, DEFAULT_OUTPUT_DIR, PIXEL_SIZES
from core.processing.prompt_utils import preset_names
from core.workflow import SpriteParams, SpriteResult
from ui.app_context import AppContext
from ui.i18n import T, tr
from ui.widgets.image_viewer import ImageViewer
from ui.workers import SpriteWorker

logger = logging.getLogger("PixelAnimIDE.ui.sprite_page")

_LOG_COLORS = {"info": "#adb2b8", "warn": "#f59e0b", "error": "#f25a5a"}
FORM_WIDTH = 380


class SpritePage(QWidget):
    sync_to_ide = Signal(object)  # 把精灵图结果同步到 IDE 模式编辑

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._worker: SpriteWorker | None = None
        self._result: SpriteResult | None = None
        self._build_ui()
        self._restore_settings()

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
        self._form_scroll = scroll
        host = QWidget()
        fl = QVBoxLayout(host)
        fl.setContentsMargins(0, 0, 6, 0)
        fl.setSpacing(10)

        input_box = T(QGroupBox(), "输入参数")
        f = QFormLayout(input_box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(10)

        self._desc_edit = QTextEdit()
        T(self._desc_edit, "例如：一只拿着剑的橙色小猫，Q 版，侧身站立", attr="placeholder")
        self._desc_edit.setMaximumHeight(80)
        f.addRow(T(QLabel(), "文本描述"), self._desc_edit)

        self._action_combo = QComboBox()
        self._action_combo.setEditable(True)
        self._action_combo.setPlaceholderText(T(None, "选择或输入动作（每帧的动作循环）…"))
        self._action_combo.addItem("")
        for name in preset_names():
            self._action_combo.addItem(tr(name), userData=name)
        f.addRow(T(QLabel(), "动作(可选)"), self._action_combo)

        self._frames_spin = QSpinBox()
        self._frames_spin.setRange(1, 256)
        self._frames_spin.setValue(16)
        f.addRow(T(QLabel(), "帧数"), self._frames_spin)

        grid_row = QHBoxLayout()
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(1, 16)
        self._rows_spin.setValue(4)
        grid_row.addWidget(QLabel("i:"))
        grid_row.addWidget(self._rows_spin, 1)
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(1, 16)
        self._cols_spin.setValue(4)
        grid_row.addWidget(QLabel("j:"))
        grid_row.addWidget(self._cols_spin, 1)
        f.addRow(T(QLabel(), "网格 i×j"), grid_row)
        grid_hint = T(QLabel(), "帧数 ≤ 行×列；多余格不裁切。如 4×4 网格、16 帧")
        grid_hint.setObjectName("HintLabel")
        grid_hint.setWordWrap(True)
        f.addRow("", grid_hint)

        self._size_combo = QComboBox()
        for size in PIXEL_SIZES:
            self._size_combo.addItem(str(size))
        self._size_combo.setCurrentText("64")
        f.addRow(T(QLabel(), "单格尺寸"), self._size_combo)

        self._colors_spin = QSpinBox()
        self._colors_spin.setRange(2, 64)
        self._colors_spin.setValue(DEFAULT_MAX_COLORS)
        f.addRow(T(QLabel(), "最大颜色数"), self._colors_spin)

        opt_box = T(QGroupBox(), "处理选项")
        of = QVBoxLayout(opt_box)
        of.setContentsMargins(12, 18, 12, 12)
        of.setSpacing(8)
        self._bg_chk = T(QCheckBox(), "一键抠图（扣除纯色背景）")
        self._bg_chk.setChecked(True)
        of.addWidget(self._bg_chk)
        self._whiten_chk = T(QCheckBox(), "背景强制纯色（主体浅色→黑底，否则白底）")
        self._whiten_chk.setChecked(True)
        of.addWidget(self._whiten_chk)
        self._loop_chk = T(QCheckBox(), "首尾帧一致（循环无缝）")
        self._loop_chk.setChecked(True)
        T(self._loop_chk, "末帧强制等于首帧；角色形象逐格一致、仅动作平滑变化", attr="tooltip")
        of.addWidget(self._loop_chk)
        fl.addWidget(input_box)
        fl.addWidget(opt_box)

        out_box = T(QGroupBox(), "输出")
        ef = QFormLayout(out_box)
        ef.setContentsMargins(12, 18, 12, 12)
        out_row = QHBoxLayout()
        self._output_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        out_row.addWidget(self._output_edit, 1)
        btn_browse = T(QPushButton(), "浏览…")
        btn_browse.clicked.connect(self._on_browse_output)
        out_row.addWidget(btn_browse)
        ef.addRow(T(QLabel(), "输出目录"), out_row)
        self._btn_open_out = T(QPushButton(), "打开输出目录")
        self._btn_open_out.clicked.connect(self._on_open_output)
        ef.addRow("", self._btn_open_out)
        fl.addWidget(out_box)
        fl.addStretch(1)
        scroll.setWidget(host)
        lp.addWidget(scroll, 1)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self._btn_start = T(QPushButton(), "生成精灵图")
        self._btn_start.setObjectName("PrimaryButton")
        self._btn_start.setMinimumHeight(36)
        self._btn_start.clicked.connect(self._on_start)
        ctrl.addWidget(self._btn_start)
        self._btn_cancel = T(QPushButton(), "取消")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        ctrl.addWidget(self._btn_cancel)
        self._btn_sync = T(QPushButton(), "同步到 IDE")
        self._btn_sync.setEnabled(False)
        T(self._btn_sync, "把精灵图帧序列同步到 IDE 模式继续编辑", attr="tooltip")
        self._btn_sync.clicked.connect(self._on_sync)
        ctrl.addWidget(self._btn_sync)
        lp.addLayout(ctrl)
        top.addWidget(left)

        # ---------- 右：预览 ----------
        self._tabs = QTabWidget()
        self._base_viewer = ImageViewer()
        self._tabs.addTab(self._base_viewer, T(None, "对象底图"))
        self._sheet_viewer = ImageViewer()
        self._tabs.addTab(self._sheet_viewer, T(None, "精灵图"))
        self._frames_viewer = ImageViewer()
        self._tabs.addTab(self._frames_viewer, T(None, "帧序列"))
        top.addWidget(self._tabs, 1)

        root.addLayout(top, 1)

        # ---------- 底部：进度 + 日志 ----------
        bottom = QHBoxLayout()
        self._step_label = T(QLabel(), "就绪")
        self._step_label.setObjectName("StepLabel")
        bottom.addWidget(self._step_label)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        bottom.addWidget(self._progress, 1)
        root.addLayout(bottom)

        self._log_view = QPlainTextEdit()
        self._log_view.setObjectName("LogView")
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(140)
        root.addWidget(self._log_view)

    # ------------------------------------------------------------------ #
    def _restore_settings(self) -> None:
        out = self._ctx.ui_settings.get("output_dir")
        if out:
            self._output_edit.setText(str(out))
        self._log(tr("精灵图模式：仅用文生图生成网格精灵图（帧数 / i×j 网格 / 一键抠图）"), "info")

    # ------------------------------------------------------------------ #
    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self._output_edit.text())
        if path:
            self._output_edit.setText(path)

    def _on_open_output(self) -> None:
        if self._result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result.output_dir)))

    def _on_start(self) -> None:
        desc = self._desc_edit.toPlainText().strip()
        if not desc:
            QMessageBox.warning(self, "提示", "请先输入文本描述")
            return
        rows, cols = self._rows_spin.value(), self._cols_spin.value()
        frames = self._frames_spin.value()
        if frames > rows * cols:
            QMessageBox.warning(self, "提示", f"帧数（{frames}）不能大于网格总数（{rows}×{cols}={rows * cols}）")
            return

        params = SpriteParams(
            description=desc,
            action=self._action_combo.currentData() or self._action_combo.currentText().strip(),
            frame_count=frames,
            grid_rows=rows,
            grid_cols=cols,
            cell_size=int(self._size_combo.currentText()),
            max_colors=self._colors_spin.value(),
            force_pure_bg=self._whiten_chk.isChecked(),
            remove_bg=self._bg_chk.isChecked(),
            loop_close=self._loop_chk.isChecked(),
            output_dir=self._run_dir(),
        )
        self._ctx.ui_settings.set("output_dir", self._output_edit.text().strip())

        self._worker = SpriteWorker(self._ctx.api, params)
        self._worker.log.connect(self._on_log)
        self._worker.base_ready.connect(self._on_base_ready)
        self._worker.sheet_ready.connect(self._on_sheet_ready)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failed)

        self._set_running(True)
        self._progress.setValue(0)
        self._log_view.clear()
        self._base_viewer.clear()
        self._sheet_viewer.clear()
        self._frames_viewer.clear()
        self._result = None
        self._btn_sync.setEnabled(False)
        self._log(f"开始生成精灵图：{rows}×{cols} 网格 / {frames} 帧")
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker:
            self._log("正在取消…")
            self._worker.cancel()

    def _on_log(self, level: str, message: str) -> None:
        color = _LOG_COLORS.get(level, "#8b949e")
        safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._log_view.appendHtml(f'<span style="color:{color};">[{level.upper()}] {safe}</span>')

    def _on_base_ready(self, path: str) -> None:
        self._base_viewer.show_path(path)
        self._tabs.setCurrentIndex(0)
        self._log(f"对象底图已生成：{path}", "info")

    def _on_sheet_ready(self, path: str) -> None:
        self._sheet_viewer.show_path(path)
        self._tabs.setCurrentIndex(1)
        self._log(f"精灵图已生成：{path}", "info")

    def _on_success(self, result: SpriteResult) -> None:
        self._result = result
        self._set_running(False)
        self._progress.setValue(100)
        self._step_label.setText("完成")
        self._log(
            f"精灵图完成：{result.frame_count} 帧 @ {result.width}x{result.height}",
            "info",
        )
        if result.gif_path:
            self._frames_viewer.show_gif(result.gif_path)
            self._log(f"GIF: {result.gif_path}")
        self._btn_open_out.setEnabled(True)
        self._btn_sync.setEnabled(True)
        QMessageBox.information(
            self, "完成", f"精灵图已生成！\n{result.frame_count} 帧\nGIF: {result.gif_path}\n抠图精灵图: {result.sheet_path}"
        )

    def _on_sync(self) -> None:
        """把精灵图帧序列同步到 IDE 模式编辑。"""
        if self._result:
            self.sync_to_ide.emit(self._result)

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._step_label.setText("失败")
        self._log(message, "error")
        QMessageBox.critical(self, "生成失败", message)

    # ------------------------------------------------------------------ #
    def _set_running(self, running: bool) -> None:
        self._btn_start.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._desc_edit.setEnabled(not running)

    def _run_dir(self) -> Path:
        base = Path(self._output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return base / f"sprite_{ts}"

    def _log(self, message: str, level: str = "info") -> None:
        color = _LOG_COLORS.get(level, "#8b949e")
        safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._log_view.appendHtml(f'<span style="color:{color};">[{level.upper()}] {safe}</span>')

    def apply_ui_scale(self, scale: float) -> None:
        """按界面比例调整左侧表单宽度。"""
        if hasattr(self, "_form_scroll"):
            self._form_scroll.setFixedWidth(max(240, int(FORM_WIDTH * scale)))
