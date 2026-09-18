"""精灵图模式页：仅用文生图生成网格精灵图（不涉及视频抽帧）。

链路：文生对象底图 → 以底图为参考图生成 i×j 网格精灵图 → 算法裁切帧序列
      → 完美像素双分辨率 → 一键抠除纯色背景 → 导出 GIF / PNG 序列 / 拼接网格图。

布局参考 Krita：
- **左停靠栏**：参数拆成 3 个可折叠 docker（输入参数 / 处理选项 / 输出），
  整栏可收起成竖排标签，宽度由外层 splitter 拖动调整；
- **右侧工作区**：预览页签（对象底图 / 精灵图 / 帧序列），随窗口弹性伸缩；
- **底部**：运行控制 + 进度 + 日志。

执行方式：
- 自动：无干涉跑完全流程；
- 手动：逐步执行，每步完成后可「重跑本步」或「继续下一步」。
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
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
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.settings import DEFAULT_MAX_COLORS, DEFAULT_OUTPUT_DIR, PIXEL_SIZES
from core.workflow import STEP_ORDER, SpriteParams, SpriteResult, SpriteWorkflow, WorkflowError
from ui.app_context import AppContext
from ui.i18n import T, tr
from ui.layout import scaled
from ui.widgets.action_combo import populate_action_combo
from ui.widgets.dock import SideDock
from ui.widgets.image_viewer import ImageViewer
from ui.workers import SPRITE_API_KINDS, IdeStepWorker, SpriteWorker, create_api_clients

logger = logging.getLogger("PixelFoundry.ui.sprite_page")

_LOG_COLORS = {"info": "#adb2b8", "warn": "#f59e0b", "error": "#f25a5a"}
FORM_WIDTH = 380

# 步骤名（zh 为 i18n ID）
SPRITE_STEP_LABELS = {
    "prompts": "生成提示词",
    "base": "生成对象底图",
    "sheet": "生成网格精灵图",
    "crop": "裁切帧序列",
    "pixelize": "完美像素化",
    "key": "抠图",
    "export": "导出",
}


class SpritePage(QWidget):
    sync_to_ide = Signal(object)  # 把精灵图结果同步到 IDE 模式编辑
    base_ready = Signal(str)      # 对象底图路径（手动模式跨线程预览）
    sheet_ready = Signal(str)     # 精灵图路径（手动模式跨线程预览）
    manual_log = Signal(str, str)  # 手动模式工作线程日志（level, message）
    running_changed = Signal(bool)  # 生成中状态变化（主窗口据此禁用/启用执行方式开关）

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._worker: SpriteWorker | None = None
        self._result: SpriteResult | None = None
        self._manual_mode = False  # 由主窗口侧栏开关驱动：False=自动，True=手动
        # 手动模式状态
        self._manual_active = False
        self._manual_step_idx = 0
        self._manual_workflow: SpriteWorkflow | None = None
        self._manual_session = None
        self._manual_params: SpriteParams | None = None
        self._manual_cancel = threading.Event()
        self._manual_clients: list = []
        self._step_worker: IdeStepWorker | None = None
        self._last_param_error: str = ""
        self._build_ui()
        self._restore_layout()
        self._restore_settings()
        self.base_ready.connect(self._on_base_ready)
        self.sheet_ready.connect(self._on_sheet_ready)
        self.manual_log.connect(self._on_log)

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        # ---------- 工作区：左停靠栏（参数）| 右侧预览页签（宽度可拖动） ----------
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("Workspace")
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(scaled(5))

        self._left_dock = SideDock(tr("参数"), side="left", default_width=FORM_WIDTH)
        self._splitter.addWidget(self._left_dock)

        # 「输入参数」字段最多：沿用原来的滚动区，面板被拖矮时仍可滚动查看全部字段
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setMinimumWidth(scaled(200))
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
        populate_action_combo(self._action_combo)
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

        # ---------- 表单分组 -> 左停靠栏的 docker ----------
        fl.addWidget(input_box)
        scroll.setWidget(host)
        input_docker = self._left_dock.add_docker("输入参数", scroll, icon_kind="pencil", stretch=1)
        input_docker.set_icon("pencil")  # 图标要显式渲染一次才可见（与像素页一致）
        # 收起按钮放在第一个面板标题里（Krita 的 docker 右上角）
        input_docker.add_header_widget(self._left_dock.add_toggle_button("chevron_left", "收起参数栏"))

        opt_docker = self._left_dock.add_docker("处理选项", opt_box, icon_kind="layers")
        opt_docker.set_icon("layers")
        out_docker = self._left_dock.add_docker("输出", out_box, icon_kind="export_image")
        out_docker.set_icon("export_image")

        # 执行方式由主窗口侧栏开关控制（向左=自动，向右=手动），此处不再放置按钮
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
        self._btn_rerun = T(QPushButton(), "重跑本步")
        self._btn_rerun.setVisible(False)
        self._btn_rerun.clicked.connect(self._on_rerun)
        ctrl.addWidget(self._btn_rerun)
        self._btn_next = T(QPushButton(), "继续下一步")
        self._btn_next.setVisible(False)
        self._btn_next.clicked.connect(self._on_next)
        ctrl.addWidget(self._btn_next)
        self._btn_sync = T(QPushButton(), "同步到 IDE")
        self._btn_sync.setEnabled(False)
        T(self._btn_sync, "把精灵图帧序列同步到 IDE 模式继续编辑", attr="tooltip")
        self._btn_sync.clicked.connect(self._on_sync)
        ctrl.addWidget(self._btn_sync)

        # ---------- 右：预览（对象底图 / 精灵图 / 帧序列） ----------
        self._tabs = QTabWidget()
        self._base_viewer = ImageViewer()
        self._tabs.addTab(self._base_viewer, T(None, "对象底图"))
        self._sheet_viewer = ImageViewer()
        self._tabs.addTab(self._sheet_viewer, T(None, "精灵图"))
        self._frames_viewer = ImageViewer()
        self._tabs.addTab(self._frames_viewer, T(None, "帧序列"))
        self._splitter.addWidget(self._tabs)

        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._left_dock.bind_splitter(self._splitter, 0, default_width=FORM_WIDTH)
        self._splitter.setSizes([scaled(FORM_WIDTH), scaled(760)])
        root.addWidget(self._splitter, 1)

        # 运行控制（固定在底部，不随预览页签切换）
        root.addLayout(ctrl)

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
    # 主窗口工具条协议（Krita 风格：工具条内容随工作区变化）
    # ------------------------------------------------------------------ #
    def toolbar_actions(self) -> list:
        """返回 [(图标, 文本, 提示, 回调, 是否主按钮), …] 供主窗口工具条渲染。

        手动模式停在本步时，主按钮跟着变成「继续下一步」（走的是同一个按钮，
        因此仍受按钮可用状态约束，不会打乱逐步执行流程）。
        """
        if self._manual_mode and self._manual_active and self._btn_next.isEnabled():
            primary = ("chevron_right", self._btn_next.text(), "继续执行下一步骤", self._btn_next.click, True)
        else:
            primary = ("grid", self._btn_start.text(), "按当前参数生成网格精灵图", self._btn_start.click, True)
        return [
            primary,
            ("copy", "同步到 IDE", "把精灵图帧序列同步到 IDE 模式继续编辑", self._on_sync, False),
            ("export_image", "打开输出目录", "打开本次生成结果的输出目录", self._on_open_output, False),
        ]

    def workspace_status(self) -> str:
        """工具条右侧的状态标识：帧数 + 网格 i×j。"""
        return tr("{0}帧 · {1}×{2}").format(
            self._frames_spin.value(), self._rows_spin.value(), self._cols_spin.value()
        )

    def _restore_settings(self) -> None:
        out = self._ctx.ui_settings.get("output_dir")
        if out:
            self._output_edit.setText(str(out))
        self._log(tr("精灵图模式：仅用文生图生成网格精灵图（帧数 / i×j 网格 / 一键抠图）"), "info")

    # ------------------------------------------------------------------ #
    # 布局持久化（停靠栏宽度 + 收起状态）
    # ------------------------------------------------------------------ #
    def _restore_layout(self) -> None:
        """恢复上次的停靠栏宽度与收起状态。"""
        try:
            sizes = self._ctx.ui_settings.get("sprite_dock_sizes") or []
            if (isinstance(sizes, (list, tuple)) and len(sizes) == 2
                    and int(sizes[1]) >= scaled(320)):
                # 预览区过窄说明上次保存的是病态布局 -> 退回默认宽度
                self._splitter.setSizes([int(v) for v in sizes])
            if self._ctx.ui_settings.get("sprite_left_collapsed"):
                self._left_dock.set_collapsed(True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("精灵图页布局恢复失败: %s", exc)

    def _remember_layout(self) -> None:
        try:
            s = self._ctx.ui_settings
            s.set("sprite_dock_sizes", list(self._splitter.sizes()))
            s.set("sprite_left_collapsed", self._left_dock.is_collapsed())
        except Exception as exc:  # noqa: BLE001
            logger.warning("精灵图页布局保存失败: %s", exc)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._remember_layout()
        super().hideEvent(event)

    # ------------------------------------------------------------------ #
    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr("选择输出目录"), self._output_edit.text())
        if path:
            self._output_edit.setText(path)

    def _on_open_output(self) -> None:
        if self._result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result.output_dir)))

    def set_manual_mode(self, manual: bool) -> None:
        """设置执行方式（由主窗口侧栏开关驱动）：False=自动，True=手动。"""
        self._manual_mode = bool(manual)
        self._log(
            tr("精灵图执行方式：{0}").format(tr("手动（逐步执行）") if manual else tr("自动（无干涉跑完全流程）")),
            "info",
        )

    def _collect_params(self) -> SpriteParams | None:
        """从表单收集参数；校验失败返回 None（原因记录在 self._last_param_error）。"""
        desc = self._desc_edit.toPlainText().strip()
        if not desc:
            self._last_param_error = tr("请先输入文本描述")
            return None
        rows, cols = self._rows_spin.value(), self._cols_spin.value()
        frames = self._frames_spin.value()
        if frames > rows * cols:
            self._last_param_error = tr("帧数（{0}）不能大于网格总数（{1}×{2}={3}）").format(
                frames, rows, cols, rows * cols
            )
            return None
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
        return params

    def _on_start(self) -> None:
        params = self._collect_params()
        if params is None:
            QMessageBox.warning(self, tr("提示"), self._last_param_error)
            return
        self._ctx.ui_settings.set("output_dir", self._output_edit.text().strip())

        self._log_view.clear()
        self._base_viewer.clear()
        self._sheet_viewer.clear()
        self._frames_viewer.clear()
        self._result = None
        self._btn_sync.setEnabled(False)
        self._progress.setValue(0)
        self._log(
            tr("开始生成精灵图：{0}×{1} 网格 / {2} 帧").format(
                params.grid_rows, params.grid_cols, params.frame_count
            )
        )

        if self._manual_mode:
            self._manual_start(params)
        else:
            self._auto_start(params)

    def _auto_start(self, params: SpriteParams) -> None:
        """自动模式：无干涉跑完全流程。"""
        self._worker = SpriteWorker(self._ctx.api, params)
        self._worker.log.connect(self._on_log)
        self._worker.base_ready.connect(self._on_base_ready)
        self._worker.sheet_ready.connect(self._on_sheet_ready)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failed)
        self._set_running(True)
        self._worker.start()

    # ------------------------------------------------------------------ #
    # 手动模式：逐步执行
    # ------------------------------------------------------------------ #
    def _manual_start(self, params: SpriteParams) -> None:
        """手动模式：创建会话与客户端，从第 1 步开始执行。"""
        try:
            clients, opened = create_api_clients(self._ctx.api, SPRITE_API_KINDS)
        except WorkflowError as exc:
            QMessageBox.warning(self, tr("提示"), str(exc))
            return
        self._manual_clients = opened
        self._manual_cancel = threading.Event()
        self._manual_workflow = SpriteWorkflow(
            clients["llm"],
            clients["image"],
            log=self.manual_log.emit,  # 信号转发：工作线程不能直接操作 Qt 控件
            cancel=self._manual_cancel,
            on_base=lambda p: self.base_ready.emit(str(p)),
            on_sheet=lambda p: self.sheet_ready.emit(str(p)),
        )
        self._manual_session = self._manual_workflow.new_session(params)
        self._manual_step_idx = 0
        self._manual_active = True
        self._set_running(True)
        self._run_manual_step()

    def _run_manual_step(self) -> None:
        """在线程中执行当前步骤；完成后由按钮决定重跑或继续。

        参数在 GUI 线程收集后缓存（工作线程禁止访问 Qt 控件）。
        """
        params = self._collect_params()
        if params is None:
            QMessageBox.warning(self, tr("提示"), self._last_param_error)
            self._btn_rerun.setEnabled(True)
            return
        self._manual_params = params
        name = STEP_ORDER[self._manual_step_idx]
        label = tr(SPRITE_STEP_LABELS[name])
        self._step_label.setText(tr("步骤 {0}/{1}：{2}…").format(self._manual_step_idx + 1, len(STEP_ORDER), label))
        self._btn_rerun.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._btn_rerun.setVisible(True)
        self._btn_next.setVisible(True)
        worker = IdeStepWorker(lambda log_cb: self._run_manual_step_in_thread(name))
        worker.log.connect(self._on_log)
        worker.succeeded.connect(self._on_manual_step_done)
        worker.failed.connect(self._on_manual_step_failed)
        self._step_worker = worker
        worker.start()

    def _run_manual_step_in_thread(self, name: str) -> str:
        """线程内执行单个步骤（使用 GUI 线程缓存的参数）。"""
        params = self._manual_params
        self._manual_session.params = params
        self._manual_workflow.step(name, params, self._manual_session)
        return name

    def _on_manual_step_done(self, name: str) -> None:
        self._step_worker = None
        if name == "export":
            # 最后一步完成：整个流程结束
            self._manual_active = False
            self._manual_finish()
            self._on_success(self._manual_session.result)
            return
        self._manual_step_idx = STEP_ORDER.index(name)
        self._progress.setValue(int((self._manual_step_idx + 1) / len(STEP_ORDER) * 100))
        label = tr(SPRITE_STEP_LABELS[name])
        self._step_label.setText(
            tr("步骤 {0}/{1}：{2} 完成 — 可重跑本步或继续").format(
                self._manual_step_idx + 1, len(STEP_ORDER), label
            )
        )
        self._btn_rerun.setEnabled(True)
        self._btn_next.setEnabled(True)
        self._log(
            tr("步骤 {0}/{1}：{2} 完成 — 可重跑本步或继续").format(
                self._manual_step_idx + 1, len(STEP_ORDER), label
            ),
            "info",
        )

    def _on_manual_step_failed(self, message: str) -> None:
        self._step_worker = None
        if self._manual_cancel.is_set():
            # 用户取消：终止手动流程
            self._manual_active = False
            self._manual_finish()
            self._step_label.setText(tr("已取消"))
            self._log(message, "warn")
            self._set_running(False)
            return
        self._btn_rerun.setEnabled(True)
        self._btn_next.setEnabled(False)
        self._step_label.setText(tr("步骤失败"))
        self._log(message, "error")

    def _manual_finish(self) -> None:
        """关闭手动模式占用的 API 客户端。"""
        for c in self._manual_clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        self._manual_clients = []
        self._manual_workflow = None

    def _on_rerun(self) -> None:
        if not self._manual_active:
            return
        self._run_manual_step()

    def _on_next(self) -> None:
        if not self._manual_active:
            return
        self._manual_step_idx += 1
        if self._manual_step_idx >= len(STEP_ORDER):
            return
        self._run_manual_step()

    def _on_cancel(self) -> None:
        if self._manual_active:
            self._log(tr("正在取消…"))
            self._manual_cancel.set()
            self._step_label.setText(tr("正在取消…"))
            return
        if self._worker:
            self._log(tr("正在取消…"))
            self._worker.cancel()

    def _on_log(self, level: str, message: str) -> None:
        color = _LOG_COLORS.get(level, "#8b949e")
        safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._log_view.appendHtml(f'<span style="color:{color};">[{level.upper()}] {safe}</span>')

    def _on_base_ready(self, path: str) -> None:
        self._base_viewer.show_path(path)
        self._tabs.setCurrentIndex(0)
        self._log(tr("对象底图已生成：{0}").format(path), "info")

    def _on_sheet_ready(self, path: str) -> None:
        self._sheet_viewer.show_path(path)
        self._tabs.setCurrentIndex(1)
        self._log(tr("精灵图已生成：{0}").format(path), "info")

    def _on_success(self, result: SpriteResult) -> None:
        self._result = result
        self._set_running(False)
        self._progress.setValue(100)
        self._step_label.setText(tr("完成"))
        self._log(
            tr("精灵图完成：{0} 帧 @ {1}x{2}").format(result.frame_count, result.width, result.height),
            "info",
        )
        if result.gif_path:
            self._frames_viewer.show_gif(result.gif_path)
            self._log(f"GIF: {result.gif_path}")
        self._btn_open_out.setEnabled(True)
        self._btn_sync.setEnabled(True)
        QMessageBox.information(
            self, tr("完成"), tr("精灵图已生成！\n{0} 帧\nGIF: {1}\n抠图精灵图: {2}").format(result.frame_count, result.gif_path, result.sheet_path)
        )

    def _on_sync(self) -> None:
        """把精灵图帧序列同步到 IDE 模式编辑。"""
        if self._result:
            self.sync_to_ide.emit(self._result)

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._step_label.setText(tr("失败"))
        self._log(message, "error")
        QMessageBox.critical(self, tr("生成失败"), message)

    # ------------------------------------------------------------------ #
    def _set_running(self, running: bool) -> None:
        self._btn_start.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._desc_edit.setEnabled(not running)
        self.running_changed.emit(running)
        if not running:
            self._btn_rerun.setVisible(False)
            self._btn_next.setVisible(False)

    def _run_dir(self) -> Path:
        base = Path(self._output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return base / f"sprite_{ts}"

    def _log(self, message: str, level: str = "info") -> None:
        color = _LOG_COLORS.get(level, "#8b949e")
        safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._log_view.appendHtml(f'<span style="color:{color};">[{level.upper()}] {safe}</span>')

    # ------------------------------------------------------------------ #
    # 键盘快捷键（设置 → 快捷键 → 精灵图可自定义；固定使用 sprite 键位）
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event) -> None:  # noqa: N802
        from ui import shortcuts as sc

        if sc.match(event, sc.get("preview_play", "sprite")):
            self._frames_viewer.toggle_play()
            event.accept()
            return
        super().keyPressEvent(event)

    def apply_ui_scale(self, scale: float) -> None:
        """按界面比例同步缩放停靠栏、分隔条与表单最小宽度（宽度交给 splitter）。"""
        if hasattr(self, "_form_scroll"):
            self._form_scroll.setMinimumWidth(scaled(200))
        self._left_dock.apply_ui_scale()
        self._splitter.setHandleWidth(scaled(5))

    def retranslate_ui(self) -> None:
        """语言切换：刷新动作预设分组下拉。"""
        populate_action_combo(self._action_combo)
