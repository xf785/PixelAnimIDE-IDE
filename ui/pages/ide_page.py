"""IDE 模式页：分步式专业工作区。

布局：
- 左侧：步骤引导栏（6 步）+ 项目（新建/打开/保存）。
- 中间：主工作区（预览 / 像素编辑 / 提示词 三个 Tab）。
- 右侧：参数面板（描述、动作、分辨率、颜色、图转视频参数、处理选项、输出目录）。
- 底部：时间轴（缩略图 / 拖动排序 / 插入 / 复制 / 删除）+ 状态 + 日志。

每一步可独立执行；中间产物保存在 IdeSession 中，可随时编辑、重跑。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image
from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from config.settings import (
    ASPECT_RATIOS,
    DEFAULT_FPS,
    DEFAULT_FRAME_COUNT,
    DEFAULT_MAX_COLORS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPEED,
    PIXEL_SIZES,
)
from core.api.factory import create_api_client
from core.processing import frame_utils as fu
from core.processing.prompt_utils import preset_names
from core.workflow import (
    IDE_STEPS,
    IdeSession,
    IdeWorkflow,
    SoloResult,
    load_ide_project,
    save_ide_project,
)
from core.workflow.solo_workflow import WorkflowError
from ui.app_context import AppContext
from ui.i18n import T, tr
from ui.widgets.image_viewer import ImageViewer
from ui.widgets.pixel_editor import PixelEditorWidget
from ui.widgets.reference_box import ReferenceImageBox
from ui.widgets.timeline import TimelineWidget
from ui.workers import IdeStepWorker

logger = logging.getLogger("PixelAnimIDE.ui.ide_page")

_LOG_COLORS = {"info": "#adb2b8", "warn": "#f59e0b", "error": "#f25a5a"}

PARAM_WIDTH = 360

# 步骤 -> 执行按钮文案（zh 原文 + 运行时翻译）
STEP_ACTIONS_ZH = ["生成提示词", "生成首帧图片", "生成动画", "像素化处理", "去除背景", "导出"]
STEP_ACTIONS = [tr(s) for s in STEP_ACTIONS_ZH]


def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


class IdePage(QWidget):
    step_changed = Signal(int)  # 当前步骤切换（供主窗口侧栏高亮）

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._session = IdeSession()
        self._worker: Optional[IdeStepWorker] = None
        self._current = 0  # 当前选中的帧索引
        self._current_step = 0
        self._playing = False
        self._play_index = 0
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._dirty = False
        self._build_ui()
        self._restore_settings()
        self._refresh_all()

    # ------------------------------------------------------------------ #
    # UI 构建
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(10)

        # ---------- 中：预览 / 编辑 / 提示词 ----------
        self._tabs = QTabWidget()
        self._build_preview_tab()
        self._build_editor_tab()
        self._build_prompt_tab()
        top.addWidget(self._tabs, 1)

        # ---------- 右：参数面板 ----------
        top.addWidget(self._build_params_panel())

        root.addLayout(top, 1)

        # ---------- 底：时间轴 + 状态 + 日志 ----------
        self._timeline = TimelineWidget()
        self._timeline.frame_selected.connect(self._on_frame_selected)
        self._timeline.reordered.connect(self._on_reordered)
        self._timeline.insert_requested.connect(self._on_insert_frame)
        self._timeline.duplicate_requested.connect(self._on_duplicate_frame)
        self._timeline.delete_requested.connect(self._on_delete_frame)
        self._timeline.add_requested.connect(self._on_add_frame)
        root.addWidget(self._timeline)

        status_row = QHBoxLayout()
        self._status_label = T(QLabel(), "就绪")
        self._status_label.setObjectName("StepLabel")
        status_row.addWidget(self._status_label)
        status_row.addStretch(1)
        self._dirty_label = QLabel("")
        self._dirty_label.setObjectName("HintLabel")
        status_row.addWidget(self._dirty_label)
        root.addLayout(status_row)

        # 日志标题行 + 收起/展开按钮
        log_header = QHBoxLayout()
        log_label = QLabel(tr("日志"))
        log_label.setObjectName("HintLabel")
        log_header.addWidget(log_label)
        log_header.addStretch(1)
        self._log_toggle_btn = QToolButton()
        self._log_toggle_btn.setText("▾")
        self._log_toggle_btn.setFixedSize(20, 20)
        self._log_toggle_btn.setToolTip("收起/展开日志")
        self._log_toggle_btn.clicked.connect(self._on_toggle_log)
        log_header.addWidget(self._log_toggle_btn)
        root.addLayout(log_header)

        self._log_view = QPlainTextEdit()
        self._log_view.setObjectName("LogView")
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(110)
        self._log_collapsed = False
        root.addWidget(self._log_view)

    # ------------------------------------------------------------------ #
    def _build_preview_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self._preview = ImageViewer()
        self._preview.zoomChanged.connect(self._on_preview_zoom_changed)
        layout.addWidget(self._preview, 1)
        row = QHBoxLayout()
        self._btn_play = T(QPushButton(), "播放")
        self._btn_play.clicked.connect(self._on_toggle_play)
        row.addWidget(self._btn_play)
        # 缩放控制（预览区放大/缩小/适应）
        zoom_out = QPushButton("−")
        zoom_out.setFixedSize(26, 26)
        zoom_out.setToolTip(tr("缩小预览"))
        zoom_out.clicked.connect(lambda: self._preview.zoom_out())
        row.addWidget(zoom_out)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(40)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._zoom_label)
        zoom_in = QPushButton("＋")
        zoom_in.setFixedSize(26, 26)
        zoom_in.setToolTip(tr("放大预览"))
        zoom_in.clicked.connect(lambda: self._preview.zoom_in())
        row.addWidget(zoom_in)
        fit_btn = QPushButton(tr("适应"))
        fit_btn.setToolTip(tr("重置为适应窗口"))
        fit_btn.clicked.connect(lambda: self._preview.reset_zoom())
        row.addWidget(fit_btn)
        # 播放倍速
        self._preview_speed_combo = QComboBox()
        for label, value in [("0.5x", 0.5), ("1x（原速）", 1.0), ("1.5x", 1.5), ("2x", 2.0), ("3x", 3.0)]:
            self._preview_speed_combo.addItem(tr(label), userData=value)
        self._preview_speed_combo.setCurrentIndex(1)
        self._preview_speed_combo.currentIndexChanged.connect(self._on_preview_speed_changed)
        row.addWidget(self._preview_speed_combo)
        self._preview_hint = QLabel("")
        self._preview_hint.setObjectName("HintLabel")
        row.addWidget(self._preview_hint)
        row.addStretch(1)
        layout.addLayout(row)
        self._tabs.addTab(tab, T(None, "预览"))
        T(self._tabs, "预览", attr="tab", index=0)

    def _build_editor_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self._editor = PixelEditorWidget()
        self._editor.edited.connect(self._on_editor_edited)
        layout.addWidget(self._editor, 1)
        self._editor_hint = QLabel("在时间轴选择帧后，在此用铅笔/橡皮/取色/填充编辑像素")
        self._editor_hint.setObjectName("HintLabel")
        layout.addWidget(self._editor_hint)
        self._tabs.addTab(tab, T(None, "编辑"))
        T(self._tabs, "编辑", attr="tab", index=1)

    def _build_prompt_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(6)
        self._prompt_edits: dict = {}
        for key, label in (
            ("image_prompt", "图片提示词"),
            ("animation_prompt", "动画提示词"),
            ("negative_prompt", "负面提示词"),
        ):
            layout.addWidget(T(QLabel(), label))
            edit = QPlainTextEdit()
            edit.setObjectName("LogView")
            edit.setMaximumHeight(90)
            self._prompt_edits[key] = edit
            layout.addWidget(edit)
        btn_row = QHBoxLayout()
        apply_btn = T(QPushButton(), "应用提示词到工作区")
        apply_btn.clicked.connect(self._on_apply_prompts)
        btn_row.addWidget(apply_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        self._tabs.addTab(tab, T(None, "提示词"))
        T(self._tabs, "提示词", attr="tab", index=2)

    # ------------------------------------------------------------------ #
    # 右侧参数面板：默认收起（仅一个三角钮），点击展开提示词/文生图等参数
    # ------------------------------------------------------------------ #
    def _on_toggle_params(self) -> None:
        self._params_collapsed = not self._params_collapsed
        self._apply_params_collapsed()

    def _apply_params_collapsed(self) -> None:
        from ui.icons import editor_icon

        self._params_scroll.setVisible(not self._params_collapsed)
        kind = "chevron_left" if self._params_collapsed else "chevron_right"
        self._params_toggle_btn.setIcon(editor_icon(kind, "#9aa0a8", size=14))
        self._params_toggle_btn.setToolTip(
            "展开参数面板" if self._params_collapsed else "收起参数面板"
        )

    def _on_toggle_log(self) -> None:
        """底部日志框收起/展开。"""
        self._log_collapsed = not self._log_collapsed
        self._log_view.setVisible(not self._log_collapsed)
        self._log_toggle_btn.setText("▴" if self._log_collapsed else "▾")

    def apply_ui_scale(self, scale: float) -> None:
        """按界面比例调整右侧参数面板宽度。"""
        if hasattr(self, "_params_scroll"):
            self._params_scroll.setFixedWidth(max(240, int(PARAM_WIDTH * scale)))

    # ------------------------------------------------------------------ #
    def _build_params_panel(self) -> QWidget:
        """右侧参数面板容器：三角钮 + 可收起参数滚动区。"""
        wrap = QWidget()
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)

        self._params_toggle_btn = QToolButton()
        self._params_toggle_btn.setFixedSize(22, 44)
        self._params_toggle_btn.clicked.connect(self._on_toggle_params)
        wl.addWidget(self._params_toggle_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setFixedWidth(PARAM_WIDTH)
        self._params_scroll = scroll
        wl.addWidget(scroll)

        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(10)

        proj_box = QGroupBox(tr("项目"))
        pf = QHBoxLayout(proj_box)
        pf.setContentsMargins(12, 18, 12, 12)
        pf.setSpacing(6)
        self._btn_new = QPushButton(tr("新建"))
        self._btn_new.clicked.connect(self._on_new)
        pf.addWidget(self._btn_new)
        self._btn_open = QPushButton(tr("打开"))
        self._btn_open.clicked.connect(self._on_open_project)
        pf.addWidget(self._btn_open)
        self._btn_save = QPushButton(tr("保存"))
        self._btn_save.clicked.connect(self._on_save_project)
        pf.addWidget(self._btn_save)
        layout.addWidget(proj_box)

        img_box = QGroupBox(tr("参考图 / 首帧图"))
        ib = QVBoxLayout(img_box)
        ib.setContentsMargins(12, 18, 12, 12)
        ib.setSpacing(8)
        ref_row = QHBoxLayout()
        ref_row.setSpacing(10)
        self._ref_box = ReferenceImageBox(size=88)
        self._ref_box.changed.connect(self._on_ref_changed)
        ref_row.addWidget(self._ref_box)
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(T(QLabel(), "点击添加自备参考图"))
        hint = T(QLabel(), "同时作为首帧图：可直接走「动画生成」步骤，\n或走「图片生成」步骤做图生图。")
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        col.addWidget(hint)
        ref_row.addLayout(col, 1)
        ib.addLayout(ref_row)
        layout.addWidget(img_box)

        # 分步骤参数：随左侧步骤切换只显示本步骤相关参数
        self._step_params = QStackedWidget()
        self._step_params.addWidget(self._build_step_text_panel())    # 0 文本
        self._step_params.addWidget(self._build_step_image_panel())   # 1 图片
        self._step_params.addWidget(self._build_step_anim_panel())    # 2 动画
        self._step_params.addWidget(self._build_step_pixel_panel())   # 3 像素
        self._step_params.addWidget(self._build_step_bg_panel())      # 4 背景
        self._step_params.addWidget(self._build_step_export_panel())  # 5 导出
        layout.addWidget(self._step_params, 1)

        # 执行按钮固定在参数面板底部
        self._btn_run = T(QPushButton(), "生成提示词")
        self._btn_run.setObjectName("PrimaryButton")
        self._btn_run.setMinimumHeight(36)
        self._btn_run.clicked.connect(self._on_run_step)
        layout.addWidget(self._btn_run)

        scroll.setWidget(host)
        # 默认收起参数面板（仅三角钮），点击展开
        self._params_collapsed = True
        self._apply_params_collapsed()
        return wrap

    # ------------------------------------------------------------------ #
    # 分步骤参数面板
    # ------------------------------------------------------------------ #
    def _build_step_text_panel(self) -> QWidget:
        box = T(QGroupBox(), "步骤 1 · 文本生成")
        f = QFormLayout(box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(8)
        self._desc_edit = QTextEdit()
        T(self._desc_edit, "例如：一只拿着剑的橙色小猫，Q 版，侧身站立", attr="placeholder")
        self._desc_edit.setMaximumHeight(80)
        f.addRow(T(QLabel(), "文本描述"), self._desc_edit)
        self._action_combo = QComboBox()
        self._action_combo.setEditable(True)
        self._action_combo.setPlaceholderText(T(None, "选择或输入动作…"))
        self._action_combo.addItem("")
        for name in preset_names():
            self._action_combo.addItem(tr(name), userData=name)
        f.addRow(T(QLabel(), "动作类型(可选)"), self._action_combo)
        tip = T(QLabel(), "生成图片/动画提示词（LLM 失败自动用本地模板）")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        f.addRow("", tip)
        return box

    def _build_step_image_panel(self) -> QWidget:
        box = T(QGroupBox(), "步骤 2 · 图片生成")
        f = QFormLayout(box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(8)
        self._aspect_combo = QComboBox()
        for ratio in ASPECT_RATIOS:
            self._aspect_combo.addItem(ratio)
        self._aspect_combo.setCurrentText("1:1")
        f.addRow(T(QLabel(), "宽高比"), self._aspect_combo)
        self._size_combo = QComboBox()
        for size in PIXEL_SIZES:
            self._size_combo.addItem(str(size))
        self._size_combo.setCurrentText("128")
        f.addRow(T(QLabel(), "像素尺寸"), self._size_combo)
        tip = T(QLabel(), "添加参考图后即走图生图（i2i）；像素风自动按像素分辨率出图")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        f.addRow("", tip)
        return box

    def _build_step_anim_panel(self) -> QWidget:
        box = T(QGroupBox(), "步骤 3 · 动画生成")
        f = QFormLayout(box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(8)
        self._frames_spin = QSpinBox()
        self._frames_spin.setRange(2, 120)
        self._frames_spin.setValue(DEFAULT_FRAME_COUNT)
        f.addRow(T(QLabel(), "帧数"), self._frames_spin)
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(DEFAULT_FPS)
        f.addRow(T(QLabel(), "帧率(fps)"), self._fps_spin)
        self._speed_combo = QComboBox()
        for label, value in [("0.5x", 0.5), ("1x（原速）", 1.0), ("1.5x", 1.5), ("2x", 2.0)]:
            self._speed_combo.addItem(tr(label), userData=value)
        self._speed_combo.setCurrentIndex(1)
        f.addRow(T(QLabel(), "播放速度"), self._speed_combo)
        self._loop_chk = T(QCheckBox(), "首尾帧一致（循环闭合）")
        self._loop_chk.setChecked(True)
        f.addRow(self._loop_chk)
        tip = T(QLabel(), "参考图将作为首帧图传入视频 API；背景强制纯色等选项在「背景」步骤")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        f.addRow("", tip)
        return box

    def _build_step_pixel_panel(self) -> QWidget:
        box = T(QGroupBox(), "步骤 4 · 像素化")
        f = QFormLayout(box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(8)
        self._pixelate_chk = T(QCheckBox(), "完美像素化")
        self._pixelate_chk.setChecked(True)
        f.addRow(self._pixelate_chk)
        self._colors_spin = QSpinBox()
        self._colors_spin.setRange(2, 64)
        self._colors_spin.setValue(DEFAULT_MAX_COLORS)
        f.addRow(T(QLabel(), "最大颜色数"), self._colors_spin)
        tip = T(QLabel(), "首帧自动检测网格大小，全部帧按同一网格精确采样；非像素风自动跳过")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        f.addRow("", tip)
        return box

    def _build_step_bg_panel(self) -> QWidget:
        box = T(QGroupBox(), "步骤 5 · 背景去除")
        f = QFormLayout(box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(8)
        self._bg_chk = T(QCheckBox(), "去除背景")
        self._bg_chk.setChecked(True)
        f.addRow(self._bg_chk)
        self._whiten_chk = T(QCheckBox(), "背景强制纯色")
        self._whiten_chk.setChecked(True)
        f.addRow(self._whiten_chk)
        self._bg_tolerance_spin = QSpinBox()
        self._bg_tolerance_spin.setRange(0, 200)
        self._bg_tolerance_spin.setValue(30)
        f.addRow(T(QLabel(), "背景容差"), self._bg_tolerance_spin)
        self._bg_feather_spin = QSpinBox()
        self._bg_feather_spin.setRange(0, 30)
        self._bg_feather_spin.setValue(8)
        f.addRow(T(QLabel(), "羽化(px)"), self._bg_feather_spin)
        self._bg_erode_spin = QSpinBox()
        self._bg_erode_spin.setRange(0, 12)
        self._bg_erode_spin.setValue(0)
        T(self._bg_erode_spin, "前景内缩像素：消掉对象边缘残留的白边/白晕", attr="tooltip")
        f.addRow(T(QLabel(), "内缩(px)"), self._bg_erode_spin)
        preview_btn = T(QPushButton(), "预览抠图效果…")
        T(preview_btn, "实时预览背景扣除效果并调整容差/内缩/羽化", attr="tooltip")
        preview_btn.clicked.connect(self._on_preview_background)
        f.addRow("", preview_btn)
        tip = T(QLabel(), "颜色键 + 容差 + 内缩去白边 + 羽化；强制纯色影响动画生成时的背景稳定约束")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        f.addRow("", tip)
        return box

    def _on_preview_background(self) -> None:
        """打开背景扣除预览弹窗；确认后应用参数并重跑背景步骤。"""
        from ui.dialogs.background_key_dialog import BackgroundKeyDialog

        src = None
        if self._session.frames:
            src = self._session.frames[0]
        elif self._session.first_frame is not None:
            src = self._session.first_frame
        if src is None:
            QMessageBox.information(self, "提示", "请先生成动画或导入首帧图再预览抠图")
            return
        dialog = BackgroundKeyDialog(
            src,
            tolerance=self._bg_tolerance_spin.value(),
            feather=self._bg_feather_spin.value(),
            erode=self._bg_erode_spin.value(),
            force_pure_bg=self._whiten_chk.isChecked(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        p = dialog.params()
        self._bg_tolerance_spin.setValue(p["tolerance"])
        self._bg_feather_spin.setValue(p["feather"])
        self._bg_erode_spin.setValue(p["erode"])
        self._whiten_chk.setChecked(p["force_pure_bg"])
        self.set_current_step(4)
        self._on_run_step()  # 用新参数重跑背景去除

    def _build_step_export_panel(self) -> QWidget:
        box = T(QGroupBox(), "步骤 6 · 导出")
        ef = QFormLayout(box)
        ef.setContentsMargins(12, 18, 12, 12)
        out_row = QHBoxLayout()
        self._output_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        out_row.addWidget(self._output_edit, 1)
        browse = T(QPushButton(), "浏览…")
        browse.clicked.connect(self._on_browse_output)
        out_row.addWidget(browse)
        ef.addRow(T(QLabel(), "输出目录"), out_row)
        self._btn_open_out = T(QPushButton(), "打开输出目录")
        self._btn_open_out.clicked.connect(self._on_open_output)
        ef.addRow("", self._btn_open_out)
        tip = T(QLabel(), "导出 GIF / APNG / PNG 序列 / 雪碧图 / JSON 元数据 / 项目文件")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        ef.addRow("", tip)
        return box

    # ------------------------------------------------------------------ #
    def _restore_settings(self) -> None:
        out = self._ctx.ui_settings.get("output_dir")
        if out:
            self._output_edit.setText(str(out))
        self._log(tr("IDE 模式：逐步执行或直接编辑帧序列"), "info")

    # ------------------------------------------------------------------ #
    # 会话同步
    # ------------------------------------------------------------------ #
    def _sync_session(self) -> None:
        """把表单参数写回 session（运行步骤前调用）。"""
        s = self._session
        s.description = self._desc_edit.toPlainText().strip()
        s.action = self._action_combo.currentData() or self._action_combo.currentText().strip()
        s.aspect_ratio = self._aspect_combo.currentText()
        s.pixel_size = int(self._size_combo.currentText())
        s.max_colors = self._colors_spin.value()
        s.frame_count = self._frames_spin.value()
        s.fps = self._fps_spin.value()
        s.speed = float(self._speed_combo.currentData() or DEFAULT_SPEED)
        s.pixelate = self._pixelate_chk.isChecked()
        s.remove_bg = self._bg_chk.isChecked()
        s.force_pure_bg = self._whiten_chk.isChecked()
        s.loop_close = self._loop_chk.isChecked()
        s.bg_tolerance = self._bg_tolerance_spin.value()
        s.bg_feather = self._bg_feather_spin.value()
        s.bg_erode = self._bg_erode_spin.value()
        s.output_dir = Path(self._output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR))

    def _load_session_to_form(self) -> None:
        s = self._session
        self._desc_edit.setPlainText(s.description)
        self._action_combo.setCurrentText(s.action)
        self._aspect_combo.setCurrentText(s.aspect_ratio)
        if str(s.pixel_size) in [self._size_combo.itemText(i) for i in range(self._size_combo.count())]:
            self._size_combo.setCurrentText(str(s.pixel_size))
        self._colors_spin.setValue(s.max_colors)
        self._frames_spin.setValue(s.frame_count)
        self._fps_spin.setValue(s.fps)
        idx = self._speed_combo.findData(s.speed)
        self._speed_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self._pixelate_chk.setChecked(s.pixelate)
        self._bg_chk.setChecked(s.remove_bg)
        self._whiten_chk.setChecked(s.force_pure_bg)
        self._loop_chk.setChecked(s.loop_close)
        self._bg_tolerance_spin.setValue(s.bg_tolerance)
        self._bg_feather_spin.setValue(s.bg_feather)
        self._bg_erode_spin.setValue(s.bg_erode)
        self._output_edit.setText(str(s.output_dir))
        self._load_prompts_to_form()

    def _load_prompts_to_form(self) -> None:
        for key, edit in self._prompt_edits.items():
            edit.setPlainText(self._session.prompts.get(key, ""))

    def _on_apply_prompts(self) -> None:
        for key, edit in self._prompt_edits.items():
            self._session.prompts[key] = edit.toPlainText().strip()
        self._log("提示词已应用到工作区", "info")
        self._mark_dirty()

    # ------------------------------------------------------------------ #
    # 刷新
    # ------------------------------------------------------------------ #
    def _refresh_all(self) -> None:
        self._refresh_preview()
        self._refresh_editor()
        self._refresh_timeline()
        self._ref_box.set_image(self._session.reference_image)
        self._update_play_button()
        self._update_status()

    def _refresh_preview(self) -> None:
        frames = self._session.frames
        if frames:
            self._show_frame(frames[min(self._play_index, len(frames) - 1)])
        elif self._session.first_frame is not None:
            self._preview.show_image(_pil_to_qpixmap(self._session.first_frame))
        else:
            self._preview.clear()

    def _refresh_editor(self) -> None:
        frames = self._session.frames
        if frames and 0 <= self._current < len(frames):
            self._editor.set_frame(frames[self._current])
            prev = frames[self._current - 1] if self._current > 0 else None
            nxt = frames[self._current + 1] if self._current < len(frames) - 1 else None
            self._editor.set_onion(prev, nxt)
            self._editor_hint.setText(tr("正在编辑帧 {cur}/{total}").format(cur=self._current + 1, total=len(frames)))
        else:
            self._editor.set_frame(Image.new("RGBA", self._session.target_size(), (0, 0, 0, 0)))
            self._editor.set_onion(None, None)
            self._editor_hint.setText(tr("暂无帧，先生成动画或添加空白帧"))

    def _refresh_timeline(self) -> None:
        self._timeline.set_frames(self._session.frames, select=self._current)

    def _show_frame(self, img: Image.Image) -> None:
        self._preview.show_image(_pil_to_qpixmap(img))

    def _on_preview_zoom_changed(self, rel: float) -> None:
        """预览缩放百分比同步（1.0 = 适应容器）。"""
        self._zoom_label.setText(tr("适应") if abs(rel - 1.0) < 1e-9 else f"{round(rel * 100)}%")

    def _preview_speed(self) -> float:
        return float(self._preview_speed_combo.currentData() or 1.0)

    def _on_preview_speed_changed(self) -> None:
        """播放中调整倍速：按新倍速重启定时器。"""
        if self._playing:
            interval = max(30, int(1000 / (max(1, self._session.fps) * self._preview_speed())))
            self._play_timer.start(interval)

    def _update_play_button(self) -> None:
        self._btn_play.setText(tr("暂停") if self._playing else tr("播放"))

    def _update_status(self) -> None:
        s = self._session
        n = len(s.frames)
        size = s.target_size()
        if n:
            self._status_label.setText(
                tr("帧 {cur}/{n}（共 {n} 帧）· {w}×{h} · {fps}fps").format(
                    cur=self._current + 1, n=n, w=size[0], h=size[1], fps=s.fps
                )
            )
        else:
            self._status_label.setText(tr("就绪 · {w}×{h}").format(w=size[0], h=size[1]))
        self._dirty_label.setText(tr("● 未保存") if self._dirty else "")

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_status()

    # ------------------------------------------------------------------ #
    # 步骤执行
    # ------------------------------------------------------------------ #
    def set_current_step(self, row: int) -> None:
        """设置当前步骤（主窗口侧栏调用），切换执行按钮文案与右侧分步参数面板。"""
        if 0 <= row < len(STEP_ACTIONS_ZH):
            self._current_step = row
            self._btn_run.setText(tr(STEP_ACTIONS_ZH[row]))
            self._step_params.setCurrentIndex(row)
            self.step_changed.emit(row)
    def _on_run_step(self) -> None:
        self._sync_session()
        step = self._current_step
        if step == 0:
            fn = lambda wf: wf.step_prompts(self._session)
        elif step == 1:
            fn = lambda wf: wf.step_image(self._session)
        elif step == 2:
            fn = lambda wf: wf.step_animation(self._session)
        elif step == 3:
            fn = lambda wf: wf.step_pixelize(self._session)
        elif step == 4:
            fn = lambda wf: wf.step_background(self._session)
        elif step == 5:
            fn = lambda wf: wf.export(
                self._session, self._export_dir(), fps=self._session.fps,
                formats=("gif", "png", "json", "apng", "sprite"),
            )
        else:
            return
        self._run_step(fn, step)

    def _run_step(self, fn, step: int) -> None:
        def job(log_cb):
            clients = self._create_clients()
            try:
                wf = IdeWorkflow(clients["llm"], clients["image"], clients["video"], log=log_cb)
                return fn(wf)
            finally:
                for c in clients.values():
                    try:
                        c.close()
                    except Exception:  # noqa: BLE001
                        pass

        self._worker = IdeStepWorker(job)
        self._worker.log.connect(self._on_log)
        self._worker.succeeded.connect(lambda result: self._on_step_success(step, result))
        self._worker.failed.connect(self._on_step_failed)
        self._set_busy(True)
        self._log(f"执行步骤：{IDE_STEPS[step]} …", "info")
        self._worker.start()

    def _create_clients(self) -> dict:
        clients = {}
        for kind in ("llm", "image", "video"):
            cfg = self._ctx.api.get_default(kind)
            if cfg is None:
                raise WorkflowError(f"未配置{kind} API，请在「设置」中配置或开启模拟 API")
            clients[kind] = create_api_client(kind, cfg)
        return clients

    def _on_log(self, level: str, message: str) -> None:
        self._log(message, level)

    def _on_step_success(self, step: int, result) -> None:
        self._set_busy(False)
        self._mark_dirty()
        if step == 0:
            self._load_prompts_to_form()
            self._tabs.setCurrentIndex(2)
            self._log("提示词已生成，可在「提示词」页签编辑", "info")
        elif step == 1:
            self._refresh_preview()
            self._tabs.setCurrentIndex(0)
            self._log("首帧图片已生成", "info")
        elif step == 2:
            self._current = 0
            self._play_index = 0
            self._refresh_all()
            self._tabs.setCurrentIndex(0)
            self._log(f"动画已生成：{len(self._session.frames)} 帧", "info")
        elif step == 3:
            self._refresh_all()
            self._log("像素化完成", "info")
        elif step == 4:
            self._refresh_all()
            self._log("背景去除完成", "info")
        elif step == 5:
            self._log("导出完成", "info")
            msg = "\n".join(f"{k}: {v}" for k, v in (result or {}).items())
            QMessageBox.information(self, "导出完成", msg or "已导出")

    def _on_step_failed(self, message: str) -> None:
        self._set_busy(False)
        self._log(message, "error")
        QMessageBox.critical(self, "步骤失败", message)

    def _set_busy(self, busy: bool) -> None:
        self._btn_run.setEnabled(not busy)
        self._btn_new.setEnabled(not busy)
        self._btn_open.setEnabled(not busy)
        self._btn_save.setEnabled(not busy)

    def _export_dir(self) -> Path:
        return Path(self._output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR)) / "export"

    # ------------------------------------------------------------------ #
    # 时间轴事件
    # ------------------------------------------------------------------ #
    def _on_frame_selected(self, index: int) -> None:
        self._current = index
        self._play_index = max(0, min(index, max(0, len(self._session.frames) - 1)))
        self._refresh_editor()
        self._refresh_preview()
        self._update_status()

    def _on_editor_edited(self) -> None:
        if self._session.frames and 0 <= self._current < len(self._session.frames):
            self._session.frames[self._current] = self._editor.frame()
            self._timeline.update_thumbnail(self._current, self._session.frames[self._current])
            self._mark_dirty()
            if not self._playing:
                self._refresh_preview()

    def _on_insert_frame(self) -> None:
        if not self._session.frames:
            self._on_add_frame()
            return
        img = self._session.frames[max(0, min(self._current, len(self._session.frames) - 1))].copy()
        self._session.insert_frame(self._current, img)
        self._mark_dirty()
        self._refresh_all()

    def _on_duplicate_frame(self) -> None:
        if not self._session.frames:
            return
        idx = max(0, min(self._current, len(self._session.frames) - 1))
        self._session.duplicate_frame(idx)
        self._current = idx + 1
        self._mark_dirty()
        self._refresh_all()

    def _on_delete_frame(self) -> None:
        if len(self._session.frames) <= 1:
            QMessageBox.information(self, "提示", "至少保留一帧")
            return
        idx = max(0, min(self._current, len(self._session.frames) - 1))
        self._session.delete_frame(idx)
        self._current = max(0, min(idx, len(self._session.frames) - 1))
        self._mark_dirty()
        self._refresh_all()

    def _on_add_frame(self) -> None:
        size = self._session.frames[0].size if self._session.frames else self._session.target_size()
        self._session.frames.append(Image.new("RGBA", size, (0, 0, 0, 0)))
        self._current = len(self._session.frames) - 1
        self._mark_dirty()
        self._refresh_all()

    def _on_reordered(self, order: List[int]) -> None:
        if len(order) != len(self._session.frames):
            return
        # Qt InternalMove 已把选中项跟随到新位置，先记录再重建
        new_sel = self._timeline.current_index()
        self._session.frames = [self._session.frames[i] for i in order]
        self._current = new_sel if new_sel >= 0 else 0
        self._mark_dirty()
        self._refresh_all()

    # ------------------------------------------------------------------ #
    # 播放
    # ------------------------------------------------------------------ #
    def _on_toggle_play(self) -> None:
        if self._playing:
            self._playing = False
            self._play_timer.stop()
        else:
            if not self._session.frames:
                return
            self._playing = True
            self._play_index = self._current
            interval = max(30, int(1000 / (max(1, self._session.fps) * self._preview_speed())))
            self._play_timer.start(interval)
        self._update_play_button()

    def _on_play_tick(self) -> None:
        n = len(self._session.frames)
        if n:
            self._play_index = (self._play_index + 1) % n
            self._show_frame(self._session.frames[self._play_index])

    # ------------------------------------------------------------------ #
    # 项目
    # ------------------------------------------------------------------ #
    def _on_new(self) -> None:
        if self._dirty and not self._confirm_discard():
            return
        self._session = IdeSession(output_dir=Path(self._output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR)))
        self._current = 0
        self._play_index = 0
        self._dirty = False
        self._stop_play()
        self._refresh_all()
        self._log("已新建工作区", "info")

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开 IDE 项目", "", "IDE 项目 (*.json);;所有文件 (*)")
        if not path:
            return
        try:
            self._session = load_ide_project(Path(path).parent)
            self._current = 0
            self._play_index = 0
            self._dirty = False
            self._load_session_to_form()
            self._refresh_all()
            self._log(f"已打开项目：{path}", "info")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "打开失败", str(exc))

    def _on_save_project(self) -> None:
        self._sync_session()
        base = self._session.output_dir
        path = QFileDialog.getExistingDirectory(self, "选择项目保存目录", str(base))
        if not path:
            return
        try:
            proj_dir = Path(path)
            # 直接存到所选目录（而非嵌套 untitled 子目录）
            if not self._session.name or self._session.name in ("untitled", "project"):
                self._session.name = proj_dir.name or "project"
            save_ide_project(self._session, proj_dir)
            self._dirty = False
            self._update_status()
            self._log(f"项目已保存到：{proj_dir}", "info")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))

    def _confirm_discard(self) -> bool:
        return QMessageBox.question(self, "未保存", "当前工作区有未保存修改，确定丢弃吗？") == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ #
    # 参考图 / 首帧图
    # ------------------------------------------------------------------ #
    def _on_ref_changed(self, img) -> None:
        """参考图卡片变更：设置图生图参考；首帧为空时同时作为首帧图。"""
        if img is not None:
            self._session.reference_image = img
            if self._session.first_frame is None:
                self._session.first_frame = img.copy()
            self._log("已添加参考图（图生图 + 首帧图）", "info")
        else:
            self._session.reference_image = None
            self._log("已移除参考图", "info")
        self._refresh_preview()
        self._mark_dirty()

    # ------------------------------------------------------------------ #
    # 从 Solo 同步
    # ------------------------------------------------------------------ #
    def set_first_frame(self, img) -> None:
        """外部导入图片作为首帧 + 图生图参考（像素板块同步用）。"""
        self._session.first_frame = img.convert("RGBA")
        self._session.reference_image = img.convert("RGBA")
        self._current = 0
        self._play_index = 0
        self._mark_dirty()
        self._refresh_all()
        self._tabs.setCurrentIndex(0)
        self._log("已导入图片（首帧 + 图生图参考），可直接走「动画生成」步骤", "info")

    def import_from_solo(self, result: SoloResult) -> None:
        """把 Solo 生成的首帧图与最终帧序列同步到 IDE 工作区。"""
        s = self._session
        # 首帧图
        if getattr(result, "first_frame", None) and Path(result.first_frame).exists():
            s.first_frame = fu.load_image(result.first_frame)
        # 最终帧序列（优先 frames_dir，其次 png_dir）
        src = getattr(result, "frames_dir", None) or getattr(result, "png_dir", None)
        frames: List[Image.Image] = []
        if src and Path(src).exists():
            frames = [fu.load_image(p) for p in sorted(Path(src).glob("*.png"))]
        if frames:
            s.frames = frames
            s.fps = int(result.fps or s.fps)
            s.frame_count = len(frames)
            w, h = frames[0].size
            s.aspect_ratio = self._aspect_from_size(w, h)
            s.pixel_size = max(w, h)
        self._current = 0
        self._play_index = 0
        self._dirty = True
        self._refresh_all()
        self._tabs.setCurrentIndex(0)  # 切到预览
        self._log(
            f"已从 Solo 同步：{len(frames)} 帧" + ("（含首帧图）" if s.first_frame is not None else ""),
            "info",
        )

    def import_from_sprite(self, result) -> None:
        """把精灵图生成的结果（底图 + 帧序列）同步到 IDE 工作区。"""
        s = self._session
        if getattr(result, "base_image", None) and Path(result.base_image).exists():
            s.first_frame = fu.load_image(result.base_image)
        src = getattr(result, "frames_dir", None)
        frames: List[Image.Image] = []
        if src and Path(src).exists():
            frames = [fu.load_image(p) for p in sorted(Path(src).glob("*.png"))]
        if frames:
            s.frames = frames
            s.frame_count = len(frames)
            w, h = frames[0].size
            s.aspect_ratio = self._aspect_from_size(w, h)
            s.pixel_size = max(w, h)
        self._current = 0
        self._play_index = 0
        self._dirty = True
        self._refresh_all()
        self._tabs.setCurrentIndex(0)
        self._log(f"已从精灵图同步：{len(frames)} 帧" + ("（含对象底图）" if s.first_frame is not None else ""), "info")

    @staticmethod
    def _aspect_from_size(w: int, h: int) -> str:
        """按宽高反向匹配最接近的预设比例。"""
        from config.settings import ASPECT_RATIOS

        best, best_err = "1:1", float("inf")
        for name, (rw, rh) in ASPECT_RATIOS.items():
            err = abs(rw / rh - w / max(1, h))
            if err < best_err:
                best_err, best = err, name
        return best

    def _stop_play(self) -> None:
        self._playing = False
        self._play_timer.stop()
        self._update_play_button()

    # ------------------------------------------------------------------ #
    # 其他
    # ------------------------------------------------------------------ #
    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self._output_edit.text())
        if path:
            self._output_edit.setText(path)

    def _on_open_output(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR))))

    def _log(self, message: str, level: str = "info") -> None:
        color = _LOG_COLORS.get(level, "#8b949e")
        safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._log_view.appendHtml(f'<span style="color:{color};">[{level.upper()}] {safe}</span>')
