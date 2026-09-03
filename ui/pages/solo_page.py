"""Solo 模式页：一键式全自动生成像素动画。

布局（全屏不变形）：
- 左侧：固定宽度(408px)参数表单，内含纵向滚动，运行按钮固定可见；
- 右侧：预览与「中间结果」面板的纵向分割区，随窗口弹性伸缩；
- 底部：步骤/进度 + 日志。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QPixmap
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

from config.settings import (
    ASPECT_RATIOS,
    DEFAULT_FPS,
    DEFAULT_FRAME_COUNT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MAX_COLORS,
    DEFAULT_SPEED,
    PIXEL_SIZES,
)
from core.processing.prompt_utils import recommended_frames
from core.workflow import SoloParams, SoloResult
from ui.app_context import AppContext
from ui.i18n import T, tr
from ui.widgets.action_combo import populate_action_combo
from ui.widgets.image_viewer import ImageViewer
from ui.widgets.reference_box import ReferenceImageBox
from ui.workers import SoloWorker

logger = logging.getLogger("PixelAnimIDE.ui.solo_page")

_LOG_COLORS = {"info": "#adb2b8", "warn": "#f59e0b", "error": "#f25a5a"}

FORM_WIDTH = 408


class SoloPage(QWidget):
    sync_to_ide = Signal(object)  # 把生成结果（SoloResult）同步到 IDE 模式

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._worker: SoloWorker | None = None
        self._result: SoloResult | None = None
        self._build_ui()
        self._restore_settings()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(14)

        # ---------- 左：参数 / 中间结果（导航栏切换） ----------
        left_panel = QWidget()
        lp = QVBoxLayout(left_panel)
        lp.setContentsMargins(0, 0, 0, 0)
        lp.setSpacing(10)

        self._tabs = QTabWidget()
        self._tabs.setFixedWidth(FORM_WIDTH)

        # Tab 0：参数表单
        self._param_tab = QWidget()
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        form_host = QWidget()
        form_layout = QVBoxLayout(form_host)
        form_layout.setContentsMargins(0, 0, 6, 0)
        form_layout.setSpacing(10)

        input_box = T(QGroupBox(), "输入参数")
        f = QFormLayout(input_box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(10)

        self._desc_edit = QTextEdit()
        T(self._desc_edit, "例如：一只拿着剑的橙色小猫，Q 版，侧身站立", attr="placeholder")
        self._desc_edit.setMaximumHeight(88)
        f.addRow(T(QLabel(), "文本描述"), self._desc_edit)

        # 参考图（豆包/即梦风格小卡片，图生图可选）
        ref_row = QHBoxLayout()
        ref_row.setSpacing(8)
        self._ref_box = ReferenceImageBox(size=72)
        ref_row.addWidget(self._ref_box)
        ref_txt = T(QLabel(), "参考图（图生图，可选）\n点击添加图片")
        ref_txt.setObjectName("HintLabel")
        ref_txt.setWordWrap(True)
        ref_row.addWidget(ref_txt, 1)
        f.addRow(ref_row)

        self._action_combo = QComboBox()
        self._action_combo.setEditable(True)
        self._action_combo.setPlaceholderText(T(None, "选择或输入动作…"))
        populate_action_combo(self._action_combo)
        self._action_combo.currentTextChanged.connect(self._on_action_changed)
        f.addRow(T(QLabel(), "动作类型(可选)"), self._action_combo)
        action_hint = T(QLabel(), "选择预设动作会自动按建议时长设置帧数（AI 视频动作较慢）")
        action_hint.setObjectName("HintLabel")
        action_hint.setWordWrap(True)
        f.addRow("", action_hint)

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

        self._colors_spin = QSpinBox()
        self._colors_spin.setRange(2, 64)
        self._colors_spin.setValue(DEFAULT_MAX_COLORS)
        f.addRow(T(QLabel(), "最大颜色数"), self._colors_spin)

        # 图转视频参数（AI 可按动作自动调整）
        video_box = T(QGroupBox(), "图转视频参数（AI 可自动调整）")
        vf = QFormLayout(video_box)
        vf.setContentsMargins(12, 18, 12, 12)
        vf.setVerticalSpacing(10)
        self._frames_spin = QSpinBox()
        self._frames_spin.setRange(2, 120)
        self._frames_spin.setValue(DEFAULT_FRAME_COUNT)
        vf.addRow(T(QLabel(), "帧数"), self._frames_spin)
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(DEFAULT_FPS)
        vf.addRow(T(QLabel(), "帧率(fps)"), self._fps_spin)
        self._speed_combo = QComboBox()
        for label, value in [("0.5x（慢放）", 0.5), ("1x（原速）", 1.0), ("1.5x", 1.5), ("2x（提速）", 2.0)]:
            self._speed_combo.addItem(tr(label), userData=value)
        self._speed_combo.setCurrentIndex(1)
        T(self._speed_combo, "AI 视频动作通常偏慢，可提速播放让动作更利落", attr="tooltip")
        vf.addRow(T(QLabel(), "播放速度"), self._speed_combo)
        video_hint = T(QLabel(), "留默认值时，LLM 会按动作自动评估时长（如 步行→2s、挥砍→1s）")
        video_hint.setObjectName("HintLabel")
        video_hint.setWordWrap(True)
        vf.addRow("", video_hint)

        opt_box = T(QGroupBox(), "处理选项")
        of = QFormLayout(opt_box)
        of.setContentsMargins(12, 18, 12, 12)
        of.setVerticalSpacing(10)
        self._pixelate_chk = T(QCheckBox(), "完美像素化（Perfect Pixel 网格采样）")
        self._pixelate_chk.setChecked(True)
        of.addRow(self._pixelate_chk)
        self._bg_chk = T(QCheckBox(), "去除背景（默认白色）")
        self._bg_chk.setChecked(True)
        of.addRow(self._bg_chk)
        self._whiten_chk = T(QCheckBox(), "背景强制纯色（主体浅色→黑底，否则白底）")
        self._whiten_chk.setChecked(True)
        T(self._whiten_chk, "检测与画面边缘相连的背景并刷成纯色；对象本身是浅色系时自动用纯黑背景保证对比度", attr="tooltip")
        of.addRow(self._whiten_chk)
        self._loop_chk = T(QCheckBox(), "首尾帧一致（循环闭合）")
        self._loop_chk.setChecked(True)
        T(self._loop_chk, "AI 视频首尾帧常不闭合，勾选后强制首尾一致，循环播放无跳变", attr="tooltip")
        of.addRow(self._loop_chk)
        self._bg_tolerance_spin = QSpinBox()
        self._bg_tolerance_spin.setRange(0, 200)
        self._bg_tolerance_spin.setValue(30)
        of.addRow(T(QLabel(), "背景容差"), self._bg_tolerance_spin)
        self._bg_feather_spin = QSpinBox()
        self._bg_feather_spin.setRange(0, 30)
        self._bg_feather_spin.setValue(8)
        of.addRow(T(QLabel(), "羽化(px)"), self._bg_feather_spin)

        exp_box = T(QGroupBox(), "导出")
        ef = QFormLayout(exp_box)
        ef.setContentsMargins(12, 18, 12, 12)
        ef.setVerticalSpacing(10)
        self._gif_chk = T(QCheckBox(), "导出 GIF 动画")
        self._gif_chk.setChecked(True)
        ef.addRow(self._gif_chk)
        self._png_chk = T(QCheckBox(), "导出 PNG 序列帧")
        self._png_chk.setChecked(True)
        ef.addRow(self._png_chk)
        self._apng_chk = T(QCheckBox(), "导出 APNG 动画")
        ef.addRow(self._apng_chk)
        self._sprite_chk = T(QCheckBox(), "导出雪碧图 (Sprite Sheet)")
        ef.addRow(self._sprite_chk)

        out_row = QHBoxLayout()
        self._output_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        out_row.addWidget(self._output_edit, 1)
        btn_browse = T(QPushButton(), "浏览…")
        btn_browse.clicked.connect(self._on_browse_output)
        out_row.addWidget(btn_browse)
        ef.addRow(T(QLabel(), "输出目录"), out_row)

        form_layout.addWidget(input_box)
        form_layout.addWidget(video_box)
        form_layout.addWidget(opt_box)
        form_layout.addWidget(exp_box)
        form_layout.addStretch(1)
        form_scroll.setWidget(form_host)
        param_layout = QVBoxLayout(self._param_tab)
        param_layout.setContentsMargins(0, 4, 0, 0)
        param_layout.addWidget(form_scroll)
        self._tabs.addTab(self._param_tab, T(None, "参数"))
        T(self._tabs, "参数", attr="tab", index=0)

        # Tab 1：中间结果（提示词 + 首帧 + 帧序列缩略图）
        self._intermediate_tab = QWidget()
        it = QVBoxLayout(self._intermediate_tab)
        it.setContentsMargins(4, 8, 4, 4)
        it.setSpacing(6)
        self._prompt_edits: dict = {}
        for key, label in (
            ("image_prompt", "图片提示词"),
            ("animation_prompt", "动画提示词"),
            ("negative_prompt", "负面提示词"),
        ):
            row = QHBoxLayout()
            name = T(QLabel(), label)
            name.setFixedWidth(76)
            row.addWidget(name)
            edit = QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setMaximumHeight(52)
            edit.setObjectName("LogView")
            edit.setPlaceholderText(T(None, "等待生成…"))
            self._prompt_edits[key] = edit
            row.addWidget(edit, 1)
            btn = T(QPushButton(), "复制")
            btn.setFixedWidth(68)  # 足够容纳「复制」文本，避免被边框裁剪
            btn.clicked.connect(lambda _=False, k=key: self._copy_prompt(k))
            row.addWidget(btn)
            it.addLayout(row)
        self._first_frame_viewer = ImageViewer()
        it.addWidget(self._first_frame_viewer, 1)
        strip_header = QHBoxLayout()
        strip_header.addWidget(T(QLabel(), "帧序列："))
        self._strip_count = QLabel("")
        self._strip_count.setObjectName("HintLabel")
        strip_header.addWidget(self._strip_count)
        strip_header.addStretch(1)
        it.addLayout(strip_header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setFixedHeight(84)
        strip_host = QWidget()
        self._frames_strip = QHBoxLayout(strip_host)
        self._frames_strip.setContentsMargins(2, 2, 2, 2)
        self._frames_strip.addStretch(1)
        scroll.setWidget(strip_host)
        it.addWidget(scroll)
        self._tabs.addTab(self._intermediate_tab, T(None, "中间结果"))
        T(self._tabs, "中间结果", attr="tab", index=1)

        lp.addWidget(self._tabs, 1)

        # 运行控制（固定在左侧底部，不随导航切换）
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self._btn_start = QPushButton(tr("开始生成"))
        self._btn_start.setObjectName("PrimaryButton")
        self._btn_start.setMinimumHeight(36)
        self._btn_start.clicked.connect(self._on_start)
        ctrl.addWidget(self._btn_start)
        self._btn_cancel = QPushButton(tr("取消"))
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        ctrl.addWidget(self._btn_cancel)
        self._btn_open_out = QPushButton(tr("打开输出目录"))
        self._btn_open_out.setEnabled(False)
        self._btn_open_out.clicked.connect(self._on_open_output)
        ctrl.addWidget(self._btn_open_out)
        self._btn_sync = T(QPushButton(), "同步到 IDE")
        self._btn_sync.setEnabled(False)
        T(self._btn_sync, "把生成的首帧图与最终帧序列同步到 IDE 模式继续编辑", attr="tooltip")
        self._btn_sync.clicked.connect(self._on_sync)
        ctrl.addWidget(self._btn_sync)
        lp.addLayout(ctrl)

        top.addWidget(left_panel)

        # ---------- 右：预览（占据整个右边，支持 GIF 倍速） ----------
        preview_box = T(QGroupBox(), "预览")
        pv = QVBoxLayout(preview_box)
        pv.setContentsMargins(8, 18, 8, 8)
        speed_row = QHBoxLayout()
        speed_row.addWidget(T(QLabel(), "GIF 播放速度:"))
        self._preview_speed_combo = QComboBox()
        for label, value in [("0.5x", 0.5), ("1x（原速）", 1.0), ("1.5x", 1.5), ("2x", 2.0), ("3x", 3.0)]:
            self._preview_speed_combo.addItem(tr(label), userData=value)
        self._preview_speed_combo.setCurrentIndex(1)
        self._preview_speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self._preview_speed_combo)
        speed_row.addStretch(1)
        pv.addLayout(speed_row)
        self._preview = ImageViewer()
        pv.addWidget(self._preview, 1)
        top.addWidget(preview_box, 1)

        root.addLayout(top, 1)

        # ---------- 底部：步骤 + 进度 + 日志 ----------
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
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

    def _restore_settings(self) -> None:
        out = self._ctx.ui_settings.get("output_dir")
        if out:
            self._output_edit.setText(str(out))
        theme = self._ctx.ui_settings.get("theme", "dark")
        self._log_html(tr("主题: {theme}；请先在「设置」中配置 API（或勾选模拟 API）").format(theme=theme), "info")

    # ------------------------------------------------------------------ #
    # 事件
    # ------------------------------------------------------------------ #
    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr("选择输出目录"), self._output_edit.text())
        if path:
            self._output_edit.setText(path)

    def _on_action_changed(self, text: str) -> None:
        """选择预设动作时，按建议时长自动设置帧数（AI 视频动作慢，时长要留足）。"""
        frames = recommended_frames(text, self._fps_spin.value())
        if frames:
            self._frames_spin.setValue(frames)
            secs = frames / max(1, self._fps_spin.value())
            self._log(tr("动作「{0}」：建议帧数已设为 {1}（约 {2}s）").format(text, frames, round(secs, 1)), "info")

    def set_reference_image(self, img) -> None:
        """外部导入参考图/首帧（像素板块「用作图生视频首帧」用）。"""
        self._ref_box.set_image(img.convert("RGBA"))
        self._log(tr("已导入像素板块图片作为参考图/首帧，可点击「开始生成」走图生视频"), "info")

    def _on_start(self) -> None:
        desc = self._desc_edit.toPlainText().strip()
        if not desc:
            QMessageBox.warning(self, tr("提示"), tr("请先输入文本描述"))
            return

        run_dir = self._run_dir()
        reference_image = None
        ref = self._ref_box.image()
        if ref is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            ref_path = run_dir / "reference.png"
            ref.save(ref_path, format="PNG")
            reference_image = ref_path

        params = SoloParams(
            description=desc,
            action=self._action_combo.currentData() or self._action_combo.currentText().strip(),
            aspect_ratio=self._aspect_combo.currentText(),
            pixel_size=int(self._size_combo.currentText()),
            frame_count=self._frames_spin.value(),
            fps=self._fps_spin.value(),
            speed=float(self._speed_combo.currentData() or DEFAULT_SPEED),
            loop_close=self._loop_chk.isChecked(),
            force_pure_bg=self._whiten_chk.isChecked(),
            max_colors=self._colors_spin.value(),
            pixelate=self._pixelate_chk.isChecked(),
            remove_bg=self._bg_chk.isChecked(),
            bg_tolerance=self._bg_tolerance_spin.value(),
            bg_feather=self._bg_feather_spin.value(),
            export_gif=self._gif_chk.isChecked(),
            export_png=self._png_chk.isChecked(),
            export_apng=self._apng_chk.isChecked(),
            export_sprite=self._sprite_chk.isChecked(),
            reference_image=reference_image,
            output_dir=run_dir,
        )

        self._ctx.ui_settings.set("output_dir", self._output_edit.text().strip())

        self._worker = SoloWorker(self._ctx.api, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.prompts_generated.connect(self._on_prompts)
        self._worker.first_frame_ready.connect(self._on_first_frame)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failed)

        self._set_running(True)
        self._progress.setValue(0)
        self._log_view.clear()
        self._preview.clear()
        self._clear_intermediate()
        self._result = None
        self._btn_sync.setEnabled(False)
        self._log(tr("开始生成：{0}").format(desc[:60]))
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker:
            self._log(tr("正在取消…"))
            self._worker.cancel()

    def _on_progress(self, step: int, total: int, name: str, pct: float, message: str) -> None:
        self._step_label.setText(tr("步骤 {0}/{1}：{2} — {3}").format(step + 1, total, name, message))
        overall = (step + pct) / total * 100
        self._progress.setValue(int(overall))

    def _on_log(self, level: str, message: str) -> None:
        self._log(message, level)

    def _on_prompts(self, prompts: dict) -> None:
        """中间结果：提示词就绪，实时展示。"""
        for key, edit in self._prompt_edits.items():
            edit.setPlainText(prompts.get(key, ""))
        self._tabs.setCurrentWidget(self._intermediate_tab)
        self._log(tr("中间结果：提示词已生成，可在「中间结果」面板查看"), "info")

    def _on_first_frame(self, path: str) -> None:
        """中间结果：首帧图就绪，实时展示。"""
        self._first_frame_viewer.show_path(path)
        self._tabs.setCurrentWidget(self._intermediate_tab)
        self._log(tr("中间结果：首帧图片已生成"), "info")

    def _copy_prompt(self, key: str) -> None:
        text = self._prompt_edits[key].toPlainText()
        if text:
            QGuiApplication.clipboard().setText(text)

    def _on_speed_changed(self) -> None:
        """预览 GIF 倍速调整。"""
        speed = float(self._preview_speed_combo.currentData() or 1.0)
        self._preview.set_speed(speed)

    def _on_sync(self) -> None:
        """把生成结果同步到 IDE 模式。"""
        if self._result:
            self.sync_to_ide.emit(self._result)

    def _on_success(self, result: SoloResult) -> None:
        self._result = result
        self._set_running(False)
        self._progress.setValue(100)
        self._step_label.setText(tr("完成"))
        self._log(tr("生成完成：{0} 帧，{1}x{2}").format(result.frame_count, result.width, result.height), "info")
        speed = float(self._preview_speed_combo.currentData() or 1.0)
        if result.gif_path:
            self._preview.show_gif(result.gif_path, speed=speed)
            self._log(f"GIF: {result.gif_path}")
        elif result.first_frame:
            self._preview.show_path(result.first_frame)
        if result.native_gif_path:
            self._log(
                tr("GIF（完美像素原生分辨率 {0}x{1}）: {2}").format(result.native_width, result.native_height, result.native_gif_path),
                "info",
            )
        self._build_frames_strip(result)
        self._btn_open_out.setEnabled(True)
        self._btn_sync.setEnabled(True)
        QMessageBox.information(
            self,
            tr("完成"),
            tr("动画已生成！\n{0}").format(result.gif_path or result.frames_dir)
            + (tr("原生分辨率版: {0}").format(result.native_gif_path) if result.native_gif_path else ""),
        )

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

    # ------------------------------------------------------------------ #
    # 中间结果面板
    # ------------------------------------------------------------------ #
    def _clear_intermediate(self) -> None:
        for edit in self._prompt_edits.values():
            edit.clear()
        self._first_frame_viewer.clear()
        self._strip_count.setText("")
        self._clear_layout(self._frames_strip)
        self._frames_strip.addStretch(1)

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_frames_strip(self, result: SoloResult) -> None:
        """完成后展示帧序列缩略图。"""
        self._clear_layout(self._frames_strip)
        src = result.frames_dir or (result.png_dir if result.png_dir else None)
        if not src or not Path(src).exists():
            self._frames_strip.addStretch(1)
            return
        paths = sorted(Path(src).glob("*.png"))[:48]
        for p in paths:
            pix = QPixmap(str(p))
            if pix.isNull():
                continue
            thumb = pix.scaled(
                64, 64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label = QLabel()
            label.setPixmap(thumb)
            label.setFixedSize(68, 68)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setToolTip(p.name)
            self._frames_strip.addWidget(label)
        self._frames_strip.addStretch(1)
        self._strip_count.setText(tr("{0} 帧").format(len(paths)))

    def _run_dir(self) -> Path:
        base = Path(self._output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR))
        # 含微秒，避免同一秒内多次生成写到同一目录互相覆盖
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return base / f"run_{ts}"

    def _log(self, message: str, level: str = "info") -> None:
        color = _LOG_COLORS.get(level, "#8b949e")
        self._log_view.appendHtml(
            f'<span style="color:{color};">[{level.upper()}] {message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</span>'
        )

    def _log_html(self, html: str, level: str = "info") -> None:
        color = _LOG_COLORS.get(level, "#8b949e")
        self._log_view.appendHtml(f'<span style="color:{color};">{html}</span>')

    def _on_open_output(self) -> None:
        if self._result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result.output_dir)))

    # ------------------------------------------------------------------ #
    # 键盘快捷键（设置 → 快捷键 → Solo 可自定义；固定使用 solo 键位）
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event) -> None:  # noqa: N802
        from ui import shortcuts as sc

        if sc.match(event, sc.get("preview_play", "solo")):
            self._preview.toggle_play()
            event.accept()
            return
        if sc.match(event, sc.get("preview_fit", "solo")):
            self._preview.reset_zoom()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    # 语言切换：刷新常驻下拉框项（动作预设分组 + 倍速等文本）
    # ------------------------------------------------------------------ #
    def retranslate_ui(self) -> None:
        populate_action_combo(self._action_combo)
        for combo, items in (
            (self._speed_combo, [("0.5x（慢放）", 0.5), ("1x（原速）", 1.0), ("1.5x", 1.5), ("2x（提速）", 2.0)]),
            (self._preview_speed_combo, [("0.5x", 0.5), ("1x（原速）", 1.0), ("1.5x", 1.5), ("2x", 2.0), ("3x", 3.0)]),
        ):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for label, value in items:
                combo.addItem(tr(label), userData=value)
            idx = combo.findData(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
