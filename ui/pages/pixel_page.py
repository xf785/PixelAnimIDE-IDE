"""像素编辑板块：独立像素画布（可设分辨率），与 IDE 双向同步，可作图生视频首帧。

布局参考 Krita：
- **顶部画布工具条**：常用动作（导入 / 同步 / 导出）收在画布上方，不再占用侧栏宽度；
- **左停靠栏**：瓦片包 / 素材包（包列表 + 包内逐级目录 + 缩略图网格），可整栏收起、可拖宽；
- **中间画布**：像素编辑器，永远占据剩余空间；
- **右停靠栏**：画布设置 + 画布信息，同样可收起、可拖宽；
- 三栏之间全部由 ``QSplitter`` 连接，**每条分隔线都能拖动**，宽度会被记住。

功能：
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
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config.settings import DEFAULT_OUTPUT_DIR
from ui.app_context import AppContext
from ui.i18n import T, tr
from ui.layout import scaled
from ui.widgets.dock import SideDock
from ui.widgets.pack_browser import PackBrowser
from ui.widgets.pixel_editor import PixelEditorWidget

logger = logging.getLogger("PixelFoundry.ui.pixel_page")

RESOLUTION_PRESETS = [16, 32, 64, 128, 256, 512]
_BG_FILLS = {"透明": (0, 0, 0, 0), "白色": (255, 255, 255, 255), "黑色": (0, 0, 0, 255)}

#: 停靠栏默认宽度（可拖动，且会被记住）
LEFT_DOCK_W = 250
RIGHT_DOCK_W = 208


class PixelPage(QWidget):
    """独立像素编辑板块。"""

    sync_from_ide_requested = Signal()            # 请求主窗口从 IDE 拉当前帧进来
    sync_to_ide = Signal(object)                  # 画布图 -> IDE 首帧/参考图
    use_as_video_first_frame = Signal(object)     # 画布图 -> Solo 图生视频首帧
    workspace_status_changed = Signal(str)        # 工具条右侧状态标识（画布尺寸）

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._build_ui()
        # 默认给一张 64×64 空画布（与「预设」下拉默认值一致），避免首次进来是一片空白小格
        self._editor.set_frame(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
        self._restore_layout()
        self._restore_packs()
        self._fit_view()
        self._refresh_info()

    # ------------------------------------------------------------------ #
    # 瓦片包 / 素材包：导入、可视化切换、送入画布
    # ------------------------------------------------------------------ #
    def _wrap_asset(self, img: Image.Image) -> Image.Image:
        """把资源居中合成进当前画布（保持画布尺寸；画布更小时裁切）。"""
        frame = self._editor.frame().convert("RGBA")
        if frame.width == 0 or frame.height == 0:
            return img.convert("RGBA")
        out = frame.copy()
        art = img.convert("RGBA")
        if art.width > out.width or art.height > out.height:
            art = art.resize((min(art.width, out.width), min(art.height, out.height)),
                             Image.Resampling.NEAREST)
        out.alpha_composite(art, ((out.width - art.width) // 2, (out.height - art.height) // 2))
        return out

    def _fit_view(self) -> None:
        """让画布以合适的整数倍缩放占满视图（新建/替换/载入后调用）。"""
        try:
            from PySide6.QtCore import QTimer

            # 延后一拍：此时视图已完成布局，才能算出合适的整数倍缩放
            QTimer.singleShot(0, self, self._fit_view_now)
        except Exception as exc:  # noqa: BLE001
            logger.debug("适应视图失败: %s", exc)

    def _fit_view_now(self) -> None:
        try:
            self._editor.fit_zoom()
        except RuntimeError:  # 控件已销毁（页面关闭/测试结束）
            pass

    def _on_pack_asset(self, img: Image.Image, name: str) -> None:
        """放入画布：居中合成，保持画布尺寸。"""
        self._editor.set_frame(self._wrap_asset(img))
        self._status_hint(tr("已放入画布：{0}").format(name))
        self._refresh_info()

    def _on_pack_asset_replace(self, img: Image.Image, name: str) -> None:
        """替换画布：画布尺寸随资源改变。"""
        self._editor.set_frame(img.convert("RGBA"))
        self._status_hint(tr("已用素材替换画布：{0}").format(name))
        self._fit_view()
        self._refresh_info()

    def _status_hint(self, text: str) -> None:
        logger.info(text)
        self._status(text)

    def _restore_packs(self) -> None:
        """恢复上次会话加载过的包（路径存在才加载）。"""
        paths = self._ctx.ui_settings.get("pixel_pack_paths", []) or []
        paths = [p for p in paths if Path(p).exists()]
        if not paths:
            return
        ok = self._pack_browser.add_paths(paths)
        if ok:
            logger.info(tr("已恢复上次的 {0} 个包").format(ok))
        self._remember_packs()

    def _remember_packs(self) -> None:
        try:
            self._ctx.ui_settings.set("pixel_pack_paths", self._pack_browser.pack_paths())
        except Exception as exc:  # noqa: BLE001
            logger.warning("包路径保存失败: %s", exc)

    # ------------------------------------------------------------------ #
    # 布局
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(scaled(10), scaled(10), scaled(10), scaled(8))
        root.setSpacing(scaled(8))

        # ---------- 工作区：左停靠栏 | 画布 | 右停靠栏（三栏可拖动调宽） ----------
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("Workspace")
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(scaled(5))

        # 左：瓦片包 / 素材包 —— 面板由 PackBrowser 提供（packs + 目录浏览）
        self._pack_browser = PackBrowser(standalone=False)
        self._pack_browser.assetChosen.connect(self._on_pack_asset)
        self._pack_browser.assetReplaceRequested.connect(self._on_pack_asset_replace)
        self._pack_browser.packChanged.connect(self._remember_packs)
        self._left_dock = SideDock(tr("瓦片包 / 素材包"), side="left", default_width=LEFT_DOCK_W)
        panels = self._pack_browser.panels()
        for i, panel in enumerate(panels):
            self._left_dock.add_panel(panel, stretch=1 if i == len(panels) - 1 else 0)
        self._splitter.addWidget(self._left_dock)

        # 中：像素画布
        self._editor = PixelEditorWidget()
        self._splitter.addWidget(self._editor)

        # 右：画布设置 + 画布信息
        self._build_right_dock()
        self._splitter.addWidget(self._right_dock)

        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._left_dock.bind_splitter(self._splitter, 0, default_width=LEFT_DOCK_W)
        self._right_dock.bind_splitter(self._splitter, 2, default_width=RIGHT_DOCK_W)
        self._splitter.setSizes([scaled(LEFT_DOCK_W), scaled(860), scaled(RIGHT_DOCK_W)])
        root.addWidget(self._splitter, 1)

    # ------------------------------------------------------------------ #
    # 主窗口工具条协议（Krita 风格：工具条内容随工作区变化）
    # ------------------------------------------------------------------ #
    def toolbar_actions(self) -> list:
        """返回 [(图标, 文本, 提示, 回调, 是否主按钮), …] 供主窗口工具条渲染。"""
        return [
            ("import_image", "导入图片…", "从本地导入图片替换当前帧", self._on_import, False),
            ("copy", "从 IDE 同步", "把 IDE 当前帧/首帧拉进画布精细编辑", self._on_sync_from_ide, False),
            ("layers", "同步到 IDE", "把画布图作为首帧 + 图生图参考导入 IDE", self._on_sync_to_ide, False),
            ("onion", "用作首帧",
             "把画布图作为首帧走图生视频（Solo）；过小会自动最近邻放大到 API 最低要求",
             self._on_use_as_video, False),
            ("export_image", "导出 PNG", "导出当前画布为 PNG", self._on_export, True),
        ]

    def workspace_status(self) -> str:
        """工具条右侧的状态标识：当前画布分辨率。"""
        w, h = self._editor.frame().size
        return f"{w} × {h}"

    def retranslate_ui(self) -> None:
        """语言切换后重刷动态文案（画布信息面板、资源统计、编辑器色族提示等）。"""
        editor = getattr(self, "_editor", None)
        if editor is not None and hasattr(editor, "retranslate_ui"):
            editor.retranslate_ui()
        self._refresh_info()
        browser = getattr(self, "_pack_browser", None)
        if browser is not None and hasattr(browser, "retranslate_ui"):
            browser.retranslate_ui()

    def _build_right_dock(self) -> None:
        """右停靠栏：画布设置 + 画布信息。"""
        self._right_dock = SideDock(tr("画布"), side="right", default_width=RIGHT_DOCK_W)

        # ---- 画布设置 ----
        box = QWidget()
        f = QFormLayout(box)
        f.setContentsMargins(0, 0, 0, 0)
        f.setVerticalSpacing(scaled(8))
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # 窄停靠栏里让标签自动换到字段上方，面板因此可以拖得更窄
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._preset_combo = QComboBox()
        for s in RESOLUTION_PRESETS:
            self._preset_combo.addItem(f"{s}×{s}", userData=s)
        self._preset_combo.setCurrentIndex(2)  # 64×64
        self._preset_combo.setMinimumWidth(scaled(72))
        self._preset_combo.currentIndexChanged.connect(self._on_preset)
        f.addRow(T(QLabel(), "预设"), self._preset_combo)

        custom_row = QWidget()
        cr = QHBoxLayout(custom_row)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.setSpacing(scaled(4))
        self._custom_w = QSpinBox()
        self._custom_w.setRange(8, 1024)
        self._custom_w.setValue(64)
        self._custom_w.setMinimumWidth(scaled(52))
        cr.addWidget(self._custom_w, 1)
        cr.addWidget(QLabel("×"))
        self._custom_h = QSpinBox()
        self._custom_h.setRange(8, 1024)
        self._custom_h.setValue(64)
        self._custom_h.setMinimumWidth(scaled(52))
        cr.addWidget(self._custom_h, 1)
        f.addRow(T(QLabel(), "宽 × 高"), custom_row)

        self._bg_combo = QComboBox()
        for key, fill in _BG_FILLS.items():
            self._bg_combo.addItem(T(None, key), userData=key)
        self._bg_combo.setMinimumWidth(scaled(72))
        f.addRow(T(QLabel(), "背景"), self._bg_combo)

        self._btn_new = T(QPushButton(), "新建画布")
        self._btn_new.setObjectName("PrimaryButton")
        self._btn_new.clicked.connect(self._on_new)
        f.addRow("", self._btn_new)

        settings_docker = self._right_dock.add_docker("画布设置", box, icon_kind="grid", stretch=1)
        settings_docker.set_icon("grid")

        # ---- 画布信息 ----
        info = QWidget()
        iv = QVBoxLayout(info)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(scaled(4))
        self._info_label = QLabel()
        self._info_label.setObjectName("HintLabel")
        self._info_label.setWordWrap(True)
        iv.addWidget(self._info_label)
        hint = T(QLabel(), "提示：在左侧资源网格里双击资源 = 直接放入画布。")
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        iv.addWidget(hint)
        iv.addStretch(1)
        info_docker = self._right_dock.add_docker("画布信息", info, icon_kind="background")
        info_docker.set_icon("background")

        # ---- 导出（倍率 + 复制到剪贴板） ----
        export = QWidget()
        ev = QVBoxLayout(export)
        ev.setContentsMargins(0, 0, 0, 0)
        ev.setSpacing(scaled(6))
        scale_row = QWidget()
        sr = QHBoxLayout(scale_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(scaled(6))
        sr.addWidget(T(QLabel(), "导出倍率"))
        self._scale_combo = QComboBox()
        for factor in (1, 2, 4, 8):
            self._scale_combo.addItem(f"{factor}×", userData=factor)
        T(self._scale_combo, "按最近邻放大导出（像素画放大不失真）", attr="tooltip")
        sr.addWidget(self._scale_combo, 1)
        ev.addWidget(scale_row)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(scaled(6))
        self._btn_export_png = T(QPushButton(), "导出 PNG")
        self._btn_export_png.setObjectName("PrimaryButton")
        T(self._btn_export_png, "按所选倍率导出当前画布为 PNG", attr="tooltip")
        self._btn_export_png.clicked.connect(self._on_export)
        self._btn_copy = T(QPushButton(), "复制到剪贴板")
        self._btn_copy.setObjectName("ToolBtn")
        T(self._btn_copy, "把当前画布（含倍率）复制到系统剪贴板，可直接粘进别的软件", attr="tooltip")
        self._btn_copy.clicked.connect(self._on_copy_clipboard)
        rl.addWidget(self._btn_export_png, 1)
        rl.addWidget(self._btn_copy, 1)
        ev.addWidget(row)
        ev.addStretch(1)
        export_docker = self._right_dock.add_docker("导出", export, icon_kind="export_image")
        export_docker.set_icon("export_image")

    # ------------------------------------------------------------------ #
    # 布局持久化
    # ------------------------------------------------------------------ #
    def _restore_layout(self) -> None:
        """恢复上次的停靠栏宽度与收起状态。"""
        try:
            sizes = self._ctx.ui_settings.get("pixel_dock_sizes") or []
            if isinstance(sizes, (list, tuple)) and len(sizes) == 3:
                sizes = [int(v) for v in sizes]
                # 兜底：保存的是「画布被挤扁」的病态布局（历史版本可能写入）时回退默认
                if sizes[1] < scaled(240):
                    logger.info("像素页保存的布局过窄（画布 %s px），改用默认宽度", sizes[1])
                    sizes = None
            if sizes:
                self._splitter.setSizes(sizes)
                # 重新绑定「上次宽度」，这样展开/首次显示时按用户拖过的宽度还原
                if sizes[0] > scaled(28):
                    self._left_dock.bind_splitter(self._splitter, 0, default_width=sizes[0])
                if sizes[2] > scaled(28):
                    self._right_dock.bind_splitter(self._splitter, 2, default_width=sizes[2])
            if self._ctx.ui_settings.get("pixel_left_collapsed"):
                self._left_dock.set_collapsed(True)
            if self._ctx.ui_settings.get("pixel_right_collapsed"):
                self._right_dock.set_collapsed(True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("像素页布局恢复失败: %s", exc)

    def _remember_layout(self) -> None:
        try:
            s = self._ctx.ui_settings
            s.set("pixel_dock_sizes", list(self._splitter.sizes()))
            s.set("pixel_left_collapsed", self._left_dock.is_collapsed())
            s.set("pixel_right_collapsed", self._right_dock.is_collapsed())
        except Exception as exc:  # noqa: BLE001
            logger.warning("像素页布局保存失败: %s", exc)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._remember_layout()
        super().hideEvent(event)

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
        self._fit_view()
        self._refresh_info()

    def set_image(self, img) -> None:
        """外部（主窗口/IDE）导入图片到画布。"""
        self._editor.set_frame(img.convert("RGBA"))
        self._status(tr("已载入 {w}×{h} 图片").format(w=img.width, h=img.height))
        self._fit_view()
        self._refresh_info()

    def image(self):
        return self._editor.frame()

    def _refresh_info(self) -> None:
        """刷新工具条尺寸标识与「画布信息」面板。"""
        try:
            frame = self._editor.frame().convert("RGBA")
        except Exception:  # noqa: BLE001
            return
        w, h = frame.size
        counts = frame.getcolors(maxcolors=65536)
        colors = len(counts) if counts is not None else -1
        self.workspace_status_changed.emit(f"{w} × {h}")
        if hasattr(self, "_info_label"):
            text = tr("尺寸：{0} × {1} 像素").format(w, h) + "\n"
            text += tr("颜色数：{0}").format(colors if colors >= 0 else tr("过多"))
            self._info_label.setText(text)

    def apply_ui_scale(self, scale: float) -> None:
        """按界面比例同步缩放内部像素编辑器控件与停靠栏。"""
        self._editor.apply_ui_scale(scale)
        self._left_dock.apply_ui_scale()
        self._right_dock.apply_ui_scale()
        self._splitter.setHandleWidth(scaled(5))

    # ------------------------------------------------------------------ #
    def _on_import(self) -> None:
        """从本地导入图片（复用编辑器导入逻辑）。"""
        self._editor.import_image()
        self._refresh_info()

    def _on_sync_from_ide(self) -> None:
        self.sync_from_ide_requested.emit()

    def _on_sync_to_ide(self) -> None:
        self.sync_to_ide.emit(self.image())

    def _on_use_as_video(self) -> None:
        self.use_as_video_first_frame.emit(self.image())

    def export_scale(self) -> int:
        """导出倍率（1/2/4/8 倍最近邻放大）。"""
        try:
            return int(self._scale_combo.currentData() or 1)
        except Exception:  # noqa: BLE001
            return 1

    def _scaled_image(self) -> Image.Image:
        img = self.image().convert("RGBA")
        factor = self.export_scale()
        if factor > 1:
            img = img.resize((img.width * factor, img.height * factor), Image.Resampling.NEAREST)
        return img

    def _on_copy_clipboard(self) -> None:
        """把画布复制到系统剪贴板（含倍率）。"""
        try:
            from ui.widgets.tilemap_view import pil_to_qpixmap
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setPixmap(pil_to_qpixmap(self._scaled_image()))
            self._status(tr("已复制到剪贴板（{0}×）").format(self.export_scale()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("复制到剪贴板失败: %s", exc)

    def _on_export(self) -> None:
        base = Path(self._ctx.ui_settings.get("output_dir") or str(DEFAULT_OUTPUT_DIR))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        factor = self.export_scale()
        suffix = f"@{factor}x" if factor > 1 else ""
        path, _ = QFileDialog.getSaveFileName(
            self, tr("导出 PNG"), str(base / f"pixel_{ts}{suffix}.png"),
            tr("PNG 图片 (*.png)"),
        )
        if not path:
            return
        self._scaled_image().save(path, format="PNG")
        self._status(f"{tr('已导出：')}{path}")

    # ------------------------------------------------------------------ #
    def _status(self, message: str) -> None:
        try:

            win = self.window()
            if win is not None and hasattr(win, "statusBar"):
                win.statusBar().showMessage(message)
        except Exception:  # noqa: BLE001
            pass
