"""主窗口：左上角 Solo/IDE 模式开关 + 自适应侧边栏 + 页面堆栈 + 设置弹窗 + 主题。

设计（参考 Trae 的 Builder/Chat 切换）：
- 左上角一个圆角分段开关：「Solo | IDE」，左 = Solo 模式，右 = IDE 模式。
- Solo 模式：左侧导航栏内缩（仅保留开关 / 设置 / 主题，步骤导航隐藏）。
- IDE 模式：导航栏展开，显示 6 个流程步骤按钮（图标 + 文字，DSH 扁平风格）。
- 设置（齿轮）按钮弹出设置对话框；底部月/日切换主题。
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from config.settings import APP_DISPLAY_NAME, APP_VERSION
from core.workflow import IDE_STEPS
from ui import layout as ui_layout
from ui import shortcuts as sc
from ui.app_context import AppContext
from ui.dialogs.settings_dialog import SettingsDialog
from ui.i18n import T, retranslate_all, tr
from ui.icons import editor_icon, logo_icon, nav_icon, step_icon, theme_fg, theme_icon
from ui.pages.ide_page import IdePage
from ui.pages.pixel_page import PixelPage
from ui.pages.solo_page import SoloPage
from ui.pages.sprite_page import SpritePage
from ui.pages.tilemap_page import TilemapPage
from ui.styles import apply_theme
from ui.widgets.segmented_toggle import SegmentedToggle

logger = logging.getLogger("PixelAnimIDE.ui.main_window")

# 侧边栏两种宽度：Solo/精灵图/像素 内缩 / IDE 展开
RAIL_COLLAPSED = 128
RAIL_EXPANDED = 200

NAV_BUTTON_SIZE = 44
NAV_ICON_SIZE = max(16, int(NAV_BUTTON_SIZE * 0.45))
STEP_ICON_SIZE = 18

# IDE 步骤短名（按钮文字）
STEP_SHORT_ZH = ["文本", "图片", "动画", "像素", "背景", "导出"]
STEP_SHORT = [tr(s) for s in STEP_SHORT_ZH]


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._mode = "solo"
        self._scale = 1.0
        self._base_font_size = QApplication.font().pointSizeF() or 10.0
        # 先设置全局比例，让所有固定尺寸（按钮/面板/图标）按比例构建
        self._scale = max(0.7, min(1.6, float(ctx.ui_settings.get("ui_scale", 1.0))))
        ui_layout.set_ui_scale(self._scale)
        # 载入用户自定义快捷键（像素编辑器等按键绑定生效）
        sc.set_shortcuts(ctx.ui_settings.get("shortcuts"))
        self.setWindowTitle(f"{APP_DISPLAY_NAME} v{APP_VERSION}")
        self.resize(1200, 780)
        self.setMinimumSize(960, 600)
        self._build_ui()
        self._apply_saved_theme()
        self._apply_ui_scale()
        self.set_mode("solo")

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---------- 左侧导航栏 ----------
        self._sidebar = QWidget()
        self._sidebar.setObjectName("Sidebar")
        self._sidebar.setFixedWidth(RAIL_COLLAPSED)
        sb = QVBoxLayout(self._sidebar)
        # 水平留 8px 内边距：按钮圆角不被裁剪、不压到侧栏右边框（避免断线/遮挡）
        sb.setContentsMargins(8, 12, 8, 12)
        sb.setSpacing(4)

        self._logo_label = QLabel()
        self._logo_label.setPixmap(logo_icon().pixmap(28, 28))
        self._logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._logo_label.setToolTip("PixelAnimIDE")
        sb.addWidget(self._logo_label, 0, Qt.AlignmentFlag.AlignHCenter)
        sb.addSpacing(8)

        # ---------- 模式开关（2×2 图标排列，横向填满侧栏、两边无缝隙） ----------
        self._mode_switch = QFrame()
        self._mode_switch.setObjectName("ModeSwitch")
        ms = QGridLayout(self._mode_switch)
        ms.setContentsMargins(0, 0, 0, 0)
        ms.setSpacing(4)
        ms.setColumnStretch(0, 1)
        ms.setColumnStretch(1, 1)

        def _mode_btn(icon_fn, tip):
            btn = QToolButton()
            btn.setObjectName("ModeSegment")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tip)
            btn.setFixedHeight(34)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setIcon(icon_fn("#9aa0a8", 20))
            btn.setIconSize(QSize(20, 20))
            return btn

        self._mode_solo_btn = _mode_btn(lambda c, s: nav_icon("solo", c, s), T(None, "Solo — 一键生成"))
        T(self._mode_solo_btn, "Solo — 一键生成", attr="tooltip")
        self._mode_solo_btn.clicked.connect(lambda: self.set_mode("solo"))

        self._mode_ide_btn = _mode_btn(lambda c, s: nav_icon("ide", c, s), T(None, "IDE — 分步工作区"))
        T(self._mode_ide_btn, "IDE — 分步工作区", attr="tooltip")
        self._mode_ide_btn.clicked.connect(lambda: self.set_mode("ide"))

        self._mode_sprite_btn = _mode_btn(lambda c, s: editor_icon("layers", c, s), T(None, "精灵图 — 文生图网格精灵图"))
        T(self._mode_sprite_btn, "精灵图 — 文生图网格精灵图", attr="tooltip")
        self._mode_sprite_btn.clicked.connect(lambda: self.set_mode("sprite"))

        self._mode_pixel_btn = _mode_btn(lambda c, s: editor_icon("grid", c, s), T(None, "像素 — 独立像素画布"))
        T(self._mode_pixel_btn, "像素 — 独立像素画布", attr="tooltip")
        self._mode_pixel_btn.clicked.connect(lambda: self.set_mode("pixel"))

        self._mode_tilemap_btn = _mode_btn(lambda c, s: editor_icon("tiles", c, s), T(None, "瓦片地图 — 文生瓦片集与地图铺设"))
        T(self._mode_tilemap_btn, "瓦片地图 — 文生瓦片集与地图铺设", attr="tooltip")
        self._mode_tilemap_btn.clicked.connect(lambda: self.set_mode("tilemap"))

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self._mode_solo_btn)
        self._mode_group.addButton(self._mode_ide_btn)
        self._mode_group.addButton(self._mode_sprite_btn)
        self._mode_group.addButton(self._mode_pixel_btn)
        self._mode_group.addButton(self._mode_tilemap_btn)

        ms.addWidget(self._mode_solo_btn, 0, 0)
        ms.addWidget(self._mode_ide_btn, 0, 1)
        ms.addWidget(self._mode_sprite_btn, 1, 0)
        ms.addWidget(self._mode_pixel_btn, 1, 1)
        ms.addWidget(self._mode_tilemap_btn, 2, 0, 1, 2)  # 第 5 模式：占满第三行
        sb.addWidget(self._mode_switch)  # 无对齐 -> 横向填满侧栏
        sb.addSpacing(8)

        # ---------- IDE 步骤导航（仅 IDE 模式展开） ----------
        self._step_nav = QWidget()
        self._step_nav.setObjectName("StepNav")
        sn = QVBoxLayout(self._step_nav)
        sn.setContentsMargins(6, 0, 6, 0)
        sn.setSpacing(4)
        self._step_buttons: dict = {}
        self._step_group = QButtonGroup(self)
        self._step_group.setExclusive(True)
        for i, full in enumerate(IDE_STEPS):
            btn = QPushButton(tr(STEP_SHORT_ZH[i]) if i < len(STEP_SHORT_ZH) else "")
            btn.setObjectName("StepButton")
            btn.setCheckable(True)
            btn.setFixedHeight(34)
            btn.setIconSize(QSize(STEP_ICON_SIZE, STEP_ICON_SIZE))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tr(full))
            btn.clicked.connect(lambda _=False, idx=i: self._on_step_clicked(idx))
            self._step_group.addButton(btn)
            self._step_buttons[i] = btn
            sn.addWidget(btn)
        self._step_nav.setVisible(False)
        sb.addWidget(self._step_nav)

        # ---------- 精灵图执行方式开关（左=自动 / 右=手动，点击切换；仅精灵图模式显示） ----------
        self._sprite_switch = QWidget()
        self._sprite_switch.setObjectName("SpriteModeSwitch")
        sw = QVBoxLayout(self._sprite_switch)
        sw.setContentsMargins(2, 4, 2, 4)
        sw.setSpacing(3)
        sw_caption = QLabel(tr("执行方式"))
        sw_caption.setObjectName("SidebarCaption")
        sw_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sw.addWidget(sw_caption)
        self._sprite_toggle = SegmentedToggle(height=30)
        self._sprite_toggle.toggled.connect(self._on_sprite_toggle)
        sw.addWidget(self._sprite_toggle, 0, Qt.AlignmentFlag.AlignHCenter)
        self._sprite_switch.setVisible(False)
        sb.addWidget(self._sprite_switch)

        sb.addStretch(1)

        # ---------- 设置 + 主题 ----------
        self._settings_btn = QPushButton()
        self._settings_btn.setObjectName("NavButton")
        self._settings_btn.setFixedSize(NAV_BUTTON_SIZE, NAV_BUTTON_SIZE)
        self._settings_btn.setIconSize(QSize(NAV_ICON_SIZE, NAV_ICON_SIZE))
        self._settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_btn.setToolTip(tr("设置 — API 配置 / 常规"))
        self._settings_btn.clicked.connect(self.open_settings)
        sb.addWidget(self._settings_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self._theme_btn = QPushButton()
        self._theme_btn.setObjectName("NavButton")
        self._theme_btn.setFixedSize(NAV_BUTTON_SIZE, NAV_BUTTON_SIZE)
        self._theme_btn.setIconSize(QSize(NAV_ICON_SIZE, NAV_ICON_SIZE))
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.setToolTip(tr("切换主题（深色 / 浅色）"))
        self._theme_btn.clicked.connect(self._on_toggle_theme)
        sb.addWidget(self._theme_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(self._sidebar)

        # ---------- 页面堆栈 ----------
        self._stack = QStackedWidget()
        self.solo_page = SoloPage(self._ctx)
        self.ide_page = IdePage(self._ctx)
        self.sprite_page = SpritePage(self._ctx)
        self.pixel_page = PixelPage(self._ctx)
        self.tilemap_page = TilemapPage(self._ctx)
        self.ide_page.step_changed.connect(self._on_ide_step_changed)
        self.solo_page.sync_to_ide.connect(self._on_sync_to_ide)
        self.sprite_page.sync_to_ide.connect(self._on_sync_sprite_to_ide)
        self.sprite_page.running_changed.connect(self._on_sprite_running_changed)
        self.pixel_page.sync_from_ide_requested.connect(self._on_pixel_sync_from_ide)
        self.pixel_page.sync_to_ide.connect(self._on_pixel_sync_to_ide)
        self.pixel_page.use_as_video_first_frame.connect(self._on_pixel_to_video)
        self._stack.addWidget(self.solo_page)  # index 0
        self._stack.addWidget(self.ide_page)   # index 1
        self._stack.addWidget(self.sprite_page)  # index 2
        self._stack.addWidget(self.pixel_page)   # index 3
        self._stack.addWidget(self.tilemap_page)  # index 4
        layout.addWidget(self._stack, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(tr("就绪"))

    # ------------------------------------------------------------------ #
    # 模式切换
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str) -> None:
        """切换 Solo / IDE / 精灵图 / 像素 / 瓦片地图 模式（分段开关 + 侧栏收展 + 页面切换）。"""
        self._mode = mode if mode in ("solo", "ide", "sprite", "pixel", "tilemap") else "solo"
        # 快捷键按当前工作模式生效（像素编辑器 / 预览等按键绑定）
        sc.set_active_mode(self._mode)
        self._mode_solo_btn.setChecked(self._mode == "solo")
        self._mode_ide_btn.setChecked(self._mode == "ide")
        self._mode_sprite_btn.setChecked(self._mode == "sprite")
        self._mode_pixel_btn.setChecked(self._mode == "pixel")
        self._mode_tilemap_btn.setChecked(self._mode == "tilemap")
        self._step_nav.setVisible(self._mode == "ide")
        self._sprite_switch.setVisible(self._mode == "sprite")
        self._sidebar.setFixedWidth(int((RAIL_EXPANDED if self._mode == "ide" else RAIL_COLLAPSED) * self._scale))
        index = {"solo": 0, "ide": 1, "sprite": 2, "pixel": 3, "tilemap": 4}[self._mode]
        self._stack.setCurrentIndex(index)
        hints = {
            "solo": tr("Solo 模式 — 一键生成"),
            "ide": tr("IDE 模式 — 分步工作区"),
            "sprite": tr("精灵图模式 — 文生图网格精灵图"),
            "pixel": tr("像素模式 — 独立像素画布"),
            "tilemap": tr("瓦片地图模式 — 文生瓦片集与地图铺设"),
        }
        self.statusBar().showMessage(hints[self._mode])
        self._refresh_icons(self._current_theme())

    def switch_page(self, key: str) -> None:
        """兼容旧接口：按 key 切换模式。"""
        self.set_mode(key if key in ("solo", "ide", "sprite", "pixel", "tilemap") else "solo")

    def _on_step_clicked(self, index: int) -> None:
        self.set_mode("ide")
        self._step_buttons[index].setChecked(True)
        self.ide_page.set_current_step(index)

    # ------------------------------------------------------------------ #
    # 精灵图执行方式开关（A=自动 / M=手动，点击切换；悬停提示信息）
    # ------------------------------------------------------------------ #
    def _on_sprite_toggle(self, checked: bool) -> None:
        """点击切换 -> 精灵图页面手动/自动模式。"""
        self.sprite_page.set_manual_mode(checked)

    def _on_sprite_running_changed(self, running: bool) -> None:
        """精灵图生成中禁用模式开关（运行中不允许切换自动/手动）。"""
        self._sprite_toggle.setEnabled(not running)

    def _on_ide_step_changed(self, index: int) -> None:
        if index in self._step_buttons:
            self._step_buttons[index].setChecked(True)
            self._refresh_icons(self._current_theme())

    def _on_sync_to_ide(self, result) -> None:
        """Solo 生成结果同步到 IDE 工作区并切换到 IDE 模式。"""
        self.ide_page.import_from_solo(result)
        self.set_mode("ide")
        self.statusBar().showMessage(tr("已同步 Solo 结果到 IDE"))

    def _on_sync_sprite_to_ide(self, result) -> None:
        """精灵图结果同步到 IDE 工作区并切换到 IDE 模式。"""
        self.ide_page.import_from_sprite(result)
        self.set_mode("ide")
        self.statusBar().showMessage(tr("已同步精灵图结果到 IDE"))

    # ------------------------------------------------------------------ #
    # 像素板块联动
    # ------------------------------------------------------------------ #
    def _on_pixel_sync_from_ide(self) -> None:
        """像素板块「从 IDE 同步」：拉 IDE 当前帧/首帧进画布。"""
        s = self.ide_page._session
        img = None
        if s.frames:
            idx = min(self.ide_page._current, len(s.frames) - 1)
            img = s.frames[idx]
        elif s.first_frame is not None:
            img = s.first_frame
        if img is None:
            self.statusBar().showMessage(tr("IDE 暂无帧可同步，请先在 IDE 生成或导入图片"))
            return
        self.pixel_page.set_image(img)
        self.statusBar().showMessage(tr("已从 IDE 同步当前帧到像素画布"))

    def _on_pixel_sync_to_ide(self, img) -> None:
        """像素板块「同步到 IDE」：画布图作为首帧 + 图生图参考。"""
        self.ide_page.set_first_frame(img)
        self.set_mode("ide")
        self.statusBar().showMessage(tr("已同步像素画布到 IDE（首帧 + 图生图参考）"))

    def _on_pixel_to_video(self, img) -> None:
        """像素板块「用作图生视频首帧」：画布图作为参考/首帧发给 Solo 走图生视频。"""
        self.solo_page.set_reference_image(img)
        self.set_mode("solo")
        self.statusBar().showMessage(tr("已设置图生视频首帧（Solo），点「开始生成」即可"))

    # ------------------------------------------------------------------ #
    # 界面布局比例（适配不同分辨率设备）
    # ------------------------------------------------------------------ #
    def _apply_ui_scale(self) -> None:
        """按设置的界面比例缩放字体与全部关键 UI 尺寸（文字 + UI 同步）。"""
        self._scale = max(0.7, min(1.6, float(self._ctx.ui_settings.get("ui_scale", 1.0))))
        ui_layout.set_ui_scale(self._scale)
        # 字体（文字大小）
        f = QApplication.font()
        f.setPointSizeF(self._base_font_size * self._scale)
        QApplication.setFont(f)
        # 侧栏导航 / logo / 步骤按钮 / 模式开关图标等固定尺寸
        self._logo_label.setPixmap(logo_icon().pixmap(ui_layout.scaled(28), ui_layout.scaled(28)))
        ns = ui_layout.scaled(NAV_BUTTON_SIZE)
        ni = max(16, ui_layout.scaled(NAV_ICON_SIZE))
        for b in (self._settings_btn, self._theme_btn):
            b.setFixedSize(ns, ns)
            b.setIconSize(QSize(ni, ni))
        si = ui_layout.scaled(STEP_ICON_SIZE)
        for b in self._step_buttons.values():
            b.setFixedHeight(ui_layout.scaled(34))
            b.setIconSize(QSize(si, si))
        mb = ui_layout.scaled(34)
        mi = ui_layout.scaled(20)
        for b in (self._mode_solo_btn, self._mode_ide_btn, self._mode_sprite_btn, self._mode_pixel_btn, self._mode_tilemap_btn):
            b.setFixedHeight(mb)
            b.setIconSize(QSize(mi, mi))
        self._refresh_icons(self._current_theme())
        self.set_mode(self._mode)
        if hasattr(self.ide_page, "apply_ui_scale"):
            self.ide_page.apply_ui_scale(self._scale)
        if hasattr(self.sprite_page, "apply_ui_scale"):
            self.sprite_page.apply_ui_scale(self._scale)
        if hasattr(self.pixel_page, "apply_ui_scale"):
            self.pixel_page.apply_ui_scale(self._scale)

    def open_settings(self) -> None:
        """弹出设置对话框（左分类 / 右表单）。"""
        dialog = SettingsDialog(self._ctx, self)
        dialog.exec()
        self._apply_theme(self._current_theme())
        self._apply_ui_scale()

    def retranslate_ui(self) -> None:
        """语言切换后立即重刷所有已注册文本并更新状态栏。"""
        retranslate_all()
        self._settings_btn.setToolTip(tr("设置 — API 配置 / 常规"))
        self._theme_btn.setToolTip(tr("切换主题（深色 / 浅色）"))
        for i, btn in self._step_buttons.items():
            btn.setText(tr(STEP_SHORT_ZH[i]) if i < len(STEP_SHORT_ZH) else "")
        # 精灵图执行方式开关文案（悬停提示由控件自身管理）
        # IDE 执行按钮按当前步骤重刷
        if hasattr(self.ide_page, "_current_step"):
            self.ide_page.set_current_step(self.ide_page._current_step)
        # 常驻下拉框项（倍速等）随语言重刷
        for page in (self.solo_page, self.ide_page, self.sprite_page):
            if hasattr(page, "retranslate_ui"):
                page.retranslate_ui()
        self.set_mode(self._mode)
        self._refresh_icons(self._current_theme())

    # ------------------------------------------------------------------ #
    # 主题与图标颜色
    # ------------------------------------------------------------------ #
    def _current_theme(self) -> str:
        return str(self._ctx.ui_settings.get("theme", "dark"))

    def _apply_saved_theme(self) -> None:
        self._apply_theme(self._current_theme())

    def _apply_theme(self, theme: str) -> None:
        apply_theme(QApplication.instance(), theme)
        self._ctx.ui_settings.set("theme", theme)
        self._sprite_toggle.setDark(theme == "dark")
        self._refresh_icons(theme)

    def _refresh_icons(self, theme: str) -> None:
        fg = theme_fg(theme)
        self._settings_btn.setIcon(nav_icon("settings", fg, size=NAV_ICON_SIZE))
        self._theme_btn.setIcon(theme_icon(theme, fg, size=NAV_ICON_SIZE))
        for i, btn in self._step_buttons.items():
            btn.setIcon(step_icon(i, theme_fg(theme, active=btn.isChecked()), size=STEP_ICON_SIZE))

    def _on_toggle_theme(self) -> None:
        new_theme = "light" if self._current_theme() == "dark" else "dark"
        self._apply_theme(new_theme)
