"""主窗口：Krita 风格外壳（菜单栏 + 上下文工具条 + 模式侧栏 + 页面堆栈 + 状态栏）。

布局（参考 Krita）：
- **菜单栏**：文件 / 编辑 / 视图 / 工作区 / 帮助，覆盖项目、撤销、主题、界面比例、模式切换；
- **工具条**：左边是当前工作区标识，右边是**当前页面提供的动作**（各页面实现
  ``toolbar_actions()``，主窗口在模式切换时重建），因此工具条总与当前工作区相关；
- **左侧模式栏**：Solo / IDE / 精灵图 / 像素 / 瓦片地图 五格 + IDE 六步导航；
- **中间页面堆栈** + **底部状态栏**（右侧常驻显示界面比例与主题）。

IDE 模式左侧栏展开显示 6 个步骤按钮；其它模式内缩只留模式开关。
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from config.settings import APP_DISPLAY_NAME, APP_FULL_NAME, APP_NAME_ZH, APP_VERSION, bundle_root
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

logger = logging.getLogger("PixelFoundry.ui.main_window")

# 侧边栏两种宽度：Solo/精灵图/像素/瓦片 内缩 / IDE 展开
RAIL_COLLAPSED = 128
RAIL_EXPANDED = 200

NAV_BUTTON_SIZE = 44
NAV_ICON_SIZE = max(16, int(NAV_BUTTON_SIZE * 0.45))
STEP_ICON_SIZE = 18
TOOLBAR_ICON_SIZE = 17

# IDE 步骤短名（按钮文字）
STEP_SHORT_ZH = ["文本", "图片", "动画", "像素", "背景", "导出"]

#: 工作区（模式）定义：(键, 中文名, 图标 kind, 图标来源)
WORKSPACES = (
    ("solo", "Solo 一键生成", "solo", "nav"),
    ("ide", "IDE 分步工作区", "ide", "nav"),
    ("sprite", "精灵图", "layers", "editor"),
    ("pixel", "独立像素画布", "grid", "editor"),
    ("tilemap", "瓦片地图", "tiles", "editor"),
)

#: 界面比例档位
UI_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25, 1.5)


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
        self.resize(1280, 820)
        self.setMinimumSize(960, 600)
        self._build_ui()
        self._build_menus()
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
        # 名称一起进 i18n（英文界面下中文名也要跟着换）
        T(self._logo_label, f"{APP_FULL_NAME}\n{APP_NAME_ZH}", attr="tooltip")
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
        self.pixel_page.workspace_status_changed.connect(self._on_workspace_status)
        self._stack.addWidget(self.solo_page)  # index 0
        self._stack.addWidget(self.ide_page)   # index 1
        self._stack.addWidget(self.sprite_page)  # index 2
        self._stack.addWidget(self.pixel_page)   # index 3
        self._stack.addWidget(self.tilemap_page)  # index 4
        layout.addWidget(self._stack, 1)

        self.setCentralWidget(central)

        # ---------- 上下文工具条（Krita 风格：只在菜单栏下方，随工作区变化） ----------
        self._toolbar = QToolBar(tr("主工具条"))
        self._toolbar.setObjectName("AppToolBar")
        self._toolbar.setMovable(False)
        self._toolbar.setIconSize(QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE))
        self._toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toolbar_widgets: list = []   # 工具条里自建的控件（重建时回收）
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        # ---------- 状态栏：右侧常驻显示界面比例 / 主题 ----------
        self._status_info = QLabel()
        self._status_info.setObjectName("HintLabel")
        self.statusBar().addPermanentWidget(self._status_info)
        self.statusBar().showMessage(tr("就绪"))

    # ------------------------------------------------------------------ #
    # 菜单栏（Krita 风格：文件 / 编辑 / 视图 / 工作区 / 帮助）
    # ------------------------------------------------------------------ #
    def _build_menus(self) -> None:
        """重建菜单栏。

        每次重建前先把旧菜单**彻底回收**（``clear()`` 只是摘掉，QMenu 仍是菜单栏的子控件），
        否则语言切换会不断累积菜单对象。
        """
        bar = self.menuBar()
        bar.clear()
        for menu in getattr(self, "_menus", []):
            menu.setParent(None)
            menu.deleteLater()
        self._menus = []
        if not hasattr(self, "_scale_group"):
            self._scale_group = QActionGroup(self)
            self._scale_group.setExclusive(True)
        if not hasattr(self, "_ws_group"):
            self._ws_group = QActionGroup(self)
            self._ws_group.setExclusive(True)

        def _menu(zh: str) -> QMenu:
            menu = bar.addMenu(T(None, zh))
            self._menus.append(menu)
            return menu

        # ---------------- 文件 ----------------
        m_file = _menu("文件")
        self._menu_action(m_file, "新建像素画布", "新建 {0}×{0} 像素画布", self._menu_new_canvas,
                                                 QKeySequence("Ctrl+N"))
        self._menu_action(m_file, "打开项目…", "打开 IDE 项目", self._menu_open_project,
                                           QKeySequence("Ctrl+O"))
        self._menu_action(m_file, "保存项目", "保存 IDE 项目", self._menu_save_project,
                                           QKeySequence("Ctrl+S"))
        m_file.addSeparator()
        self._menu_action(m_file, "导出…", "跳到 IDE 的导出步骤并导出", self._menu_export)
        self._menu_action(m_file, "打开输出目录", "在文件管理器里打开输出目录",
                                                  self._menu_open_output)
        m_file.addSeparator()
        self._menu_action(m_file, "退出", "退出程序", self.close,
                                           QKeySequence("Ctrl+Q"))

        # ---------------- 编辑 ----------------
        m_edit = _menu("编辑")
        self._menu_action(m_edit, "撤销", "撤销上一步像素编辑", self._menu_undo,
                                           QKeySequence("Ctrl+Z"))
        self._menu_action(m_edit, "重做", "重做被撤销的像素编辑", self._menu_redo,
                                           QKeySequence("Ctrl+Shift+Z"))
        m_edit.addSeparator()
        self._menu_action(m_edit, "复制选区", "把当前选区复制到剪贴板", self._menu_copy,
                                           QKeySequence("Ctrl+C"))
        self._menu_action(m_edit, "粘贴为浮动图层", "把剪贴板内容粘贴成半透明浮动图层",
                                            self._menu_paste, QKeySequence("Ctrl+V"))
        self._menu_action(m_edit, "合并浮动图层", "把浮动图层合并进当前帧", self._menu_merge,
                                            QKeySequence("Ctrl+M"))

        # ---------------- 视图 ----------------
        m_view = _menu("视图")
        self._menu_action(m_view, "切换深色 / 浅色主题", "在深色与浅色主题之间切换",
                                            self._on_toggle_theme)
        m_scale = m_view.addMenu(T(None, "界面比例"))
        self._scale_actions = {}
        for value in UI_SCALES:
            act = QAction(f"{int(value * 100)}%", m_scale)
            act.setCheckable(True)
            act.setChecked(abs(value - self._scale) < 0.01)
            act.triggered.connect(lambda _=False, v=value: self._set_ui_scale(v))
            self._scale_group.addAction(act)
            m_scale.addAction(act)
            self._scale_actions[value] = act
        m_view.addSeparator()
        self._menu_action(m_view, "全屏", "切换全屏显示", self._menu_fullscreen,
                                                 QKeySequence("F11"))
        self._menu_action(m_view, "重置面板布局", "恢复默认的停靠面板宽度",
                                                   self._menu_reset_layout)

        # ---------------- 工作区 ----------------
        m_ws = _menu("工作区")
        self._ws_actions = {}
        for key, zh, icon_kind, source in WORKSPACES:
            act = QAction(T(None, zh), m_ws)
            act.setCheckable(True)
            act.setChecked(key == "solo")
            act.triggered.connect(lambda _=False, k=key: self.set_mode(k))
            self._ws_group.addAction(act)
            m_ws.addAction(act)
            self._ws_actions[key] = act

        # ---------------- 帮助 ----------------
        m_help = _menu("帮助")
        self._menu_action(m_help, "快捷键设置…", "打开设置里的快捷键面板",
                                                lambda: self.open_settings(category=4))
        self._menu_action(m_help, "打开使用文档", "用系统默认程序打开 README", self._menu_readme)
        m_help.addSeparator()
        self._menu_action(m_help, "关于", "版本与项目信息", self._menu_about)

    def _menu_action(self, menu: QMenu, zh: str, tip: str, slot, shortcut=None) -> QAction:
        """建一个带图标/提示/快捷键的菜单项（挂在菜单上，随菜单一起回收）。"""
        act = QAction(T(None, zh), menu)
        act.setToolTip(T(None, tip))
        act.triggered.connect(slot)
        if shortcut is not None:
            act.setShortcut(shortcut)
        menu.addAction(act)
        return act

    # ------------------------------------------------------------------ #
    # 工具条：工作区标识 + 当前页面动作（Krita 的上下文工具条）
    # ------------------------------------------------------------------ #
    def _rebuild_toolbar(self) -> None:
        """重建工具条内容。

        **关键**：所有控件都必须以工具条为父控件。``QToolBar.addWidget()`` 内部走
        ``QWidgetAction.setDefaultWidget()``，会先 ``show()`` 再接管父子关系；若控件此时
        还没有父控件，Qt 会把它当成顶层窗口显示一帧 —— 在 Windows 上就是「切换页面时
        弹出一个空白窗口又立刻消失」。旧控件也要显式回收，否则每次切换都会留下游离控件。
        """
        self._toolbar.clear()
        for widget in self._toolbar_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._toolbar_widgets = []

        fg = theme_fg(self._current_theme())

        chip = QLabel(f"  {self._workspace_name()}  ", self._toolbar)
        chip.setObjectName("WorkspaceTitle")
        self._add_toolbar_widget(chip)
        self._toolbar.addSeparator()

        page = self._current_page()
        actions = []
        getter = getattr(page, "toolbar_actions", None)
        if callable(getter):
            try:
                actions = list(getter() or [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("工具条动作构建失败: %s", exc)
        for item in actions:
            kind, text, tip, slot, primary = (list(item) + [False])[:5]
            btn = QToolButton(self._toolbar)
            btn.setObjectName("ToolbarButtonPrimary" if primary else "ToolbarButton")
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setIcon(editor_icon(kind, "#ffffff" if primary else fg.name(), size=TOOLBAR_ICON_SIZE))
            btn.setIconSize(QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE))
            # 工具条每次切换工作区都会重建：直接用 tr() 取文案即可（无需注册到 i18n 注册表）
            btn.setText(tr(text))
            btn.setToolTip(tr(tip))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, fn=slot: fn())
            self._add_toolbar_widget(btn)

        spacer = QWidget(self._toolbar)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._add_toolbar_widget(spacer)

        status = ""
        getter = getattr(page, "workspace_status", None)
        if callable(getter):
            try:
                status = str(getter() or "")
            except Exception:  # noqa: BLE001
                status = ""
        self._toolbar_status = QLabel(status, self._toolbar)
        self._toolbar_status.setObjectName("InfoChip")
        self._toolbar_status.setVisible(bool(status))
        self._add_toolbar_widget(self._toolbar_status)

    def _add_toolbar_widget(self, widget: QWidget) -> None:
        """加入工具条并登记，便于下次重建时回收。"""
        self._toolbar.addWidget(widget)
        self._toolbar_widgets.append(widget)

    def _workspace_name(self) -> str:
        for key, zh, _icon, _src in WORKSPACES:
            if key == self._mode:
                return tr(zh)
        return tr("工作区")

    def _current_page(self) -> QWidget:
        return self._stack.currentWidget()

    def _on_workspace_status(self, text: str) -> None:
        """页面状态标识变化（如画布分辨率）-> 更新工具条右侧胶囊。"""
        if not hasattr(self, "_toolbar_status"):
            return
        self._toolbar_status.setText(text or "")
        self._toolbar_status.setVisible(bool(text))

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
        if hasattr(self, "_ws_actions"):
            self._ws_actions[self._mode].setChecked(True)
        hints = {
            "solo": tr("Solo 模式 — 一键生成"),
            "ide": tr("IDE 模式 — 分步工作区"),
            "sprite": tr("精灵图模式 — 文生图网格精灵图"),
            "pixel": tr("像素模式 — 独立像素画布"),
            "tilemap": tr("瓦片地图模式 — 文生瓦片集与地图铺设"),
        }
        self.statusBar().showMessage(hints[self._mode])
        self._refresh_icons(self._current_theme())
        if hasattr(self, "_toolbar"):
            self._rebuild_toolbar()

    def switch_page(self, key: str) -> None:
        """兼容旧接口：按 key 切换模式。"""
        self.set_mode(key if key in ("solo", "ide", "sprite", "pixel", "tilemap") else "solo")

    def _on_step_clicked(self, index: int) -> None:
        self.set_mode("ide")
        self._step_buttons[index].setChecked(True)
        self.ide_page.set_current_step(index)

    # ------------------------------------------------------------------ #
    # 菜单动作实现
    # ------------------------------------------------------------------ #
    def _active_editor(self):
        """当前工作区里的像素编辑器（没有则返回 None）。"""
        page = self._current_page()
        for key in ("_editor",):
            editor = getattr(page, key, None)
            if editor is not None and hasattr(editor, "undo"):
                return editor
        return None

    @staticmethod
    def _focused_text_widget():
        """当前获得焦点的文本输入控件（有的话，编辑菜单要让位给它）。"""
        w = QApplication.focusWidget()
        if w is None:
            return None
        for cls in (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox):
            if isinstance(w, cls):
                return w
        return None

    def _menu_new_canvas(self) -> None:
        self.set_mode("pixel")
        self.pixel_page._on_new()

    def _menu_open_project(self) -> None:
        self.set_mode("ide")
        self.ide_page._on_open_project()

    def _menu_save_project(self) -> None:
        self.set_mode("ide")
        self.ide_page._on_save_project()

    def _menu_export(self) -> None:
        self.set_mode("ide")
        self._step_buttons[5].setChecked(True)
        self.ide_page.set_current_step(5)

    def _menu_open_output(self) -> None:
        self.ide_page._on_open_output()

    def _menu_undo(self) -> None:
        text = self._focused_text_widget()
        if text is not None:
            text.undo()
            return
        editor = self._active_editor()
        if editor is not None:
            editor.undo()

    def _menu_redo(self) -> None:
        text = self._focused_text_widget()
        if text is not None:
            text.redo()
            return
        editor = self._active_editor()
        if editor is not None:
            editor.redo()

    def _menu_copy(self) -> None:
        text = self._focused_text_widget()
        if text is not None:
            text.copy()
            return
        editor = self._active_editor()
        if editor is not None and hasattr(editor, "copy_selection"):
            editor.copy_selection()

    def _menu_paste(self) -> None:
        text = self._focused_text_widget()
        if text is not None:
            text.paste()
            return
        editor = self._active_editor()
        if editor is not None and hasattr(editor, "paste_layer"):
            editor.paste_layer()

    def _menu_merge(self) -> None:
        if self._focused_text_widget() is not None:
            return
        editor = self._active_editor()
        if editor is not None and hasattr(editor, "merge_float_layer"):
            editor.merge_float_layer()

    def _menu_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _menu_reset_layout(self) -> None:
        """恢复默认面板宽度（像素页三栏）。"""
        page = self.pixel_page
        try:
            page._splitter.setSizes([ui_layout.scaled(250), ui_layout.scaled(860), ui_layout.scaled(208)])
            page._left_dock.set_collapsed(False)
            page._right_dock.set_collapsed(False)
            page._remember_layout()
        except Exception as exc:  # noqa: BLE001
            logger.warning("重置面板布局失败: %s", exc)
        self.statusBar().showMessage(tr("已重置面板布局"))

    def _menu_readme(self) -> None:
        for name in ("README_CN.md", "README.md"):
            path = bundle_root() / name
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
                return
        self.statusBar().showMessage(tr("未找到 README 文件"))

    def _menu_about(self) -> None:
        QMessageBox.about(
            self, f"{tr('关于')} {APP_DISPLAY_NAME}",
            f"<b>{APP_FULL_NAME}</b> · {tr(APP_NAME_ZH)}<br/>"
            f"{tr('版本')} v{APP_VERSION}<br/><br/>"
            f"{tr('从文本到像素资产的一站式桌面工作台：文生图 → 动画 → 严格像素化 → 抠图 → GIF / APNG / 序列帧 / 雪碧图。')}<br/>"
            f"{tr('界面参考 Krita 的停靠面板与工作区布局；MIT 许可，开源免费。')}",
        )

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
    def _set_ui_scale(self, value: float) -> None:
        """菜单里选择界面比例 -> 写入设置并立即生效。"""
        self._ctx.ui_settings.set("ui_scale", float(value))
        self._apply_ui_scale()

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
        self._toolbar.setIconSize(QSize(ui_layout.scaled(TOOLBAR_ICON_SIZE), ui_layout.scaled(TOOLBAR_ICON_SIZE)))
        self._refresh_icons(self._current_theme())
        self.set_mode(self._mode)
        for page in (self.solo_page, self.ide_page, self.sprite_page, self.pixel_page, self.tilemap_page):
            if hasattr(page, "apply_ui_scale"):
                page.apply_ui_scale(self._scale)
        self._update_status_info()

    def open_settings(self, category: int = None) -> None:
        """弹出设置对话框（左分类 / 右表单）；category 可指定初始分类。"""
        dialog = SettingsDialog(self._ctx, self)
        if category is not None:
            try:
                dialog._cat_list.setCurrentRow(category)
            except Exception:  # noqa: BLE001
                pass
        dialog.exec()
        self._apply_theme(self._current_theme())
        self._apply_ui_scale()

    def retranslate_ui(self) -> None:
        """语言切换后立即重刷所有已注册文本并更新状态栏。"""
        retranslate_all()
        self.menuBar().clear()
        self._build_menus()
        self._settings_btn.setToolTip(tr("设置 — API 配置 / 常规"))
        self._theme_btn.setToolTip(tr("切换主题（深色 / 浅色）"))
        for i, btn in self._step_buttons.items():
            btn.setText(tr(STEP_SHORT_ZH[i]) if i < len(STEP_SHORT_ZH) else "")
        # IDE 执行按钮按当前步骤重刷
        if hasattr(self.ide_page, "_current_step"):
            self.ide_page.set_current_step(self.ide_page._current_step)
        # 常驻下拉框项（倍速等）随语言重刷
        for page in (self.solo_page, self.ide_page, self.sprite_page, self.pixel_page, self.tilemap_page):
            if hasattr(page, "retranslate_ui"):
                page.retranslate_ui()
        self.set_mode(self._mode)
        self._refresh_icons(self._current_theme())
        self._update_status_info()   # 状态栏右侧的动态文案（界面比例 / 主题）

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
        self._update_status_info()

    def _refresh_icons(self, theme: str) -> None:
        fg = theme_fg(theme)
        self._settings_btn.setIcon(nav_icon("settings", fg, size=NAV_ICON_SIZE))
        self._theme_btn.setIcon(theme_icon(theme, fg, size=NAV_ICON_SIZE))
        for i, btn in self._step_buttons.items():
            btn.setIcon(step_icon(i, theme_fg(theme, active=btn.isChecked()), size=STEP_ICON_SIZE))
        if hasattr(self, "_toolbar"):
            self._rebuild_toolbar()

    def _update_status_info(self) -> None:
        """状态栏右侧常驻信息：界面比例 + 主题。"""
        theme_zh = "深色" if self._current_theme() == "dark" else "浅色"
        self._status_info.setText(
            f"{tr('界面')} {int(round(self._scale * 100))}% · {tr(theme_zh)}"
        )

    def _on_toggle_theme(self) -> None:
        new_theme = "light" if self._current_theme() == "dark" else "dark"
        self._apply_theme(new_theme)
