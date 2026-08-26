"""设置弹窗：点击按钮弹出；左侧分类导航（三类 API + 常规），右侧对应配置表单。"""
from __future__ import annotations

import logging

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config.settings import DATA_DIR, DEFAULT_OUTPUT_DIR, API_KIND_LABELS
from ui.app_context import AppContext
from ui.i18n import tr
from ui.icons import category_icon, theme_fg
from ui.styles import apply_theme
from ui.widgets.api_config_widget import ApiConfigWidget

logger = logging.getLogger("PixelAnimIDE.ui.settings_dialog")

_CATEGORIES = [
    ("llm", "通用文本 API"),
    ("image", "图片生成 API"),
    ("video", "图转视频 API"),
    ("general", "常规设置"),
]


class SettingsDialog(QDialog):
    """设置弹窗：左分类导航 / 右配置表单。"""

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self.setWindowTitle("设置")
        self.setMinimumSize(920, 700)
        self._build_ui()
        self._on_cat_changed(0)

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        body = QHBoxLayout()
        body.setSpacing(12)

        # ---------- 左：分类导航 ----------
        theme = self._ctx.ui_settings.get("theme", "dark")
        fg = theme_fg(theme)
        self._cat_list = QListWidget()
        self._cat_list.setObjectName("SettingsCatList")
        self._cat_list.setFixedWidth(200)
        for key, label in _CATEGORIES:
            item = QListWidgetItem(category_icon(key, fg), tr(label))
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setSizeHint(QSize(180, 38))
            self._cat_list.addItem(item)
        self._cat_list.currentRowChanged.connect(self._on_cat_changed)
        body.addWidget(self._cat_list)

        # ---------- 右：分类内容（滚动，防止字段过多溢出） ----------
        self._stack = QStackedWidget()
        self._api_widgets: dict = {}
        for kind in ("llm", "image", "video"):
            widget = ApiConfigWidget(self._ctx.api, kind)
            self._api_widgets[kind] = widget
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(widget)
            self._stack.addWidget(scroll)
        self._general_panel = self._build_general_panel()
        self._stack.addWidget(self._general_panel)
        body.addWidget(self._stack, 1)

        root.addLayout(body, 1)

        # ---------- 底部 ----------
        bottom = QHBoxLayout()
        self._status = QLabel("")
        self._status.setObjectName("HintLabel")
        bottom.addWidget(self._status)
        bottom.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
        save_btn.setText(tr("保存"))
        save_btn.clicked.connect(self._on_save)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        close_btn.setText(tr("关闭"))
        close_btn.clicked.connect(self._on_close)
        bottom.addWidget(buttons)
        root.addLayout(bottom)

    # ------------------------------------------------------------------ #
    def _build_general_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(12)

        ui_box = QGroupBox(tr("界面"))
        f = QFormLayout(ui_box)
        f.setContentsMargins(12, 18, 12, 12)
        self._theme_combo = QComboBox()
        self._theme_combo.addItem(tr("深色"), userData="dark")
        self._theme_combo.addItem(tr("浅色"), userData="light")
        theme = self._ctx.ui_settings.get("theme", "dark")
        idx = self._theme_combo.findData(theme)
        self._theme_combo.setCurrentIndex(idx if idx >= 0 else 0)
        f.addRow(tr("主题"), self._theme_combo)
        # 语言（中/英，切换后立即生效）
        from ui.i18n import available_languages

        self._lang_combo = QComboBox()
        for code, label in available_languages():
            self._lang_combo.addItem(label, userData=code)
        lang = self._ctx.ui_settings.get("language", "zh")
        idx = self._lang_combo.findData(lang)
        self._lang_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._lang_combo.setToolTip(tr("切换界面语言（重启后全局生效）"))
        f.addRow(tr("语言"), self._lang_combo)
        # 界面布局比例（适配不同分辨率设备）
        self._scale_combo = QComboBox()
        self._scale_combo.addItem(tr("小（0.8×）"), userData=0.8)
        self._scale_combo.addItem(tr("标准（1.0×）"), userData=1.0)
        self._scale_combo.addItem(tr("大（1.25×）"), userData=1.25)
        self._scale_combo.addItem(tr("特大（1.5×）"), userData=1.5)
        scale = float(self._ctx.ui_settings.get("ui_scale", 1.0))
        idx = self._scale_combo.findData(scale)
        self._scale_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self._scale_combo.setToolTip(tr("缩放界面字体与布局，适配高分辨率/小屏幕设备"))
        f.addRow(tr("界面布局比例"), self._scale_combo)
        v.addWidget(ui_box)

        out_box = QGroupBox(tr("输出"))
        of = QFormLayout(out_box)
        of.setContentsMargins(12, 18, 12, 12)
        out_row = QHBoxLayout()
        self._output_edit = QLineEdit(str(self._ctx.ui_settings.get("output_dir") or DEFAULT_OUTPUT_DIR))
        out_row.addWidget(self._output_edit, 1)
        btn = QPushButton(tr("浏览…"))
        btn.clicked.connect(self._on_browse_output)
        out_row.addWidget(btn)
        of.addRow(tr("默认输出目录"), out_row)
        v.addWidget(out_box)

        info = QLabel(f"{tr('数据目录：')}{DATA_DIR}")
        info.setObjectName("HintLabel")
        info.setWordWrap(True)
        v.addWidget(info)
        v.addStretch(1)
        return panel

    # ------------------------------------------------------------------ #
    def _on_cat_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)
        self._status.setText("")

    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr("选择输出目录"), self._output_edit.text())
        if path:
            self._output_edit.setText(path)

    def _save_settings(self) -> None:
        """保存常规设置：输出目录 / 界面比例 / 语言 / 主题（语言立即生效并同步主窗口）。"""
        out = self._output_edit.text().strip()
        if out:
            self._ctx.ui_settings.set("output_dir", out)
        self._ctx.ui_settings.set("ui_scale", float(self._scale_combo.currentData() or 1.0))
        lang = str(self._lang_combo.currentData() or "zh")
        self._ctx.ui_settings.set("language", lang)
        theme = self._theme_combo.currentData()
        current = self._ctx.ui_settings.get("theme", "dark")
        if theme != current:
            self._ctx.ui_settings.set("theme", theme)
            apply_theme(QApplication.instance(), theme)
            parent = self.parent()
            if parent is not None and hasattr(parent, "_apply_theme"):
                parent._apply_theme(theme)
        # 语言与布局立即生效
        from ui.i18n import set_language

        set_language(lang)
        parent = self.parent()
        if parent is not None and hasattr(parent, "retranslate_ui"):
            parent.retranslate_ui()

    def _on_save(self) -> None:
        """保存按钮：写入设置并立即生效。"""
        self._save_settings()
        self._status.setText(tr("已保存"))

    def _on_close(self) -> None:
        self._save_settings()
        self.accept()
