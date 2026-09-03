"""快捷键调整面板：类别（一级下拉）→ 条目（二级下拉）→ 当前键 + 修改/恢复。

设置对话框与「按键调整」页面共用；mode 参数指定当前编辑的键位范围
（solo / ide / sprite / pixel），切换模式时调用 set_mode 刷新。
"""
from __future__ import annotations

from typing import Optional

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
    QVBoxLayout,
    QWidget,
)

from ui import shortcuts as sc
from ui.i18n import tr


class ShortcutSettingsPanel(QWidget):
    """多级下拉快捷键编辑面板。"""

    status_changed = Signal(str)  # 提示信息（供宿主页面显示）

    def __init__(self, mode: str = "pixel", parent=None):
        super().__init__(parent)
        self._mode = mode if mode in sc.MODES else "pixel"
        self._build_ui()
        self._refresh_actions()

    # ------------------------------------------------------------------ #
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """切换当前编辑的键位范围（solo/ide/sprite/pixel）。"""
        mode = mode if mode in sc.MODES else "pixel"
        if mode == self._mode:
            return
        self._mode = mode
        self._refresh_actions()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        box = QGroupBox(tr("快捷键"))
        f = QFormLayout(box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(10)

        # 一级下拉：类别
        self._cat_combo = QComboBox()
        self._cat_combo.currentIndexChanged.connect(self._refresh_actions)
        f.addRow(tr("类别"), self._cat_combo)

        # 二级下拉：该类别下的条目
        self._action_combo = QComboBox()
        self._action_combo.currentIndexChanged.connect(self._refresh_current)
        f.addRow(tr("条目"), self._action_combo)

        # 当前快捷键 + 修改 / 恢复默认
        row = QHBoxLayout()
        self._current_label = QLabel("—")
        self._current_label.setObjectName("StepLabel")
        row.addWidget(self._current_label, 1)
        self._btn_change = QPushButton(tr("修改…"))
        self._btn_change.clicked.connect(self._change)
        row.addWidget(self._btn_change)
        self._btn_reset = QPushButton(tr("恢复默认"))
        self._btn_reset.clicked.connect(self._reset_one)
        row.addWidget(self._btn_reset)
        f.addRow(tr("当前快捷键"), row)
        v.addWidget(box)

        self._btn_reset_all = QPushButton(tr("恢复全部默认"))
        self._btn_reset_all.clicked.connect(self._reset_all)
        v.addWidget(self._btn_reset_all)

        hint = QLabel(
            tr("修改立即生效；点「保存」持久化。同键被多个条目使用时后设置的覆盖先设置的。")
        )
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch(1)

    # ------------------------------------------------------------------ #
    def _current_action(self) -> str:
        return str(self._action_combo.currentData() or "")

    def _refresh_actions(self) -> None:
        """类别切换 -> 刷新条目下拉（保持当前条目尽量不跳变）。"""
        cat = str(self._cat_combo.currentData() or "")
        current = self._current_action()
        cats = sc.categories(self._mode)
        if cat not in cats:
            cat = cats[0] if cats else ""
        self._cat_combo.blockSignals(True)
        self._cat_combo.clear()
        for c in cats:
            self._cat_combo.addItem(tr(c), userData=c)
        if cat:
            idx = self._cat_combo.findData(cat)
            self._cat_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._cat_combo.blockSignals(False)

        actions = sc.actions_in(cat, self._mode) if cat else []
        self._action_combo.blockSignals(True)
        self._action_combo.clear()
        for aid in actions:
            self._action_combo.addItem(tr(sc.action_name(aid, self._mode)), userData=aid)
        if current in actions:
            self._action_combo.setCurrentIndex(actions.index(current))
        self._action_combo.blockSignals(False)
        self._refresh_current()

    def _refresh_current(self) -> None:
        aid = self._current_action()
        self._current_label.setText(sc.format_shortcut(sc.get(aid, self._mode)) if aid else "—")

    def _change(self) -> None:
        aid = self._current_action()
        if not aid:
            return
        dialog = ShortcutCaptureDialog(sc.get(aid, self._mode), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        seq = dialog.sequence()
        if not seq:
            return
        # 冲突检测：同键已被同模式其他条目占用
        for other, other_seq in sc.all_entries(self._mode).items():
            if other != aid and other_seq == seq:
                ret = QMessageBox.question(
                    self,
                    tr("提示"),
                    tr("快捷键 {0} 已被「{1}」使用，将覆盖原绑定。继续？").format(
                        seq, tr(sc.action_name(other, self._mode))
                    ),
                )
                if ret != QMessageBox.StandardButton.Yes:
                    return
                break
        self._set_entry(aid, seq)
        self._refresh_current()
        self.status_changed.emit(tr("已设置：{0}").format(seq))

    def _reset_one(self) -> None:
        aid = self._current_action()
        if not aid:
            return
        self._set_entry(aid, sc.DEFAULTS[self._mode].get(aid, ""))
        self._refresh_current()
        self.status_changed.emit(tr("已恢复默认"))

    def _reset_all(self) -> None:
        cfg = dict(sc._config.get(self._mode, {}))
        for aid in cfg:
            self._set_entry(aid, sc.DEFAULTS[self._mode].get(aid, ""))
        self._refresh_current()
        self.status_changed.emit(tr("已恢复全部默认"))

    def _set_entry(self, aid: str, seq: str) -> None:
        """写入当前模式配置（缓存即时生效）。"""
        cfg = dict(sc._config.get(self._mode, {}))
        if seq:
            cfg[aid] = seq
        else:
            cfg.pop(aid, None)
        sc._config[self._mode] = cfg


class ShortcutCaptureDialog(QDialog):
    """快捷键录制弹窗：按下组合键即提交；Esc 取消。"""

    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("修改快捷键"))
        self.setMinimumWidth(320)
        self._sequence: str = ""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        label = QLabel(tr("按新的快捷键…（Esc 取消）"))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self._current_label = QLabel(sc.format_shortcut(current))
        self._current_label.setObjectName("StepLabel")
        self._current_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._current_label)
        self.setFocus()

    def sequence(self) -> str:
        return self._sequence

    def keyPressEvent(self, event) -> None:  # noqa: N802
        mods = event.modifiers()
        key = event.key()
        # 纯修饰键忽略；Esc 取消
        if key in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
            Qt.Key.Key_AltGr,
        ):
            event.accept()
            return
        if key == Qt.Key.Key_Escape and not mods:
            self.reject()
            return
        seq = sc.key_to_text(mods, key)
        if seq:
            self._sequence = seq
            self.accept()
        else:
            event.accept()
