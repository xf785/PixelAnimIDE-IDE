"""动作预设下拉的共享填充：按分类分组，分类为禁用表头。

供 Solo / IDE / 精灵图 页面的动作下拉复用：
首项为空（不选动作），随后每个分类一个禁用表头（— 分类 —），
表头下为该分类的动作项（显示翻译名，userData 存中文 ID）。
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from core.processing.prompt_utils import preset_categories
from ui.i18n import tr


def populate_action_combo(combo: QComboBox) -> None:
    """按分类分组填充动作预设下拉（保留当前选中的动作，若无则置空）。"""
    current = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem("")  # 空首项 = 不选动作
    model = combo.model()
    for cat, names in preset_categories():
        head = combo.count()
        combo.addItem("— " + tr(cat) + " —")
        item = model.item(head) if model is not None else None
        if item is not None:
            item.setEnabled(False)  # 分类表头不可选
        for name in names:
            combo.addItem(tr(name), userData=name)
    if current is not None:
        idx = combo.findData(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    combo.blockSignals(False)
