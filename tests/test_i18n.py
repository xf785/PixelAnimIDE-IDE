"""中英文适配测试。"""
from ui.i18n import available_languages, set_language, tr


def test_tr_zh_default():
    set_language("zh")
    assert tr("开始生成") == "开始生成"
    assert tr("未收录文案XYZ") == "未收录文案XYZ"  # 回退原文


def test_tr_en():
    set_language("en")
    assert tr("开始生成") == "Start generating"
    assert tr("新建画布") == "New canvas"
    # 未收录文案回退中文原文
    assert tr("未收录文案XYZ") == "未收录文案XYZ"


def test_language_invalid_falls_back_zh():
    set_language("fr")
    assert tr("铅笔") == "铅笔"
    set_language("zh")  # 恢复


def test_available_languages_includes_zh_en():
    """语言包架构：注册表语言自动出现在可用列表。"""
    langs = dict(available_languages())
    assert "zh" in langs and langs["zh"] == "中文"
    assert "en" in langs and langs["en"] == "English"


def test_editor_tooltips_retranslate_on_language_switch(qapp):
    """语言切换后常驻控件的 tooltip 自动重译（T() 注册机制）。

    回归：像素编辑器的「撤销/重做」等 tooltip 之前用 tr() 直接设置，
    切到 English 后仍显示中文；现改为 T() 注册，retranslate_all 后刷新。
    """
    from ui import shortcuts as sc
    from ui.i18n import retranslate_all
    from ui.widgets.pixel_editor import PixelEditorWidget

    sc.set_shortcuts(None)
    editor = PixelEditorWidget()

    # 英文环境
    set_language("en")
    retranslate_all()
    assert editor._undo_btn.toolTip() == "Undo (Ctrl+Z)"
    assert editor._redo_btn.toolTip() == "Redo (Ctrl+Shift+Z)"
    assert editor._color_swatch.toolTip() == "Current color"
    assert editor._bg_btn.toolTip().startswith("Background:")

    # 中文环境
    set_language("zh")
    retranslate_all()
    assert "撤销" in editor._undo_btn.toolTip()
    assert "重做" in editor._redo_btn.toolTip()
    editor.close()
    set_language("zh")
