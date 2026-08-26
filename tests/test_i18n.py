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
