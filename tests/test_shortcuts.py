"""快捷键注册表 / 解析 / 编辑器应用测试（多模式：Solo/IDE/精灵图/像素）。"""
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from ui import shortcuts as sc


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    """确保 QApplication 存在（编辑器测试需要）。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _key_event(key, mods=Qt.KeyboardModifier.NoModifier, text=""):
    return QKeyEvent(QEvent.Type.KeyPress, key, mods, text)


def test_parse_shortcut():
    mods, key = sc.parse_shortcut("Ctrl+Shift+Z")
    assert mods == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    assert key == Qt.Key.Key_Z
    mods, key = sc.parse_shortcut("Esc")
    assert mods == Qt.KeyboardModifier.NoModifier
    assert key == Qt.Key.Key_Escape
    mods, key = sc.parse_shortcut("Ctrl+=")
    assert key == Qt.Key.Key_Equal
    assert sc.parse_shortcut("") is None
    assert sc.parse_shortcut("Ctrl+NoSuchKey") is None


def test_match_shortcut():
    ev = _key_event(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier, "z")
    assert sc.match(ev, "Ctrl+Z")
    assert not sc.match(ev, "Ctrl+Shift+Z")   # 修饰键精确匹配
    ev2 = _key_event(
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        "Z",
    )
    assert sc.match(ev2, "Ctrl+Shift+Z")
    assert not sc.match(ev2, "Ctrl+Z")
    assert not sc.match(ev, "Ctrl+C")
    # 等号键：部分键盘需 Shift，允许额外 Shift
    ev3 = _key_event(
        Qt.Key.Key_Equal,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        "=",
    )
    assert sc.match(ev3, "Ctrl+=")


def test_modes_and_defaults():
    """每个模式有独立条目与默认值。"""
    assert sc.MODES == ("solo", "ide", "sprite", "pixel")
    assert sc.get("undo", "pixel") == "Ctrl+Z"
    assert sc.get("preview_play", "ide") == "Space"
    assert sc.get("timeline_insert", "ide") == "I"
    assert sc.get("preview_play", "solo") == "Space"
    assert sc.get("tool_pencil", "pixel") == ""        # 默认不绑定
    # 像素模式没有 preview_play，solo 没有 undo
    assert sc.get("preview_play", "pixel") == ""
    assert sc.get("undo", "solo") == ""


def test_set_shortcuts_new_and_legacy_format():
    """新格式按模式配置；旧格式自动迁移到 pixel 模式。"""
    sc.set_shortcuts(None)
    # 新格式
    sc.set_shortcuts({
        "pixel": {"undo": "Ctrl+U"},
        "ide": {"preview_play": "P"},
    })
    assert sc.get("undo", "pixel") == "Ctrl+U"
    assert sc.get("preview_play", "ide") == "P"
    assert sc.get("redo", "pixel") == "Ctrl+Shift+Z"    # 未配置用默认
    # 旧格式 {action_id: seq} -> pixel
    sc.set_shortcuts({"undo": "Ctrl+Y"})
    assert sc.get("undo", "pixel") == "Ctrl+Y"
    assert sc.get("preview_play", "ide") == "Space"     # 其他模式不受影响
    saved = sc.to_settings()
    assert saved.get("pixel", {}).get("undo") == "Ctrl+Y"
    assert "ide" not in saved                            # 全默认的模式不持久化
    sc.set_shortcuts(None)
    assert sc.get("undo", "pixel") == "Ctrl+Z"


def test_active_mode():
    """get() 默认使用当前生效模式。"""
    sc.set_shortcuts({"ide": {"preview_play": "P"}, "solo": {"preview_play": "S"}})
    sc.set_active_mode("ide")
    assert sc.active_mode() == "ide"
    assert sc.get("preview_play") == "P"
    sc.set_active_mode("solo")
    assert sc.get("preview_play") == "S"
    sc.set_shortcuts(None)


def test_categories_and_actions_per_mode():
    ids = sc.actions_in("编辑", "pixel")
    assert "undo" in ids and "merge" in ids
    assert sc.action_name("undo", "pixel") == "撤销"
    assert "预览" in sc.categories("ide")
    assert "timeline_insert" in sc.actions_in("时间轴", "ide")
    assert sc.all_entries("ide").get("preview_fit") == "F"


def test_key_to_text_roundtrip():
    text = sc.key_to_text(
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier, Qt.Key.Key_Z
    )
    assert text == "Ctrl+Shift+Z"
    assert sc.parse_shortcut(text) == (
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        Qt.Key.Key_Z,
    )


def test_editor_uses_custom_shortcuts():
    """像素编辑器按键绑定跟随自定义快捷键（当前模式为 pixel）。"""
    from PIL import Image

    from ui.widgets.pixel_editor import PixelEditorWidget

    sc.set_shortcuts(None)
    sc.set_active_mode("pixel")
    editor = PixelEditorWidget()
    editor.set_frame(Image.new("RGBA", (8, 8), (255, 255, 255, 255)))
    canvas = editor._canvas

    # 默认 Ctrl+Z 撤销
    canvas.set_pixel(0, 0, (0, 0, 0, 255))
    assert canvas.image.getpixel((0, 0))[0] == 0
    editor.keyPressEvent(_key_event(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier, "z"))
    assert canvas.image.getpixel((0, 0))[0] == 255

    # 自定义为 Ctrl+U 后，Ctrl+Z 不再撤销、Ctrl+U 撤销
    sc.set_shortcuts({"pixel": {"undo": "Ctrl+U"}})
    canvas.set_pixel(0, 0, (10, 0, 0, 255))
    editor.keyPressEvent(_key_event(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier, "z"))
    assert canvas.image.getpixel((0, 0))[0] == 10   # Ctrl+Z 失效
    editor.keyPressEvent(_key_event(Qt.Key.Key_U, Qt.KeyboardModifier.ControlModifier, "u"))
    assert canvas.image.getpixel((0, 0))[0] == 255  # Ctrl+U 生效

    # 工具快捷键：默认不绑定，配置后生效
    sc.set_shortcuts({"pixel": {"tool_eraser": "E"}})
    editor.keyPressEvent(_key_event(Qt.Key.Key_E))
    assert editor.tool().value == "eraser"
    sc.set_shortcuts(None)
