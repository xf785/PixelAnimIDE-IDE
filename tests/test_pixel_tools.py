"""像素编辑器实用功能回归：形状工具 / 对称绘制 / 环绕绘制 / 画布变换 / 导出倍率。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from core.editing import PixelCanvas
from core.editing.canvas import SYM_BOTH, SYM_H, SYM_NONE, SYM_V
from ui.widgets.pixel_editor import SHAPE_TOOLS, PixelEditorWidget, Tool


def _canvas(w=8, h=8, color=(0, 0, 0, 0)) -> PixelCanvas:
    return PixelCanvas(Image.new("RGBA", (w, h), color))


# --------------------------------------------------------------------------- #
# 数据层：形状 / 对称 / 环绕
# --------------------------------------------------------------------------- #
def test_shapes_draw_stroke_and_fill():
    c = _canvas(8, 8)
    c.draw_rect((1, 1), (6, 6), (255, 0, 0, 255))
    assert c.get_pixel(1, 1) == (255, 0, 0, 255)
    assert c.get_pixel(3, 3) == (0, 0, 0, 0)      # 空心描边
    assert c.can_undo

    c2 = _canvas(8, 8)
    c2.draw_rect((1, 1), (6, 6), (0, 255, 0, 255), filled=True)
    assert c2.get_pixel(3, 3) == (0, 255, 0, 255)

    c3 = _canvas(8, 8)
    c3.draw_ellipse((0, 0), (7, 7), (0, 0, 255, 255), filled=True)
    assert c3.get_pixel(3, 3) == (0, 0, 255, 255)
    assert c3.get_pixel(0, 0)[3] == 0             # 椭圆四角不填


def test_shape_drawing_is_single_undo_step():
    c = _canvas(16, 16)
    c.draw_ellipse((0, 0), (15, 15), (10, 20, 30, 255), filled=True)
    assert c.can_undo
    c.undo()
    assert np.asarray(c.image)[..., 3].max() == 0  # 一次撤销即整块回退


def test_symmetry_mirrors_stroke():
    for mode, expect, absent in (
        (SYM_H, [(1, 1), (6, 1)], (1, 6)),
        (SYM_V, [(1, 1), (1, 6)], (6, 1)),
        (SYM_BOTH, [(1, 1), (6, 1), (1, 6), (6, 6)], None),
    ):
        c = _canvas(8, 8)
        c.set_pixel(1, 1, (255, 255, 0, 255), symmetry=mode)
        for x, y in expect:
            assert c.get_pixel(x, y)[3] == 255, (mode, x, y)
        if absent is not None:
            assert c.get_pixel(*absent)[3] == 0, (mode, absent)


def test_wrap_around_stroke():
    c = _canvas(8, 8)
    c.set_pixel(8, 8, (7, 8, 9, 255), wrap=True)      # 越界 -> 环绕到 (0,0)
    assert c.get_pixel(0, 0) == (7, 8, 9, 255)
    c2 = _canvas(8, 8)
    c2.set_pixel(8, 8, (7, 8, 9, 255))                # 不环绕则丢弃
    assert c2.get_pixel(0, 0)[3] == 0


def test_canvas_transforms_and_undo():
    c = _canvas(4, 2, (0, 0, 0, 0))
    c.set_pixel(0, 0, (255, 0, 0, 255), wrap=False)
    assert c.rotate_90(clockwise=True) is True
    assert c.size == (2, 4)
    assert c.flip_horizontal() is True
    assert c.crop_to((0, 0, 0, 3)) is True
    assert c.size == (1, 4)
    assert c.resize_canvas(4, 4, "center") is True
    assert c.size == (4, 4)
    assert c.scale_content(2) is True
    assert c.size == (8, 8)
    # 逐步撤销回初始状态
    for _ in range(6):
        c.undo()
    assert c.size == (4, 2)


def test_transform_skips_noop_and_rejects_bad_scale():
    c = _canvas(4, 4, (5, 5, 5, 255))
    assert c.flip_horizontal() is False       # 纯色翻转无变化 -> 不产生撤销
    assert c.crop_to((0, 0, 3, 3)) is False   # 全画布裁剪
    assert c.resize_canvas(4, 4) is False
    assert c.scale_content(1) is False
    assert c.scale_content(0) is False
    assert not c.can_undo


# --------------------------------------------------------------------------- #
# 控件层：形状工具 / 对称 / 环绕 / 变换
# --------------------------------------------------------------------------- #
def _editor(qtbot, size=(16, 16), view=(400, 320)):
    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    editor.set_frame(Image.new("RGBA", size, (255, 255, 255, 255)))
    host = editor._canvas_host
    host.resize(*view)
    from PySide6.QtGui import QPaintEvent

    host.paintEvent(QPaintEvent(host.rect()))   # 初始化居中偏移
    return editor, host


def test_shape_tools_registered_and_preview(qtbot):
    editor, _host = _editor(qtbot)
    for tool in SHAPE_TOOLS:
        assert tool in editor._tool_buttons
    editor.set_tool(Tool.RECT)
    editor._shape_start = (2, 2)
    editor._shape_cur = (5, 5)
    editor._set_shape_filled(False)
    assert len(editor._shape_preview_cells()) == 12   # 4×4 空心边框
    editor._set_shape_filled(True)
    assert len(editor._shape_preview_cells()) == 16   # 4×4 填充
    editor._set_shape_filled(False)
    assert len(editor._shape_preview_cells()) == 12


def test_shape_drag_commits_on_release(qtbot):
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    from tests.test_gui import _mouse_ev  # 复用现有鼠标事件构造

    editor, host = _editor(qtbot)
    editor.set_tool(Tool.LINE)
    editor.set_color((255, 0, 0, 255))
    ox, oy = host._ox, host._oy

    QApplication.sendEvent(host, _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(ox + 2, oy + 2),
                                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    QApplication.sendEvent(host, _mouse_ev(QEvent.Type.MouseMove, QPoint(ox + 6, oy + 2),
                                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert editor._shape_start == (2, 2) and editor._shape_cur == (6, 2)
    assert editor.canvas().get_pixel(2, 2) == (255, 255, 255, 255)  # 未松开前不落笔
    QApplication.sendEvent(host, _mouse_ev(QEvent.Type.MouseButtonRelease, QPoint(ox + 6, oy + 2),
                                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert editor.canvas().get_pixel(2, 2) == (255, 0, 0, 255)
    assert editor.canvas().get_pixel(6, 2) == (255, 0, 0, 255)
    assert editor._shape_start is None
    editor.undo()
    assert editor.canvas().get_pixel(2, 2) == (255, 255, 255, 255)  # 整条线一次撤销


def test_symmetry_and_wrap_toggles(qtbot):
    editor, _host = _editor(qtbot)
    assert editor.symmetry() == SYM_NONE
    editor._sym_btn.setChecked(True)
    editor._on_symmetry_clicked(True)
    assert editor.symmetry() == SYM_H
    editor._on_symmetry_clicked(True)
    assert editor.symmetry() == SYM_V
    editor._on_symmetry_clicked(True)
    assert editor.symmetry() == SYM_BOTH
    editor._on_symmetry_clicked(True)
    assert editor.symmetry() == SYM_H
    editor.set_symmetry(SYM_NONE)
    assert not editor._sym_btn.isChecked()

    assert editor.wrap_enabled() is False
    editor._wrap_btn.setChecked(True)
    assert editor.wrap_enabled() is True


def test_editor_transform_actions(qtbot, monkeypatch):
    """变换动作走画布 API（弹窗类动作用 monkeypatch 替掉，避免阻塞）。"""
    from PySide6.QtWidgets import QInputDialog

    editor, _host = _editor(qtbot, size=(8, 4))
    editor.canvas().fill_rect(0, 0, 0, 0, (255, 0, 0, 255))
    assert editor.transform("flip_h") is True
    assert editor.transform("rotate_cw") is True
    assert editor.canvas().size == (4, 8)

    # 画布尺寸：宽度 8 / 高度 8 / 内容居中
    answers = iter([(8, True), (8, True), ("居中", True)])
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: next(answers))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: next(answers))
    assert editor.transform("resize") is True
    assert editor.canvas().size == (8, 8)

    # 内容缩放：2 倍
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (2, True))
    assert editor.transform("scale") is True
    assert editor.canvas().size == (16, 16)

    assert editor.transform("nonexistent") is False

    # 裁剪到选区
    editor._selection = np.zeros((16, 16), dtype=bool)
    editor._selection[0:4, 0:4] = True
    assert editor.transform("crop_selection") is True
    assert editor.canvas().size == (4, 4)


def test_pixel_page_export_scale_and_clipboard(qtbot, tmp_path):
    from config.api_config import APIConfigManager
    from core.storage.keyring import Keyring
    from ui.app_context import AppContext, UISettings
    from ui.pages.pixel_page import PixelPage

    ctx = AppContext(
        api=APIConfigManager(config_file=tmp_path / "api.json", keyring=Keyring(tmp_path / ".keyring")),
        ui_settings=UISettings(tmp_path / "ui.json"),
    )
    page = PixelPage(ctx)
    qtbot.addWidget(page)
    assert page.export_scale() == 1
    assert page._scaled_image().size == page.image().size
    page._scale_combo.setCurrentIndex(2)                 # 4×
    assert page.export_scale() == 4
    scaled = page._scaled_image()
    assert scaled.size == (page.image().width * 4, page.image().height * 4)
    page._on_copy_clipboard()
    from PySide6.QtWidgets import QApplication

    assert not QApplication.clipboard().pixmap().isNull()


def test_new_tool_shortcuts_have_defaults():
    """新工具（直线/矩形/椭圆）与对称/环绕开关都有可绑定的条目。"""
    from ui import shortcuts as sc

    for aid, key in (("tool_line", "L"), ("tool_rect", "U"), ("tool_ellipse", "O"),
                     ("tool_pencil", "B"), ("tool_eraser", "E"), ("tool_select", "M")):
        assert sc.get(aid, "pixel") == key
    for aid in ("toggle_symmetry", "toggle_wrap"):
        assert aid in sc.SHORTCUT_DEFS["pixel"]


# --------------------------------------------------------------------------- #
# 调色板导入 / 导出（GIMP .gpl）
# --------------------------------------------------------------------------- #
def test_parse_gpl_variants():
    from ui.widgets.pixel_editor import parse_gpl

    colors = parse_gpl(
        "GIMP Palette\nName: X\nColumns: 4\n#\n"
        "255   0   0\tRed\n  0 255   0\tGreen\n0 0 255\n"
        "not a color\n999 0 0\n"
    )
    assert colors == [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
    assert parse_gpl("") == []
    assert parse_gpl("# only comments\n") == []


def test_palette_dialog_export_and_import(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from ui.widgets.pixel_editor import _PaletteDialog

    editor, _host = _editor(qtbot)
    img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    img.putpixel((0, 0), (0, 0, 255, 255))
    editor.set_frame(img)
    dialog = _PaletteDialog(editor._families(), editor)
    qtbot.addWidget(dialog)
    assert dialog.palette_colors()                     # 至少红/蓝两色

    out = tmp_path / "pal.gpl"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    dialog._on_export_palette()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("GIMP Palette")
    assert "255   0   0" in text

    # 再导回来：颜色一致
    loaded = []
    dialog.palette_loaded.connect(loaded.append)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(out), ""))
    dialog._on_import_palette()
    assert loaded and (255, 0, 0, 255) in loaded[0]


def test_editor_applies_imported_palette(qtbot):
    editor, _host = _editor(qtbot)
    editor.set_frame(Image.new("RGBA", (8, 8), (200, 30, 30, 255)))
    editor.apply_imported_palette([(255, 0, 0, 255), (0, 255, 0, 255)])
    assert editor._palette_locked is True
    assert editor.canvas().palette == [(255, 0, 0, 255), (0, 255, 0, 255)]
    assert editor.canvas().snap_color((250, 5, 5, 255)) == (255, 0, 0, 255)
