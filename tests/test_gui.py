"""GUI 冒烟测试（QT_QPA_PLATFORM=offscreen）。"""
import pytest
from PySide6.QtWidgets import QApplication

from config.api_config import APIConfig, APIConfigManager
from core.storage.keyring import Keyring
from core.workflow import SoloParams
from ui.app_context import AppContext, UISettings
from ui.main_window import MainWindow
from ui.widgets.api_config_widget import ApiConfigWidget
from ui.workers import SoloWorker


@pytest.fixture()
def ctx(tmp_path):
    """带默认 mock 配置的应用上下文。"""
    api = APIConfigManager(
        config_file=tmp_path / "api_config.json",
        keyring=Keyring(tmp_path / ".keyring"),
    )
    for kind in ("llm", "image", "video"):
        api.add(
            APIConfig(
                kind=kind,
                name=f"mock-{kind}",
                base_url="mock",
                model="mock-model",
                params={"mock": True, "frames": 8, "fps": 8},
            )
        )
    ui = UISettings(tmp_path / "ui_settings.json")
    return AppContext(api=api, ui_settings=ui)


def test_main_window_structure(qtbot, ctx):
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    assert window._stack.count() == 5  # solo / ide / sprite / pixel / tilemap
    # 默认在 Solo 模式
    assert window._stack.currentIndex() == 0
    assert window._mode == "solo"
    # Solo 模式：侧栏内缩，步骤导航隐藏
    assert window._step_nav.isHidden()
    assert window._sidebar.width() == 128

    window.switch_page("ide")
    assert window._stack.currentIndex() == 1
    assert window._mode == "ide"
    # IDE 模式：侧栏展开，显示步骤导航
    assert window._step_nav.isVisible()
    assert window._sidebar.width() == 200

    window.switch_page("sprite")
    assert window._stack.currentIndex() == 2
    assert window._mode == "sprite"
    assert window._step_nav.isHidden()
    assert window._sidebar.width() == 128

    window.switch_page("pixel")
    assert window._stack.currentIndex() == 3
    assert window._mode == "pixel"
    assert window._step_nav.isHidden()
    assert window._sidebar.width() == 128

    window.switch_page("tilemap")
    assert window._stack.currentIndex() == 4
    assert window._mode == "tilemap"
    assert window._step_nav.isHidden()
    assert window._sidebar.width() == 128

    window.switch_page("solo")
    assert window._stack.currentIndex() == 0
    assert window._step_nav.isHidden()


def test_mode_switch_and_step_buttons(qtbot, ctx):
    """左上角 Solo/IDE/精灵图 分段开关；IDE 展开后显示 6 个步骤按钮。"""
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    # 模式开关五格（2×2 + 瓦片地图占第三行）
    assert window._mode_solo_btn.isChecked()
    assert not window._mode_ide_btn.isChecked()
    assert not window._mode_sprite_btn.isChecked()
    for b in (window._mode_solo_btn, window._mode_ide_btn, window._mode_sprite_btn, window._mode_pixel_btn, window._mode_tilemap_btn):
        assert b.icon() is not None and not b.icon().isNull()  # 开源风格图标
        assert b.toolTip()
    assert window._mode_switch.layout().count() == 5  # 2×2 + 第 5 模式整行
    # 6 个步骤按钮（含图标 + 短名）
    assert len(window._step_buttons) == 6
    for i, btn in window._step_buttons.items():
        assert btn.text()
        assert not btn.icon().isNull()
        assert btn.toolTip()

    # 切到 IDE：展开
    window.set_mode("ide")
    assert window._mode_ide_btn.isChecked()
    assert window._step_nav.isVisible()

    # 切到精灵图：内缩、页面切换
    window.set_mode("sprite")
    assert window._mode_sprite_btn.isChecked()
    assert window._stack.currentIndex() == 2
    assert window._step_nav.isHidden()

    # 点击步骤按钮 → 同步到 IDE 页 + 高亮
    window._on_step_clicked(3)
    assert window.ide_page._current_step == 3
    assert window._step_buttons[3].isChecked()
    assert window.ide_page._btn_run.text() == "像素化处理"

    # 设置按钮：齿轮图标、非勾选、有提示
    assert not window._settings_btn.isCheckable()
    assert not window._settings_btn.icon().isNull()
    assert window._settings_btn.toolTip()


def test_theme_switch(qtbot, ctx):
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    app = QApplication.instance()
    # 初始深色
    assert ctx.ui_settings.get("theme") == "dark"
    window._on_toggle_theme()
    assert ctx.ui_settings.get("theme") == "light"
    assert app.styleSheet()  # 应用了 QSS
    # 浅色主题下图标仍非空
    assert not window._settings_btn.icon().isNull()
    window._on_toggle_theme()
    assert ctx.ui_settings.get("theme") == "dark"


def test_settings_dialog_structure(qtbot, ctx):
    """设置弹窗：左侧分类导航，右侧对应表单。"""
    from ui.dialogs.settings_dialog import SettingsDialog

    dialog = SettingsDialog(ctx)
    qtbot.addWidget(dialog)
    dialog.show()
    # 5 个分类 + 「快捷键」分类下的 4 个模式子项（默认隐藏）
    assert dialog._cat_list.count() == 9
    assert len(dialog._shortcut_mode_items) == 4
    # 右侧堆栈含 3 个 ApiConfigWidget + 常规面板 + 快捷键面板
    assert set(dialog._api_widgets.keys()) == {"llm", "image", "video"}
    assert dialog._stack.count() == 5
    # 选中分类 -> 切换右侧面板
    dialog._cat_list.setCurrentRow(2)
    assert dialog._stack.currentIndex() == 2
    dialog._cat_list.setCurrentRow(3)
    assert dialog._stack.currentIndex() == 3
    # 常规面板：深色模式 iOS 风格开关（默认深色）
    assert dialog._dark_switch.isChecked() is True
    dialog._dark_switch.setChecked(False, animate=False)
    assert dialog._dark_switch.isChecked() is False
    dialog._dark_switch.setChecked(True, animate=False)
    # 快捷键分类：两级交互（点击进入 → 再点展开模式子菜单 → 选模式 → 再点收起）
    dialog._cat_list.setCurrentRow(4)
    assert dialog._stack.currentIndex() == 4
    panel = dialog._shortcuts_panel_widget
    assert panel._cat_combo.count() >= 2            # 类别一级下拉
    assert panel._action_combo.count() >= 1         # 条目二级下拉
    assert panel._current_label.text()              # 当前快捷键显示
    assert panel._btn_change.text() == "修改…"
    # 第二级：再次点击「快捷键」分类项 -> 展开模式子菜单（该项目正下方）
    shortcuts_item = dialog._cat_list.item(4)
    pixel_row = dialog._shortcut_mode_items["pixel"][0]
    assert dialog._cat_list.isRowHidden(pixel_row)
    assert "▸" in shortcuts_item.text()
    dialog._cat_list.itemClicked.emit(shortcuts_item)
    assert not dialog._cat_list.isRowHidden(pixel_row)
    assert "▾" in shortcuts_item.text()
    # 第三级：选择像素模式 -> 面板切换 + 子菜单高亮，右侧停留
    dialog._cat_list.setCurrentItem(dialog._shortcut_mode_items["pixel"][1])
    assert panel.mode() == "pixel"
    assert dialog._stack.currentIndex() == 4
    # 再点分类项 -> 收起子菜单，右侧表单保持不变
    dialog._cat_list.itemClicked.emit(shortcuts_item)
    assert dialog._cat_list.isRowHidden(pixel_row)
    assert "▸" in shortcuts_item.text()
    assert panel.mode() == "pixel"


def test_api_config_widget_shows_configs(qtbot, ctx):
    widget = ApiConfigWidget(ctx.api, "llm")
    qtbot.addWidget(widget)
    widget.refresh()
    assert widget._combo.count() == 1
    assert "mock" in widget._combo.currentText()


def test_api_config_widget_save_new(qtbot, ctx):
    widget = ApiConfigWidget(ctx.api, "llm")
    qtbot.addWidget(widget)
    # 先删除默认，从空开始
    for cfg in list(ctx.api.list("llm")):
        ctx.api.delete(cfg.id)
    widget.refresh()
    assert widget._combo.count() == 0

    widget._set_field("base_url", "http://openai.example/v1")
    widget._set_field("api_key", "sk-test")
    widget._set_field("model", "gpt-test")
    widget._on_save()
    configs = ctx.api.list("llm")
    assert len(configs) == 1
    assert configs[0].base_url == "http://openai.example/v1"
    assert configs[0].api_key == "sk-test"
    # 保存后刷新并选中
    assert widget._combo.count() == 1


def test_api_config_widget_has_models_button(qtbot, ctx):
    """配置控件提供「查询模型」按钮。"""
    widget = ApiConfigWidget(ctx.api, "video")
    qtbot.addWidget(widget)
    assert widget._btn_models.text() == "查询模型"


def test_api_config_widget_custom_request_fields(qtbot, ctx):
    """完全自定义字段：复选框 + 多行模板/请求头（textarea）+ 路径可读写。"""
    from PySide6.QtWidgets import QCheckBox, QPlainTextEdit

    widget = ApiConfigWidget(ctx.api, "llm")
    qtbot.addWidget(widget)
    # 字段存在且类型正确
    assert isinstance(widget._fields["custom_request"], QCheckBox)
    assert isinstance(widget._fields["payload_template"], QPlainTextEdit)
    assert isinstance(widget._fields["extra_headers"], QPlainTextEdit)
    # 读写往返
    widget._set_field("custom_request", True)
    assert widget._get_field("custom_request") is True
    template = '{"model": "$model", "prompt": "$prompt"}'
    widget._set_field("payload_template", template)
    assert widget._get_field("payload_template") == template
    widget._set_field("text_path", "data.answer")
    assert widget._get_field("text_path") == "data.answer"


def test_api_config_widget_advanced_collapsible(qtbot, ctx):
    """字段多的 API（视频）：高级选项默认折叠，字段仍正常读写，展开可调。"""
    widget = ApiConfigWidget(ctx.api, "video")
    qtbot.addWidget(widget)
    widget.refresh()
    assert widget._adv_box is not None
    assert widget._adv_box.isChecked() is False          # 默认折叠
    assert widget._adv_container.isHidden() is True
    # 折叠状态下也能读写高级字段
    widget._set_field("submit_url", "https://x/submit")
    assert widget._get_field("submit_url") == "https://x/submit"
    assert widget._get_field("payload_template") == ""
    # 展开后可见
    widget._adv_box.setChecked(True)
    assert widget._adv_container.isHidden() is False
    assert "高级选项（收起）" in widget._adv_box.title()
    # 收起
    widget._adv_box.setChecked(False)
    assert widget._adv_container.isHidden() is True


def test_api_config_widget_provider_preset(qtbot, ctx):
    """选择服务商预设自动填充 Base URL / 模型 / 适配参数。"""
    from config.api_config import PROVIDER_PRESETS

    widget = ApiConfigWidget(ctx.api, "video")
    qtbot.addWidget(widget)
    widget.refresh()
    # 找到 gpt.ge 预设并选择
    index = widget._preset_combo.findData("gptge_doubao")
    assert index >= 0
    widget._preset_combo.setCurrentIndex(index)
    assert widget._get_field("base_url") == "https://api.gpt.ge"
    assert widget._get_field("model") == "doubao-seedance-1-5-pro-251215"
    assert widget._get_field("provider") == "gptge"

    # 可灵预设填入请求体模板与完整映射
    index = widget._preset_combo.findData("kling")
    widget._preset_combo.setCurrentIndex(index)
    assert widget._get_field("base_url") == "https://api.klingai.com"
    assert widget._get_field("provider") == "generic"
    assert "$model" in widget._get_field("payload_template")
    assert widget._get_field("status_success") == "succeed,success"
    # 没有 API Key 时提示而不弹查询
    assert "API Key" in widget._test_result.text()


def test_model_picker_dialog_filter(qtbot):
    from ui.widgets.api_config_widget import ModelPickerDialog

    dialog = ModelPickerDialog(
        ["doubao-seedance-1-5-pro-251215", "gpt-4o", "kling-v1-6", "doubao-seedance-1-0-pro-250528"]
    )
    qtbot.addWidget(dialog)
    dialog._filter.setText("seedance")
    items = [dialog._list.item(i).text() for i in range(dialog._list.count())]
    assert items == ["doubao-seedance-1-5-pro-251215", "doubao-seedance-1-0-pro-250528"]
    dialog._filter.setText("kling")
    items = [dialog._list.item(i).text() for i in range(dialog._list.count())]
    assert items == ["kling-v1-6"]


def test_api_config_widget_video_provider_choice(qtbot, ctx):
    """视频配置支持「服务商适配」下拉并正确存取。"""
    widget = ApiConfigWidget(ctx.api, "video")
    qtbot.addWidget(widget)
    widget.refresh()
    provider_widget = widget._fields["provider"]
    assert provider_widget.count() == 4  # generic / doubao / gptge / custom
    widget._set_field("provider", "gptge")
    assert widget._get_field("provider") == "gptge"
    widget._set_field("base_url", "https://api.gpt.ge")
    widget._set_field("api_key", "gptge-key")
    widget._set_field("model", "doubao-seedance-1-5-pro-251215")
    widget._on_save()
    cfg = ctx.api.get_default("video")
    assert cfg.params["provider"] == "gptge"
    assert cfg.base_url == "https://api.gpt.ge"
    # 重新加载表单仍显示 gptge
    widget._load_config(ctx.api.get_default("video"))
    assert widget._get_field("provider") == "gptge"


def test_solo_page_defaults_and_action_frames(qtbot, ctx):
    """Solo 页默认 1s 动画；选择预设动作自动设置建议帧数；含「图转视频参数」分组。"""
    from ui.pages.solo_page import SoloPage

    page = SoloPage(ctx)
    qtbot.addWidget(page)
    page.show()
    # 默认 8 帧 @ 8fps = 1s；播放速度默认 1x
    assert page._frames_spin.value() == 8
    assert page._fps_spin.value() == 8
    assert page._speed_combo.currentData() == 1.0
    # 选择预设动作 -> 帧数自动跟随建议时长
    page._action_combo.setCurrentText("步行")
    assert page._frames_spin.value() == 16  # 2.0s * 8fps
    page._action_combo.setCurrentText("攻击")
    assert page._frames_spin.value() == 10  # 1.2s * 8fps
    # 自定义动作不干预
    page._action_combo.setCurrentText("自定义动作")
    assert page._frames_spin.value() == 10


def test_action_combo_grouped_presets(qtbot, ctx):
    """动作下拉按分类分组：分类为禁用表头，动作项紧随其后且 userData 为中文 ID。"""
    from PySide6.QtCore import Qt

    from ui.pages.solo_page import SoloPage

    page = SoloPage(ctx)
    qtbot.addWidget(page)
    combo = page._action_combo
    model = combo.model()
    assert combo.itemText(0) == ""  # 空首项 = 不选动作
    texts = [combo.itemText(i) for i in range(combo.count())]
    # 六个分类表头（禁用）
    for cat in ("待机", "移动", "战斗", "魔法", "表情", "互动"):
        idx = next((i for i, t in enumerate(texts) if "— " + cat in t), None)
        assert idx is not None, f"缺少分类表头 {cat}"
        assert not (model.item(idx).flags() & Qt.ItemFlag.ItemIsEnabled)
    # 分类表头下方紧跟该分类第一个动作（userData 为中文 ID）
    head_idx = next(i for i, t in enumerate(texts) if "— 待机" in t)
    assert combo.itemData(head_idx + 1) == "站立待机"
    # 动作总数 = 1 空项 + 6 表头 + 全部预设动作
    from core.processing.prompt_utils import preset_names

    assert combo.count() == 1 + 6 + len(preset_names())
    assert combo.count() >= 50


def test_solo_worker_end_to_end(qtbot, ctx, tmp_out):
    """通过后台线程跑完整 Solo 流程（mock API），验证 UI 侧入口可用。"""
    params = SoloParams(
        description="一只拿剑的橙色小猫",
        action="步行",
        frame_count=4,
        fps=8,
        pixel_size=64,
        max_colors=8,
        output_dir=tmp_out,
    )
    worker = SoloWorker(ctx.api, params)
    with qtbot.waitSignal(worker.succeeded, timeout=120_000) as blocker:
        worker.start()
    result = blocker.args[0]
    assert result.gif_path is not None
    assert result.gif_path.exists()
    assert result.frame_count == 4


def test_solo_worker_missing_config_fails(qtbot, ctx, tmp_out):
    """未配置任何 API 时报错而不是崩溃。"""
    for kind in ("llm", "image", "video"):
        for cfg in list(ctx.api.list(kind)):
            ctx.api.delete(cfg.id)
    params = SoloParams(description="x", output_dir=tmp_out)
    worker = SoloWorker(ctx.api, params)
    with qtbot.waitSignal(worker.failed, timeout=30_000) as blocker:
        worker.start()
    assert "未配置" in blocker.args[0]


def test_solo_worker_intermediate_signals(qtbot, ctx, tmp_out):
    """SoloWorker 应发出提示词与首帧图的中间结果信号。"""
    params = SoloParams(
        description="一只橙色小猫",
        action="步行",
        frame_count=4,
        fps=8,
        pixel_size=64,
        max_colors=8,
        output_dir=tmp_out,
    )
    worker = SoloWorker(ctx.api, params)
    received = {"prompts": [], "frames": []}
    worker.prompts_generated.connect(received["prompts"].append)
    worker.first_frame_ready.connect(received["frames"].append)
    with qtbot.waitSignal(worker.succeeded, timeout=120_000):
        worker.start()
    assert len(received["prompts"]) == 1
    prompts = received["prompts"][0]
    assert prompts["image_prompt"] and prompts["animation_prompt"] and prompts["negative_prompt"]
    assert len(received["frames"]) == 1
    from pathlib import Path

    assert Path(received["frames"][0]).exists()


def test_solo_page_intermediate_panel(qtbot, ctx, tmp_out):
    """Solo 页面「中间结果」面板：提示词展示、首帧图展示、帧缩略图。"""
    from PIL import Image as PILImage

    from ui.pages.solo_page import SoloPage

    page = SoloPage(ctx)
    qtbot.addWidget(page)
    page.show()

    # 提示词
    page._on_prompts({"image_prompt": "pixel cat", "animation_prompt": "walking", "negative_prompt": "blurry"})
    assert page._prompt_edits["image_prompt"].toPlainText() == "pixel cat"
    assert page._prompt_edits["animation_prompt"].toPlainText() == "walking"
    # 切换到左侧「中间结果」页签
    assert page._tabs.currentWidget() is page._intermediate_tab

    # 首帧图
    first_path = tmp_out / "first.png"
    PILImage.new("RGB", (32, 32), (255, 0, 0)).save(first_path)
    page._on_first_frame(str(first_path))
    assert page._tabs.currentWidget() is page._intermediate_tab
    assert page._first_frame_viewer._source_pixmap is not None
    assert not page._first_frame_viewer._source_pixmap.isNull()

    # 帧缩略图
    from core.workflow import SoloResult

    frames_dir = tmp_out / "frames"
    frames_dir.mkdir()
    for i in range(4):
        PILImage.new("RGBA", (16, 16), (i * 60, 30, 40, 255)).save(frames_dir / f"frame_{i:04d}.png")
    result = SoloResult(output_dir=tmp_out, frames_dir=frames_dir, frame_count=4, width=16, height=16)
    page._build_frames_strip(result)
    assert page._strip_count.text() == "4 帧"


def test_solo_page_preview_speed_combo(qtbot, ctx):
    """预览 GIF 支持倍速调整。"""
    from ui.pages.solo_page import SoloPage

    page = SoloPage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._preview_speed_combo is not None
    assert page._preview_speed_combo.currentData() == 1.0
    page._preview_speed_combo.setCurrentIndex(3)  # 2x
    assert page._preview_speed_combo.currentData() == 2.0


def test_ide_page_import_from_solo(qtbot, ctx, tmp_out):
    """Solo 生成的首帧图与最终帧序列同步到 IDE 工作区。"""
    from PIL import Image

    from core.workflow import SoloResult
    from ui.pages.ide_page import IdePage

    frames_dir = tmp_out / "frames"
    frames_dir.mkdir()
    for i in range(3):
        Image.new("RGBA", (16, 16), (i * 80, 0, 0, 255)).save(frames_dir / f"frame_{i:04d}.png")
    first = tmp_out / "first.png"
    Image.new("RGB", (64, 64), (0, 255, 0)).save(first)
    result = SoloResult(
        output_dir=tmp_out, first_frame=first, frames_dir=frames_dir,
        frame_count=3, width=16, height=16, fps=8,
    )

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    page.import_from_solo(result)
    assert len(page._session.frames) == 3
    assert page._session.frames[1].getpixel((0, 0)) == (80, 0, 0, 255)
    assert page._session.first_frame is not None
    assert page._session.first_frame.size == (64, 64)
    assert page._session.fps == 8
    assert page._dirty is True


def test_main_window_sync_switches_to_ide(qtbot, ctx, tmp_out):
    """主窗口收到 Solo 同步信号后切换到 IDE 模式。"""
    from PIL import Image

    from core.workflow import SoloResult
    from ui.main_window import MainWindow

    frames_dir = tmp_out / "frames"
    frames_dir.mkdir()
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(frames_dir / "frame_0000.png")
    result = SoloResult(output_dir=tmp_out, frames_dir=frames_dir, frame_count=1, width=8, height=8)

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    window._on_sync_to_ide(result)
    assert window._mode == "ide"
    assert len(window.ide_page._session.frames) == 1


# --------------------------------------------------------------------------- #
# IDE 模式
# --------------------------------------------------------------------------- #
def test_ide_page_structure(qtbot, ctx):
    """IDE 页：时间轴、像素编辑器、参数面板与执行按钮（步骤由主窗口侧栏驱动）。"""
    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._timeline is not None
    assert page._editor is not None
    assert page._btn_run.text() == "生成提示词"  # 默认第 1 步
    page.set_current_step(5)
    assert page._btn_run.text() == "导出"
    page.set_current_step(2)
    assert page._btn_run.text() == "生成动画"
    # 步骤切换发信号
    emitted = []
    page.step_changed.connect(emitted.append)
    page.set_current_step(1)
    assert emitted[-1] == 1


def test_ide_page_frame_editing_syncs_session(qtbot, ctx):
    """编辑器修改像素 -> session 帧同步、时间轴刷新。"""
    from PIL import Image

    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    page._session.frames = [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(3)]
    page._current = 0
    page._refresh_all()
    assert page._timeline.frame_count() == 3
    assert page._editor.frame().getpixel((0, 0)) == (255, 0, 0, 255)

    page._editor.canvas().set_pixel(5, 5, (0, 255, 0, 255))
    page._on_editor_edited()
    assert page._session.frames[0].getpixel((5, 5)) == (0, 255, 0, 255)


def test_ide_page_reorder_frames(qtbot, ctx):
    """时间轴拖拽重排 -> session 帧顺序同步，且选中帧跟随新位置。"""
    from PIL import Image

    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    red = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    green = Image.new("RGBA", (8, 8), (0, 255, 0, 255))
    blue = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
    page._session.frames = [red, green, blue]
    page._current = 1
    page._refresh_all()
    # 模拟拖拽：把 blue（原索引 2）拖到最前，选中项落在索引 0
    page._timeline._list.setCurrentRow(0)
    page._on_reordered([2, 0, 1])
    assert page._session.frames[0] is blue
    assert page._session.frames[1] is red
    assert page._current == 0  # 选中帧跟随到新位置


def test_ide_page_delete_guard_single_frame(qtbot, ctx, monkeypatch):
    """仅剩一帧时删除应被拦截。"""
    from PIL import Image

    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    page._session.frames = [Image.new("RGBA", (8, 8), (0, 0, 0, 255))]
    import PySide6.QtWidgets as qw

    monkeypatch.setattr(qw.QMessageBox, "information", lambda *a, **k: None)
    page._on_delete_frame()
    assert len(page._session.frames) == 1


def test_ide_step_worker_runs_and_emits(qtbot):
    """IdeStepWorker：后台执行函数并转发日志/结果。"""
    from ui.workers import IdeStepWorker

    logs = []

    def fn(log):
        log("info", "hello")
        return 42

    worker = IdeStepWorker(fn)
    worker.log.connect(lambda level, msg: logs.append((level, msg)))
    with qtbot.waitSignal(worker.succeeded, timeout=10_000) as blocker:
        worker.start()
    assert blocker.args[0] == 42
    assert logs == [("info", "hello")]


def test_ide_step_worker_reports_workflow_error(qtbot):
    """步骤抛 WorkflowError -> failed 信号带步骤名。"""
    from ui.workers import IdeStepWorker
    from core.workflow.solo_workflow import WorkflowError

    def fn(log):
        raise WorkflowError("没有可导出的帧", step="导出")

    worker = IdeStepWorker(fn)
    with qtbot.waitSignal(worker.failed, timeout=10_000) as blocker:
        worker.start()
    assert "没有可导出的帧" in blocker.args[0]
    assert "导出" in blocker.args[0]


def test_timeline_widget_basic(qtbot):
    """时间轴：刷新缩略图、选中、增删信号。"""
    from PIL import Image

    from ui.widgets.timeline import TimelineWidget

    widget = TimelineWidget()
    qtbot.addWidget(widget)
    widget.show()
    frames = [Image.new("RGBA", (8, 8), (i * 40, 0, 0, 255)) for i in range(4)]
    widget.set_frames(frames)
    assert widget.frame_count() == 4
    assert widget.current_index() == -1
    widget.select(2)
    assert widget.current_index() == 2

    signals = []
    widget.frame_selected.connect(signals.append)
    widget._list.setCurrentRow(1)
    assert signals[-1] == 1

    insert = []
    widget.insert_requested.connect(lambda: insert.append(True))
    widget._insert_btn.click()
    assert insert == [True]


def test_ide_page_run_step_integration(qtbot, ctx):
    """通过 IDE 页面执行「文本生成」步骤：客户端创建 + 后台线程 + 会话同步。"""
    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    page._desc_edit.setPlainText("一只橙色小猫")
    page.set_current_step(0)
    page._on_run_step()
    worker = page._worker
    with qtbot.waitSignal(worker.succeeded, timeout=30_000):
        pass
    assert page._session.prompts["image_prompt"]
    assert "EXACT 128x128 pixel grid" in page._session.prompts["image_prompt"]
    # 成功回调（队列连接）把提示词写入表单
    qtbot.waitUntil(lambda: page._prompt_edits["image_prompt"].toPlainText() != "")


# --------------------------------------------------------------------------- #
# 阶段三：洋葱皮 / 调色板锁定 / 导出选项
# --------------------------------------------------------------------------- #
def test_pixel_editor_onion_skin(qtbot):
    from PIL import Image

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    prev = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    nxt = Image.new("RGBA", (8, 8), (0, 255, 0, 255))
    editor.set_onion(prev, nxt)
    assert editor._onion_prev_qimg is not None
    assert editor._onion_next_qimg is not None
    assert editor.onion_enabled() is False
    editor.set_onion_enabled(True)
    assert editor.onion_enabled() is True
    assert editor._onion_btn.isChecked()


def test_pixel_editor_palette_lock_ui(qtbot):
    from PIL import Image

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    editor.set_frame(Image.new("RGBA", (4, 4), (10, 20, 30, 255)))
    editor._on_extract_palette()
    assert editor._palette_locked is True
    assert editor._canvas.palette is not None
    # 锁定后绘制吸附到最近锁定色
    editor._canvas.set_pixel(1, 1, (9, 20, 30, 255))
    assert editor._canvas.get_pixel(1, 1) == (10, 20, 30, 255)
    # 解锁
    editor._on_palette_lock_toggled(False)
    assert editor._canvas.palette is None


def test_ide_page_onion_wiring(qtbot, ctx):
    """选中帧时编辑器加载相邻帧为洋葱皮幽灵。"""
    from PIL import Image

    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    page._session.frames = [
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 255, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 0, 255, 255)),
    ]
    page._current = 1
    page._refresh_editor()
    assert page._editor._onion_prev_qimg is not None  # 帧 0
    assert page._editor._onion_next_qimg is not None  # 帧 2


def test_solo_page_export_options(qtbot, ctx):
    """Solo 页新增 APNG/雪碧图导出与背景容差/羽化参数。"""
    from ui.pages.solo_page import SoloPage

    page = SoloPage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._apng_chk is not None and page._apng_chk.isChecked() is False
    assert page._sprite_chk is not None and page._sprite_chk.isChecked() is False
    assert page._bg_tolerance_spin.value() == 30
    assert page._bg_feather_spin.value() == 8


def test_ide_page_import_reference(qtbot, ctx):
    """IDE 导入参考图：同时作为图生图参考与首帧图（无需全链路）。"""
    from PIL import Image

    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    img = Image.new("RGBA", (16, 16), (1, 2, 3, 255))
    page._on_ref_changed(img)
    assert page._session.reference_image is not None
    assert page._session.reference_image.size == (16, 16)
    assert page._session.first_frame is not None  # 同时作为首帧图
    assert page._session.first_frame.size == (16, 16)

    # 移除参考图
    page._on_ref_changed(None)
    assert page._session.reference_image is None


def test_reference_image_box(qtbot):
    """参考图卡片：set_image 显示、clear 发 changed(None)。"""
    from PIL import Image

    from ui.widgets.reference_box import ReferenceImageBox

    box = ReferenceImageBox(size=72)
    qtbot.addWidget(box)
    box.show()
    img = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
    box.set_image(img)
    assert box.image() is not None
    assert box.image().size == (8, 8)
    events = []
    box.changed.connect(lambda v: events.append(v))
    box.clear()
    assert box.image() is None
    assert events == [None]


def test_solo_page_reference_box(qtbot, ctx):
    """Solo 页支持参考图卡片。"""
    from PIL import Image

    from ui.pages.solo_page import SoloPage

    page = SoloPage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._ref_box is not None
    page._ref_box.set_image(Image.new("RGBA", (8, 8), (9, 8, 7, 255)))
    assert page._ref_box.image() is not None


def test_sprite_page_structure(qtbot, ctx):
    """精灵图页：帧数 / i×j 网格 / 单格尺寸 / 一键抠图。"""
    from ui.pages.sprite_page import SpritePage

    page = SpritePage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._frames_spin.value() == 16
    assert page._rows_spin.value() == 4 and page._cols_spin.value() == 4
    assert page._size_combo.currentText() == "64"
    assert page._bg_chk.isChecked()  # 一键抠图默认开
    assert page._btn_start.text() == "生成精灵图"
    assert page._tabs.count() == 3  # 底图 / 精灵图 / 帧序列


def test_sprite_worker_end_to_end(qtbot, ctx, tmp_out):
    """通过后台线程跑完整精灵图流程（mock API）。"""
    from core.workflow import SpriteParams
    from ui.workers import SpriteWorker

    params = SpriteParams(
        description="一只小猫", action="步行", frame_count=4, grid_rows=2, grid_cols=2, output_dir=tmp_out
    )
    worker = SpriteWorker(ctx.api, params)
    with qtbot.waitSignal(worker.succeeded, timeout=60_000) as blocker:
        worker.start()
    result = blocker.args[0]
    assert result.frame_count == 4
    assert result.gif_path is not None and result.gif_path.exists()


def test_editor_palette_shows_colors_by_frequency(qtbot):
    """调色板按图片颜色频次展示（最多 PALETTE_SHOW 个 + 省略号）。"""
    from PIL import Image
    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()

    # 主色红色(256 像素) + 少量绿色(4 像素)
    img = Image.new("RGBA", (16, 16), (200, 30, 30, 255))
    for i in range(4):
        img.putpixel((i, 0), (20, 200, 30, 255))
    editor.set_frame(img)
    editor._refresh_palette()

    counts = editor._extract_color_counts()
    assert counts[0][1] == (200, 30, 30, 255)  # 最高频
    assert counts[1][1] == (20, 200, 30, 255)
    assert len(editor._palette_swatches) == 2  # 当前色 + 顶部颜色
    assert editor._palette_more_btn is not None  # 省略号按钮


def test_editor_replace_color_global_ui(qtbot):
    """右键替换颜色：整图同色替换 + 撤销恢复。"""
    from PIL import Image
    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()

    img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    editor.set_frame(img)
    editor._refresh_palette()

    n = editor.canvas().replace_color((255, 0, 0, 255), (0, 0, 255, 255))
    assert n == 64
    editor._rebuild()
    assert editor.frame().getpixel((3, 3)) == (0, 0, 255, 255)
    # 撤销后恢复原色
    editor.undo()
    assert editor.frame().getpixel((3, 3)) == (255, 0, 0, 255)


def test_main_window_sync_sprite_to_ide(qtbot, ctx, tmp_out):
    """精灵图结果一键同步到 IDE 模式并切换过去。"""
    from PIL import Image
    from core.workflow import SpriteResult
    from ui.main_window import MainWindow

    frames_dir = tmp_out / "frames"
    frames_dir.mkdir()
    for i in range(2):
        Image.new("RGBA", (8, 8), (i * 50, 0, 0, 255)).save(
            frames_dir / f"frame_{i:04d}.png"
        )
    result = SpriteResult(
        output_dir=tmp_out, frames_dir=frames_dir, frame_count=2, width=8, height=8
    )

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    window._on_sync_sprite_to_ide(result)
    assert window._mode == "ide"
    assert len(window.ide_page._session.frames) == 2
    assert window.ide_page._session.frames[0] is not None


def test_image_viewer_zoom(qtbot):
    """预览缩放：放大 / 缩小 / 复位。"""
    from PySide6.QtGui import QPixmap
    from ui.widgets.image_viewer import ImageViewer

    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    viewer.show()
    viewer.show_image(QPixmap(100, 100))
    assert viewer.zoom() == 1.0
    viewer.zoom_in()
    assert viewer.zoom() > 1.0
    viewer.zoom_out()
    viewer.zoom_out()
    assert viewer.zoom() < 1.0
    viewer.reset_zoom()
    assert viewer.zoom() == 1.0


def test_ide_preview_zoom_buttons(qtbot, ctx):
    """IDE 预览页具备缩放按钮且可联动预览。"""
    from PySide6.QtGui import QPixmap

    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._zoom_label is not None
    page._preview.show_image(QPixmap(200, 150))
    page._preview.reset_zoom()
    page._preview.zoom_in()
    assert page._preview.zoom() > 1.0


def test_image_viewer_wheel_zoom_focus(qtbot):
    """滚轮缩放：以鼠标位置为焦点（光标下的图像点保持不动）。"""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QPixmap, QWheelEvent
    from PySide6.QtWidgets import QApplication

    from ui.widgets.image_viewer import ImageViewer

    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(300, 300)
    viewer.show()
    viewer.show_image(QPixmap(400, 400))

    def img_coords(v, wx, wy):
        scale = v._current_scale()
        return (wx - v._ox) / (v._source_pixmap.width() * scale), (wy - v._oy) / (v._source_pixmap.height() * scale)

    cursor = QPointF(210, 140)
    before = img_coords(viewer, cursor.x(), cursor.y())
    event = QWheelEvent(
        cursor, cursor, QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(viewer, event)
    assert viewer.zoom() > 1.0
    after = img_coords(viewer, cursor.x(), cursor.y())
    assert abs(after[0] - before[0]) < 0.01 and abs(after[1] - before[1]) < 0.01


def test_pixel_editor_palette_shows_families(qtbot):
    """调色板按色族显示：相近色合并为一个色族色块。"""
    from PIL import Image

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    # 红族两色（相近）+ 白色 + 绿色 -> 3 个色族
    img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    for i in range(8):
        img.putpixel((i, 0), (250, 10, 5, 255))
    img.putpixel((0, 1), (255, 255, 255, 255))
    img.putpixel((1, 1), (0, 200, 0, 255))
    editor.set_frame(img)
    families = editor._families()
    assert len(families) == 3  # 红色族（两色合并）/ 白 / 绿
    # 色块数 = 色族数（≤ PALETTE_SHOW）
    assert len(editor._palette_swatches) == 3
    # 第一族代表色为最高频红
    assert families[0][0] == (255, 0, 0, 255)
    assert len(families[0][2]) == 2  # 族内包含相近红


def test_ide_step_params_switch_with_step(qtbot, ctx):
    """IDE 右侧参数面板随左侧步骤切换：只显示对应步骤的参数。"""
    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._step_params.count() == 6
    page.set_current_step(0)
    assert page._step_params.currentIndex() == 0
    page.set_current_step(2)
    assert page._step_params.currentIndex() == 2
    page.set_current_step(5)
    assert page._step_params.currentIndex() == 5
    assert page._btn_run.text() == "导出"


def test_ide_params_panel_default_collapsed(qtbot, ctx):
    """IDE 右侧参数面板（提示词/文生图等）默认收起，仅三角钮；点击展开。"""
    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._params_collapsed is True
    assert not page._params_scroll.isVisible()
    page._on_toggle_params()
    assert page._params_collapsed is False
    assert page._params_scroll.isVisible()
    page._on_toggle_params()
    assert page._params_collapsed is True
    assert not page._params_scroll.isVisible()


def test_ide_log_collapsible(qtbot, ctx):
    """IDE 底部日志框可收起/展开。"""
    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._log_collapsed is False
    assert page._log_view.isVisible()
    page._on_toggle_log()
    assert page._log_collapsed is True
    assert not page._log_view.isVisible()
    page._on_toggle_log()
    assert page._log_collapsed is False
    assert page._log_view.isVisible()


def test_pixel_editor_set_zoom_with_focus(qtbot):
    """画布缩放以焦点为中心：保持焦点处画布格不动。"""
    from PySide6.QtCore import QPoint

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    editor._canvas_host.resize(400, 300)
    editor._set_zoom(2, focus=QPoint(200, 150))
    assert editor._zoom == 2
    # 焦点处的画布坐标在缩放前后一致
    x_before = (200 - editor._canvas_host._ox) / 2
    y_before = (150 - editor._canvas_host._oy) / 2
    assert abs(x_before - round(x_before)) < 1e-6 and abs(y_before - round(y_before)) < 1e-6


def test_color_wheel_geometry():
    """取色圆盘坐标映射：色环=色相，方块=饱和度/明度，底条=最近色。"""
    from PySide6.QtCore import QPoint

    from ui.widgets.color_wheel import ColorWheelPopup

    popup = ColorWheelPopup((255, 0, 0, 255), recent=[(1, 2, 3, 255), (4, 5, 6, 255)])
    c = popup.SIZE / 2
    mr = (popup.R_OUT + popup.R_IN) / 2
    # 色环 0° -> 红；90°（dy>0）-> 黄绿
    assert popup._color_at(QPoint(int(c + mr), int(c)))[:3] == (255, 0, 0)
    assert popup._color_at(QPoint(int(c), int(c + mr)))[:3] == (128, 255, 0)
    # 方块中心 -> 当前色相 50% 饱和 / 50% 明度
    sq = popup._sq0 + popup.SQUARE / 2
    mid = popup._color_at(QPoint(int(sq), int(sq)))
    assert mid[:3] == (128, 64, 64)  # 红(255,0,0) 50% s / 50% v
    # 底条第一个最近色
    strip = popup._color_at(QPoint(12, popup.SIZE + popup.PAD + popup.STRIP_H // 2))
    assert strip == (1, 2, 3, 255)


def test_color_wheel_pick_and_commit(qtbot):
    """取色圆盘：移动实时预览，松开提交颜色。"""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    from ui.widgets.color_wheel import ColorWheelPopup

    popup = ColorWheelPopup((255, 255, 255, 255), recent=[])
    qtbot.addWidget(popup)
    popup.show()
    previews = []
    popup.color_preview.connect(previews.append)
    commits = []
    popup.color_selected.connect(commits.append)

    c = popup.SIZE / 2
    mr = (popup.R_OUT + popup.R_IN) / 2
    pos = QPointF(c + mr, c)  # 色环 0° -> 红
    QApplication.sendEvent(
        popup,
        QMouseEvent(QEvent.Type.MouseMove, pos, QPointF(0, 0), Qt.MouseButton.NoButton,
                    Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier),
    )
    assert previews and previews[-1] == (255, 0, 0, 255)
    QApplication.sendEvent(
        popup,
        QMouseEvent(QEvent.Type.MouseButtonRelease, pos, QPointF(0, 0), Qt.MouseButton.RightButton,
                    Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier),
    )
    assert commits and commits[-1] == (255, 0, 0, 255)


def test_editor_right_click_opens_wheel(qtbot):
    """画布右键弹出取色圆盘；提交后更新当前颜色。"""
    from PySide6.QtCore import QPoint

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    editor.open_color_wheel(QPoint(200, 200))
    assert editor._wheel_popup is not None
    editor._on_wheel_commit((10, 20, 30, 255))
    assert editor.color() == (10, 20, 30, 255)
    assert editor._recent_colors[0] == (10, 20, 30, 255)
    editor._wheel_popup.close()


def test_editor_recent_colors_capped(qtbot):
    """最近使用色去重并限制 10 个。"""
    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    editor.set_color((255, 0, 0, 255))
    editor.set_color((0, 255, 0, 255))
    editor.set_color((255, 0, 0, 255))  # 重复 -> 移到最前
    assert editor._recent_colors == [(255, 0, 0, 255), (0, 255, 0, 255)]
    for i in range(12):
        editor.set_color((i, i, i, 255))
    assert len(editor._recent_colors) == 10


def _make_editor_with_painted_canvas(qtbot, size=(400, 300)):
    """构造已排版/已绘制一次的像素编辑器（_ox/_oy 有效）。"""
    from PIL import Image

    from PySide6.QtGui import QPaintEvent

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    editor.set_frame(Image.new("RGBA", (16, 16), (255, 255, 255, 255)))
    host = editor._canvas_host
    host.resize(*size)
    host.paintEvent(QPaintEvent(host.rect()))  # 初始化居中偏移
    return editor, host


def _mouse_ev(ev_type, pos, button, modifiers):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(ev_type, QPointF(pos), QPointF(pos), button, button, modifiers)


def test_editor_ctrl_left_drag_pans_canvas(qtbot):
    """Ctrl + 左键拖动 = 平移画布视图。"""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    editor, host = _make_editor_with_painted_canvas(qtbot)
    QApplication.sendEvent(
        host,
        _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(200, 150), Qt.MouseButton.LeftButton,
                  Qt.KeyboardModifier.ControlModifier),
    )
    QApplication.sendEvent(
        host,
        _mouse_ev(QEvent.Type.MouseMove, QPoint(212, 160), Qt.MouseButton.LeftButton,
                  Qt.KeyboardModifier.ControlModifier),
    )
    assert editor._pan_x == 12 and editor._pan_y == 10
    QApplication.sendEvent(
        host,
        _mouse_ev(QEvent.Type.MouseButtonRelease, QPoint(212, 160), Qt.MouseButton.LeftButton,
                  Qt.KeyboardModifier.ControlModifier),
    )
    assert not editor._panning


def test_editor_right_drag_select_fill(qtbot):
    """右键拖拽框选区域并填充当前颜色。"""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    editor, host = _make_editor_with_painted_canvas(qtbot)
    editor.set_color((255, 0, 0, 255))
    # 画布 16x16 @ zoom1，居中偏移 _ox=192, _oy=142
    ox, oy = host._ox, host._oy
    QApplication.sendEvent(
        host,
        _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(ox + 2, oy + 2), Qt.MouseButton.RightButton,
                  Qt.KeyboardModifier.NoModifier),
    )
    QApplication.sendEvent(
        host,
        _mouse_ev(QEvent.Type.MouseMove, QPoint(ox + 5, oy + 5), Qt.MouseButton.RightButton,
                  Qt.KeyboardModifier.NoModifier),
    )
    assert editor._rb_cell0 == (2, 2) and editor._rb_cell1 == (5, 5)
    QApplication.sendEvent(
        host,
        _mouse_ev(QEvent.Type.MouseButtonRelease, QPoint(ox + 5, oy + 5), Qt.MouseButton.RightButton,
                  Qt.KeyboardModifier.NoModifier),
    )
    # 框选区域被当前颜色填充
    assert editor.canvas().get_pixel(2, 2) == (255, 0, 0, 255)
    assert editor.canvas().get_pixel(5, 5) == (255, 0, 0, 255)
    assert editor.canvas().get_pixel(0, 0) == (255, 255, 255, 255)  # 区域外不变
    assert not editor._rb_active
    # 可撤销
    editor.undo()
    assert editor.canvas().get_pixel(2, 2) == (255, 255, 255, 255)


def test_editor_select_copy_move_merge(qtbot):
    """选择档：框选 -> Ctrl+C 复制浮动图层 -> 移动 -> Ctrl+M 合并（复制保留原图）。"""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    from ui.widgets.pixel_editor import Tool

    editor, host = _make_editor_with_painted_canvas(qtbot)
    editor.canvas().set_pixel(2, 2, (255, 0, 0))
    editor.canvas().set_pixel(3, 3, (255, 0, 0))
    editor._rebuild()
    ox, oy = host._ox, host._oy
    editor.set_tool(Tool.SELECT)
    # 框选 (2,2)-(3,3)
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(ox + 2, oy + 2),
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseMove, QPoint(ox + 3, oy + 3),
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonRelease, QPoint(ox + 3, oy + 3),
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert editor._selection is not None and editor._selection.sum() == 4
    # Ctrl+C 复制到剪贴板（原图与选区保留），Ctrl+V 粘贴为浮动图层
    editor._copy_selection()
    assert editor._clipboard is not None and editor._clipboard.size == (2, 2)
    assert editor._float_layer is None  # 复制不直接建层
    editor._paste_layer()
    assert editor._float_layer is not None
    assert editor._float_layer.size == (2, 2)
    assert editor._float_pos == (2, 2)  # 显示在原选区位置
    assert editor._float_opacity < 1.0  # 半透明显示
    assert editor.canvas().get_pixel(2, 2) == (255, 0, 0, 255)  # 复制非剪切
    # 移动浮动图层到 (6,6) 后合并
    editor._float_pos = (6, 6)
    editor._merge_float_layer()
    assert editor._float_layer is None and editor._selection is None
    assert editor.canvas().get_pixel(6, 6) == (255, 0, 0, 255)
    assert editor.canvas().get_pixel(7, 7) == (255, 0, 0, 255)
    assert editor.canvas().get_pixel(2, 2) == (255, 0, 0, 255)  # 原位置仍保留
    # 合并可撤销
    editor.undo()
    assert editor.canvas().get_pixel(6, 6) == (255, 255, 255, 255)


def test_editor_select_ctrl_click_toggle(qtbot):
    """选择档：Ctrl+左键逐个切换像素选择状态。"""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    from ui.widgets.pixel_editor import Tool

    editor, host = _make_editor_with_painted_canvas(qtbot)
    editor.set_tool(Tool.SELECT)
    ox, oy = host._ox, host._oy
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(ox + 3, oy + 4),
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier))
    assert bool(editor._selection[4, 3]) is True
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(ox + 3, oy + 4),
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier))
    assert bool(editor._selection[4, 3]) is False


def test_editor_lasso_selection(qtbot):
    """选择档：套索多边形选区。"""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    from ui.widgets.pixel_editor import Tool

    editor, host = _make_editor_with_painted_canvas(qtbot)
    editor.set_tool(Tool.SELECT)
    editor._set_sel_mode("lasso")
    ox, oy = host._ox, host._oy
    pts = [(2, 2), (6, 2), (6, 6), (2, 6)]
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(ox + 2, oy + 2),
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    for x, y in pts[1:]:
        QApplication.sendEvent(
            host, _mouse_ev(QEvent.Type.MouseMove, QPoint(ox + x, oy + y),
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonRelease, QPoint(ox + 6, oy + 6),
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert editor._selection is not None and editor._selection.sum() > 0
    assert bool(editor._selection[4, 4]) is True  # 多边形内部被选中


def test_editor_keyboard_shortcuts(qtbot):
    """键盘快捷键：Ctrl+C 复制 / Ctrl+V 粘贴半透明图层 / Ctrl+M 合并 / Ctrl+Z 撤销 / Esc 清选。"""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from ui.widgets.pixel_editor import Tool

    editor, _host = _make_editor_with_painted_canvas(qtbot)
    editor.canvas().set_pixel(0, 0, (255, 0, 0))
    editor.set_tool(Tool.SELECT)
    editor._select_all()
    QTest.keyClick(editor, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert editor._clipboard is not None and editor._float_layer is None
    QTest.keyClick(editor, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert editor._float_layer is not None
    assert editor._float_opacity < 1.0  # 半透明显示
    editor._float_pos = (1, 0)  # 向右移一格（实时拖拽由 _on_float_move 处理）
    QTest.keyClick(editor, Qt.Key.Key_M, Qt.KeyboardModifier.ControlModifier)
    assert editor._float_layer is None
    assert editor.canvas().get_pixel(1, 0) == (255, 0, 0, 255)  # 合并后红点移到新位置
    QTest.keyClick(editor, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert editor.canvas().get_pixel(1, 0) == (255, 255, 255, 255)  # 撤销合并
    editor._select_all()
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    assert editor._selection is None


def test_editor_right_quick_click_opens_wheel(qtbot):
    """右键快速点击（未拖动）仍弹出取色圆盘。"""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    editor, host = _make_editor_with_painted_canvas(qtbot)
    QApplication.sendEvent(
        host,
        _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(200, 150), Qt.MouseButton.RightButton,
                  Qt.KeyboardModifier.NoModifier),
    )
    QApplication.sendEvent(
        host,
        _mouse_ev(QEvent.Type.MouseButtonRelease, QPoint(200, 150), Qt.MouseButton.RightButton,
                  Qt.KeyboardModifier.NoModifier),
    )
    assert editor._wheel_popup is not None
    editor._wheel_popup.close()


def test_editor_background_cycle(qtbot):
    """背景循环切换：灰黑网格 -> 纯白 -> 纯黑 -> 纯绿。"""
    from ui.widgets.pixel_editor import BG_MODES
    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    assert editor._bg_mode == "checker"
    editor._on_bg_cycle()
    assert editor._bg_mode == "white"
    editor._on_bg_cycle()
    assert editor._bg_mode == "black"
    editor._on_bg_cycle()
    assert editor._bg_mode == "green"
    editor._on_bg_cycle()  # 回到起点
    assert editor._bg_mode == BG_MODES[0][0]


def test_editor_grid_toggle(qtbot):
    """网格显示/隐藏开关。"""
    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    assert editor._grid_visible is True
    editor._on_grid_toggled(False)
    assert editor._grid_visible is False
    editor._on_grid_toggled(True)
    assert editor._grid_visible is True


def test_editor_side_panel_collapse(qtbot):
    """编辑器右侧图标列默认展开显示，可点击收起/展开。"""
    from ui.widgets.pixel_editor import SIDE_MIN, SIDE_W
    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    # 默认展开
    assert editor._side_collapsed is False
    assert editor._side_panel.width() == SIDE_W
    editor._on_toggle_side()
    assert editor._side_collapsed is True
    assert editor._side_panel.width() == SIDE_MIN
    assert not editor._tool_buttons[list(editor._tool_buttons)[0]].isVisible()
    editor._on_toggle_side()
    assert editor._side_collapsed is False
    assert editor._side_panel.width() == SIDE_W


def test_editor_palette_bar_toggle(qtbot):
    """底部调色板条可隐藏/显示。"""
    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    assert editor._palette_bar.isVisible()
    editor._on_toggle_palette(False)
    assert not editor._palette_bar.isVisible()
    editor._on_toggle_palette(True)
    assert editor._palette_bar.isVisible()
    editor._on_collapse_palette()
    assert not editor._palette_bar.isVisible()


def test_ide_preview_speed_control(qtbot, ctx):
    """IDE 预览支持播放倍速调整。"""
    from ui.pages.ide_page import IdePage

    page = IdePage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page._preview_speed_combo is not None
    assert page._preview_speed() == 1.0
    idx = page._preview_speed_combo.findData(2.0)
    page._preview_speed_combo.setCurrentIndex(idx)
    assert page._preview_speed() == 2.0


def test_editor_float_layer_prominent_border(qtbot):
    """浮动图层突出显示（半透明）；粘贴后清空选区、合并后清除。"""
    from PIL import Image

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    editor.set_frame(Image.new("RGBA", (16, 16), (255, 255, 255, 255)))
    editor._selection = __import__("numpy").zeros((16, 16), dtype=bool)
    editor._selection[2:6, 2:6] = True
    editor._copy_selection()
    editor._paste_layer()
    assert editor._float_layer is not None
    assert editor._float_qimg is not None
    assert editor._float_opacity < 1.0          # 半透明显示新图层
    assert editor._selection is None            # 选区已清空，只突出浮动层
    editor._merge_float_layer()
    assert editor._float_layer is None


def test_editor_selection_border_segments(qtbot):
    """选区边框为屏幕空间线段（细线），非逐格分辨率。"""
    import numpy as np

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    editor._selection = np.zeros((16, 16), dtype=bool)
    editor._selection[2:6, 2:6] = True
    editor._rebuild_sel_overlay()
    segs = editor._sel_border_segments
    assert len(segs) > 0
    # 4x4 方块 -> 周长 16 条格边线段
    assert len(segs) == 16
    # 高亮为逐格半透明蓝
    assert editor._sel_hl_qimg is not None
    editor._clear_selection()
    assert editor._sel_border_segments == []


def test_background_key_dialog_live_preview(qtbot):
    """背景抠图预览弹窗：实时预览 + 参数读取。"""
    from PIL import Image

    from ui.dialogs.background_key_dialog import BackgroundKeyDialog

    img = Image.new("RGBA", (24, 24), (255, 255, 255, 255))
    for y in range(6, 18):
        for x in range(6, 18):
            img.putpixel((x, y), (200, 50, 50, 255))
    dialog = BackgroundKeyDialog(img, tolerance=30, feather=8, erode=1)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog._update_preview()
    assert dialog._viewer._source_pixmap is not None
    p = dialog.params()
    assert p["tolerance"] == 30 and p["erode"] == 1 and p["feather"] == 8
    dialog._erode_spin.setValue(3)
    p = dialog.params()
    assert p["erode"] == 3
    dialog.close()


def test_pixel_page_structure_and_actions(qtbot, ctx, tmp_out):
    """像素板块：新建画布（分辨率/背景）、图片载入、信号发出。"""
    from PIL import Image

    from ui.pages.pixel_page import PixelPage

    page = PixelPage(ctx)
    qtbot.addWidget(page)
    page.show()
    # 新建画布（自定义分辨率 + 白底）
    page._custom_w.setValue(32)
    page._custom_h.setValue(16)
    page._bg_combo.setCurrentText("白色")
    page._on_new()
    assert page.image().size == (32, 16)
    assert page.image().getpixel((0, 0)) == (255, 255, 255, 255)
    # 载入外部图
    page.set_image(Image.new("RGBA", (8, 8), (1, 2, 3, 255)))
    assert page.image().size == (8, 8)
    # 信号
    to_ide, to_video = [], []
    page.sync_to_ide.connect(to_ide.append)
    page.use_as_video_first_frame.connect(to_video.append)
    page._on_sync_to_ide()
    page._on_use_as_video()
    assert len(to_ide) == 1 and len(to_video) == 1
    assert to_ide[0].size == (8, 8)
    # 导入/导出按钮位于左侧操作栏
    assert page._btn_import is not None and page._btn_export is not None
    # 导入走编辑器导入逻辑（传路径免弹窗）
    imp_path = tmp_out / "imp.png"
    Image.new("RGBA", (10, 6), (9, 8, 7, 255)).save(imp_path)
    page._editor.import_image(str(imp_path))
    assert page.image().size == (10, 6)


def test_main_window_pixel_mode_sync(qtbot, ctx):
    """像素板块与 IDE / Solo 联动：从 IDE 同步、同步到 IDE、作图生视频首帧。"""
    from PIL import Image

    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    window.ide_page._session.frames = [Image.new("RGBA", (16, 16), (10, 20, 30, 255))]
    # 像素 -> IDE（画布图作为首帧）
    window.pixel_page.set_image(Image.new("RGBA", (8, 8), (9, 8, 7, 255)))
    window._on_pixel_sync_to_ide(window.pixel_page.image())
    assert window._mode == "ide"
    assert window.ide_page._session.first_frame.size == (8, 8)
    # 像素 -> 图生视频首帧（Solo 参考图）
    window.pixel_page.set_image(Image.new("RGBA", (8, 8), (6, 5, 4, 255)))
    window._on_pixel_to_video(window.pixel_page.image())
    assert window._mode == "solo"
    assert window.solo_page._ref_box.image() is not None
    assert window.solo_page._ref_box.image().size == (8, 8)
    # 从 IDE 同步回像素板块
    window._on_pixel_sync_from_ide()
    assert window.pixel_page.image().size == (16, 16)


def test_editor_move_selection_by_ctrl_right_drag(qtbot):
    """选中像素后（无需复制）Ctrl+右键拖拽：自动提起为浮动层并实时移动（任意工具）。"""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    from ui.widgets.pixel_editor import Tool

    editor, host = _make_editor_with_painted_canvas(qtbot)
    ox, oy = host._ox, host._oy
    # 非选择工具 + 已有选区
    editor.set_tool(Tool.PENCIL)
    editor._selection = __import__("numpy").zeros((16, 16), dtype=bool)
    editor._selection[2:5, 2:5] = True
    # Ctrl+右键按下 -> 自动提起为浮动层
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(ox + 3, oy + 3),
                        Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier))
    assert editor._float_layer is not None
    assert editor._moving_float is True
    # 拖动实时更新位置
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseMove, QPoint(ox + 9, oy + 3),
                        Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier))
    assert editor._float_pos == (8, 2)
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonRelease, QPoint(ox + 9, oy + 3),
                        Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier))
    assert editor._moving_float is False
    assert editor._float_pos == (8, 2)


def test_editor_paste_layer_ctrl_right_drag(qtbot):
    """粘贴出的图层：Ctrl+右键拖拽实时移动（选择档）。"""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtWidgets import QApplication

    from ui.widgets.pixel_editor import Tool

    editor, host = _make_editor_with_painted_canvas(qtbot)
    ox, oy = host._ox, host._oy
    editor.set_tool(Tool.SELECT)
    editor._selection = __import__("numpy").zeros((16, 16), dtype=bool)
    editor._selection[6:8, 6:8] = True
    editor._copy_selection()
    editor._paste_layer()
    assert editor._float_pos == (6, 6)
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonPress, QPoint(ox + 7, oy + 7),
                        Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier))
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseMove, QPoint(ox + 12, oy + 7),
                        Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier))
    assert editor._float_pos == (11, 6)
    QApplication.sendEvent(
        host, _mouse_ev(QEvent.Type.MouseButtonRelease, QPoint(ox + 12, oy + 7),
                        Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier))


def test_editor_import_export_image(qtbot, tmp_path):
    """像素编辑器：本地导入图片替换当前帧、导出 PNG 可读回（往返一致）。"""
    from pathlib import Path

    from PIL import Image

    from ui.widgets.pixel_editor import PixelEditorWidget

    editor = PixelEditorWidget()
    qtbot.addWidget(editor)
    editor.show()
    # 导入
    src = tmp_path / "in.png"
    Image.new("RGBA", (12, 8), (200, 30, 30, 255)).save(src)
    img = editor.import_image(str(src))
    assert img is not None
    assert editor.frame().size == (12, 8)
    assert editor.frame().getpixel((0, 0)) == (200, 30, 30, 255)
    # 导出
    out = tmp_path / "out.png"
    p = editor.export_image(str(out))
    assert p is not None and Path(out).exists()
    loaded = Image.open(out).convert("RGBA")
    assert loaded.size == (12, 8)
    assert loaded.getpixel((5, 4)) == (200, 30, 30, 255)


def test_ui_scale_setting_scales_layout(qtbot, ctx):
    """界面布局比例：缩放侧栏宽度、模式开关填满侧栏、IDE 参数面板宽度。"""
    from ui.main_window import RAIL_COLLAPSED

    ctx.ui_settings.set("ui_scale", 1.25)
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    assert window._scale == 1.25
    # 侧栏按比例放大
    assert window._sidebar.width() == int(RAIL_COLLAPSED * 1.25)
    # 模式开关横向填满侧栏可用宽度（8px 内边距内不留缝隙、不压边框）
    assert window._mode_switch.width() >= window._sidebar.width() - 20
    assert window._mode_switch.width() <= window._sidebar.width()
    # 编辑器控件尺寸同步放大（UI 大小随比例缩放）
    from ui.widgets.pixel_editor import Tool

    btn = window.pixel_page._editor._tool_buttons[Tool.PENCIL]
    assert btn.width() > 40  # 基准 34 → 1.25× 约 42
    assert btn.height() > 40
    # 切回标准比例
    ctx.ui_settings.set("ui_scale", 1.0)
    window._apply_ui_scale()
    assert window._sidebar.width() == RAIL_COLLAPSED
    assert window.pixel_page._editor._tool_buttons[Tool.PENCIL].width() == 34


def test_language_switch_applies_immediately(qtbot, ctx):
    """语言切换立即生效：设置保存后按钮/页签/分组标题同步变英文。"""
    from ui.i18n import set_language

    set_language("zh")
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    assert window.pixel_page._btn_export.text() == "导出 PNG"
    assert window.pixel_page._btn_new.text() == "新建画布"
    assert window.ide_page._btn_run.text() == "生成提示词"
    # 切英文 + retranslate（设置保存时触发）→ 立即生效
    set_language("en")
    window.retranslate_ui()
    assert window.pixel_page._btn_export.text() == "Export PNG"
    assert window.pixel_page._btn_new.text() == "New canvas"
    assert window.pixel_page._btn_video.text() == "Use as video first frame"
    assert window.ide_page._btn_run.text() == "Generate prompts"
    assert window.ide_page._tabs.tabText(0) == "Preview"
    # 复位中文
    set_language("zh")
    window.retranslate_ui()


def test_settings_dialog_has_save_button(qtbot, ctx):
    """设置弹窗具备「保存」按钮。"""
    from ui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog(ctx)
    qtbot.addWidget(dlg)
    assert dlg._save_settings is not None
