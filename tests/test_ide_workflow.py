"""IDE 工作流分步测试（模拟 API）。"""
from pathlib import Path

import pytest
from PIL import Image

from core.api.mock_clients import MockImageAPI, MockLLMAPI, MockVideoAPI
from core.processing import frame_utils as fu
from core.workflow import (
    IdeSession,
    IdeWorkflow,
    load_ide_project,
    save_ide_project,
)
from core.workflow.solo_workflow import WorkflowError


def make_wf():
    return IdeWorkflow(MockLLMAPI(), MockImageAPI(), MockVideoAPI())


def make_session(tmp_path, **overrides):
    base = dict(
        description="一只拿剑的橙色小猫",
        action="步行",
        aspect_ratio="1:1",
        pixel_size=64,
        frame_count=4,
        fps=8,
        max_colors=8,
        output_dir=tmp_path,
    )
    base.update(overrides)
    return IdeSession(**base)


def test_step_prompts_populates_and_injects_resolution(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)
    prompts = wf.step_prompts(s)
    assert prompts["image_prompt"] and prompts["animation_prompt"] and prompts["negative_prompt"]
    # 用户像素尺寸被严格写入提示词（共享 finalize_prompts）
    assert "EXACT 64x64 pixel grid" in prompts["image_prompt"]
    assert s.prompts == prompts


def test_step_prompts_fallback_on_llm_failure(tmp_path):
    from core.api.base import APIResult

    class FailingLLM(MockLLMAPI):
        def call(self, prompt, **kwargs):
            return APIResult(ok=False, error="down")

    wf = IdeWorkflow(FailingLLM(), MockImageAPI(), MockVideoAPI())
    s = make_session(tmp_path)
    prompts = wf.step_prompts(s)
    assert prompts["image_prompt"]  # fallback 提示词
    assert "EXACT 64x64 pixel grid" in prompts["image_prompt"]


def test_step_image_generates_first_frame(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)
    wf.step_prompts(s)
    img = wf.step_image(s)
    assert isinstance(img, Image.Image)
    assert s.first_frame is not None
    assert s.first_frame.size == img.size


def test_step_image_requires_prompt(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)  # 无提示词
    with pytest.raises(WorkflowError) as ei:
        wf.step_image(s)
    assert "提示词" in ei.value.message


def test_step_image_passes_reference_image(tmp_path):
    """文字生图支持参考图（图生图）：参考图字节传入 image API。"""
    captured = {}

    class CapturingImage(MockImageAPI):
        def call(self, prompt, size=None, n=1, **kwargs):
            captured["image"] = kwargs.get("image")
            return super().call(prompt, size=size, n=n, **kwargs)

    wf = IdeWorkflow(MockLLMAPI(), CapturingImage(), MockVideoAPI())
    s = make_session(tmp_path)
    wf.step_prompts(s)
    ref = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
    s.reference_image = ref
    img = wf.step_image(s)
    assert captured["image"] is not None  # PNG 字节
    assert captured["image"].startswith(b"\x89PNG")
    assert isinstance(img, Image.Image)
    assert s.first_frame is not None


def test_step_animation_works_with_imported_first_frame(tmp_path):
    """不跑全链路：自备首帧图即可直接生成动画。"""
    wf = make_wf()
    s = make_session(tmp_path)
    s.first_frame = Image.new("RGBA", (64, 64), (200, 100, 50, 255))  # 用户自备
    # 无提示词：动画提示词回退通用默认
    frames = wf.step_animation(s)
    assert len(frames) == 4
    assert s.frames == frames


def test_step_animation_produces_frames(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)
    wf.step_prompts(s)
    wf.step_image(s)
    frames = wf.step_animation(s)
    assert len(frames) == 4
    assert s.frames == frames


def test_step_animation_requires_first_frame(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)
    with pytest.raises(WorkflowError) as ei:
        wf.step_animation(s)
    assert "首帧" in ei.value.message


def test_step_pixelize_resizes_to_target(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)
    wf.step_prompts(s)
    wf.step_image(s)
    wf.step_animation(s)
    frames = wf.step_pixelize(s)
    assert len(frames) == 4
    assert frames[0].size == (64, 64)  # 目标像素尺寸


def test_step_background_adds_alpha(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)
    wf.step_prompts(s)
    wf.step_image(s)
    wf.step_animation(s)
    wf.step_pixelize(s)
    frames = wf.step_background(s)
    assert frames[0].mode == "RGBA"


def test_export_produces_artifacts(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)
    wf.step_prompts(s)
    wf.step_image(s)
    wf.step_animation(s)
    wf.step_pixelize(s)
    wf.step_background(s)
    paths = wf.export(s, export_dir=tmp_path / "export", fps=8)
    assert Path(paths["gif"]).exists()
    assert Path(paths["png_dir"]).is_dir()
    assert len(list(Path(paths["png_dir"]).glob("*.png"))) == 4
    assert Path(paths["metadata"]).exists()


def test_export_apng_and_sprite(tmp_path):
    """APNG 与雪碧图导出。"""
    wf = make_wf()
    s = make_session(tmp_path)
    wf.step_prompts(s)
    wf.step_image(s)
    wf.step_animation(s)
    wf.step_pixelize(s)
    paths = wf.export(s, export_dir=tmp_path / "export", formats=("apng", "sprite"))
    assert Path(paths["apng"]).exists()
    assert fu.apng_frame_count(paths["apng"]) == 4
    assert Path(paths["sprite"]).exists()
    with Image.open(paths["sprite"]) as sheet:
        assert sheet.size == (64 * 4, 64)  # 4 帧横向排列


def test_full_pipeline_end_to_end(tmp_path):
    wf = make_wf()
    s = make_session(tmp_path)
    wf.step_prompts(s)
    wf.step_image(s)
    wf.step_animation(s)
    wf.step_pixelize(s)
    wf.step_background(s)
    paths = wf.export(s, export_dir=tmp_path / "export")
    assert s.frames and s.frames[0].size == (64, 64)
    assert Path(paths["gif"]).exists()


# --------------------------------------------------------------------------- #
# 帧序列操作
# --------------------------------------------------------------------------- #
def test_session_frame_ops():
    s = IdeSession()
    a = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    b = Image.new("RGBA", (16, 16), (0, 255, 0, 255))
    s.insert_frame(0, a)
    s.insert_frame(1, b)
    assert len(s.frames) == 2

    dup = s.duplicate_frame(0)
    assert len(s.frames) == 3
    assert s.frames[1] is dup

    s.move_frame(0, 2)  # a 移到末尾
    assert s.frames[-1].getpixel((0, 0)) == (255, 0, 0, 255)

    removed = s.delete_frame(1)
    assert removed is not None
    assert len(s.frames) == 2


def test_session_delete_empty_returns_none():
    s = IdeSession()
    assert s.delete_frame(0) is None


# --------------------------------------------------------------------------- #
# 项目保存 / 加载
# --------------------------------------------------------------------------- #
def test_save_and_load_project_roundtrip(tmp_path):
    s = make_session(tmp_path)
    s.prompts = {"image_prompt": "pixel cat", "animation_prompt": "walk", "negative_prompt": "blur"}
    s.first_frame = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
    s.frames = [Image.new("RGBA", (16, 16), (i * 40, 0, 0, 255)) for i in range(4)]

    project_dir = tmp_path / "proj" / "my_anim"
    save_ide_project(s, project_dir)

    loaded = load_ide_project(project_dir)
    assert loaded.name == s.name
    assert loaded.prompts == s.prompts
    assert loaded.pixel_size == 64
    assert len(loaded.frames) == 4
    assert loaded.frames[2].getpixel((0, 0)) == (80, 0, 0, 255)
    assert loaded.first_frame is not None
    assert loaded.first_frame.size == (32, 32)


def test_session_bg_erode_roundtrip():
    """bg_erode 参数随会话保存/加载。"""
    s = IdeSession(bg_tolerance=25, bg_feather=6, bg_erode=3)
    data = s.to_dict()
    assert data["bg_erode"] == 3
    loaded = IdeSession.from_dict(data)
    assert loaded.bg_erode == 3
    assert loaded.bg_tolerance == 25 and loaded.bg_feather == 6


def test_step_background_erode_removes_fringe(tmp_path):
    """背景步骤支持内缩：消掉对象边缘残留白边。"""
    wf = make_wf()
    s = make_session(tmp_path, remove_bg=True, force_pure_bg=False, bg_tolerance=30, bg_erode=1)
    img = Image.new("RGBA", (20, 20), (255, 255, 255, 255))
    for y in range(5, 15):
        for x in range(5, 15):
            img.putpixel((x, y), (200, 50, 50, 255))
    img.putpixel((15, 10), (255, 225, 225, 255))  # 白晕像素（容差外）
    s.frames = [img]
    out = wf.step_background(s)
    arr = __import__("numpy").array(out[0])
    assert arr[10, 15, 3] == 0    # 内缩后白晕被去掉
    assert arr[10, 10, 3] == 255  # 主体核心保留
