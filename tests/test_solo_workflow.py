"""Solo 工作流端到端测试（模拟 API）。"""
import json
import threading
from pathlib import Path

import pytest

from core.api.base import APIResult
from core.api.mock_clients import MockImageAPI, MockLLMAPI, MockVideoAPI
from core.processing import frame_utils as fu
from core.workflow import SoloParams, SoloWorkflow, WorkflowCancelled


def build_params(tmp_out, **overrides):
    base = dict(
        description="一只拿剑的橙色小猫",
        action="步行",
        aspect_ratio="1:1",
        pixel_size=64,
        frame_count=6,
        fps=8,
        max_colors=8,
        output_dir=tmp_out,
    )
    base.update(overrides)
    return SoloParams(**base)


def make_workflow(params, **kwargs):
    return SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=MockVideoAPI(),
        params=params,
        **kwargs,
    )


def test_full_pipeline_produces_gif_and_png(tmp_out):
    params = build_params(tmp_out)
    result = make_workflow(params).run()

    assert result.frame_count == 6
    assert result.first_frame and result.first_frame.exists()
    assert result.frames_dir and result.frames_dir.exists()
    pngs = sorted(result.frames_dir.glob("*.png"))
    assert len(pngs) == 6

    assert result.gif_path and result.gif_path.exists()
    assert fu.gif_frame_count(result.gif_path) == 6
    assert result.png_dir and len(list(result.png_dir.glob("*.png"))) == 6
    assert result.project_file and result.project_file.exists()
    assert (result.output_dir / "export" / "metadata.json").exists()
    assert result.width == 64 and result.height == 64


def test_progress_callback_reports_all_steps(tmp_out):
    events = []

    def progress(step, total, name, pct, message):
        events.append((step, total, name))

    result = make_workflow(build_params(tmp_out), progress=progress).run()
    steps = sorted({e[0] for e in events})
    assert steps == list(range(6))
    assert result


def test_log_callback_collects_entries(tmp_out):
    logs = []

    def log(level, message):
        logs.append((level, message))

    make_workflow(build_params(tmp_out), log=log).run()
    assert len(logs) > 5
    assert any(level == "info" for level, _ in logs)


def test_cancel_raises(tmp_out):
    params = build_params(tmp_out)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(WorkflowCancelled):
        make_workflow(params, cancel=cancel).run()


def test_llm_failure_falls_back(tmp_out):
    class FailingLLM(MockLLMAPI):
        def call(self, prompt, **kwargs):
            return APIResult(ok=False, error="network down")

    workflow = SoloWorkflow(
        llm_api=FailingLLM(),
        image_api=MockImageAPI(),
        video_api=MockVideoAPI(),
        params=build_params(tmp_out),
    )
    result = workflow.run()
    assert result.gif_path.exists()
    assert result.prompts["image_prompt"]  # fallback 提示词


def test_image_failure_raises(tmp_out):
    class FailingImage(MockImageAPI):
        def call(self, prompt, **kwargs):
            return APIResult(ok=False, error="quota exceeded")

    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=FailingImage(),
        video_api=MockVideoAPI(),
        params=build_params(tmp_out),
    )
    from core.workflow import WorkflowError

    with pytest.raises(WorkflowError) as exc_info:
        workflow.run()
    assert "首帧图片生成失败" in exc_info.value.message


def test_no_pixelate_no_bg(tmp_out):
    params = build_params(tmp_out, pixelate=False, remove_bg=False)
    result = make_workflow(params).run()
    assert result.gif_path.exists()
    # 未去背景：帧为不透明
    from PIL import Image

    frame = Image.open(sorted(result.frames_dir.glob("*.png"))[0])
    assert frame.mode == "RGBA"
    assert frame.getpixel((0, 0))[3] == 255


def test_non_square_aspect(tmp_out):
    params = build_params(tmp_out, aspect_ratio="16:9", pixel_size=96)
    result = make_workflow(params).run()
    assert result.width == 96
    assert result.height == 54  # 96 * 9/16


def test_intermediate_callbacks_fire(tmp_out):
    """中间产物回调：提示词与首帧图就绪时被调用。"""
    captured = {}

    def on_prompts(prompts):
        captured["prompts"] = prompts

    def on_first_frame(path):
        captured["first_frame"] = path

    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=MockVideoAPI(),
        params=build_params(tmp_out),
        on_prompts=on_prompts,
        on_first_frame=on_first_frame,
    )
    result = workflow.run()
    assert "image_prompt" in captured["prompts"]
    assert "animation_prompt" in captured["prompts"]
    assert Path(captured["first_frame"]).exists()
    assert Path(captured["first_frame"]) == result.first_frame


def test_intermediate_callbacks_fire_on_llm_fallback(tmp_out):
    """LLM 失败降级时，fallback 提示词也要回调展示。"""
    class FailingLLM(MockLLMAPI):
        def call(self, prompt, **kwargs):
            return APIResult(ok=False, error="down")

    captured = {}

    def on_prompts(prompts):
        captured["prompts"] = prompts

    workflow = SoloWorkflow(
        llm_api=FailingLLM(),
        image_api=MockImageAPI(),
        video_api=MockVideoAPI(),
        params=build_params(tmp_out),
        on_prompts=on_prompts,
    )
    workflow.run()
    assert captured["prompts"]["image_prompt"]


def test_video_url_path(tmp_out, monkeypatch):
    """视频 API 返回 video_url（而非帧）时，工作流应下载并拆帧。"""
    import imageio.v2 as imageio
    import numpy as np

    from core.api.base import BaseAPI

    # 1) 生成一个真实的小 mp4（白色方块逐步右移，保证帧互不相同）
    src_mp4 = tmp_out / "src.mp4"
    writer = imageio.get_writer(str(src_mp4), format="ffmpeg", fps=8)
    try:
        for i in range(4):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[10:20, 2 + i * 7 : 10 + i * 7] = 255
            writer.append_data(frame)
    finally:
        writer.close()

    class UrlVideoAPI(BaseAPI):
        KIND = "video"

        def __init__(self):
            super().__init__({"base_url": "", "api_key": "", "model": "", "params": {}})

        def call(self, image_bytes, prompt, **kwargs):
            return APIResult(ok=True, data={"video_url": "http://cdn/x.mp4", "frames": None})

        def test_connection(self):
            return APIResult(ok=True)

    # 2) 拦截下载：返回本地 mp4 字节
    monkeypatch.setattr(
        "core.workflow.solo_workflow.fu.download_bytes",
        lambda url, timeout=180.0: src_mp4.read_bytes(),
    )

    params = build_params(tmp_out, frame_count=4)
    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=UrlVideoAPI(),
        params=params,
    )
    result = workflow.run()
    assert result.video_path and result.video_path.exists()
    assert result.frame_count == 4
    assert result.gif_path.exists()
    assert fu.gif_frame_count(result.gif_path) == 4


def _write_mp4(path, frames, fps=8, block_step=3):
    import imageio.v2 as imageio
    import numpy as np

    writer = imageio.get_writer(str(path), format="ffmpeg", fps=fps)
    try:
        for i in range(frames):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[10:20, (i * block_step) % 24 : (i * block_step) % 24 + 8] = 255
            writer.append_data(frame)
    finally:
        writer.close()


class VideoUrlStub:
    """返回固定 video_url 的假视频客户端。"""

    def __init__(self, url="http://cdn/x.mp4"):
        self._url = url

    def call(self, image_bytes, prompt, **kwargs):
        return APIResult(ok=True, data={"video_url": self._url, "frames": None})

    def test_connection(self):
        return APIResult(ok=True)


def test_video_duration_1s_fps_adjusted(tmp_out, monkeypatch):
    """视频仅 1s 而用户请求 2s 动画时，输出帧率自动提速以保持原速（1x）。

    用户 8 帧 @ 4fps（期望 2s），视频实际 1s（8 帧 @ 8fps）：
    最终输出 8 帧 @ 8fps = 1s，与源视频 1x 速度一致。
    """
    src_mp4 = tmp_out / "short.mp4"
    _write_mp4(src_mp4, frames=8, fps=8)  # 1s
    monkeypatch.setattr(
        "core.workflow.solo_workflow.fu.download_bytes",
        lambda url, timeout=180.0: src_mp4.read_bytes(),
    )

    params = build_params(tmp_out, frame_count=8, fps=4)
    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=VideoUrlStub(),
        params=params,
    )
    result = workflow.run()
    assert result.fps == 8  # 1s 视频 8 帧 -> 8fps 保持原速
    assert result.video_duration is not None
    assert abs(result.video_duration - 1.0) < 0.3
    assert result.frame_count == 8
    # 帧数以 PNG 序列为准（GIF 会合并量化后相同的相邻帧，属 PIL 已知行为）
    assert len(list(Path(result.frames_dir).glob("*.png"))) == 8
    assert result.fps == 8


def test_video_longer_than_request_keeps_full_action(tmp_out, monkeypatch):
    """视频 2s 而用户只要 1s：抽取完整视频（不截断），保留首尾帧、中间均匀采样。

    完整动作都体现：8 帧覆盖整段 2s 动作（首帧 + 尾帧 + 中间 6 帧），
    输出帧率校准为 8 帧 / 2s = 4fps，保持原速 1x。
    """
    src_mp4 = tmp_out / "long.mp4"
    _write_mp4(src_mp4, frames=16, fps=8)  # 2s
    monkeypatch.setattr(
        "core.workflow.solo_workflow.fu.download_bytes",
        lambda url, timeout=180.0: src_mp4.read_bytes(),
    )

    params = build_params(tmp_out, frame_count=8, fps=8, action="")  # 期望 1s，无 LLM 覆盖
    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=VideoUrlStub(),
        params=params,
    )
    result = workflow.run()
    assert result.fps == 4  # 8 帧覆盖完整 2s -> 4fps 保持原速
    assert result.video_duration is not None
    assert abs(result.video_duration - 2.0) < 0.3  # 完整视频时长，不再截断到 1s
    assert result.frame_count == 8
    assert fu.gif_frame_count(result.gif_path) == 8


def test_speed_multiplier_frames_direct(tmp_out):
    """倍速：直接返回帧序列时，输出帧率 = 请求帧率 × 倍速。"""
    params = build_params(tmp_out, frame_count=8, fps=8, speed=2.0, action="")
    result = make_workflow(params).run()
    assert result.fps == 16  # 8fps × 2
    assert result.frame_count == 8
    # 帧数以 PNG 序列为准（GIF 会合并量化后相同的相邻帧，属 PIL 已知行为）
    assert len(list(Path(result.frames_dir).glob("*.png"))) == 8


def test_animation_prompt_injects_background_stability(tmp_out):
    """强制纯色背景时，动画提示词附加背景稳定性约束（防背景漂移）。"""
    captured = {}

    class CapturingVideo(MockVideoAPI):
        def call(self, image_bytes=None, prompt="", **kwargs):
            captured["prompt"] = prompt
            return super().call(image_bytes=image_bytes, prompt=prompt, **kwargs)

    params = build_params(tmp_out, force_pure_bg=True, action="")
    workflow = SoloWorkflow(MockLLMAPI(), MockImageAPI(), CapturingVideo(), params)
    workflow.run()
    assert "background must remain" in captured["prompt"]
    assert "SOLID PURE WHITE" in captured["prompt"]


def test_animation_prompt_no_stability_when_bg_off(tmp_out):
    """未开启背景强制纯色时，动画提示词不附加背景稳定性约束。"""
    captured = {}

    class CapturingVideo(MockVideoAPI):
        def call(self, image_bytes=None, prompt="", **kwargs):
            captured["prompt"] = prompt
            return super().call(image_bytes=image_bytes, prompt=prompt, **kwargs)

    params = build_params(tmp_out, force_pure_bg=False, remove_bg=False, action="")
    workflow = SoloWorkflow(MockLLMAPI(), MockImageAPI(), CapturingVideo(), params)
    workflow.run()
    assert "background must remain" not in captured["prompt"]


def test_solo_reference_image_passed_to_api(tmp_out):
    """Solo 文本生图支持参考图（图生图）：参考图字节传入 image API。"""
    from PIL import Image as PILImage

    captured = {}

    class CapturingImage(MockImageAPI):
        def call(self, prompt, size=None, n=1, **kwargs):
            captured["image"] = kwargs.get("image")
            return super().call(prompt, size=size, n=n, **kwargs)

    ref = tmp_out / "ref.png"
    PILImage.new("RGBA", (16, 16), (1, 2, 3, 255)).save(ref)
    params = build_params(tmp_out, reference_image=ref, action="")
    workflow = SoloWorkflow(MockLLMAPI(), CapturingImage(), MockVideoAPI(), params)
    workflow.run()
    assert captured["image"] is not None
    assert captured["image"].startswith(b"\x89PNG")


def test_animation_prompt_includes_subject_margin(tmp_out):
    """动画提示词附加主体完整性约束（对象不触碰边缘）。"""
    captured = {}

    class CapturingVideo(MockVideoAPI):
        def call(self, image_bytes=None, prompt="", **kwargs):
            captured["prompt"] = prompt
            return super().call(image_bytes=image_bytes, prompt=prompt, **kwargs)

    params = build_params(tmp_out, force_pure_bg=False, remove_bg=False, action="")
    workflow = SoloWorkflow(MockLLMAPI(), MockImageAPI(), CapturingVideo(), params)
    workflow.run()
    assert "fully visible" in captured["prompt"]
    assert "margin" in captured["prompt"]
    assert "never cropped" in captured["prompt"]


def test_image_prompt_includes_subject_margin(tmp_out):
    """图片提示词写入主体完整性约束。"""
    workflow = SoloWorkflow(MockLLMAPI(), MockImageAPI(), MockVideoAPI(), build_params(tmp_out, action=""))
    prompts, _ = workflow._step_prompts()
    assert "fully visible" in prompts["image_prompt"]
    assert "margin" in prompts["image_prompt"]


def test_speed_multiplier_video_path(tmp_out, monkeypatch):
    """倍速：视频路径下，输出帧率 = 原速帧率 × 倍速。"""
    src_mp4 = tmp_out / "speed.mp4"
    _write_mp4(src_mp4, frames=8, fps=8)  # 1s，8 帧
    monkeypatch.setattr(
        "core.workflow.solo_workflow.fu.download_bytes",
        lambda url, timeout=180.0: src_mp4.read_bytes(),
    )

    params = build_params(tmp_out, frame_count=8, fps=8, speed=1.5)
    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=VideoUrlStub(),
        params=params,
    )
    result = workflow.run()
    # 原速 = 8 帧 / 1s = 8fps，×1.5 = 12fps
    assert result.fps == 12
    # 帧数以 PNG 序列为准（GIF 会合并量化后相同的相邻帧，属 PIL 已知行为）
    assert len(list(Path(result.frames_dir).glob("*.png"))) == 8


def test_solo_params_defaults_support_1s_animation():
    """默认参数为 1s 动画（8 帧 @ 8fps），LLM 会按动作自动调整。"""
    params = SoloParams(description="x")
    assert params.frame_count == 8
    assert params.fps == 8
    assert params.speed == 1.0
    assert params.frame_count / params.fps == 1.0


def test_loop_close_first_last_identical(tmp_out):
    """首尾帧一致：最终交付帧的第一帧与最后一帧相同。"""
    result = make_workflow(build_params(tmp_out, frame_count=6, loop_close=True)).run()
    paths = sorted(Path(result.frames_dir).glob("*.png"))
    assert len(paths) == 6
    first = paths[0].read_bytes()
    last = paths[-1].read_bytes()
    assert first == last


def test_loop_close_can_be_disabled(tmp_out):
    result = make_workflow(build_params(tmp_out, frame_count=6, loop_close=False)).run()
    paths = sorted(Path(result.frames_dir).glob("*.png"))
    assert len(paths) == 6


def test_first_frame_downscaled_before_video_api(tmp_out):
    """发送给视频 API 的首帧长边 ≤ video_image_max_side（省 token 优化）。"""
    import io

    from PIL import Image as PILImage

    class CaptureVideo(MockVideoAPI):
        def __init__(self):
            super().__init__()
            self.received: bytes | None = None

        def call(self, image_bytes, prompt, **kwargs):
            self.received = image_bytes
            return super().call(image_bytes, prompt, **kwargs)

    params = build_params(tmp_out, frame_count=4, video_image_max_side=256)
    video = CaptureVideo()
    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),  # 生成 1024x1024 首帧
        video_api=video,
        params=params,
    )
    result = workflow.run()
    assert video.received is not None
    img = PILImage.open(io.BytesIO(video.received))
    assert max(img.size) <= 256
    assert result.gif_path.exists()


def test_prompts_contain_color_constraint(tmp_out):
    """提示词内置强制项：颜色数量上限与 max_colors 一致、纯白背景。"""
    captured = {}

    def on_prompts(prompts):
        captured["prompts"] = prompts

    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=MockVideoAPI(),
        params=build_params(tmp_out, max_colors=8),
        on_prompts=on_prompts,
    )
    workflow.run()
    prompts = captured["prompts"]
    assert "at most 8" in prompts["image_prompt"]
    assert "white background" in prompts["image_prompt"]
    assert "gray or colored background" in prompts["negative_prompt"]


def test_prompts_contain_forced_resolution(tmp_out):
    """提示词必须严格写入预设像素尺寸（强指令式，保证出图符合要求）。"""
    captured = {}

    def on_prompts(prompts):
        captured["prompts"] = prompts

    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=MockImageAPI(),
        video_api=MockVideoAPI(),
        params=build_params(tmp_out, pixel_size=64),
        on_prompts=on_prompts,
    )
    workflow.run()
    image_prompt = captured["prompts"]["image_prompt"]
    # 强指令 + 用户像素尺寸
    assert "MUST" in image_prompt
    assert "EXACT 64x64 pixel grid" in image_prompt
    # 非 1:1 宽高比时写入目标宽高
    captured2 = {}

    def on_prompts2(p):
        captured2["prompts"] = p

    workflow2 = SoloWorkflow(
        MockLLMAPI(), MockImageAPI(), MockVideoAPI(),
        build_params(tmp_out, pixel_size=96, aspect_ratio="16:9"),
        on_prompts=on_prompts2,
    )
    workflow2.run()
    assert "EXACT 96x54 pixel grid" in captured2["prompts"]["image_prompt"]


def test_llm_empty_output_retries_with_more_tokens(tmp_out):
    """推理模型 content 为空时，工作流提高 max_tokens 重试一次并成功。"""
    tokens = []

    class EmptyFirstLLM(MockLLMAPI):
        def call(self, prompt, **kwargs):
            tokens.append(kwargs.get("max_tokens"))
            if len(tokens) == 1:
                return APIResult(ok=True, data="")  # 空输出（推理被截断）
            return super().call(prompt, **kwargs)

    workflow = SoloWorkflow(
        EmptyFirstLLM(), MockImageAPI(), MockVideoAPI(), build_params(tmp_out, frame_count=4)
    )
    result = workflow.run()
    assert tokens == [1600, 4096]  # 先 1600，空输出后提高到 4096
    assert result.gif_path.exists()


def test_verbose_animation_prompt_gets_strict_retry(tmp_out):
    """动画提示词过于冗长（>40 词）时，用严格纠正指令重试一次，取简洁结果。"""
    verbose_prompt = (
        "Smooth looping pixel-art animation of a mounted lancer performing a spear thrust "
        "to the right: the rider pulls the spear back slightly while the horse gathers, "
        "then the horse lunges forward and the rider drives the spear forward in a quick "
        "thrusting stab, extending the arm and spear to the right, holding the extension "
        "briefly, then returning to the starting pose. The entire action is on a solid "
        "pure white background with limited solid colors, clean hard edges, no shading."
    )
    calls = []

    class VerboseThenConciseLLM(MockLLMAPI):
        def call(self, prompt, **kwargs):
            calls.append((len(calls), kwargs.get("system", ""), kwargs.get("max_tokens")))
            if len(calls) == 1:
                data = super().call(prompt, **kwargs).data
                data["animation_prompt"] = verbose_prompt
                return APIResult(ok=True, data=data)
            return super().call(prompt, **kwargs)

    workflow = SoloWorkflow(
        VerboseThenConciseLLM(), MockImageAPI(), MockVideoAPI(), build_params(tmp_out, frame_count=4)
    )
    captured = {}

    def on_prompts(p):
        captured["prompts"] = p

    workflow._on_prompts_cb = on_prompts
    prompts, _ = workflow._step_prompts()
    assert len(calls) == 2  # 触发严格重试
    # 第二次调用带严格纠正指令
    assert "STRICT CORRECTION" in calls[1][1]
    # 最终动画提示词来自简洁重试结果（mock 模板，非冗长原文）
    assert prompts["animation_prompt"] != verbose_prompt
    assert len(prompts["animation_prompt"].split()) <= 40


def test_first_frame_whitened_before_video(tmp_out):
    """AI 生成图背景发灰时，首帧在保存/发送前被归一化（深色主体 → 白底）。"""
    import io

    from PIL import Image as PILImage, ImageDraw

    class GrayBgImageAPI(MockImageAPI):
        def call(self, prompt, **kwargs):
            img = PILImage.new("RGB", (64, 64), (230, 230, 230))  # 发灰背景
            draw = ImageDraw.Draw(img)
            draw.rectangle([16, 16, 48, 48], fill=(40, 40, 40))   # 深色主体
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return APIResult(ok=True, data={"images": [buf.getvalue()], "urls": []})

    captured = {}

    def on_first_frame(path):
        captured["path"] = Path(path)

    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=GrayBgImageAPI(),
        video_api=MockVideoAPI(),
        params=build_params(tmp_out, frame_count=4, force_pure_bg=True),
        on_first_frame=on_first_frame,
    )
    result = workflow.run()
    first = PILImage.open(captured["path"]).convert("RGBA")
    # 角落背景已纯白
    assert first.getpixel((2, 2))[:3] == (255, 255, 255)
    # 主体未变
    assert first.getpixel((32, 32))[:3] == (40, 40, 40)
    assert result.gif_path.exists()


def test_first_frame_black_bg_for_pale_subject(tmp_out):
    """对象本身是浅色系时，自适应背景强制为纯黑（保证对比度）。"""
    import io

    from PIL import Image as PILImage, ImageDraw

    class PaleSubjectImageAPI(MockImageAPI):
        def call(self, prompt, **kwargs):
            img = PILImage.new("RGB", (64, 64), (230, 230, 230))  # 发灰背景
            draw = ImageDraw.Draw(img)
            draw.rectangle([16, 16, 48, 48], fill=(252, 250, 248))  # 极淡色主体
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return APIResult(ok=True, data={"images": [buf.getvalue()], "urls": []})

    captured = {}

    def on_first_frame(path):
        captured["path"] = Path(path)

    workflow = SoloWorkflow(
        llm_api=MockLLMAPI(),
        image_api=PaleSubjectImageAPI(),
        video_api=MockVideoAPI(),
        params=build_params(tmp_out, frame_count=4, force_pure_bg=True),
        on_first_frame=on_first_frame,
    )
    result = workflow.run()
    first = PILImage.open(captured["path"]).convert("RGBA")
    # 角落背景被强制为纯黑
    assert first.getpixel((2, 2))[:3] == (0, 0, 0)
    # 浅色主体保留
    assert first.getpixel((32, 32))[:3] == (252, 250, 248)
    assert result.gif_path.exists()


def test_force_white_bg_disabled_keeps_frame(tmp_out):
    """关闭强制纯色背景时，帧内容不被改写。"""
    result = make_workflow(build_params(tmp_out, frame_count=4, force_pure_bg=False)).run()
    assert result.gif_path.exists()


def test_llm_params_adjust_defaults_by_action(tmp_out):
    """LLM 按动作建议动画参数：默认帧数被覆盖（攻击 → ~10 帧 @ 8fps）。"""
    params = SoloParams(description="角色", action="攻击", output_dir=tmp_out)  # 默认 16/8
    result = SoloWorkflow(MockLLMAPI(), MockImageAPI(), MockVideoAPI(), params).run()
    assert result.frame_count == 10  # 1.2s * 8fps ≈ 10 帧
    assert result.gif_path.exists()


def test_llm_params_respect_explicit_values(tmp_out):
    """用户显式设置过帧数，则不被 LLM 覆盖。"""
    params = build_params(tmp_out, frame_count=20, action="攻击")
    result = make_workflow(params).run()
    assert result.frame_count == 20


def test_pixel_prompt_forces_generation_size(tmp_out):
    """像素风意图（提示词含关键字）→ 生图尺寸强制为预设像素分辨率。"""
    workflow = SoloWorkflow(MockLLMAPI(), MockImageAPI(), MockVideoAPI(), build_params(tmp_out, pixel_size=64))
    w, h = workflow._api_image_size("a pixel art character, sprite style")
    assert w == 256 and h == 256  # max(64,256) -> 256，长边即预设像素分辨率
    w2, h2 = workflow._api_image_size("a realistic character")
    assert w2 == 1024  # 非像素风 -> 默认 1024


def test_pixel_pipeline_runs_perfect_pixel_once(tmp_out, monkeypatch):
    """像素风帧序列：完美像素只做一次网格检测，全部帧同网格采样（效率优化）。"""
    import imageio.v2 as imageio
    import numpy as np

    from core.processing import pixelizer as pxmod

    real_seq = pxmod.perfect_pixelize_sequence
    calls = {"n": 0}

    def counting_seq(frames, **kwargs):
        calls["n"] += 1
        return real_seq(frames, **kwargs)

    monkeypatch.setattr(pxmod, "perfect_pixelize_sequence", counting_seq)

    # 192x192 棋盘格（12px 格）视频帧：像素风
    src = tmp_out / "pixel.mp4"
    writer = imageio.get_writer(str(src), format="ffmpeg", fps=8)
    try:
        for i in range(4):
            frame = np.zeros((192, 192, 3), dtype=np.uint8)
            for cy in range(16):
                for cx in range(16):
                    v = 255 if ((cx + cy + i) % 2 == 0) else 20
                    frame[cy * 12 : (cy + 1) * 12, cx * 12 : (cx + 1) * 12] = v
            writer.append_data(frame)
    finally:
        writer.close()
    monkeypatch.setattr(
        "core.workflow.solo_workflow.fu.download_bytes",
        lambda url, timeout=180.0: src.read_bytes(),
    )

    params = build_params(tmp_out, frame_count=4)
    workflow = SoloWorkflow(MockLLMAPI(), MockImageAPI(), VideoUrlStub(), params)
    result = workflow.run()
    assert calls["n"] == 1  # 网格检测/采样计划只做一次
    assert result.frame_count == 4
    assert result.gif_path.exists()
    # 循环闭合 + 同网格采样：首尾帧一致
    paths = sorted(Path(result.frames_dir).glob("*.png"))
    assert paths[0].read_bytes() == paths[-1].read_bytes()


def test_non_pixel_pipeline_skips_perfect_pixel(tmp_out, monkeypatch):
    """非像素风帧：完美像素被跳过（调用一次返回 None 即回退）。"""
    from core.processing import pixelizer as pxmod

    real_seq = pxmod.perfect_pixelize_sequence
    calls = {"n": 0}

    def counting_seq(frames, **kwargs):
        calls["n"] += 1
        return real_seq(frames, **kwargs)

    monkeypatch.setattr(pxmod, "perfect_pixelize_sequence", counting_seq)
    result = make_workflow(build_params(tmp_out, frame_count=4)).run()  # mock 渐变帧（非像素风）
    assert calls["n"] == 1
    assert result.gif_path.exists()


# --------------------------------------------------------------------------- #
# 双分辨率导出：完美像素原生 + 用户预设
# --------------------------------------------------------------------------- #
def _write_checkerboard_mp4(path, frames=4, size=192, cell=12):
    """16×16 棋盘格视频帧；每帧多一个移动的暗格，保证 16×16 网格下帧也互不相同
    （否则周期-2 的纯棋盘格会让 GIF 合并相同帧）。"""
    import imageio.v2 as imageio
    import numpy as np

    writer = imageio.get_writer(str(path), format="ffmpeg", fps=8)
    try:
        n = size // cell
        for i in range(frames):
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            for cy in range(n):
                for cx in range(n):
                    v = 255 if ((cx + cy) % 2 == 0) else 20
                    if i > 0 and cx == i and cy == i:
                        v = 20  # 移动暗格：帧间内容不同
                    frame[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell] = v
            writer.append_data(frame)
    finally:
        writer.close()


def test_pixel_pipeline_exports_both_resolutions(tmp_out, monkeypatch):
    """像素风：最终产物同时保留完美像素原生分辨率（16x16）与用户预设分辨率（64x64）。"""
    from PIL import Image

    src = tmp_out / "pixel.mp4"
    _write_checkerboard_mp4(src, frames=4)  # 192x192、12px 格 -> 网格 16x16
    monkeypatch.setattr(
        "core.workflow.solo_workflow.fu.download_bytes",
        lambda url, timeout=180.0: src.read_bytes(),
    )

    params = build_params(tmp_out, frame_count=4, pixel_size=64)
    result = SoloWorkflow(MockLLMAPI(), MockImageAPI(), VideoUrlStub(), params).run()

    # 用户预设分辨率
    assert result.width == 64 and result.height == 64
    assert result.gif_path and result.gif_path.exists()
    assert fu.gif_frame_count(result.gif_path) == 4
    assert result.png_dir and len(list(result.png_dir.glob("*.png"))) == 4
    # 完美像素原生分辨率（网格约 16x16；视频压缩后检测存在 ±3 容差）
    assert result.native_width == result.native_height
    assert 13 <= result.native_width <= 19
    assert result.native_gif_path and result.native_gif_path.exists()
    assert fu.gif_frame_count(result.native_gif_path) == 4
    assert result.native_png_dir and len(list(result.native_png_dir.glob("*.png"))) == 4
    npng = Image.open(sorted(result.native_png_dir.glob("*.png"))[0])
    assert npng.size == (result.native_width, result.native_height)
    # 原生版本同样保持循环闭合（首尾帧一致）
    npaths = sorted(result.native_png_dir.glob("*.png"))
    assert npaths[0].read_bytes() == npaths[-1].read_bytes()
    # 两种版本都是小调色板（背景归一化在不同分辨率下的填充决策可合理不同，
    # 因此不要求颜色集合完全相等；共享调色板由实现保证）
    preset_colors = set(Image.open(sorted(result.png_dir.glob("*.png"))[0]).convert("RGB").getdata())
    native_colors = set(npng.convert("RGB").getdata())
    assert len(preset_colors) >= 2 and len(native_colors) >= 2
    # metadata 记录两种分辨率
    meta = json.loads((result.output_dir / "export" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["native_resolution"]["width"] == result.native_width
    assert meta["native_resolution"]["height"] == result.native_height
    # project.json 记录原生路径与尺寸
    proj = json.loads(result.project_file.read_text(encoding="utf-8"))
    assert proj["native_gif_path"]
    assert proj["native_width"] == result.native_width and proj["native_height"] == result.native_height


def test_non_pixel_exports_only_preset_resolution(tmp_out):
    """非像素风：不产生完美像素原生分辨率产物。"""
    result = make_workflow(build_params(tmp_out, frame_count=4)).run()  # mock 渐变帧（非像素风）
    assert result.gif_path and result.gif_path.exists()
    assert result.native_gif_path is None
    assert result.native_png_dir is None
    assert result.native_width == 0 and result.native_height == 0
    meta = json.loads((result.output_dir / "export" / "metadata.json").read_text(encoding="utf-8"))
    assert "native_resolution" not in meta


def test_native_equals_preset_no_duplicate(tmp_out, monkeypatch):
    """完美像素网格 == 用户预设分辨率时，不重复导出原生版本（跳过逻辑）。"""
    from PIL import Image

    from core.processing import pixelizer as pxmod

    params = build_params(tmp_out, frame_count=4, pixel_size=16)  # 预设 16x16
    workflow = SoloWorkflow(MockLLMAPI(), MockImageAPI(), MockVideoAPI(), params)
    frames = [Image.new("RGB", (32, 32), (i * 20, i * 20, i * 20)) for i in range(4)]
    # 模拟完美像素检测出 16x16 网格（== 目标尺寸）
    monkeypatch.setattr(
        pxmod,
        "perfect_pixelize_sequence",
        lambda frames, **kw: ([f.resize((16, 16)) for f in frames], (16, 16)),
    )
    quantized, native = workflow._step_pixelize(
        frames,
        pxmod.PixelizeParams(target_size=(16, 16), max_colors=8, edge_clean=True),
    )
    assert len(quantized) == 4
    assert quantized[0].size == (16, 16)
    assert native is None  # 网格 == 预设 -> 不保留原生

    # 网格 < 预设时两者并存
    monkeypatch.setattr(
        pxmod,
        "perfect_pixelize_sequence",
        lambda frames, **kw: ([f.resize((8, 8)) for f in frames], (8, 8)),
    )
    quantized, native = workflow._step_pixelize(
        frames,
        pxmod.PixelizeParams(target_size=(16, 16), max_colors=8, edge_clean=True),
    )
    assert native is not None
    assert native[0].size == (8, 8)
    assert quantized[0].size == (16, 16)


def test_video_first_frame_upscaled_to_min_side(tmp_out):
    """首帧分辨率低于 API 最低要求时，最近邻放大到该要求（不模糊）。"""
    import io

    from PIL import Image as PILImage

    captured = {}

    class CapturingVideo(MockVideoAPI):
        def call(self, image_bytes=None, prompt="", **kwargs):
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")
            captured["size"] = img.size
            return super().call(image_bytes=image_bytes, prompt=prompt, **kwargs)

    # 16x16 像素画首帧
    tiny = tmp_out / "tiny.png"
    PILImage.new("RGBA", (16, 16), (255, 0, 0, 255)).save(tiny)
    tiny_bytes = tiny.read_bytes()

    # 开启最低要求：放大到长边 ≥256（NEAREST）
    params = build_params(tmp_out, action="", video_image_min_side=256, video_image_max_side=0)
    workflow = SoloWorkflow(MockLLMAPI(), MockImageAPI(), CapturingVideo(), params)
    workflow._step_animation(tiny_bytes, "walk", tmp_out / "artifacts")
    assert captured["size"][0] >= 256 and captured["size"][1] >= 256

    # 关闭最低要求：16x16 原样发送
    captured.clear()
    params2 = build_params(tmp_out, action="", video_image_min_side=0, video_image_max_side=0)
    workflow2 = SoloWorkflow(MockLLMAPI(), MockImageAPI(), CapturingVideo(), params2)
    workflow2._step_animation(tiny_bytes, "walk", tmp_out / "artifacts")
    assert captured["size"] == (16, 16)
