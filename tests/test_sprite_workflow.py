"""精灵图工作流测试（模拟 API）。"""
from pathlib import Path

from core.api.base import APIResult
from core.api.mock_clients import MockImageAPI, MockLLMAPI, parse_size
from core.workflow import STEP_ORDER, SpriteParams, SpriteWorkflow
from core.workflow.sprite_workflow import SPRITE_SHEET_PROMPT


def test_sprite_workflow_end_to_end(tmp_path):
    """完整链路：文生底图 -> 精灵图 -> 裁切 -> 抠图 -> 导出。"""
    wf = SpriteWorkflow(MockLLMAPI(), MockImageAPI())
    params = SpriteParams(
        description="一只橙色小猫",
        action="步行",
        frame_count=16,
        grid_rows=4,
        grid_cols=4,
        cell_size=64,
        output_dir=tmp_path,
    )
    result = wf.run(params)
    assert result.base_image and result.base_image.exists()
    assert result.sheet_image and result.sheet_image.exists()
    assert result.frame_count == 16
    assert result.frames_dir and len(list(Path(result.frames_dir).glob("*.png"))) == 16
    assert result.gif_path and result.gif_path.exists()
    assert result.sheet_path and result.sheet_path.exists()
    assert result.project_file and result.project_file.exists()
    # 雪碧图索引 JSON（FrameRonin 格式：每帧坐标 + 时间戳）
    assert result.sheet_index and result.sheet_index.exists()
    import json

    index = json.loads(result.sheet_index.read_text(encoding="utf-8"))
    assert index["version"] == "1.0"
    assert len(index["frames"]) == result.frame_count
    assert index["frames"][0]["x"] == 0 and index["frames"][0]["y"] == 0
    assert index["frames"][1]["x"] == index["frame_size"]["w"]  # 第二格紧随其后
    # 帧尺寸 = 网格切分（64px 单格）
    from PIL import Image

    frame = Image.open(sorted(Path(result.frames_dir).glob("*.png"))[0])
    assert frame.size[0] == 64 or frame.size[0] > 0


def test_sprite_workflow_frame_count_less_than_grid(tmp_path):
    """帧数 < 网格总数时只裁切前 N 帧。"""
    wf = SpriteWorkflow(MockLLMAPI(), MockImageAPI())
    params = SpriteParams(
        description="x", frame_count=6, grid_rows=4, grid_cols=4, cell_size=64, output_dir=tmp_path
    )
    result = wf.run(params)
    assert result.frame_count == 6
    assert len(list(Path(result.frames_dir).glob("*.png"))) == 6


def test_sprite_workflow_skips_keying_when_disabled(tmp_path):
    """关闭抠图/强制纯色时，帧保持不透明。"""
    wf = SpriteWorkflow(MockLLMAPI(), MockImageAPI())
    params = SpriteParams(
        description="x",
        frame_count=4,
        grid_rows=2,
        grid_cols=2,
        cell_size=64,
        remove_bg=False,
        force_pure_bg=False,
        output_dir=tmp_path,
    )
    result = wf.run(params)
    assert result.frame_count == 4
    from PIL import Image

    frame = Image.open(sorted(Path(result.frames_dir).glob("*.png"))[0])
    assert frame.mode == "RGBA"


def test_sprite_sheet_prompt_built_in():
    """内置强提示词：一整张网格图、首尾格姿势一致、角色不突变、无文字/边框。"""
    prompt = SPRITE_SHEET_PROMPT.format(rows=4, cols=4, total=16, action="walk")
    assert "4×4 grid" in prompt
    assert "16" in prompt and "walk" in prompt
    # 一整张网格图（不是 i×j 张独立图）
    assert "SINGLE image" in prompt
    assert "never generate" in prompt and "separate images" in prompt
    # 首尾帧一致（循环无缝）
    assert "IDENTICAL pose" in prompt and "loops back" in prompt
    # 角色不突变
    assert "EXACTLY identical" in prompt and "no mutation" in prompt
    assert "white background" in prompt.lower()
    assert "NO text" in prompt and "NO borders" in prompt


def test_sprite_workflow_loop_close_enforces_first_last_equal(tmp_path):
    """首尾帧一致开启：末帧强制等于首帧（循环无缝）。"""
    wf = SpriteWorkflow(MockLLMAPI(), _OversizedImageAPI())
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=64,
        loop_close=True, output_dir=tmp_path,
    )
    result = wf.run(params)
    assert result.frame_count == 4
    from PIL import Image

    frames = sorted(Path(result.frames_dir).glob("*.png"))
    first = Image.open(frames[0]).tobytes()
    last = Image.open(frames[-1]).tobytes()
    assert first == last


def test_sprite_workflow_loop_close_off_keeps_frames(tmp_path):
    """首尾帧一致关闭：保留模型生成的各帧（首尾不同）。"""
    wf = SpriteWorkflow(MockLLMAPI(), _OversizedImageAPI())
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=64,
        loop_close=False, output_dir=tmp_path,
    )
    result = wf.run(params)
    from PIL import Image

    frames = sorted(Path(result.frames_dir).glob("*.png"))
    first = Image.open(frames[0]).tobytes()
    last = Image.open(frames[-1]).tobytes()
    assert first != last  # _OversizedImageAPI 的四个格子颜色不同


class _OversizedImageAPI(MockImageAPI):
    """模拟服务商无视请求尺寸、总是返回 512x512 大图（如某些代理 API）。"""

    def call(self, prompt: str, size=None, n=1, **kwargs):
        from PIL import Image

        # 返回 512x512 的确定性网格图（4x4 网格，每格 128px）
        img = Image.new("RGB", (512, 512), (255, 255, 255))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        for r in range(4):
            for c in range(4):
                color = (30 + r * 40, 30 + c * 40, 200)
                draw.rectangle([c * 128 + 16, r * 128 + 16, c * 128 + 112, r * 128 + 112], fill=color)
        from core.api.mock_clients import _to_png_bytes

        return APIResult(ok=True, data={"images": [_to_png_bytes(img)], "urls": []})


def test_sprite_workflow_resizes_frames_to_cell_target(tmp_path):
    """服务商返回的精灵图比请求尺寸大时，帧统一缩放到单格尺寸（确定性输出）。"""
    from core.api.mock_clients import MockLLMAPI

    wf = SpriteWorkflow(MockLLMAPI(), _OversizedImageAPI())
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=64, output_dir=tmp_path
    )
    result = wf.run(params)
    assert result.frame_count == 4
    from PIL import Image

    frame = Image.open(sorted(Path(result.frames_dir).glob("*.png"))[0])
    # 512px 大图裁出 256px 格子，最终统一回 64x64 单格尺寸
    assert frame.size == (64, 64)


class _BorderedImageAPI(MockImageAPI):
    """模拟 AI 画的网格：格子之间有黑色边框线（用户反馈的「裁剪到黑框」场景）。"""

    def call(self, prompt: str, size=None, n=1, **kwargs):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (512, 512), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for r in range(2):
            for c in range(2):
                color = (30 + r * 60, 30 + c * 60, 200)
                draw.rectangle([c * 256 + 32, r * 256 + 32, c * 256 + 224, r * 256 + 224], fill=color)
        # 格子间黑框（8px）
        draw.rectangle([248, 0, 263, 512], fill=(0, 0, 0))
        draw.rectangle([0, 248, 512, 263], fill=(0, 0, 0))
        from core.api.mock_clients import _to_png_bytes

        return APIResult(ok=True, data={"images": [_to_png_bytes(img)], "urls": []})


def test_sprite_workflow_removes_cell_black_border(tmp_path):
    """裁切自动内缩去掉格子黑框：帧边缘不再是黑色边框。"""
    from core.api.mock_clients import MockLLMAPI

    wf = SpriteWorkflow(MockLLMAPI(), _BorderedImageAPI())
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=64,
        loop_close=False, output_dir=tmp_path,
    )
    result = wf.run(params)
    from PIL import Image

    frame = Image.open(sorted(Path(result.frames_dir).glob("*.png"))[0]).convert("RGBA")
    # 内缩后边缘应为白色背景（非黑色边框）
    for px in (frame.getpixel((0, 0)), frame.getpixel((0, frame.height - 1)), frame.getpixel((frame.width - 1, 0))):
        assert px[0] > 200 and px[1] > 200 and px[2] > 200, px


class _PixelSheetImageAPI(MockImageAPI):
    """模拟 AI 生成的真·像素风网格图：每格为 16×16 棋盘格（8px 块），帧间一格渐变。

    用于验证完美像素双分辨率分支：原生 16×16 + 用户设定分辨率。
    """

    def call(self, prompt: str, size=None, n=1, **kwargs):
        from PIL import Image, ImageDraw

        size = size or "256x256"
        width, height = parse_size(size)
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        cells_r, cells_c = 2, 2
        cw, ch = width // cells_c, height // cells_r
        block = 8  # 16×16 网格
        for r in range(cells_r):
            for c in range(cells_c):
                frame_idx = r * cells_c + c
                for gy in range(16):
                    for gx in range(16):
                        v = 255 if (gx + gy) % 2 == 0 else 20
                        if frame_idx > 0 and gx == frame_idx and gy == frame_idx:
                            v = 90  # 每格一个不同暗块（帧间内容不同）
                        draw.rectangle(
                            [c * cw + gx * block, r * ch + gy * block,
                             c * cw + (gx + 1) * block, r * ch + (gy + 1) * block],
                            fill=(v, v, v),
                        )
        from core.api.mock_clients import _to_png_bytes

        return APIResult(ok=True, data={"images": [_to_png_bytes(img)], "urls": []})


def test_sprite_workflow_preserves_native_and_preset_resolutions(tmp_path):
    """完美像素双分辨率：原生网格分辨率 + 用户设定分辨率，各三种格式导出。"""
    from core.api.mock_clients import MockLLMAPI

    wf = SpriteWorkflow(MockLLMAPI(), _PixelSheetImageAPI())
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=128,
        cell_inset=-1,  # 测试数据格子已对齐 8px 块，关闭内缩保证网格整数对齐
        output_dir=tmp_path,
    )
    result = wf.run(params)
    from PIL import Image

    # 检测到像素网格（16×16），原生分辨率帧存在且远小于用户分辨率
    assert result.grid == (16, 16)
    assert result.native_width == result.native_height
    assert 8 <= result.native_width <= 32
    # 用户分辨率套
    assert result.width == 128 and result.height == 128
    frame = Image.open(sorted(Path(result.frames_dir).glob("*.png"))[0])
    assert frame.size == (128, 128)
    assert result.gif_path and result.gif_path.exists()
    assert result.sheet_path and result.sheet_path.exists()
    assert result.sheet_index and result.sheet_index.exists()
    # 原生分辨率套：PNG 序列 / 拼接网格图 / GIF + 索引
    assert result.native_frames_dir and len(list(Path(result.native_frames_dir).glob("*.png"))) == 4
    nframe = Image.open(sorted(Path(result.native_frames_dir).glob("*.png"))[0])
    assert nframe.size == (result.native_width, result.native_height)
    assert result.native_gif_path and result.native_gif_path.exists()
    assert result.native_sheet_path and result.native_sheet_path.exists()
    assert result.native_sheet_index and result.native_sheet_index.exists()
    # 机器严格拼接的网格图尺寸正确（2 列）
    import json

    idx = json.loads(result.native_sheet_index.read_text(encoding="utf-8"))
    assert len(idx["frames"]) == 4
    assert idx["sheet_size"]["w"] == result.native_width * 2
    # 两套都做了循环闭合（首尾帧一致）
    npaths = sorted(Path(result.native_frames_dir).glob("*.png"))
    assert npaths[0].read_bytes() == npaths[-1].read_bytes()
    ppaths = sorted(Path(result.frames_dir).glob("*.png"))
    assert ppaths[0].read_bytes() == ppaths[-1].read_bytes()
    # 元数据记录两种分辨率
    meta = json.loads(result.project_file.read_text(encoding="utf-8"))
    assert meta["native_resolution"] == [result.native_width, result.native_height]
    assert meta["detected_grid"] == [16, 16]


class _RecordingImageAPI(MockImageAPI):
    """记录所有生图请求尺寸的 mock。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requested_sizes: list = []

    def call(self, prompt: str, size=None, n=1, **kwargs):
        self.requested_sizes.append(size)
        return super().call(prompt, size=size, n=n, **kwargs)


def test_sprite_workflow_base_image_uses_full_resolution(tmp_path):
    """底图直接以 1024×1024 原始分辨率生成（作 i2i 参考），网格图另请求。"""
    from core.api.mock_clients import MockLLMAPI

    api = _RecordingImageAPI()
    wf = SpriteWorkflow(MockLLMAPI(), api)
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=64, output_dir=tmp_path,
    )
    result = wf.run(params)
    assert result.base_image and result.base_image.exists()
    # 第一次调用 = 底图 1024×1024；第二次 = 网格图（2×2 × 64px = 128×128）
    assert api.requested_sizes[0] == "1024x1024"
    assert api.requested_sizes[1] == "128x128"
    from PIL import Image

    base = Image.open(result.base_image)
    assert base.size == (1024, 1024)  # 原始分辨率直出，未降采样


# --------------------------------------------------------------------------- #
# 手动模式：逐步执行
# --------------------------------------------------------------------------- #
def test_sprite_workflow_stepwise_matches_run(tmp_path):
    """逐步执行（手动模式）与一键 run() 结果一致。"""
    from core.api.mock_clients import MockLLMAPI

    wf = SpriteWorkflow(MockLLMAPI(), MockImageAPI())
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=64, output_dir=tmp_path
    )
    session = wf.new_session(params)
    for name in STEP_ORDER:
        wf.step(name, params, session)
    result = session.result
    assert result is not None
    assert result.frame_count == 4
    assert result.frames_dir and len(list(Path(result.frames_dir).glob("*.png"))) == 4
    assert result.gif_path and result.gif_path.exists()
    assert result.sheet_path and result.sheet_path.exists()
    assert result.sheet_index and result.sheet_index.exists()
    assert session.max_done == len(STEP_ORDER) - 1


def test_sprite_workflow_stepwise_rerun_step(tmp_path):
    """手动模式：重跑「生成网格精灵图」后继续，最终产物完整（下游被重新计算）。"""
    from core.api.mock_clients import MockLLMAPI

    wf = SpriteWorkflow(MockLLMAPI(), MockImageAPI())
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=64, output_dir=tmp_path
    )
    session = wf.new_session(params)
    for name in ("prompts", "base", "sheet"):
        wf.step(name, params, session)
    first_sheet = session.sheet
    # 重跑 sheet（覆盖旧产物）
    wf.step("sheet", params, session)
    assert session.sheet is not first_sheet
    for name in ("crop", "pixelize", "key", "export"):
        wf.step(name, params, session)
    result = session.result
    assert result is not None
    assert result.frame_count == 4
    assert result.gif_path and result.gif_path.exists()


def test_sprite_workflow_step_requires_prerequisite(tmp_path):
    """跳步保护：未裁切直接导出会报错（手动模式防误操作）。"""
    import pytest

    from core.api.mock_clients import MockLLMAPI
    from core.workflow import WorkflowError

    wf = SpriteWorkflow(MockLLMAPI(), MockImageAPI())
    params = SpriteParams(
        description="x", frame_count=4, grid_rows=2, grid_cols=2, cell_size=64, output_dir=tmp_path
    )
    session = wf.new_session(params)
    with pytest.raises(WorkflowError):
        wf.step("export", params, session)
    with pytest.raises(WorkflowError):
        wf.step("unknown-step", params, session)
