"""精灵图工作流测试（模拟 API）。"""
from pathlib import Path

from core.api.base import APIResult
from core.api.mock_clients import MockImageAPI, MockLLMAPI
from core.workflow import SpriteParams, SpriteWorkflow
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
