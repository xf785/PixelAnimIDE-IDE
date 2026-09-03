"""确定性模拟客户端：无需任何 API Key 即可跑通全流程（演示 / 测试 / 离线试用）。

- MockLLMAPI   根据描述生成结构化的图片/动画提示词（模板化）
- MockImageAPI 生成确定性的合成图像（PNG 字节）
- MockVideoAPI 生成确定性的帧序列（PNG 字节列表）
"""
from __future__ import annotations

import io
import logging
import zlib
from typing import List, Optional

from PIL import Image, ImageDraw

from .base import APIResult, BaseAPI

logger = logging.getLogger("PixelAnimIDE.api.mock")


def _hash_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# 提示词模板（与 prompt_utils 中 fallback 逻辑保持一致）
# --------------------------------------------------------------------------- #
def build_mock_prompts(description: str, action: str = "") -> dict:
    """根据用户描述生成结构化提示词。"""
    from core.processing.prompt_utils import build_animation_prompt, get_preset, preset_duration

    desc = _clean_description(description)
    if action.strip():
        preset = get_preset(action)
        hint = preset if preset else f"{action.strip()} animation"
    else:
        hint = "subtle idle animation"
    image_prompt = (
        f"Pixel art of {desc or 'a character'}, "
        "clean hard edges, limited color palette, game sprite style, centered, "
        "solid pure white background (#FFFFFF), no background objects, no anti-aliasing"
    )
    animation_prompt = f"{hint}, smooth looping, consistent character design"
    negative_prompt = (
        "blurry, anti-aliasing, gradients, photorealism, text, watermark, "
        "extra limbs, distorted anatomy, inconsistent colors, "
        "background objects, gray or colored background"
    )
    # LLM 参数建议：按动作建议时长推导流畅循环的帧数（模拟真实 LLM 的 frame_count/fps 字段）
    secs = preset_duration(action) or 1.0  # 无动作/未知动作时保持默认 1s
    frame_count = max(4, min(48, int(round(secs * 8))))
    return {
        "image_prompt": image_prompt,
        "animation_prompt": animation_prompt,
        "negative_prompt": negative_prompt,
        "frame_count": frame_count,
        "fps": 8,
    }


def _clean_description(prompt: str) -> str:
    """去掉 build_user_prompt 附加的 'Description:'/'Action/motion:' 标记。"""
    text = (prompt or "").strip()
    if text.startswith("Description:"):
        text = text.split("Description:", 1)[1]
        if "Action/motion:" in text:
            text = text.split("Action/motion:", 1)[0]
    return text.strip()


# --------------------------------------------------------------------------- #
# 合成图像
# --------------------------------------------------------------------------- #
PALETTE = [
    (255, 255, 255),  # 白
    (34, 40, 49),     # 深灰蓝
    (238, 238, 238),  # 浅灰
    (214, 48, 49),    # 红
    (0, 148, 255),    # 蓝
    (255, 205, 0),    # 黄
    (38, 166, 91),    # 绿
    (150, 80, 200),   # 紫
]


def _make_frame(width: int, height: int, seed: int, t: int, n_frames: int) -> Image.Image:
    """生成一帧确定性像素风图像：渐变背景 + 弹跳圆 + 大幅移动的主体方块。

    运动幅度足够大，保证小尺寸像素化后各帧依然可区分（GIF 不会合并帧）。
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    bg_top, bg_bottom = PALETTE[seed % 4 + 2], PALETTE[(seed // 4) % 4 + 2]
    for y in range(height):
        k = y / max(1, height - 1)
        color = tuple(int(bg_top[i] + (bg_bottom[i] - bg_top[i]) * k) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)

    phase = t / max(1, n_frames - 1)

    # 弹跳圆（横向大跨度往返 + 纵向抛物线）
    cx = int(width * (0.15 + 0.7 * phase))
    cy = int(height * (0.72 - 0.5 * abs((phase * 2) % 2 - 1)))
    r = max(8, min(width, height) // 8)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PALETTE[(seed + t) % len(PALETTE)])

    # 主体方块：从左侧大幅移动到右侧
    bx = int(width * (0.15 + 0.6 * phase))
    bw, bh = width // 5, height // 4
    draw.rectangle([bx, height - bh - height // 5, bx + bw, height - height // 5], fill=PALETTE[seed % 4])

    # 顶部装饰线（随帧左右偏移，提供额外运动线索）
    bar_x = int(width * 0.1 * ((t * 2) % 5))
    draw.rectangle([bar_x, 0, bar_x + width // 6, max(2, height // 40)], fill=PALETTE[(seed // 7) % len(PALETTE)])
    return img


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Mock 客户端
# --------------------------------------------------------------------------- #
def _extract_action(prompt: str, kwargs_action: str) -> str:
    """动作提示：优先取显式参数，其次从 prompt 文本中解析。"""
    if kwargs_action and kwargs_action.strip():
        return kwargs_action.strip()
    text = prompt or ""
    marker = "Action/motion:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return ""


class MockLLMAPI(BaseAPI):
    KIND = "llm"

    def __init__(self, config=None, **kwargs):
        super().__init__(config or {}, **kwargs)

    def call(self, prompt: str, **kwargs) -> APIResult:
        action = _extract_action(prompt, kwargs.get("action", ""))
        return APIResult(ok=True, data=build_mock_prompts(prompt, action))

    def test_connection(self) -> APIResult:
        return APIResult(ok=True, data="模拟 API 可用（无需密钥）")


class MockImageAPI(BaseAPI):
    KIND = "image"

    def __init__(self, config=None, **kwargs):
        super().__init__(config or {}, **kwargs)

    def call(self, prompt: str, size: Optional[str] = None, n: int = 1, **kwargs) -> APIResult:
        size = size or self.params.get("size", "1024x1024")
        width, height = parse_size(size)
        seed = _hash_seed(prompt)
        images = [_to_png_bytes(_make_frame(width, height, seed + i, t=0, n_frames=1)) for i in range(max(1, int(n)))]
        return APIResult(ok=True, data={"images": images, "urls": []})

    def test_connection(self) -> APIResult:
        return APIResult(ok=True, data="模拟 API 可用（无需密钥）")


class MockVideoAPI(BaseAPI):
    KIND = "video"

    def __init__(self, config=None, **kwargs):
        super().__init__(config or {}, **kwargs)

    def call(
        self,
        image_bytes: Optional[bytes] = None,
        prompt: str = "",
        frames: Optional[int] = None,
        fps: Optional[int] = None,
        **kwargs,
    ) -> APIResult:
        n_frames = int(frames if frames is not None else self.params.get("frames", 8))
        # 帧尺寸上限 256：保持运动幅度在后续降采样后依然可见
        width = height = 256
        if image_bytes:
            try:
                src = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                scale = min(1.0, 256.0 / max(src.size))
                width = max(16, int(src.width * scale))
                height = max(16, int(src.height * scale))
            except Exception:  # noqa: BLE001
                pass
        seed = _hash_seed(prompt or "mock-video")
        out: List[bytes] = []
        for t in range(n_frames):
            frame = _make_frame(width, height, seed + t, t, n_frames)
            out.append(_to_png_bytes(frame))
        return APIResult(ok=True, data={"video_url": None, "frames": out})

    def test_connection(self) -> APIResult:
        return APIResult(ok=True, data="模拟 API 可用（无需密钥）")


def parse_size(size: str):
    """'1024x1024' -> (1024, 1024)；也兼容 '1024*1024'、'1024,1024'。"""
    size = str(size).strip().lower().replace("*", "x").replace(",", "x")
    for sep in ("x", "×"):
        if sep in size:
            parts = size.split(sep)
            if len(parts) == 2:
                try:
                    return int(parts[0].strip()), int(parts[1].strip())
                except ValueError:
                    break
    return 1024, 1024
