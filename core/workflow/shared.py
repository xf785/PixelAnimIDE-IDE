"""Solo / IDE 工作流共用的小工具：提示词强制项、生图尺寸解析。"""
from __future__ import annotations

from typing import Optional, Tuple

from config.settings import ASPECT_RATIOS, DEFAULT_ASPECT
from core.processing.prompt_utils import SUBJECT_MARGIN_RULE, is_pixel_prompt


def parse_size(text: str) -> Optional[Tuple[int, int]]:
    """解析 '宽x高' / '宽*高' / '宽,高' 尺寸字符串，失败返回 None。"""
    if not text:
        return None
    for sep in ("x", "×", "*", ","):
        if sep in str(text):
            parts = str(text).lower().split(sep)
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                return int(parts[0].strip()), int(parts[1].strip())
    return None


def finalize_prompts(
    prompts: dict,
    target_size: Tuple[int, int],
    aspect_ratio: str,
    max_colors: int,
) -> dict:
    """内置强制项：像素尺寸 + 颜色数量 + 纯白背景，严格写入图片提示词。

    用户设置的「像素尺寸」以强指令形式写入（醒目、不可忽略），
    保证 AI 按目标像素网格出图（最终尺寸仍由像素化管线强制保证）。
    """
    prompts = dict(prompts)
    w, h = int(target_size[0]), int(target_size[1])
    res_rule = (
        f"IMPORTANT — the artwork MUST be true pixel art on an EXACT {w}x{h} pixel grid "
        f"(aspect {aspect_ratio}, each pixel a solid square, no anti-aliasing, no blur), "
        f"using at most {max_colors} distinct solid colors, "
        "on a solid pure white background (#FFFFFF), no gradients, no shading."
    )
    prompts["image_prompt"] = f"{prompts['image_prompt']} {res_rule} {SUBJECT_MARGIN_RULE}"
    prompts["negative_prompt"] = (
        f"{prompts['negative_prompt']}, background objects, gray or colored background, gradients, "
        "anti-aliasing, wrong resolution, non-pixel-art, cropped subject, cut-off subject, "
        "subject touching frame edges"
    )
    return prompts


def resolve_api_image_size(
    cfg_size: Optional[str],
    aspect_ratio: str,
    pixel_size: int,
    prompt_text: str,
) -> Tuple[int, int]:
    """生图 API 的请求尺寸。

    1) 优先使用图片 API 配置的「默认尺寸」（用户可在设置里调小以节省成本）；
    2) 否则若判定为像素风意图，强制用预设像素分辨率（长边 max(pixel_size, 256)）；
    3) 否则按宽高比计算（上限 1024）。
    """
    parsed = parse_size(cfg_size) if cfg_size else None
    if parsed:
        return parsed
    rw, rh = ASPECT_RATIOS.get(aspect_ratio, ASPECT_RATIOS[DEFAULT_ASPECT])

    if is_pixel_prompt(prompt_text):
        side = max(256, min(int(pixel_size), 768))
        if rw >= rh:
            return side, max(64, round(side * rh / rw))
        return max(64, round(side * rw / rh)), side

    max_side = 1024
    if rw >= rh:
        return max_side, max(64, round(max_side * rh / rw))
    return max(64, round(max_side * rw / rh)), max_side
