"""Solo / IDE / 精灵图工作流共用的小工具：提示词强制项、生图尺寸解析、LLM 提示词生成。"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from config.settings import ASPECT_RATIOS, DEFAULT_ASPECT
from core.api.base import APIResult, BaseAPI
from core.processing.prompt_utils import (
    STRICT_ANIMATION_CORRECTION,
    SUBJECT_MARGIN_RULE,
    SYSTEM_PROMPT,
    build_user_prompt,
    is_pixel_prompt,
    parse_json_response,
)
from ui.i18n import tr

# 动画提示词最大词数：超过则附加「严格纠正」指令重试
MAX_ANIMATION_PROMPT_WORDS = 40
_PROMPT_FIRST_TOKENS = 1600
_PROMPT_RETRY_TOKENS = 4096


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


# --------------------------------------------------------------------------- #
# LLM 提示词生成（solo / ide / sprite 共用）
# --------------------------------------------------------------------------- #
def _extract_prompt_data(result: APIResult) -> Optional[dict]:
    """把 LLM 结果归一化为提示词 dict（dict 直返 / 文本解析 JSON），失败返回 None。"""
    if not result.ok:
        return None
    data = result.data
    if isinstance(data, dict):
        return data
    parsed = parse_json_response(data)
    return parsed or None


def _prompts_usable(result: APIResult) -> bool:
    """LLM 结果是否含可用数据（dict 含提示词键，或非空文本）。"""
    if not result.ok:
        return False
    data = result.data
    if isinstance(data, dict):
        return bool(data.get("image_prompt") or data.get("animation_prompt"))
    text = (data or "").strip() if isinstance(data, str) else ""
    return bool(text)


def generate_prompt_data(
    llm_api: BaseAPI,
    description: str,
    action: str,
    log: Optional[Callable[[str, str], None]] = None,
) -> Tuple[Optional[dict], APIResult]:
    """LLM 生成提示词：一次调用 + 两重兜底重试（三个工作流共用同一逻辑）。

    1. 首次 max_tokens=1600；输出为空 / 调用失败 / 不可解析 -> 提高到 4096
       重试一次（推理模型的 max_tokens 常被思考吃光导致 content 为空）；
    2. animation_prompt 词数 > MAX_ANIMATION_PROMPT_WORDS -> 附加
       STRICT_ANIMATION_CORRECTION 重试一次，保证只描述用户动作、不虚构上下文。

    返回 (解析后的提示词 dict, 最后一次 APIResult)；dict 为 None 时调用方应
    降级本地模板（可用 last 的 message/ok 组织日志）。
    log: 可选回调 (level, message)，用于输出重试提示。
    """

    def _call(max_tokens: int, strict: bool = False) -> APIResult:
        system = SYSTEM_PROMPT + (STRICT_ANIMATION_CORRECTION if strict else "")
        return llm_api.call(
            prompt=build_user_prompt(description, action),
            system=system,
            action=action,
            max_tokens=max_tokens,
        )

    result = _call(_PROMPT_FIRST_TOKENS)
    if not _prompts_usable(result):
        if log:
            log("warn", tr("LLM 输出为空或不可解析，提高 max_tokens 重试一次"))
        result = _call(_PROMPT_RETRY_TOKENS)

    data = _extract_prompt_data(result)
    if data is not None:
        animation = str(data.get("animation_prompt") or "")
        word_count = len([w for w in animation.split() if w.strip()])
        if word_count > MAX_ANIMATION_PROMPT_WORDS:
            if log:
                log(
                    "warn",
                    tr(
                        "动画提示词过于冗长（{0} 词），按「简洁且忠实于动作」要求重试一次"
                    ).format(word_count),
                )
            result = _call(_PROMPT_FIRST_TOKENS, strict=True)
            data = _extract_prompt_data(result)
    return data, result
