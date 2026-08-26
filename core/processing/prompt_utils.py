"""提示词生成与预设管理。

- 从 assets/prompts.json 加载预设动作提示词（可扩展）。
- LLM 提示词工程：把用户描述转成结构化 JSON（图片提示词/动画提示词/负面提示词）。
- 解析 LLM 返回的 JSON（兼容 markdown 代码块）。
- LLM 不可用时的本地模板 fallback。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from config.settings import ASSETS_DIR

logger = logging.getLogger("PixelAnimIDE.processing.prompt_utils")

_PRESETS_CACHE: Optional[dict] = None


# --------------------------------------------------------------------------- #
# 预设提示词库
# --------------------------------------------------------------------------- #
# 各预设动作建议的动画时长（秒）。
# AI 视频生成的运动普遍偏慢、偏柔和，时长不足会截到动作起步阶段、循环不完整。
ACTION_RECOMMENDED_SECONDS = {
    # 移动类
    "步行": 2.0,
    "奔跑": 1.5,
    "跳跃": 1.2,
    "突进": 1.0,
    "爬行": 2.5,
    # 战斗类
    "攻击": 1.2,
    "格挡": 1.0,
    "昏迷": 1.5,
}


def recommended_frames(action: str, fps: int) -> Optional[int]:
    """按动作类型建议帧数（帧数 = 建议时长 × 帧率）；未知动作返回 None。"""
    secs = ACTION_RECOMMENDED_SECONDS.get((action or "").strip())
    if not secs:
        return None
    return max(2, int(round(secs * max(1, int(fps)))))


def load_presets() -> dict:
    """加载预设提示词库（assets/prompts.json）。"""
    global _PRESETS_CACHE
    if _PRESETS_CACHE is not None:
        return _PRESETS_CACHE
    path = ASSETS_DIR / "prompts.json"
    try:
        with open(path, encoding="utf-8") as f:
            _PRESETS_CACHE = json.load(f)
    except FileNotFoundError:
        logger.warning("未找到预设提示词文件 %s", path)
        _PRESETS_CACHE = {}
    return _PRESETS_CACHE


def get_preset(name: str) -> Optional[str]:
    """按名称查预设提示词（可在多个分类中查找）。"""
    name = name.strip()
    for category, items in load_presets().items():
        if isinstance(items, dict) and name in items:
            return str(items[name])
    return None


def preset_names() -> list:
    """返回所有预设动作名。"""
    names: list = []
    for items in load_presets().values():
        if isinstance(items, dict):
            names.extend(items.keys())
    return names


def build_animation_prompt(action: str) -> str:
    """动作名 -> 动画提示词；找不到预设时给出通用默认。"""
    preset = get_preset(action)
    if preset:
        return preset
    return "subtle idle animation, smooth looping, consistent character design"


# --------------------------------------------------------------------------- #
# LLM 提示词工程
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a professional pixel-art game asset prompt engineer. "
    "Given the user's description, produce a compact JSON object (no markdown fences) "
    'with exactly these keys: "image_prompt" (detailed English prompt for generating '
    "the first frame; MUST request pixel-art style, clean hard edges, a very limited "
    "solid color palette, and a SOLID PURE WHITE background (#FFFFFF) — no background "
    "objects, no gradients, no shading, no gray or colored background), "
    '"animation_prompt" (a concise English prompt, at most ~30 words, describing ONLY '
    "the looping motion the user stated. It MUST faithfully and exactly reflect the "
    "user's action — never invent characters, mounts, weapons, environments or any "
    "context the user did not mention (e.g. if the user says 'thrust attack', a single "
    "character performs a quick thrust, nothing else). Describe only the essential "
    "motion cycle, no unnecessary micro-details), "
    '"negative_prompt" (things to avoid, including textured/gray/colored backgrounds), '
    '"frame_count" (int 4-60) and "fps" (int 4-30) — estimate the natural duration of '
    "this specific action from the description AND action, then set frame_count/fps so "
    "that duration = frame_count / fps: quick actions like a slash or dash are ~1s "
    "(e.g. 8 frames @ 8fps), cyclic/slow actions like walking or crawling are 1.5-2.5s "
    "(e.g. 16-20 frames @ 8fps), a jump is ~1.2s. "
    "Respond with the JSON object only."
)


# 严格纠正指令：动画提示词冗长或偏离用户动作时附加
STRICT_ANIMATION_CORRECTION = (
    "\nSTRICT CORRECTION for animation_prompt: rewrite it to be concise (under 25 words) "
    "and to describe ONLY the exact action the user stated — nothing more. Do NOT invent "
    "characters, mounts, weapons, settings, or any context not mentioned by the user. "
    "Keep the core looping motion only."
)


# 图转视频背景稳定性强约束：首帧背景为纯色时，防止中间帧背景漂移
BACKGROUND_STABILITY_RULE = (
    "IMPORTANT: the background must remain a SOLID PURE WHITE (#FFFFFF) and completely "
    "unchanged in every frame — only the subject moves. No background shift, no camera "
    "movement, no background color/texture change."
)


# 主体完整性约束：主体必须完整可见、居中、与画面边缘留白（不裁切、不贴边）
SUBJECT_MARGIN_RULE = (
    "IMPORTANT: the entire subject must be fully visible and centered with a clear margin "
    "from all frame edges — never cropped, never touching or cut off by the border, "
    "keep the whole object complete in view at all times."
)


# 像素风格意图关键字（提示词中出现即判定为像素风）
PIXEL_KEYWORDS = (
    "pixel art",
    "pixelart",
    "pixel-art",
    "pixelated",
    "像素",
    "sprite",
    "perler",
    "8-bit",
    "8bit",
    "chunky pixel",
    "retro game",
    "dot art",
)


def is_pixel_prompt(text: str) -> bool:
    """提示词/描述中是否出现像素风意图。"""
    t = (text or "").lower()
    return any(k in t for k in PIXEL_KEYWORDS)


def build_user_prompt(description: str, action: str = "") -> str:
    text = f"Description: {description.strip()}"
    if action.strip():
        text += f"\nAction/motion: {action.strip()}"
    return text


def parse_json_response(text: str) -> dict:
    """从 LLM 回复中提取 JSON（兼容 ```json 代码块与前后噪声）。"""
    if not isinstance(text, str) or not text.strip():
        return {}
    text = text.strip()
    # 去掉 markdown 代码块
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 从第一个 { 到最后一个 } 截取
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        logger.warning("LLM 返回无法解析为 JSON: %s", text[:200])
        return {}


# --------------------------------------------------------------------------- #
# 本地 fallback（LLM 不可用时使用）
# --------------------------------------------------------------------------- #
def build_fallback_prompts(description: str, action: str = "") -> dict:
    """不依赖 LLM 的模板化提示词，保证 Solo 流程可离线跑通。

    内置强制项：纯白背景、有限纯色调色板（AI 生成背景常发灰/发黄）。
    动画提示词必须契合用户填写的动作（预设名用预设，自定义动作原文嵌入）。
    """
    desc = (description or "").strip()
    if action.strip():
        preset = get_preset(action)
        animation = preset if preset else f"{action.strip()} animation, smooth looping, consistent character design"
    else:
        animation = "subtle idle animation, smooth looping, consistent character design"
    image_prompt = (
        f"Pixel art of {desc or 'a game character'}, clean hard edges, "
        "limited solid color palette, game sprite style, centered, "
        "solid pure white background (#FFFFFF), plain white canvas, "
        "no background objects, no gradients, no anti-aliasing"
    )
    negative_prompt = (
        "blurry, anti-aliasing, gradients, photorealism, text, watermark, extra limbs, distorted, "
        "background objects, textured background, gray or colored background"
    )
    return {
        "image_prompt": image_prompt,
        "animation_prompt": animation,
        "negative_prompt": negative_prompt,
    }


def normalize_prompts(data: dict, description: str, action: str = "") -> dict:
    """把 LLM 返回的 dict 归一化为固定三键结构（缺失字段用 fallback 补齐）。"""
    fallback = build_fallback_prompts(description, action)
    result = {
        "image_prompt": str(data.get("image_prompt") or fallback["image_prompt"]).strip(),
        "animation_prompt": str(data.get("animation_prompt") or fallback["animation_prompt"]).strip(),
        "negative_prompt": str(data.get("negative_prompt") or fallback["negative_prompt"]).strip(),
    }
    if not result["image_prompt"]:
        result["image_prompt"] = fallback["image_prompt"]
    return result
