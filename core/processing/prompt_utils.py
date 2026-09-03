"""提示词生成与预设管理。

- 从 assets/prompts.json 加载预设动作提示词库（可扩展）：
  {"categories": {分类: {动作: 提示词}}, "durations": {动作: 建议时长秒}}；
  兼容旧结构（顶层即分类 dict，无 durations）。
- LLM 提示词工程：把用户描述转成结构化 JSON（图片提示词/动画提示词/负面提示词）。
- 解析 LLM 返回的 JSON（兼容 markdown 代码块）。
- LLM 不可用时的本地模板 fallback。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from config.settings import ASSETS_DIR

logger = logging.getLogger("PixelAnimIDE.processing.prompt_utils")

_PRESETS_CACHE: Optional[dict] = None     # {分类: {动作: 提示词}}
_DURATIONS_CACHE: Optional[dict] = None   # {动作: 建议时长（秒）}


# --------------------------------------------------------------------------- #
# 预设提示词库
# --------------------------------------------------------------------------- #
def load_presets() -> dict:
    """加载预设提示词库（assets/prompts.json），返回 {分类: {动作: 提示词}}。"""
    global _PRESETS_CACHE, _DURATIONS_CACHE
    if _PRESETS_CACHE is not None:
        return _PRESETS_CACHE
    path = ASSETS_DIR / "prompts.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("未找到预设提示词文件 %s", path)
        _PRESETS_CACHE, _DURATIONS_CACHE = {}, {}
        return _PRESETS_CACHE
    except json.JSONDecodeError as exc:
        logger.warning("预设提示词文件解析失败 %s: %s", path, exc)
        _PRESETS_CACHE, _DURATIONS_CACHE = {}, {}
        return _PRESETS_CACHE
    if isinstance(data, dict) and "categories" in data:
        cats = data.get("categories") or {}
        _DURATIONS_CACHE = data.get("durations") or {}
    else:
        # 旧结构兼容：顶层即分类 dict，无建议时长
        cats = data
        _DURATIONS_CACHE = {}
    _PRESETS_CACHE = {
        cat: items for cat, items in cats.items() if isinstance(items, dict)
    }
    return _PRESETS_CACHE


def preset_duration(name: str) -> Optional[float]:
    """预设动作的建议动画时长（秒）；未收录返回 None。"""
    load_presets()
    try:
        v = float(_DURATIONS_CACHE.get((name or "").strip(), 0) or 0)
    except (TypeError, ValueError):
        return None
    return v if 0.1 <= v <= 10.0 else None


def preset_categories() -> List[Tuple[str, List[str]]]:
    """返回 [(分类, [动作名…])]，按文件顺序；动作数为 0 的分类不返回。"""
    out: List[Tuple[str, List[str]]] = []
    for cat, items in load_presets().items():
        names = [n for n in items.keys() if str(n).strip()]
        if names:
            out.append((cat, names))
    return out


def recommended_frames(action: str, fps: int) -> Optional[int]:
    """按动作建议帧数（帧数 = 建议时长 × 帧率）；未知动作返回 None。"""
    secs = preset_duration(action)
    if not secs:
        return None
    return max(2, int(round(secs * max(1, int(fps)))))


def get_preset(name: str) -> Optional[str]:
    """按名称查预设提示词（可在多个分类中查找）。"""
    name = name.strip()
    for category, items in load_presets().items():
        if isinstance(items, dict) and name in items:
            return str(items[name])
    return None


def preset_names() -> list:
    """返回所有预设动作名（跨分类摊平，保持加载顺序）。"""
    names: list = []
    for _cat, items in preset_categories():
        names.extend(items)
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

# 黑色版：主体为浅色系、首帧背景被归一化为纯黑时使用（提示词必须与首帧背景一致，
# 否则模型会困惑并在中间帧漂移回白色/灰色背景）
BACKGROUND_STABILITY_RULE_DARK = (
    "IMPORTANT: the background must remain a SOLID PURE BLACK (#000000) and completely "
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
