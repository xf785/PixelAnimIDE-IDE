"""建筑素材（道具）生成：文生图 → 算法抠底 → 像素完美、背景透明的独立素材。

用于树木 / 草丛 / 花 / 石头 / 箱子这类**可叠加在地块上的物件**：
- 提示词要求 N 个等大方格、纯色背景、每个物件完整落在格内、硬边像素风；
- 处理后每个物件都是**独立 PNG（alpha 精确 0/255，无白边、无半透明灰边）**，
  按内容裁紧后等比缩放到瓦片尺寸，并**底部居中**对齐 —— 这样贴到地图上恰好「站在」地面格；
- 结果放进 `TilePack.pieces`，与建筑拼件同一套机制：可保存成 `.tilepack`，
  地图预览用「添加瓦片包」加载后即可当画笔铺。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from core.processing import background as bg

logger = logging.getLogger("PixelAnimIDE.tilemap.props")

#: 主体可能含浅色/白色的关键词 —— 这类素材必须用**纯黑底**（否则白底会把主体一起删掉）
LIGHT_SUBJECT_KEYWORDS = (
    "雪", "白", "冰", "云", "雾", "霜", "骨", "纸", "奶", "光", "银", "棉", "花", "羊", "蛋", "石", "晶", "月",
    "snow", "white", "ice", "cloud", "fog", "frost", "bone", "paper", "milk", "light", "silver",
    "cotton", "sheep", "egg", "crystal", "moon", "ghost", "skull", "pearl", "chrome",
)


def subject_is_light(description: str) -> bool:
    """从描述/提示词粗判主体是否可能含浅色（雪、云、白骨…）→ 决定用黑底还是白底。"""
    text = (description or "").lower()
    return any(k.lower() in text for k in LIGHT_SUBJECT_KEYWORDS)


PROP_BACKGROUNDS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "magenta": (255, 0, 255),
    "green": (0, 255, 0),
}


def build_prop_prompts(
    description: str,
    style: str = "game sprite",
    variants: int = 4,
    tile_size: Optional[int] = None,
    cell_px: Optional[int] = None,
    background: str = "auto",
) -> dict:
    """素材（道具）提示词：N 个变体排成网格，纯色底、格内完整、像素硬边、不要文字。"""
    desc = (description or "").strip() or "a tree"
    style = (style or "").strip() or "game sprite"
    variants = max(1, min(16, int(variants)))
    cols = 2 if variants <= 4 else (3 if variants <= 9 else 4)
    rows = int(np.ceil(variants / cols))
    if background in ("auto", "", None):
        # 主体可能含浅色 -> 纯黑底（黑底不会误删主体）；否则纯白底
        background = "black" if subject_is_light(description) else "white"
    bg_rgb = PROP_BACKGROUNDS.get(background, PROP_BACKGROUNDS["white"])
    bg_name = {"white": "SOLID PURE WHITE", "black": "SOLID PURE BLACK",
               "magenta": "SOLID PURE MAGENTA",
               "green": "SOLID PURE GREEN"}.get(background, "SOLID PURE WHITE")
    cell_note = (
        f" Each cell is exactly {int(cell_px)}x{int(cell_px)} pixels "
        f"(the whole image is exactly {int(cell_px) * cols}x{int(cell_px) * rows} pixels)."
        if cell_px
        else ""
    )
    prompt = (
        f"Create ONE single pixel-art sprite sheet of {variants} variants of '{desc}', in {style} "
        f"style, for a top-down / side-view game. IMAGE LAYOUT (critical): the image is ONE square "
        f"divided into an EXACT {cols}x{rows} grid of equal square cells.{cell_note} "
        f"Every cell contains exactly ONE complete object, fully inside its cell with a small margin, "
        f"standing on the bottom of the cell (bottom-aligned), centred horizontally, and all variants "
        f"share the same palette, same scale and same light direction.\n"
        f"BACKGROUND RULE (critical): the background of every cell is {bg_name} "
        f"(RGB {bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]}), perfectly flat, no gradient, no shadow cast on the "
        f"background, no outline around the cell, no grid lines, no frames. The object must be sharply "
        f"separated from the background (a clean 1px darker outline around the object is good).\n"
        f"Style rules: crisp hard pixel edges, NO anti-aliasing, no blur, no transparency, no drop "
        f"shadow, limited palette, readable silhouette. ABSOLUTE RULE — NO TEXT: no letters, no words, "
        f"no numbers, no labels, no signature, no watermark, no UI.\n"
        f"Negative: text, watermark, grid lines, cell borders, gradient background, soft shadows, blur, "
        f"anti-aliasing, photorealism, multiple objects per cell, cut-off object, empty cells."
    )
    return {
        "image_prompt": prompt,
        "negative_prompt": (
            "text, watermark, letters, numbers, grid lines, frames, borders between cells, "
            "gradient background, drop shadow, blurry, anti-aliasing, photorealism, "
            "cropped object, empty cell"
        ),
        "variants": variants,
        "grid_rows": rows,
        "grid_cols": cols,
        "background": background,
        "tile_size": tile_size,
        "cell_px": cell_px,
    }


# --------------------------------------------------------------------------- #
# 抠底与裁切
# --------------------------------------------------------------------------- #
def _border_color(arr: np.ndarray, patch: int = 6) -> np.ndarray:
    """背景色估计：取**四角小块**的中位色（比整条边稳——物件压到边上也不会被带偏）。"""
    h, w = arr.shape[:2]
    p = max(2, min(patch, h // 4, w // 4))
    corners = np.concatenate([
        arr[:p, :p, :3].reshape(-1, 3), arr[:p, w - p:, :3].reshape(-1, 3),
        arr[h - p:, :p, :3].reshape(-1, 3), arr[h - p:, w - p:, :3].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    return np.median(corners, axis=0)


def _flood_transparent(rgba: Image.Image, bg_rgb: np.ndarray, tolerance: float) -> Image.Image:
    """把与图像边缘连通的、接近 bg_rgb 的像素变透明（不动物件内部的同色区域）。"""
    arr = np.asarray(rgba.convert("RGBA")).astype(np.int16)
    dist = np.abs(arr[..., :3] - bg_rgb[None, None, :]).max(axis=2)
    near = dist <= tolerance
    h, w = near.shape
    mask = np.zeros((h, w), dtype=bool)
    stack = [(y, x) for y in (0, h - 1) for x in range(w) if near[y, x]]
    stack += [(y, x) for x in (0, w - 1) for y in range(h) if near[y, x]]
    for y, x in stack:
        mask[y, x] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and near[ny, nx] and not mask[ny, nx]:
                mask[ny, nx] = True
                stack.append((ny, nx))
    out = np.array(rgba.convert("RGBA"))
    out[mask] = (0, 0, 0, 0)
    return Image.fromarray(out, "RGBA")


def _harden_alpha(img: Image.Image) -> Image.Image:
    """alpha 归一化到 0/255（像素画硬边；叠加到地块上不会出现灰边）。"""
    arr = np.asarray(img.convert("RGBA")).copy()
    alpha = arr[..., 3]
    arr[..., 3] = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    arr[arr[..., 3] == 0] = (0, 0, 0, 0)
    return Image.fromarray(arr, "RGBA")


def key_background_range(tile: Image.Image, background: str = "white", hard: int = 240, soft: int = 214) -> Image.Image:
    """按底色**直接删除特定范围内的像素**（用户建议的算法）：

    - 白底（主体不含浅色）：删除所有通道都 ≥ `soft` 的近白像素（≥`hard` 必然删）；
      这样即使 AI 画了带噪点/渐变的"纯白"底也能彻底清干净；
    - 黑底（主体可能含浅色）：删除所有通道都 ≤ `255 - soft` 的近黑像素。
    之后再叠加 flood-fill 兜底（去掉与边缘连通的中间调底色）。
    """
    arr = np.asarray(tile.convert("RGBA"))
    out = np.array(arr)
    rgb = out[..., :3].astype(np.int16)
    if background == "black":
        near = rgb.max(axis=2) <= (255 - soft)
        strong = rgb.max(axis=2) <= (255 - hard)
    else:
        near = rgb.min(axis=2) >= soft
        strong = rgb.min(axis=2) >= hard
    out[near] = (0, 0, 0, 0)
    # 强判定再放宽一点（连边缘中间调也一起清掉）
    out[strong] = (0, 0, 0, 0)
    return _harden_alpha(Image.fromarray(out, "RGBA"))


def key_background(
    tile: Image.Image,
    tolerance: float = 38,
    max_opaque_ratio: float = 0.72,
    tolerances: Sequence[float] = (38, 58, 84, 120),
) -> Image.Image:
    """抠除纯色背景（自动升级容差，保证「一定有背景被删掉」）。

    1. 背景色取四角小块中位色；从图像四边**洪泛**，只删与边缘连通的同色像素
       （物件内部同色区域不会被挖空）；
    2. 容差从小到大自动重试：若抠完仍然「几乎整格不透明」，说明背景不是严格纯色
       （AI 常见：轻微渐变/噪点），加大容差继续；
    3. 最后仍然不透明（物件与背景同色、无法分离）时，做一次**全局**同色删除兜底，
       宁可留下干净的剪影也不要整块底色。
    """
    rgba = tile.convert("RGBA")
    bg_rgb = _border_color(np.asarray(rgba))
    best: Optional[Image.Image] = None
    for tol in tolerances:
        keyed = _flood_transparent(rgba, bg_rgb, float(tol))
        ratio = float((np.asarray(keyed)[..., 3] > 0).mean())
        if best is None or ratio < float((np.asarray(best)[..., 3] > 0).mean()):
            best = keyed
        if ratio <= max_opaque_ratio:
            return _harden_alpha(keyed)
    # 兜底：全局删除接近背景色的像素
    arr = np.asarray(rgba).astype(np.int16)
    dist = np.abs(arr[..., :3] - bg_rgb[None, None, :]).max(axis=2)
    out = np.array(rgba)
    out[dist <= max(tolerances)] = (0, 0, 0, 0)
    return _harden_alpha(Image.fromarray(out, "RGBA"))


def trim_to_content(tile: Image.Image, pad: int = 0) -> Optional[Image.Image]:
    """裁掉四周全透明区域（保留内容 + 可选留边）。"""
    arr = np.asarray(tile.convert("RGBA"))
    ys, xs = np.nonzero(arr[..., 3] > 0)
    if not ys.size:
        return None
    y0, y1 = max(0, int(ys.min()) - pad), min(arr.shape[0], int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(arr.shape[1], int(xs.max()) + 1 + pad)
    return tile.convert("RGBA").crop((x0, y0, x1, y1))


def fit_to_tile(tile: Image.Image, tile_size: int, bottom_align: bool = True) -> Image.Image:
    """等比缩放到 tile_size 内并底部居中放置（物件「站」在格子上）。"""
    t = trim_to_content(tile)
    canvas = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    if t is None:
        return canvas
    scale = min(tile_size / t.width, tile_size / t.height)
    w = max(1, int(round(t.width * scale)))
    h = max(1, int(round(t.height * scale)))
    t = t.resize((w, h), Image.Resampling.NEAREST)
    x = (tile_size - w) // 2
    y = tile_size - h if bottom_align else (tile_size - h) // 2
    canvas.alpha_composite(t, (x, y))
    return canvas


def process_prop_sheet(
    img: Image.Image,
    rows: int,
    cols: int,
    names: Sequence[str],
    tile_size: int = 32,
    tolerance: int = 38,
    bottom_align: bool = True,
    background: str = "white",
) -> Dict[str, Image.Image]:
    """把素材底图切成 rows×cols 格，逐格抠底 + 裁紧 + 归一化，返回 {名字: RGBA}。"""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    cw, ch = w // max(1, cols), h // max(1, rows)
    props: Dict[str, Image.Image] = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(names):
                break
            cell = rgba.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
            keyed = key_background_range(cell, background=background)
            keyed = key_background(keyed, tolerance=tolerance)
            alpha = np.asarray(keyed)[..., 3]
            if not (alpha > 0).any():
                idx += 1
                continue                       # 空格子（AI 少画了）直接跳过
            opaque = float((alpha > 0).mean())
            if opaque > 0.85:
                logger.warning("素材 %s 抠底后仍占 %.0f%% 格面积（背景可能与物件同色），已按剪影输出",
                               names[idx], opaque * 100)
            props[str(names[idx])] = fit_to_tile(keyed, tile_size, bottom_align=bottom_align)
            idx += 1
    return props


def prop_names(base: str, count: int) -> List[str]:
    """素材命名：`名字_1`、`名字_2`…（作为包内拼件名，导出即文件名）。"""
    stem = (base or "prop").strip() or "prop"
    return [f"{stem}_{i + 1}" for i in range(max(1, count))]
