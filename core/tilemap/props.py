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

PROP_BACKGROUNDS = {
    "white": (255, 255, 255),
    "magenta": (255, 0, 255),
    "green": (0, 255, 0),
}


def build_prop_prompts(
    description: str,
    style: str = "game sprite",
    variants: int = 4,
    tile_size: Optional[int] = None,
    cell_px: Optional[int] = None,
    background: str = "white",
) -> dict:
    """素材（道具）提示词：N 个变体排成网格，纯色底、格内完整、像素硬边、不要文字。"""
    desc = (description or "").strip() or "a tree"
    style = (style or "").strip() or "game sprite"
    variants = max(1, min(16, int(variants)))
    cols = 2 if variants <= 4 else (3 if variants <= 9 else 4)
    rows = int(np.ceil(variants / cols))
    bg_rgb = PROP_BACKGROUNDS.get(background, PROP_BACKGROUNDS["white"])
    bg_name = {"white": "SOLID PURE WHITE", "magenta": "SOLID PURE MAGENTA",
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
def _border_color(arr: np.ndarray) -> np.ndarray:
    """取四边像素的中位色作为背景色（比写死白/洋红稳）。"""
    h, w = arr.shape[:2]
    edge = np.concatenate([
        arr[0, :, :3].reshape(-1, 3), arr[h - 1, :, :3].reshape(-1, 3),
        arr[:, 0, :3].reshape(-1, 3), arr[:, w - 1, :3].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    return np.median(edge, axis=0)


def key_background(tile: Image.Image, tolerance: int = 38) -> Image.Image:
    """抠背景：按四边中位色做 flood-fill 抠除（只删与边缘连通的背景，保留物件内部同色）。"""
    rgba = tile.convert("RGBA")
    arr = np.asarray(rgba).astype(np.int16)
    bg_rgb = _border_color(np.asarray(rgba))
    dist = np.abs(arr[..., :3] - bg_rgb[None, None, :]).max(axis=2)
    near = dist <= tolerance
    h, w = near.shape
    # 从四边洪泛：只有与边缘连通的背景像素才透明（避免挖空物件内部同色区域）
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
    out = np.array(rgba)
    out[mask] = (0, 0, 0, 0)
    # 硬边：alpha 只有 0/255（避免叠加到地块上出现灰边）
    alpha = out[..., 3]
    out[..., 3] = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    out[out[..., 3] == 0] = (0, 0, 0, 0)
    return Image.fromarray(out, "RGBA")


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
            keyed = key_background(cell, tolerance=tolerance)
            if not (np.asarray(keyed)[..., 3] > 0).any():
                idx += 1
                continue                       # 空格子（AI 少画了）直接跳过
            props[str(names[idx])] = fit_to_tile(keyed, tile_size, bottom_align=bottom_align)
            idx += 1
    return props


def prop_names(base: str, count: int) -> List[str]:
    """素材命名：`名字_1`、`名字_2`…（作为包内拼件名，导出即文件名）。"""
    stem = (base or "prop").strip() or "prop"
    return [f"{stem}_{i + 1}" for i in range(max(1, count))]
