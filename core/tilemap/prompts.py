"""瓦片集严格提示词：把用户描述（纹理/风格）嵌入内置 3×3 九宫格生成规范。

生成的图将按 3×3 等分裁切，因此提示词必须强制：
- 一整张 3×3 网格图（9 个等大方格，禁止画格线/边框/文字）；
- 中心格：可无缝平铺的纯纹理（全向无缝）；
- 上下左右格：同纹理 + 一侧直线边界的「墙面/交界」瓦片（边界线统一风格）；
- 四角格：同纹理 + 圆角切角的「转角」瓦片；
- 纯白背景、有限纯色调色板、像素风硬边。
"""
from __future__ import annotations

from typing import Optional

TILESET_SHEET_PROMPT = (
    "Create ONE single pixel-art tileset image: an EXACT 3x3 grid of 9 equal square "
    "cells (the whole image is one square, each cell is exactly 1/3 of the width and "
    "height), NO grid lines, NO gaps, NO borders between cells, NO text, NO labels, "
    "NO watermarks. Every cell uses the SAME material/texture and the SAME limited "
    "solid color palette: '{description}', in {style} pixel-art style, clean hard "
    "edges, no anti-aliasing, no gradients, no photorealism.\n"
    "Cell layout (row-major, 3 columns x 3 rows):\n"
    "  - CENTER cell (row 1, col 1): a plain, fully SEAMLESS tileable texture — the "
    "texture must wrap perfectly both horizontally and vertically with no visible "
    "seam, no border, no vignette, no focal object.\n"
    "  - TOP / BOTTOM / LEFT / RIGHT cells: the SAME texture but ending in a straight, "
    "clean boundary along the outer side of the cell (a uniform 1-2px dark outline "
    "line runs along that boundary); the texture continues all the way to the "
    "boundary, and the cell must be horizontally tileable along the boundary "
    "direction so it can be repeated seamlessly.\n"
    "  - FOUR CORNER cells: the SAME texture with ONE rounded (quarter-disc) corner "
    "cut off (transparent/white cut region at the corner facing the cell's outer "
    "corner of the 3x3 grid); the same uniform dark outline follows the rounded "
    "cut edge, matching the edge cells exactly.\n"
    "The background of every cell is a SOLID PURE WHITE (#FFFFFF) where the texture "
    "is cut away; texture colors must be consistent across all 9 cells; keep the "
    "entire subject fully inside its cell with a small margin."
)

TILESET_NEGATIVE_PROMPT = (
    "grid lines, cell borders, gaps between cells, text, numbers, labels, watermark, "
    "inconsistent palette, different texture between cells, missing cells, scattered "
    "cells, gradients, anti-aliasing, blurry, photorealism, gray background, "
    "colored background, shading, shadows, perspective"
)


def build_tileset_prompts(
    description: str,
    style: str = "game sprite",
    tile_size: Optional[int] = None,
) -> dict:
    """生成瓦片集提示词 dict。

    description: 用户描述的纹理内容（如「草地」「石墙」）；
    style: 风格补充（如 pixel / game sprite / retro）；
    tile_size: 目标单格像素（仅作为附加指令写入，最终尺寸由裁切/缩放强制）。
    """
    desc = (description or "").strip() or "a simple texture"
    style = (style or "").strip() or "game sprite"
    prompt = TILESET_SHEET_PROMPT.format(description=desc, style=style)
    if tile_size:
        prompt += (
            f" Each cell must be drawn at exactly {int(tile_size)}x{int(tile_size)} "
            f"pixels (EXACT {int(tile_size)}x{int(tile_size)} pixel grid per cell)."
        )
    return {
        "image_prompt": prompt,
        "negative_prompt": TILESET_NEGATIVE_PROMPT,
        "grid_rows": 3,
        "grid_cols": 3,
        "tile_size": tile_size,
    }
