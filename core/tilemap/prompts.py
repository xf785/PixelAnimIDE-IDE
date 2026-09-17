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
    "  - FOUR CORNER cells: the SAME texture with ONE rounded corner transition "
    "(a quarter-disc rounded boundary painted INSIDE the cell, fully opaque — "
    "the terrain fills the entire cell); the same uniform dark outline follows "
    "the rounded edge, matching the edge cells exactly.\n"
    "The background of every cell is FULLY PAINTED (no transparency, no empty "
    "areas); texture colors must be consistent across all 9 cells; keep the "
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
    cell_px: Optional[int] = None,
) -> dict:
    """生成瓦片集提示词 dict（经典 3×3）。

    description: 用户描述的纹理内容（如「草地」「石墙」）；
    style: 风格补充（如 pixel / game sprite / retro）；
    cell_px: 本次请求的单格像素（= 请求边长/3）；写入提示词以避免
        「文字说 32px、实际请求 768px（每格 256px）」的尺寸矛盾导致网格错位。
    """
    desc = (description or "").strip() or "a simple texture"
    style = (style or "").strip() or "game sprite"
    prompt = TILESET_SHEET_PROMPT.format(description=desc, style=style)
    if cell_px:
        prompt += (
            f" Each cell is exactly {int(cell_px)}x{int(cell_px)} pixels "
            f"(the whole image is exactly {int(cell_px) * 3}x{int(cell_px) * 3} pixels)."
        )
    return {
        "image_prompt": prompt,
        "negative_prompt": TILESET_NEGATIVE_PROMPT,
        "grid_rows": 3,
        "grid_cols": 3,
        "tile_size": tile_size,
        "cell_px": cell_px,
    }


# --------------------------------------------------------------------------- #
# 地块生态图（2×2 块，每块 3×3）与建筑图（2×2 块）严格提示词
# --------------------------------------------------------------------------- #
_GROUND_LAYOUT = (
    "IMAGE LAYOUT (critical): the image is ONE square divided into an EXACT 2x2 grid "
    "of blocks; every block is an EXACT 3x3 grid of 9 equal square cells — that is "
    "6 columns x 6 rows of equal cells{cell_note}. Perfectly aligned to the image edges, "
    "NO grid lines, NO gaps, NO borders between cells, NO text, NO numbers, NO labels, "
    "NO watermark. Every cell is FULLY PAINTED (opaque, no transparency, no empty "
    "areas) and all cells share ONE limited solid color palette.\n"
)


def _ground_base_block(base: str) -> str:
    return (
        f"- BLOCK 1 (top-left) = BASE TERRAIN '{base}': all 9 cells show the SAME plain "
        f"seamless '{base}' ground texture (same colors, same pattern phase, seamless in "
        f"every direction, no rim, no border band, no objects). The CENTER cell must be a "
        f"clean, perfectly tileable '{base}' texture. This block is the reference texture "
        f"for the whole ecosystem.\n"
        f"    * CRITICAL: this is the ONLY block without any boundary/rim/outline band — the "
        f"tool locates it automatically by that plainness, so it must contain NO pebbles, "
        f"rocks, decals, blobs or edge band anywhere in the block; if this block is not "
        f"unmistakably plain, the whole tileset is unusable."
    )


def _ground_feature_block(index: int, position: str, feature: str, base: str) -> str:
    """特征块（3×3）：整块拼成一个带深色边缘的圆角团块（参考示意图）。

    几何契约：边界的条带位于边格的**外侧约 1/4**（1/5~1/3 之间），角格外侧
    1/4 区域为基础地形 + 圆弧过渡 —— 算法按「同侧四分之一块」拼接，因此条带
    在格内多深都要求各格严格一致。
    """
    return (
        f"- BLOCK {index} ({position}) = FEATURE '{feature}': the 9 cells together form "
        f"ONE rounded blob of '{feature}' sitting on the base terrain '{base}', like a "
        f"3x3 autotile template of that blob. The blob's boundary band lies in the OUTER "
        f"part of the outer cells: about the outer 1/4 of the cell (anywhere between 1/5 "
        f"and 1/3, but the SAME depth for all outer cells).\n"
        f"    * CENTER cell: 100% pure '{feature}' (no base terrain).\n"
        f"    * TOP edge cell: the OUTER (top) band — about the top 1/4 of the cell — is "
        f"pure base terrain '{base}'; the remaining lower part is '{feature}'; the "
        f"'{feature}' rim/outline runs along that boundary.\n"
        f"    * BOTTOM edge cell: mirror of the top one — '{base}' in the outer (bottom) "
        f"band, '{feature}' above it.\n"
        f"    * LEFT edge cell: '{base}' in the outer (left) band, '{feature}' in the rest.\n"
        f"    * RIGHT edge cell: '{base}' in the outer (right) band, '{feature}' in the rest.\n"
        f"    * FOUR CORNER cells: the OUTER corner quarter region (facing that block "
        f"corner of the 2x2 layout) is base terrain '{base}', rounded with a "
        f"quarter-circle arc of radius about 1/4 of the cell; the rest of the cell is "
        f"'{feature}' and its rim follows the arc.\n"
        f"    * Feature rendering: give '{feature}' a clear darker 1-2 px outline (rim) "
        f"wherever it meets the base terrain, plus simple 2-3 tone shading and internal "
        f"texture detail (e.g. ripples, facets, highlights) so it reads as a real object "
        f"on the ground.\n"
        f"    * The '{base}' parts of these 9 cells must be drawn EXACTLY like BLOCK 1 "
        f"(same colors, same texture, same scale) so the blob joins the ground without "
        f"any seam or color shift."
    )


def build_ecosystem_prompts(
    description: str,
    features: dict,
    style: str = "game sprite",
    tile_size: Optional[int] = None,
    cell_px: Optional[int] = None,
) -> dict:
    """地块生态图提示词：一整张 2×2 网格，每块是 3×3 瓦片组（共 6×6 格）。

    description: 生态基础地形（如「草地」）；
    features: {特征名: 特征描述}（默认 水塘/稀疏草地/岩石，对应右上/左下/右下块）；
    cell_px: 单格像素（应与实际请求边长/6 一致，写进提示词避免文字与图像尺寸矛盾）。

    几何契约（与裁切/构图算法严格对应）：
    - 左上块 = 纯基础地形（9 格同一无缝纹理）；
    - 特征块 = 整块拼成一个带深色描边的圆角团块：边格的外侧约 1/4 条带为基础地形、
      角格外侧 1/4 区域为基础地形 + 圆弧；算法按「同侧四分之一块」拼接，
      因此条带深度在各格必须一致（不能有的格 1/4、有的格 1/2）。
    """
    base = (description or "").strip() or "grass"
    style = (style or "").strip() or "game sprite"
    feats = dict(features) if features else {}
    if not feats:
        feats = {
            "水塘": "a pond of clear water",
            "稀疏草地": "sparse patchy grass",
            "岩石": "a rocky outcrop",
        }
    positions = ["top-right", "bottom-left", "bottom-right"]
    feature_list = [(n, d) for n, d in list(feats.items())[:3]]
    while len(feature_list) < 3:
        feature_list.append((base, f"another patch of {base}"))
    cell_note = (
        f", each cell exactly {int(cell_px)}x{int(cell_px)} pixels "
        f"(the whole image is exactly {int(cell_px) * 6}x{int(cell_px) * 6} pixels)"
        if cell_px
        else ""
    )
    prompt = (
        f"Create ONE single pixel-art ecosystem tileset image in {style} style, for the "
        f"terrain ecosystem '{base}'. Style reference: hand-pixelled game tileset with "
        f"crisp hard edges, a dark outline around each feature blob, simple 2-3 tone "
        f"shading, and readable texture detail.\n"
        + _GROUND_LAYOUT.format(cell_note=cell_note)
        + "Block layout (row-major, 2 columns x 2 rows):\n"
        + _ground_base_block(base) + "\n"
        + _ground_feature_block(2, positions[0], feature_list[0][1], base) + "\n"
        + _ground_feature_block(3, positions[1], feature_list[1][1], base) + "\n"
        + _ground_feature_block(4, positions[2], feature_list[2][1], base) + "\n"
        + "Style rules: clean hard pixel edges, no anti-aliasing, no gradients, no grid "
        "lines or frames between cells, no text; features keep their dark rim and "
        "shading (this is a tileset, not a flat color chart); the same base-terrain "
        "texture must be pixel-consistent across all 4 blocks."
    )
    return {
        "image_prompt": prompt,
        "negative_prompt": (
            TILESET_NEGATIVE_PROMPT + ", transparency, empty areas, off-center boundary, "
            "inconsistent boundary depth, misaligned grid, uneven cells, flat unshaded "
            "blobs, missing outline"
        ),
        "block_rows": 2,
        "block_cols": 2,
        "tile_size": tile_size,
        "cell_px": cell_px,
        "features": [n for n, _d in list(feats.items())[:3]],  # 仅真实特征（补齐块不入元数据）
    }


def build_building_prompts(
    description: str,
    style: str = "game sprite",
    tile_size: Optional[int] = None,
    cell_px: Optional[int] = None,
) -> dict:
    """建筑类图提示词：2×2 网格，每块 3×3（共 6×6 格），建筑外背景为纯白
    （算法抠除后透明，可叠放在地块上）。

    块布局（每块的 CENTER 格是算法直接使用的拼件来源）：
    - 左上 墙体组：中心=填满整格的墙面纹理（可左右重复，掩码据此派生直段/端头/转角），
      四边=墙体端头/收边，四角=墙体转角（外转角圆角）；
    - 右上 墙顶/顶面组：中心=墙顶顶面纹理（作 overlay 顶面件），边角=顶面过渡；
    - 左下 开口组：中心=门洞（墙体上开洞，作 overlay 门件），边角=门框/窗框；
    - 右下 立柱组：中心=独立立柱，边角=柱础/柱头。
    """
    desc = (description or "").strip() or "a stone wall"
    style = (style or "").strip() or "game sprite"
    cell_note = (
        f", each cell exactly {int(cell_px)}x{int(cell_px)} pixels "
        f"(the whole image is exactly {int(cell_px) * 6}x{int(cell_px) * 6} pixels)"
        if cell_px
        else ""
    )
    prompt = (
        f"Create ONE single pixel-art building tileset image in {style} style for the "
        f"building material '{desc}'.\n"
        "IMAGE LAYOUT (critical): the image is ONE square divided into an EXACT 2x2 grid "
        "of blocks; every block is an EXACT 3x3 grid of 9 equal square cells — that is "
        f"6 columns x 6 rows of equal cells{cell_note}. Perfectly aligned, NO grid lines, "
        "NO gaps, NO borders between cells, NO text, NO numbers, NO labels.\n"
        "Background rule: wherever there is no building part, the cell MUST be SOLID "
        "PURE WHITE (#FFFFFF) — it will be keyed out to transparency later so the piece "
        "can be overlaid on terrain. No shadows or gradients in the white areas.\n"
        "Block layout (row-major, 2 columns x 2 rows):\n"
        "- BLOCK 1 (top-left) WALL SET: the CENTER cell must be a square wall segment "
        "that FILLS THE ENTIRE CELL edge-to-edge (fully opaque, no white margin inside "
        "the cell) and repeats horizontally with no visible seam; the 4 edge cells are "
        "wall end caps / top-bottom caps of the same wall; the 4 corner cells are outer "
        "wall corners of the same wall.\n"
        "- BLOCK 2 (top-right) WALL TOP SET: CENTER = the wall's top surface texture as "
        "seen from above (fills its cell); edge/corner cells = top-surface variants.\n"
        "- BLOCK 3 (bottom-left) OPENING SET: CENTER = a doorway opening cut into the "
        "same wall (the wall fills its cell, with a dark doorway hole of at most 2/3 of "
        "the cell width); edge/corner cells = door/window frames on the same wall.\n"
        "- BLOCK 4 (bottom-right) PILLAR SET: CENTER = one free-standing pillar of the "
        "same material, centred in its cell with pure white around it; edge/corner "
        "cells = pillar base/capital variants.\n"
        "Keep the whole material family consistent (same palette, same lighting) across "
        "all 4 blocks; clean hard pixel edges, no anti-aliasing."
    )
    return {
        "image_prompt": prompt,
        "negative_prompt": (
            TILESET_NEGATIVE_PROMPT
            + ", transparent background, missing white background, gradients, shadows"
        ),
        "block_rows": 2,
        "block_cols": 2,
        "tile_size": tile_size,
        "cell_px": cell_px,
    }
