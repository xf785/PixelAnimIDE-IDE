"""建筑类瓦片处理：抠白底 → 墙体纹理无缝 → 派生透明拼件（可旋转、叠放于地块）。

拼件全部由「无缝墙体纹理 + 程序化 alpha 掩码 + 轮廓描边」推导：
straight（直墙段，旋转 0/90）、end（端头，旋转 0/90/180/270）、
corner（外转角，旋转 4 向）、pillar（立柱）。同一纹理 + 同一掩码几何
=> 拼件之间、拼件与直墙之间构造性无缝；透明边缘可叠放在任何地块上。
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from PIL import Image

from . import art_utils
from .seamless import make_texture_seamless
from .tiles import BuildingSheet

logger = logging.getLogger("PixelFoundry.tilemap.buildings")

PIECE_NAMES = ("straight", "end", "corner", "pillar")


def _key_white(img: Image.Image, tolerance: int = 42) -> Image.Image:
    return art_utils.key_white(img, tolerance)


def opaque_ratio(tile: Image.Image, tol: int = 18) -> float:
    """瓦片中「非纯白背景」像素的比例（0~1）。

    建筑图约定：墙体组的中心格必须**填满整格**（比例≈1），立柱组中心格是白底上的
    一根柱子（比例明显 <1）。用它做底图自检，避免 AI 没按位置画时静默产出坏拼件。
    """
    arr = np.asarray(tile.convert("RGB"), dtype=np.int16)
    white = np.abs(arr - 255).max(axis=2) <= tol
    return float(1.0 - white.mean())


def _outline_color(img: Image.Image) -> Tuple[int, int, int]:
    return art_utils.outline_color(img, fallback=(0, 0, 0))


def _apply_mask(texture: Image.Image, mask: np.ndarray, outline: Tuple[int, int, int]) -> Image.Image:
    """纹理 × 掩码 + 掩码边界描边（掩码 True=建筑主体）。"""
    s = texture.size[0]
    src = np.asarray(texture.convert("RGBA")).astype(np.float32)
    out = np.zeros((s, s, 4), dtype=np.float32)
    out[mask] = src[mask]
    # 边界描边：掩码内部距边界 1px 的环
    from PIL import ImageFilter

    eroded = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
    ) > 127
    ring = mask & (~eroded)
    out[ring] = (outline[0], outline[1], outline[2], 255)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def _corner_disc_mask(s: int, radius: int, corners) -> np.ndarray:
    """全 1 掩码上在给定角切圆盘（radius 半径）。"""
    mask = np.ones((s, s), dtype=bool)
    xs = np.arange(s)
    ys = np.arange(s)
    for (cx, cy) in corners:
        d2 = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2
        mask[d2 < radius * radius] = False
    return mask


def process_building_sheet(sheet: BuildingSheet, key_tolerance: int = 42) -> dict:
    """处理建筑图：抠白底 + 墙体无缝 + 派生拼件。

    返回 {"pieces": {name: RGBA}, "core": 无缝墙体, "outline": 轮廓色,
          "opening": 门洞件, "top": 顶面件, "pillar_raw": 立柱件}。
    """
    s = sheet.wall.size
    # 1) 抠白底
    wall_core = _key_white(sheet.wall.center, key_tolerance)
    top = _key_white(sheet.top.center, key_tolerance)
    opening = _key_white(sheet.opening.center, key_tolerance)
    pillar = _key_white(sheet.pillar.center, key_tolerance)

    # 2) 墙体纹理全向无缝（纹理细节保留的偏移错位缝合）
    core = make_texture_seamless(wall_core, max_colors=32)
    outline = _outline_color(core)

    # 3) 派生拼件（同一纹理 + 掩码，构造性无缝）
    r = s // 2
    straight = _apply_mask(core, np.ones((s, s), dtype=bool), outline)
    end = _apply_mask(core, _corner_disc_mask(s, r, [(s - 1, 0), (s - 1, s - 1)]), outline)
    corner = _apply_mask(core, _corner_disc_mask(s, r, [(0, 0)]), outline)

    return {
        "pieces": {
            "straight": straight,
            "end": end,
            "corner": corner,
            "pillar": pillar,
        },
        "core": core,
        "outline": outline,
        "opening": opening,
        "top": top,
    }


def rotate_piece(piece: Image.Image, rot: int) -> Image.Image:
    """旋转拼件（0/90/180/270，逆时针）。"""
    rot = int(rot) % 4
    if rot == 0:
        return piece
    return piece.transpose(
        {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}[rot]
    )
