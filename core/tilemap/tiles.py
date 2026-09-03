"""3×3 基础九宫格裁切与规格归一。

AI 生图尺寸通常不是 3 的整倍数；「自适应规格」策略：
1. 以整图短边为基准计算单格尺寸 cell = floor(min(w, h) / 3)，并向下取到偶数
   （后续四分块构图需要 2 的倍数）；
2. 在整图中心取 3*cell × 3*cell 区域（AI 常在四周留白/水印边缘，居中裁切
   更稳），等分裁出 9 张瓦片；
3. 归一化到目标尺寸（默认 32，偶数），最近邻缩放保持像素硬边。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image

from core.processing.pixelizer import resize_nearest

logger = logging.getLogger("PixelAnimIDE.tilemap.tiles")

# 九宫格语义位置（行, 列）：角 / 边 / 中心
GRID_POSITIONS = {
    "tl": (0, 0), "top": (0, 1), "tr": (0, 2),
    "left": (1, 0), "center": (1, 1), "right": (1, 2),
    "bl": (2, 0), "bottom": (2, 1), "br": (2, 2),
}

EDGE_NAMES = ("top", "bottom", "left", "right")
CORNER_NAMES = ("tl", "tr", "bl", "br")


@dataclass
class BaseTileSet:
    """处理后的 9 张基础瓦片（RGBA，同一尺寸）。"""

    size: int
    center: Image.Image
    edges: Dict[str, Image.Image] = field(default_factory=dict)    # top/bottom/left/right
    corners: Dict[str, Image.Image] = field(default_factory=dict)  # tl/tr/bl/br
    line_color: Tuple[int, int, int] = (0, 0, 0)                   # 统一边界线色
    line_width: int = 1

    def tile(self, name: str) -> Image.Image:
        """按名字取瓦片：'center' / 'top'… / 'tl'…。"""
        if name == "center":
            return self.center
        if name in self.edges:
            return self.edges[name]
        if name in self.corners:
            return self.corners[name]
        raise KeyError(f"未知瓦片: {name}")

    def all(self) -> List[Image.Image]:
        order = ["tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"]
        return [self.tile(n) for n in order]


def _snap_even(value: int, minimum: int = 8) -> int:
    value = max(minimum, int(value))
    return value - (value % 2)


def compute_cell_size(w: int, h: int, rows: int = 3, cols: int = 3) -> int:
    """按图短边自适应计算单格尺寸（偶数，≥8）。"""
    side = min(w // cols, h // rows)
    return _snap_even(side)


def crop_base_3x3(
    img: Image.Image,
    tile_size: Optional[int] = None,
) -> Tuple[List[Image.Image], int]:
    """把整图裁切成 3×3 瓦片列表（行优先，tl..br），返回 (tiles, cell)。

    tile_size 给定时按该尺寸在图中居中裁 3×3；缺省时按短边自适应。
    裁出的每格尺寸为 cell（自适应时 cell = min(w,h)/3 取偶）。
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if tile_size:
        cell = _snap_even(min(tile_size, w // 3, h // 3), minimum=4)
    else:
        cell = compute_cell_size(w, h)
    x0 = (w - cell * 3) // 2
    y0 = (h - cell * 3) // 2
    tiles: List[Image.Image] = []
    for r in range(3):
        for c in range(3):
            box = (x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell, y0 + (r + 1) * cell)
            tiles.append(rgba.crop(box))
    return tiles, cell


def to_base_set(tiles: List[Image.Image]) -> BaseTileSet:
    """9 张行优先瓦片 -> BaseTileSet（命名映射，尺寸取第一张）。"""
    if len(tiles) != 9:
        raise ValueError(f"需要 9 张瓦片，实际 {len(tiles)}")
    size = tiles[4].size
    if size[0] != size[1]:
        raise ValueError(f"瓦片必须为正方形: {size}")
    names = ["tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"]
    named = {n: t.convert("RGBA") for n, t in zip(names, tiles)}
    return BaseTileSet(
        size=size[0],
        center=named["center"],
        edges={n: named[n] for n in EDGE_NAMES},
        corners={n: named[n] for n in CORNER_NAMES},
    )


def normalize_tileset(base: BaseTileSet, target_size: int = 32) -> BaseTileSet:
    """把 BaseTileSet 全部瓦片归一化到 target_size（偶数，最近邻）。"""
    target = _snap_even(target_size, minimum=8)
    if base.size == target:
        return base
    scale = lambda im: resize_nearest(im, (target, target))
    return BaseTileSet(
        size=target,
        center=scale(base.center),
        edges={n: scale(t) for n, t in base.edges.items()},
        corners={n: scale(t) for n, t in base.corners.items()},
        line_color=base.line_color,
        line_width=base.line_width,
    )
