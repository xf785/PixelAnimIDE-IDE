"""柏林噪声大地图：把程序化地形（Perlin/FBM/河谷/山地）铺成一张大预览图。

- `generate_perlin_map()`：按 `ProceduralTerrain`（Perlin + FBM + 河谷 + 山地场）生成
  一张 `width×height` 的地形网格，映射到「水 / 平原 / 山地」三类地形 id；
- 可选把建筑包（墙 16-tile 族）沿地形边界自动铺在叠加层上（村庄墙、遗迹等），
  用来预览「地块 + 建筑」的综合效果；
- 渲染走 `compose_art_tile_cached`，因此几十万格也能秒级出图。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .autotile import canonical_mask
from .terrain_noise import ProceduralTerrain
from .map import TileMapModel
from .walls import W16_SLOTS

logger = logging.getLogger("PixelAnimIDE.tilemap.bigmap")


def generate_perlin_map(
    model: TileMapModel,
    *,
    seed: str = "pixelgifide",
    sea_level: float = 0.38,
    mountain_threshold: float = 0.55,
    water_id: int = 2,
    plain_id: int = 1,
    mountain_id: Optional[int] = 3,
    origin: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """把程序化地形写进模型（返回 kinds 网格：0=水 1=平原 2=山地）。"""
    w, h = model.width, model.height
    terrain = ProceduralTerrain(seed=seed, sea_level=sea_level, mountain_threshold=mountain_threshold)
    kinds, _masks = terrain.grid(w, h, origin=origin)
    mapping = {0: int(water_id), 1: int(plain_id)}
    if mountain_id is not None:
        mapping[2] = int(mountain_id)
    else:
        mapping[2] = int(plain_id)
    for y in range(h):
        for x in range(w):
            model.set_cell(x, y, mapping[int(kinds[y, x])])
    return kinds


def scatter_walls(
    model: TileMapModel,
    pieces: Dict[str, object],
    *,
    kinds: Optional[np.ndarray] = None,
    every: int = 9,
    seed: str = "walls",
    place_on: Sequence[int] = (1,),
) -> int:
    """在平原上按网格散布小建筑（用 16-tile 族的墙件拼成小屋），返回放置的墙格数。

    用于预览「地块 + 建筑」的叠加效果：每隔 `every` 格放一间 3×5 的小屋（带门洞），
    墙件按 4 邻接自动选型（转角/T 形/直墙/端头都由同一套规则决定）。
    """
    if not pieces:
        return 0
    w, h = model.width, model.height
    grid = np.asarray(model.grid)
    walls = np.zeros((h, w), dtype=bool)
    step = max(4, int(every))
    count = 0
    for by in range(2, h - 6, step):
        for bx in range(2, w - 6, step):
            if kinds is not None and int(kinds[by + 3, bx + 2]) not in place_on:
                continue
            if (grid[by:by + 5, bx:bx + 4] != 0).sum() and int(grid[by + 2, bx + 1]) != int(
                getattr(model, "base_terrain", 1) or 1
            ):
                continue
            for x in range(bx, bx + 4):
                walls[by, x] = True
                walls[by + 4, x] = True
            for y in range(by, by + 5):
                walls[y, bx] = True
                walls[y, bx + 3] = True
            walls[by + 4, bx + 1] = False          # 门洞
            count += 1
    if not count:
        return 0
    total = 0
    for y in range(h):
        for x in range(w):
            if not walls[y, x]:
                continue
            m = 0
            for (dy, dx, bit) in ((-1, 0, 2), (1, 0, 64), (0, -1, 8), (0, 1, 16),
                                  (-1, -1, 1), (-1, 1, 4), (1, -1, 32), (1, 1, 128)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and walls[ny, nx]:
                    m |= bit
            m = canonical_mask(m)
            name = W16_SLOTS[m & 0b1111]
            piece = pieces.get(name)
            if piece is not None:
                model.set_overlay(x, y, piece, 0, name=name)
                total += 1
    logger.info("程序化地图散布建筑：%d 间（墙格 %d）", count, total)
    return total
