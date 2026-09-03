"""瓦片地图数据模型：铺设、自动衔接（位掩码）、预览渲染与序列化。"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .autotile import BIT, compose_tile, dual_grid_map, mask_from_neighbors

FILLED = 1
EMPTY = 0


class TileMapModel:
    """地形瓦片地图（单一地形 + 空白）。绘制后按 8 邻域位掩码自动衔接。"""

    def __init__(self, width: int, height: int, tile_size: int = 32):
        if width < 1 or height < 1:
            raise ValueError("地图尺寸必须 ≥1")
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(tile_size)
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    def cell(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return EMPTY
        return int(self.grid[y, x])

    def set_cell(self, x: int, y: int, value: int = FILLED) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = FILLED if value else EMPTY

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, value: int = FILLED) -> None:
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.width - 1, x1)
        y1 = min(self.height - 1, y1)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.grid[y, x] = FILLED if value else EMPTY

    def clear(self) -> None:
        self.grid[...] = EMPTY

    # ------------------------------------------------------------------ #
    def neighbors(self, x: int, y: int) -> List[List[int]]:
        """3×3 邻域（越界视为空）。"""
        return [
            [self.cell(nx, ny) for nx in range(x - 1, x + 2)]
            for ny in range(y - 1, y + 2)
        ]

    def mask(self, x: int, y: int) -> int:
        return mask_from_neighbors(self.neighbors(x, y))

    # ------------------------------------------------------------------ #
    def render(
        self,
        center: Image.Image,
        line_color: Tuple[int, int, int] = (0, 0, 0),
        line_width: int = 1,
        mode: str = "47",
    ) -> Image.Image:
        """渲染整张地图（mode='47' 单格构图 / 'dual' 双网格 2× 分辨率）。"""
        if mode == "dual":
            return self._render_dual(center, line_color, line_width)
        s = self.tile_size
        canvas = Image.new("RGBA", (self.width * s, self.height * s), (0, 0, 0, 0))
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x]:
                    tile = compose_tile(center, self.mask(x, y), line_color, line_width)
                    canvas.paste(tile, (x * s, y * s), tile)
        return canvas

    def _render_dual(self, center: Image.Image, line_color: Tuple[int, int, int], line_width: int) -> Image.Image:
        """双网格渲染：每格按 4 个四分之一块构图（总像素尺寸与 47 模式一致）。"""
        s = self.tile_size
        half = s // 2
        pieces = self._dual_pieces(center, line_color, line_width, half)
        idx = dual_grid_map(self.grid)
        canvas = Image.new("RGBA", (self.width * s, self.height * s), (0, 0, 0, 0))
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y, x]:
                    continue
                for qy in range(2):
                    for qx in range(2):
                        kind = int(idx[2 * y + qy, 2 * x + qx])
                        piece = pieces[(qy, qx, kind)]
                        canvas.paste(piece, (x * s + qx * half, y * s + qy * half), piece)
        return canvas

    @staticmethod
    def _dual_pieces(center: Image.Image, line_color: Tuple[int, int, int], line_width: int, half: int) -> Dict:
        """生成 2×2×4 种四分之一块（kind: 0 填充 / 1 上切 / 2 左切 / 3 角盘）。"""
        s = center.size[0]
        all_sides = BIT["T"] | BIT["B"] | BIT["L"] | BIT["R"]
        pieces: Dict = {}
        # 每格的 mask：填充 / 上边界（T 空）/ 左边界（L 空）/ 外角（T+L 空）
        kind_masks = {
            0: all_sides,
            1: all_sides & ~BIT["T"],
            2: all_sides & ~BIT["L"],
            3: all_sides & ~BIT["T"] & ~BIT["L"],
        }
        box = {
            (0, 0): (0, 0, half, half),
            (0, 1): (half, 0, s, half),
            (1, 0): (0, half, half, s),
            (1, 1): (half, half, s, s),
        }
        for qy in range(2):
            for qx in range(2):
                for kind in range(4):
                    tile = compose_tile(center, kind_masks[kind], line_color, line_width)
                    l, t, r, b = box[(qy, qx)]
                    pieces[(qy, qx, kind)] = tile.crop((l, t, r, b))
        return pieces

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "grid": self.grid.astype(int).tolist(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "TileMapModel":
        model = cls(int(data["width"]), int(data["height"]), int(data.get("tile_size", 32)))
        grid = data.get("grid") or []
        for y, row in enumerate(grid[: model.height]):
            for x, v in enumerate(row[: model.width]):
                model.grid[y, x] = FILLED if v else EMPTY
        return model

    @classmethod
    def from_json(cls, text: str) -> "TileMapModel":
        return cls.from_dict(json.loads(text))
