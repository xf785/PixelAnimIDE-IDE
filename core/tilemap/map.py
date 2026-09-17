"""瓦片地图数据模型：铺设、自动衔接（位掩码）、多地形/建筑 overlay、预览渲染与序列化。"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .autotile import (
    BIT,
    compose_art_tile,
    compose_art_tile_cached,
    compose_tile,
    dual_grid_map,
    mask_for_terrain,
    mask_from_neighbors,
)
from .buildings import rotate_piece
from .tiles import BaseTileSet

FILLED = 1
EMPTY = 0


class TileMapModel:
    """瓦片地图：多地形网格（0=空白，1..=地形 id）+ 建筑 overlay 层。

    - 单地形兼容模式：不设置 terrain_sets 时，grid 只有 0/1，
      render(center, ...) 用程序化构图（旧行为）；
    - 多地形模式：set_terrain(id, BaseTileSet) 注册各地形瓦片组，
      set_base_terrain(id) 指定基础地形；render() 用九宫格艺术片构图。
    - overlay：set_overlay(x, y, piece, rot) 在格子上叠放透明建筑拼件
      （像图层一样覆盖在地块上）。
    """

    def __init__(self, width: int, height: int, tile_size: int = 32):
        if width < 1 or height < 1:
            raise ValueError("地图尺寸必须 ≥1")
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(tile_size)
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.terrain_sets: Dict[int, BaseTileSet] = {}
        self.base_terrain: Optional[int] = None
        # 建筑 overlay：{(x,y): (拼件图, 旋转 0..3, 拼件名或 None)}
        self.overlay: Dict[Tuple[int, int], Tuple[Image.Image, int, Optional[str]]] = {}

    # ------------------------------------------------------------------ #
    def cell(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return EMPTY
        return int(self.grid[y, x])

    def set_cell(self, x: int, y: int, value: int = FILLED) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = int(value)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, value: int = FILLED) -> None:
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.width - 1, x1)
        y1 = min(self.height - 1, y1)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.grid[y, x] = int(value)

    def clear(self) -> None:
        self.grid[...] = EMPTY

    # ------------------------------------------------------------------ #
    # 多地形与 overlay
    # ------------------------------------------------------------------ #
    def set_terrain(self, terrain_id: int, base: BaseTileSet) -> None:
        self.terrain_sets[int(terrain_id)] = base

    def set_base_terrain(self, terrain_id: int) -> None:
        self.base_terrain = int(terrain_id)

    def set_overlay(
        self, x: int, y: int, piece: Image.Image, rot: int = 0, name: Optional[str] = None
    ) -> None:
        """在格子上叠放建筑拼件（name 用于序列化后按名称恢复）。"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.overlay[(x, y)] = (piece, int(rot) % 4, name)

    def remove_overlay(self, x: int, y: int) -> None:
        self.overlay.pop((x, y), None)

    def clear_overlay(self) -> None:
        self.overlay.clear()

    # ------------------------------------------------------------------ #
    def neighbors(self, x: int, y: int) -> List[List[int]]:
        """3×3 邻域（越界视为空）。"""
        return [
            [self.cell(nx, ny) for nx in range(x - 1, x + 2)]
            for ny in range(y - 1, y + 2)
        ]

    def mask(self, x: int, y: int) -> int:
        tid = int(self.grid[y, x])
        if self.terrain_sets:
            return mask_for_terrain(self.neighbors(x, y), tid, self.base_terrain)
        return mask_from_neighbors(self.neighbors(x, y))

    # ------------------------------------------------------------------ #
    def render(
        self,
        center: Optional[Image.Image] = None,
        line_color: Tuple[int, int, int] = (0, 0, 0),
        line_width: int = 1,
        mode: str = "47",
    ) -> Image.Image:
        """渲染整张地图（**任何模式都会叠加建筑 overlay 层**）。

        - 多地形模式（已注册 terrain_sets）：各地形九宫格艺术片构图 + overlay；
        - 单地形兼容模式：center 给定时程序化构图（47 / dual）；center 为 None
          时只画 overlay（建筑拼件预览）。
        """
        if self.terrain_sets:
            canvas = self._render_terrains()
        elif center is None:
            canvas = Image.new("RGBA", (self.width * self.tile_size, self.height * self.tile_size), (0, 0, 0, 0))
        elif mode == "dual":
            canvas = self._render_dual(center, line_color, line_width)
        else:
            s = self.tile_size
            canvas = Image.new("RGBA", (self.width * s, self.height * s), (0, 0, 0, 0))
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y, x]:
                        tile = compose_tile(center, self.mask(x, y), line_color, line_width)
                        canvas.paste(tile, (x * s, y * s), tile)
        self._paste_overlay(canvas)
        return canvas

    def _paste_overlay(self, canvas: Image.Image) -> None:
        """把建筑 overlay 拼件按旋转叠放到画布上。"""
        s = self.tile_size
        for (x, y), (piece, rot, _name) in self.overlay.items():
            p = rotate_piece(piece, rot)
            canvas.paste(p, (x * s, y * s), p)

    def _render_terrains(self) -> Image.Image:
        """多地形艺术片渲染（overlay 由 render 统一叠加）。"""
        s = self.tile_size
        canvas = Image.new("RGBA", (self.width * s, self.height * s), (0, 0, 0, 0))
        for y in range(self.height):
            for x in range(self.width):
                tid = int(self.grid[y, x])
                if not tid or tid not in self.terrain_sets:
                    continue
                tile = compose_art_tile_cached(self.terrain_sets[tid], self.mask(x, y))
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
        data = {
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "grid": self.grid.astype(int).tolist(),
        }
        if self.base_terrain is not None:
            data["base_terrain"] = int(self.base_terrain)
        # 建筑 overlay：按拼件名 + 旋转序列化（恢复时用 pieces 注册表还原图像）
        overlays = [
            {"x": x, "y": y, "piece": name, "rot": rot}
            for (x, y), (_img, rot, name) in self.overlay.items()
            if name
        ]
        if overlays:
            data["overlay"] = overlays
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict, pieces: Optional[Dict[str, Image.Image]] = None) -> "TileMapModel":
        """从 dict 恢复；pieces 为 {名称: 拼件图} 时同时恢复建筑 overlay。"""
        model = cls(int(data["width"]), int(data["height"]), int(data.get("tile_size", 32)))
        grid = data.get("grid") or []
        for y, row in enumerate(grid[: model.height]):
            for x, v in enumerate(row[: model.width]):
                model.grid[y, x] = int(v) if v else EMPTY
        if data.get("base_terrain") is not None:
            model.base_terrain = int(data["base_terrain"])
        for item in data.get("overlay") or []:
            name = item.get("piece")
            if pieces and name in pieces:
                model.set_overlay(
                    int(item.get("x", 0)), int(item.get("y", 0)),
                    pieces[name], int(item.get("rot", 0)), name=name,
                )
        return model

    @classmethod
    def from_json(cls, text: str, pieces: Optional[Dict[str, Image.Image]] = None) -> "TileMapModel":
        return cls.from_dict(json.loads(text), pieces=pieces)
